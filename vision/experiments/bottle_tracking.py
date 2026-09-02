"""First bottle tracker: a COCO object detector (EfficientDet-Lite via MediaPipe) plus hand-built tracking rules.

Superseded by ``markos_vision.perception.bottle_yolo`` (segmentation gives the outline and a stable track ID directly).
Kept as a record of the earlier approach; the hover and calibration tools no longer use it.
"""

import math

import cv2
import mediapipe as mp
import numpy as np

from markos_vision.config import EFFICIENTDET_MODEL as MODEL_PATH
from markos_vision.config import SCENE_REFERENCE as SCENE_PATH
from markos_vision.perception.bottle_types import BottleLibrary, descriptor
from markos_vision.perception.cap_finder import find_cap

# COCO detectors often call a bottle a vase or a wine glass. That's useful for
# keeping hold of a bottle already being tracked, but far too loose to start
# one from -- it's what made the robot and other tall things look like bottles.
CATEGORY_WEIGHT = {"bottle": 1.0, "vase": 0.8, "wine glass": 0.8}
START_CATEGORIES = ("bottle",)
CANDIDATE_SCORE = 0.15  # weakest detection worth considering once a bottle is already being tracked
START_SCORE = 0.50  # to start a track with nothing else to go on...
START_SCORE_WITH_SCENE = 0.35  # ...or this, once the empty scene is known and static things are ruled out
START_ASPECT = 2.0  # box height / width to start on: real bottles run about 2.5 to 4.5
MIN_UPRIGHT_RATIO = 1.3  # looser once tracking, since boxes wobble and a hand may cover the bottle
HIT_GAIN = 0.4  # how far one detection pulls the hit-rate up toward 1
MISS_DECAY = 0.9  # what one missed frame does to it; ~14 misses in a row drop a solid track
LIVE_ON = 0.5  # hit-rate at which a track is trusted...
LIVE_OFF = 0.15  # ...and the level below which it's given up on
GATE_FRAC = 0.8  # a detection counts as the same bottle if it lands within this many box-heights of the track
BOX_SMOOTH = 0.5  # how far each new detection pulls the tracked box
ZOOM_PX = 320  # the model looks at 320x320, so a crop this size is seen at full resolution
REJECT_SHOW_SCORE = 0.4  # only strong "bottle" detections that get turned down are worth showing
SCENE_W, SCENE_H = 160, 120  # the scene reference is kept this small: it only needs to notice what changed
CHANGE_THRESH = 25  # gray-level change that counts as "something is different here"
MIN_NOVELTY = 0.30  # share of a candidate's box that must be new, not part of the empty scene
RECHECK_EVERY = 8  # frames between re-verifying a track against the enrolled bottles
RECHECK_STRIKES = 2  # consecutive failed re-checks before a track is dropped
REJECT_MEMORY_FRAMES = 20  # a box turned down this recently isn't re-examined every frame
REJECT_SAME_IOU = 0.6
MAX_NECK_RATIO = 0.75  # neck-to-body width; real bottles are ~0.3-0.5, a straight column ~1.0


def bottle_from_box(x, y, w, h, score):
    """Where the basic points of an upright bottle are, from its box: top of
    the box, the base on the desk, and the vertical axis between them (a
    bottle is close to symmetric). The cap itself is found more precisely by
    cap_finder once the bottle is known. Returns None for a box that isn't a
    bottle standing up."""
    if h < MIN_UPRIGHT_RATIO * w:
        return None
    cx = x + w / 2
    return {
        "cx": cx,
        "cy": y + h / 2,
        "box": (x, y, w, h),
        "top": (cx, y),
        "base": (cx, y + h),
        "score": score,
    }


def _iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    iw = min(ax0 + aw, bx0 + bw) - max(ax0, bx0)
    ih = min(ay0 + ah, by0 + bh) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter)


def _small_gray(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (SCENE_W, SCENE_H), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (5, 5), 0).astype(np.float32)


class BottleTracker:
    """Finds a bottle with a pretrained COCO object detector, then decides
    whether to believe it. A detector can't tell a bottle from a
    bottle-shaped robot, so a detection has to get through several checks
    before it becomes a track: it must be new in the scene (learn_scene), and
    once bottle types are enrolled it has to look like one of them and not
    like anything marked as a false alarm. And because a detector on a small
    bottle flickers, the track keeps a hit-rate that rises on a detection and
    decays on a miss, instead of trusting single frames."""

    def __init__(self, model_path=MODEL_PATH, detect=None, library=None):
        # `detect` is a stand-in for the real model, so the tracking logic can
        # be tested without one: frame -> [((x, y, w, h), score, category), ...]
        if detect is None:
            options = mp.tasks.vision.ObjectDetectorOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                category_allowlist=list(CATEGORY_WEIGHT),
                score_threshold=CANDIDATE_SCORE,
                max_results=10,
            )
            self._model = mp.tasks.vision.ObjectDetector.create_from_options(options)
            detect = self._model_detect
        self._detect = detect
        self.shape_check = True  # the neck test; switchable in case a bottle's outline doesn't cut out well
        self.library = library if library is not None else BottleLibrary()
        self.rejected = []  # (box, reason) for strong bottle detections that were turned down, for drawing
        self._recent_rejects = []  # (box, reason, frame index)
        self._track = None
        self._reference = None
        self._changed = None
        self._frame = None
        self._frame_shape = (480, 640)
        self._frame_idx = 0
        self.stats = {"frames": 0, "full_hits": 0, "zoom_rescues": 0, "live_frames": 0, "fresh_frames": 0, "drops": 0}

    # ── the empty-scene reference ──────────────────────────────────────────

    @property
    def has_scene(self):
        return self._reference is not None

    def learn_scene(self, frames_bgr):
        self._reference = np.mean([_small_gray(f) for f in frames_bgr], axis=0)

    def forget_scene(self):
        self._reference = None

    def save_scene(self, path=SCENE_PATH):
        if self._reference is not None:
            np.savez(path, reference=self._reference)

    def load_scene(self, path=SCENE_PATH):
        try:
            self._reference = np.load(path)["reference"].astype(np.float32)
            return True
        except (FileNotFoundError, KeyError):
            return False

    def _changed_mask(self, frame_bgr):
        d = _small_gray(frame_bgr) - self._reference
        d -= np.median(d)  # auto-exposure shifts the whole picture; that isn't a new object
        return np.abs(d) > CHANGE_THRESH

    def _novelty(self, box):
        x, y, w, h = box
        fh, fw = self._frame_shape
        sx, sy = SCENE_W / fw, SCENE_H / fh
        region = self._changed[int(y * sy):int((y + h) * sy) + 1, int(x * sx):int((x + w) * sx) + 1]
        return float(region.mean()) if region.size else 0.0

    # ── what counts as a bottle ────────────────────────────────────────────

    def _verify(self, box):
        """Colour first (cheap), then shape. Colour can't tell a white robot
        from a white bottle, but the outline can: a bottle narrows to a neck,
        a column doesn't."""
        verdict = self.library.classify(descriptor(self._frame, box), box)
        verdict["neck_ratio"] = None
        if self.shape_check and verdict["verdict"] in ("ok", "unverified"):
            ratio = find_cap(self._frame, box)["neck_ratio"]
            verdict["neck_ratio"] = ratio
            if ratio is not None and ratio > MAX_NECK_RATIO:
                verdict = {**verdict, "verdict": "not bottle-shaped", "type": None}
        return verdict

    @staticmethod
    def _reason(verdict):
        return {
            "no match": f"not an enrolled bottle ({verdict['sim']:.2f})",
            "known non-bottle": f"known non-bottle ({verdict['neg_sim']:.2f})",
            "not bottle-shaped": f"not bottle-shaped (neck {verdict['neck_ratio']:.2f})"
                                  if verdict.get("neck_ratio") is not None else "not bottle-shaped",
        }.get(verdict["verdict"])

    def enrol(self, name, height_cm, cap_height_cm, diameter_cm=None):
        """Adds what the tracked bottle looks like right now to a type."""
        t = self._track
        if t is None or self._frame is None:
            return None
        desc = descriptor(self._frame, t["box"])
        if desc is None:
            return None
        # Measure the bottle's true height/width from a clean cut-out now, while
        # the cap is easy to see, so a hard-to-see cap can be placed later.
        cut = find_cap(self._frame, t["box"], cap_height_cm / height_cm)
        aspect = cut["height_px"] / cut["body_w"] if cut["method"] == "grabcut" and cut["body_w"] > 0 else None
        bottle_type = self.library.add_sample(name, height_cm, cap_height_cm, t["box"], desc, aspect, diameter_cm=diameter_cm)
        t["verdict"] = self._verify(t["box"])
        return bottle_type

    def mark_not_a_bottle(self):
        """The tracked 'bottle' is a false alarm (the robot, say): remember
        what it looks like as a non-bottle, and drop the track."""
        t = self._track
        if t is None or self._frame is None:
            return False
        desc = descriptor(self._frame, t["box"])
        if desc is None:
            return False
        self.library.add_negative(desc)
        self._remember_reject(t["box"], "known non-bottle")
        self._track = None
        return True

    def _remember_reject(self, box, reason):
        self._recent_rejects.append((box, reason, self._frame_idx))

    def _recently_rejected(self, box):
        self._recent_rejects = [r for r in self._recent_rejects if self._frame_idx - r[2] <= REJECT_MEMORY_FRAMES]
        for old_box, reason, _ in self._recent_rejects:
            if _iou(old_box, box) >= REJECT_SAME_IOU:
                return reason
        return None

    # ── detection ──────────────────────────────────────────────────────────

    def _model_detect(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        out = []
        for d in result.detections:
            b, c = d.bounding_box, d.categories[0]
            out.append(((b.origin_x, b.origin_y, b.width, b.height),
                        c.score * CATEGORY_WEIGHT.get(c.category_name, 0.0), c.category_name))
        return out

    def _reject(self, box, score, category, reason):
        if category in START_CATEGORIES and score >= REJECT_SHOW_SCORE:
            self.rejected.append((box, reason))

    def _pick(self, detections, offset=(0, 0)):
        """The detection to use this frame, as (box, score, verdict), or
        None: near the current track if there is one, else the most
        convincing one that gets through the (much stricter) checks for
        starting a track. Boxes are shifted by offset to full-frame
        coordinates."""
        t = self._track
        start_score = START_SCORE_WITH_SCENE if self._reference is not None else START_SCORE
        best = None
        for (x, y, w, h), score, category in detections:
            if score < CANDIDATE_SCORE:
                continue
            box = (x + offset[0], y + offset[1], w, h)
            verdict = None
            if t is not None:
                if h < MIN_UPRIGHT_RATIO * w:
                    continue
                tx, ty, tw, th = t["box"]
                if math.hypot(box[0] + w / 2 - (tx + tw / 2), box[1] + h / 2 - (ty + th / 2)) > GATE_FRAC * th:
                    continue
            else:
                if category not in START_CATEGORIES or score < start_score:
                    continue
                if h < START_ASPECT * w:
                    self._reject(box, score, category, "not upright")
                    continue
                if self._changed is not None and self._novelty(box) < MIN_NOVELTY:
                    self._reject(box, score, category, "already in the scene")
                    continue
                earlier = self._recently_rejected(box)
                if earlier is not None:
                    self._reject(box, score, category, earlier)
                    continue
                verdict = self._verify(box)
                reason = self._reason(verdict)
                if reason is not None:
                    self._remember_reject(box, reason)
                    self._reject(box, score, category, reason)
                    continue
            if best is None or score > best[1]:
                best = (box, score, verdict)
        return best

    def _zoom_pass(self, frame_bgr):
        """Full-resolution look around where the bottle was, for when the
        whole-frame pass came up empty."""
        fh, fw = frame_bgr.shape[:2]
        tx, ty, tw, th = self._track["box"]
        x0 = int(min(max(tx + tw / 2 - ZOOM_PX / 2, 0), max(fw - ZOOM_PX, 0)))
        y0 = int(min(max(ty + th / 2 - ZOOM_PX / 2, 0), max(fh - ZOOM_PX, 0)))
        crop = frame_bgr[y0:y0 + ZOOM_PX, x0:x0 + ZOOM_PX]
        return self._pick(self._detect(crop), offset=(x0, y0))

    def find(self, frame_bgr):
        """Dict for the tracked upright bottle -- cx, cy, box, top, base,
        score, hit_rate, missed (frames since a real detection; 0 = seen this
        frame), verified and type (the enrolled bottle it matches, or None)
        -- or None. Anything that needs the bottle to be where it currently
        is, not where it was, should check missed == 0."""
        self.stats["frames"] += 1
        self._frame_idx += 1
        self.rejected = []
        self._frame = frame_bgr
        self._frame_shape = frame_bgr.shape[:2]
        # Only a track that's just starting needs the novelty check.
        self._changed = self._changed_mask(frame_bgr) if (self._reference is not None and self._track is None) else None

        hit = self._pick(self._detect(frame_bgr))
        if hit is not None:
            self.stats["full_hits"] += 1
        elif self._track is not None:
            hit = self._zoom_pass(frame_bgr)
            if hit is not None:
                self.stats["zoom_rescues"] += 1
        result = self._update(hit)
        self._recheck()
        return result if self._track is not None else None

    def _recheck(self):
        """A track that started as the right thing can drift onto the wrong
        one (a moving robot passing behind the bottle, say), so look again
        now and then and let go of it if it no longer matches."""
        t = self._track
        if t is None or t["missed"] != 0 or self._frame_idx % RECHECK_EVERY:
            return
        verdict = self._verify(t["box"])
        if self._reason(verdict) is None:
            t["strikes"], t["verdict"] = 0, verdict
            return
        t["strikes"] += 1
        if t["strikes"] >= RECHECK_STRIKES:
            self._remember_reject(t["box"], "no longer matches")
            self._track = None
            self.stats["drops"] += 1

    def _update(self, hit):
        t = self._track
        if hit is not None:
            box, score, verdict = hit
            if t is None:
                t = self._track = {"box": box, "hits": HIT_GAIN, "live": False, "strikes": 0, "verdict": verdict}
            else:
                t["box"] = tuple(o + BOX_SMOOTH * (n - o) for o, n in zip(t["box"], box))
                t["hits"] += HIT_GAIN * (1 - t["hits"])
            t["score"], t["missed"] = score, 0
        elif t is not None:
            t["hits"] *= MISS_DECAY
            t["missed"] += 1
            if t["hits"] < LIVE_OFF:
                self._track = None
                self.stats["drops"] += 1
                return None

        if t is None:
            return None
        if t["hits"] >= LIVE_ON:
            t["live"] = True
        if not t["live"]:
            return None
        bottle = bottle_from_box(*t["box"], t["score"])
        if bottle is None:
            return None
        self.stats["live_frames"] += 1
        self.stats["fresh_frames"] += t["missed"] == 0
        verdict = t["verdict"]
        return {**bottle, "hit_rate": t["hits"], "missed": t["missed"],
                "verified": verdict["verdict"] == "ok", "type": verdict["type"], "sim": verdict["sim"]}

    def summary(self):
        s = self.stats
        n = max(s["frames"], 1)
        return (f"{s['frames']} frames: tracked {s['live_frames'] / n:.0%}, seen fresh {s['fresh_frames'] / n:.0%} "
                f"(whole-frame hits {s['full_hits'] / n:.0%}, zoom rescues {s['zoom_rescues'] / n:.0%}), "
                f"{s['drops']} drop(s)")
