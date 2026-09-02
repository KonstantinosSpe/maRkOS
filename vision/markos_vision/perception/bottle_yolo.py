"""Finds the bottle in a camera frame with YOLO11 segmentation and decides whether to believe it.

``YoloBottleFinder.find`` runs the model, tracks each detection with a stable ID, and accepts only a detection that
looks like an enrolled bottle type, has a bottle's neck-and-body outline and is not a known false alarm. The outline gives
the cap position and the foot of the bottle directly. A recognised bottle keeps its identity while it stays put (the
identity latch), because the arm's shadow over it would otherwise make it look unknown to the colour check.
"""

import math
import os
import time

import cv2
import numpy as np

from markos_vision.config import BOTTLE_REASONS_LOG as LOG_PATH
from markos_vision.config import YOLO_WEIGHTS as WEIGHTS
from markos_vision.perception.bottle_types import BottleLibrary, descriptor
from markos_vision.perception.cap_finder import DEFAULT_CAP_FRAC, analyse_mask

LOG_MAX_BYTES = 200_000
BOTTLE_CLASS = 39  # COCO
CONF = 0.25
IMGSZ = 480  # a 640x480 webcam frame, in a multiple of 32
# Neck-to-body width of the outline. A clean bottle is ~0.3-0.5 and a straight column ~1.0, but the model's
# outline is a soft blob that fattens a thin neck: a real, nearly straight-sided bottle measured 0.84-0.89 in
# use. So a fixed cut-off in the middle flickers. Instead an enrolled bottle is judged against its own ratio
# (measured when it was enrolled), and a bottle nobody has enrolled only fails if it's a true column.
NECK_LIMIT_DEFAULT = 0.95
NECK_TOLERANCE = 0.15  # how much more column-like than its enrolled self a bottle may look
NECK_LIMIT_CEILING = 0.97
NECK_SMOOTH_FRAMES = 7  # recent per-frame ratios a verdict is smoothed over
NECK_HISTORY = 30  # per-track ratios kept, which is what an enrolment is calibrated from
RECHECK_EVERY = 8  # frames between re-verifying a tracked bottle
RECHECK_STRIKES = 2  # consecutive failed re-checks before a tracked bottle is dropped
HOLD_FRAMES = 10  # keep reporting the target through a brief dropout
REJECT_SHOW_CONF = 0.4  # only strong detections that get turned down are worth showing
# A recognised bottle keeps its identity while it stays put: the arm hovering over it casts a shadow that drops its
# colour similarity to an enrolled type from ~0.9 to ~0.5, which made the finder call it "unknown" and the hover
# program stop sending the arm halfway. Moved (or replaced) bottles still have to be recognised afresh.
LATCH_MOVE_FRAC = 0.12  # how far the foot may drift from where it was last recognised, as a fraction of its height
LATCH_MIN_PX = 12.0
LATCH_SECONDS = 90.0


class YoloBottleFinder:
    """YOLO11 segmentation finds bottles, outlines each one and tracks it
    with a stable ID, which is most of what used to be hand-built here. What's
    left is deciding which detections to believe: it has to look like an
    enrolled bottle type (once any exist), have a bottle's neck-and-body
    outline, and not be something you've marked as a false alarm. The
    outline the model returns also gives the cap position directly."""

    def __init__(self, weights=WEIGHTS, library=None, imgsz=IMGSZ, run_model=None):
        # `run_model` stands in for YOLO so the decision logic can be tested
        # without it: frame -> [{"id", "box": (x, y, w, h), "conf", "polygon"}, ...]
        self._run_model = run_model
        if run_model is None:
            from ultralytics import YOLO, settings
            settings.update({"sync": False})  # no telemetry, no account
            self._model = YOLO(weights)
            self._run_model = self._yolo_instances
        self.imgsz = imgsz
        self.library = library if library is not None else BottleLibrary()
        self.shape_check = True  # the neck test; switchable in case a bottle's outline doesn't cut out well
        self.rejected = []  # (box, reason) for strong detections that were turned down, for drawing
        self._banned = set()  # track IDs marked as "not a bottle"
        self._neck_hist = {}  # per track ID: recent neck ratios, frame by frame
        self._info = {}  # per track ID: last verdict, strikes, when it was last checked
        self._target_id = None
        self._latch = None  # the last recognised bottle: its track ID, type, where its foot was and when
        self._last = None
        self._missed = 0
        self._frame = None
        self._frame_idx = 0
        self.stats = {"frames": 0, "target_frames": 0, "fresh_frames": 0}

    def _yolo_instances(self, frame_bgr):
        r = self._model.track(frame_bgr, persist=True, classes=[BOTTLE_CLASS], conf=CONF, imgsz=self.imgsz,
                              retina_masks=True, tracker="bytetrack.yaml", verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
        polygons = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
        out = []
        for (x0, y0, x1, y1), conf, tid, poly in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), ids, polygons):
            out.append({"id": tid, "box": (x0, y0, x1 - x0, y1 - y0), "conf": conf, "polygon": poly})
        return out

    # ── what counts as a bottle ────────────────────────────────────────────

    @staticmethod
    def _mask_of(inst):
        """The outline as a boolean image cropped to the box, plus its offset."""
        poly = inst["polygon"]
        if poly is None or len(poly) < 3:
            return None
        x, y, w, h = inst["box"]
        x0, y0 = int(max(0, np.floor(x))), int(max(0, np.floor(y)))
        x1, y1 = int(np.ceil(x + w)) + 1, int(np.ceil(y + h)) + 1
        canvas = np.zeros((max(y1 - y0, 1), max(x1 - x0, 1)), np.uint8)
        cv2.fillPoly(canvas, [np.round(poly - np.array([x0, y0])).astype(np.int32)], 255)
        return canvas > 0, x0, y0

    def _measure(self, inst, bottle_type=None):
        """Cap and neck-ratio measurements for one detection."""
        x, y, w, h = inst["box"]
        cap_frac = bottle_type["cap_height_cm"] / bottle_type["height_cm"] if bottle_type else DEFAULT_CAP_FRAC
        prior = bottle_type.get("silhouette_aspect") if bottle_type else None
        cut = self._mask_of(inst)
        if cut is None:
            return analyse_mask(None, 0, 0, 1.0, inst["box"], cap_frac, prior)
        mask, x0, y0 = cut
        return analyse_mask(mask, x0, y0, 1.0, inst["box"], cap_frac, prior, "yolo mask")

    def _verdict(self, inst):
        """{'verdict', 'type', 'sim', 'neck_ratio', 'cap'} -- colour first
        (cheap), then shape. Colour can't tell a white robot from a white
        bottle, but the outline can: a bottle narrows to a neck, a column
        doesn't."""
        v = self.library.classify(descriptor(self._frame, inst["box"]), inst["box"])
        cap = self._measure(inst, v["type"])
        # One frame's ratio is noisy (it moves a few hundredths as the outline wobbles), so judge the median of the
        # recent frames for this track, with this frame included.
        ratio = cap["neck_ratio"]
        recent = self._neck_hist.get(inst["id"], [])[-NECK_SMOOTH_FRAMES:]
        if ratio is not None and recent:
            ratio = float(np.median(recent + [ratio]))
        v = {**v, "neck_ratio": ratio, "cap": cap}
        limit = NECK_LIMIT_DEFAULT
        if v["type"] is not None and v["type"].get("neck_ratio") is not None:
            limit = min(NECK_LIMIT_CEILING, v["type"]["neck_ratio"] + NECK_TOLERANCE)
        v["neck_limit"] = limit
        if self.shape_check and v["verdict"] in ("ok", "unverified", "unknown bottle") \
                and cap["neck_ratio"] is not None and cap["neck_ratio"] > limit:
            v = {**v, "verdict": "not bottle-shaped", "type": None}
        self._log(inst, v)
        return v

    @staticmethod
    def _log(inst, v):
        """One text line per verdict, numbers only, so what the finder decided
        and why can be read back afterwards without anyone having to describe
        the screen. Never images."""
        try:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > LOG_MAX_BYTES:
                os.replace(LOG_PATH, LOG_PATH + ".old")
            neck = "n/a" if v["neck_ratio"] is None else f"{v['neck_ratio']:.2f}"
            with open(LOG_PATH, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} id={inst['id']} conf={inst['conf']:.2f} verdict={v['verdict']!r} "
                        f"sim={v['sim']:.2f} neg={v['neg_sim']:.2f} neck={neck}/{v.get('neck_limit', 0):.2f} "
                        f"box={inst['box'][2]:.0f}x{inst['box'][3]:.0f} cap_via={v['cap']['method']!r}\n")
        except OSError:
            pass

    @staticmethod
    def _reason(v):
        """Why a detection is turned down, or None if it's kept. Only a
        non-bottle is turned down: a bottle that isn't enrolled is kept, as
        'unknown'."""
        return {
            "known non-bottle": f"known non-bottle ({v['neg_sim']:.2f})",
            "not bottle-shaped": f"column-like (neck {v['neck_ratio']:.2f} > {v['neck_limit']:.2f})"
                                  if v.get("neck_ratio") is not None else "not bottle-shaped",
        }.get(v["verdict"])

    def _assess(self, inst):
        """The current verdict for a detection, cached per track ID and
        refreshed every few frames. A new track is judged on its first look;
        an accepted one has to fail twice in a row to be dropped, so one bad
        frame doesn't lose it."""
        tid = inst["id"]
        info = self._info.get(tid) if tid is not None else None
        if info is not None and self._frame_idx - info["checked_at"] < RECHECK_EVERY:
            return info
        v = self._verdict(inst)
        failed = self._reason(v) is not None
        if info is None:
            info = {"ok": not failed, "strikes": 0}
        elif failed:
            info["strikes"] += 1
            info["ok"] = info["ok"] and info["strikes"] < RECHECK_STRIKES
        else:
            info["strikes"], info["ok"] = 0, True
        info.update(v=v, checked_at=self._frame_idx)
        if tid is not None:
            self._info[tid] = info
        return info

    def _latch_holds(self, tid, v, foot, height):
        """True if this is the bottle recognised a moment ago, still in the same place, that only looks different now:
        same track, only 'unknown' (never a known non-bottle or a column), foot within a few percent of its height."""
        lt = self._latch
        if tid is None or tid != lt["id"] or v["verdict"] not in ("unknown bottle", "unverified"):
            return False
        if time.time() - lt["t"] > LATCH_SECONDS:
            return False
        moved = math.hypot(foot[0] - lt["foot"][0], foot[1] - lt["foot"][1])
        return moved < max(LATCH_MIN_PX, LATCH_MOVE_FRAC * height)

    # ── the loop ───────────────────────────────────────────────────────────

    def find(self, frame_bgr):
        """Dict for the target bottle -- id, box, conf, top, base, cx, cy,
        cap (from cap_finder), verified, type, sim, missed (frames since it
        was actually seen; 0 = this frame) -- or None."""
        self.stats["frames"] += 1
        self._frame_idx += 1
        self._frame = frame_bgr
        self.rejected = []

        eligible = []
        for inst in self._run_model(frame_bgr):
            if inst["id"] in self._banned:
                continue
            info = self._assess(inst)
            if info["ok"]:
                eligible.append((inst, info))
            elif inst["conf"] >= REJECT_SHOW_CONF:
                self.rejected.append((inst["box"], self._reason(info["v"]) or "rejected"))

        chosen = None
        if eligible:
            # Stick with the bottle already being followed while it's still there.
            chosen = next((e for e in eligible if e[0]["id"] == self._target_id and self._target_id is not None), None)
            chosen = chosen or max(eligible, key=lambda e: e[0]["conf"])

        if chosen is not None:
            inst, info = chosen
            self._target_id = inst["id"]
            self._missed = 0
            v = info["v"]
            # The cap is worth refreshing every frame: it's cheap, and it's what the arm goes for.
            cap = self._measure(inst, v["type"])
            if inst["id"] is not None and cap["neck_ratio"] is not None:
                hist = self._neck_hist.setdefault(inst["id"], [])
                hist.append(cap["neck_ratio"])
                del hist[:-NECK_HISTORY]
            x, y, w, h = inst["box"]
            verified, btype, latched = v["verdict"] == "ok", v["type"], False
            foot = (x + w / 2, y + h)
            if verified:
                self._latch = {"id": inst["id"], "type": btype, "foot": foot, "t": time.time()}
            elif self._latch is not None and self._latch_holds(inst["id"], v, foot, h):
                verified, btype, latched = True, self._latch["type"], True
            elif self._latch is not None and inst["id"] != self._latch["id"]:
                self._latch = None
            self._last = {
                "id": inst["id"], "box": inst["box"], "conf": inst["conf"], "cx": x + w / 2, "cy": y + h / 2,
                "top": (x + w / 2, y), "base": (x + w / 2, y + h), "cap": cap, "polygon": inst["polygon"],
                "verified": verified, "type": btype, "sim": v["sim"], "latched": latched,
            }
            self.stats["target_frames"] += 1
            self.stats["fresh_frames"] += 1
            return {**self._last, "missed": 0}

        if self._last is not None and self._missed < HOLD_FRAMES:
            self._missed += 1
            self.stats["target_frames"] += 1
            return {**self._last, "missed": self._missed}
        self._last = None
        self._target_id = None
        return None

    # ── teaching it ────────────────────────────────────────────────────────

    def enrol(self, name, height_cm, cap_height_cm, diameter_cm=None):
        """Adds what the target bottle looks like right now to a type."""
        t = self._last
        if t is None or self._frame is None:
            return None
        desc = descriptor(self._frame, t["box"])
        if desc is None:
            return None
        # Measure the true height/width from the model's own outline now, while
        # the cap is easy to see, so a hard-to-see cap can be placed later.
        cap = t["cap"]
        aspect = cap["height_px"] / cap["body_w"] if cap["method"] == "yolo mask" and cap["body_w"] > 0 else None
        # Calibrate on the median over the frames this bottle has been tracked, not on this one frame.
        hist = self._neck_hist.get(t["id"], [])
        neck = float(np.median(hist)) if len(hist) >= 5 else cap["neck_ratio"]
        if cap["method"] != "yolo mask":
            neck = None
        bottle_type = self.library.add_sample(name, height_cm, cap_height_cm, t["box"], desc, aspect, neck, diameter_cm)
        if t["id"] in self._info:
            self._info[t["id"]]["checked_at"] = -RECHECK_EVERY   # look again with the type known
        return bottle_type

    def mark_not_a_bottle(self):
        """The target is a false alarm (the robot, say): remember what it
        looks like as a non-bottle, and never follow that track again."""
        t = self._last
        if t is None or self._frame is None:
            return False
        desc = descriptor(self._frame, t["box"])
        if desc is not None:
            self.library.add_negative(desc)
        if t["id"] is not None:
            self._banned.add(t["id"])
        self._last, self._target_id = None, None
        return True

    def summary(self):
        s = self.stats
        n = max(s["frames"], 1)
        return f"{s['frames']} frames: target on {s['target_frames'] / n:.0%}, seen fresh on {s['fresh_frames'] / n:.0%}"
