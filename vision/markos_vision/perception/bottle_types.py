"""The bottle types you have enrolled, and how to tell whether what the camera sees is one of them.

A type holds a bottle's real height, cap height and foot diameter plus a few colour signatures (``descriptor``: hue /
saturation and brightness histograms over a strip down the bottle's middle). ``BottleLibrary.classify`` compares a new
signature with the enrolled ones and with the signatures of things that were wrongly taken for a bottle, so a colour match
is needed before the arm is allowed to act on a detection.
"""

import json
import os

import cv2
import numpy as np

from markos_vision.config import BOTTLE_TYPES as LIBRARY_PATH
from markos_vision.config import ensure_parent

BASE_HEIGHT_CM = 0.0  # how far the robot's base sits above the surface the bottles stand on

H_BINS, S_BINS, V_BINS = 18, 8, 8
NEUTRAL_SAT = 30  # below this saturation (of 255) a pixel is treated as grey, whatever its hue says
NEUTRAL_VAL = 40  # ...and below this brightness as black
STRIP_HALF = 0.25  # the descriptor looks at this fraction of the width either side of the bottle's axis...
STRIP_TRIM = 0.10  # ...and skips this fraction of the height at the top and bottom
ACCEPT_SIM = 0.50  # colour similarity to an enrolled bottle needed to count as that bottle
NEG_MARGIN = 0.05  # and it has to beat the closest known non-bottle by at least this much
NEG_ONLY_SIM = 0.75  # with no types enrolled yet, how alike a known false alarm has to be to be turned down
ASPECT_TOL = 0.45  # allowed |ln(box aspect / enrolled aspect)|; a bottle's box changes a little with tilt and distance


def descriptor(frame_bgr, box):
    """A colour signature of the bottle: hue/saturation plus brightness
    histograms over a strip down its middle. The strip is mostly bottle even
    when the box is loose, which a whole-box histogram wouldn't be."""
    x, y, w, h = box
    fh, fw = frame_bgr.shape[:2]
    x0, x1 = int(max(0, x + w * (0.5 - STRIP_HALF))), int(min(fw, x + w * (0.5 + STRIP_HALF)))
    y0, y1 = int(max(0, y + h * STRIP_TRIM)), int(min(fh, y + h * (1 - STRIP_TRIM)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    # Hue means nothing on a near-white, grey or near-black pixel -- sensor
    # noise alone scatters it across every hue bin -- so those all count as
    # one neutral colour. White bottles and the grey robot are exactly the
    # cases that would otherwise flicker between "match" and "no match".
    neutral = (hsv[..., 1] < NEUTRAL_SAT) | (hsv[..., 2] < NEUTRAL_VAL)
    hsv[..., 0][neutral] = 0
    hsv[..., 1][neutral] = 0
    hs =cv2.calcHist([hsv], [0, 1], None, [H_BINS, S_BINS], [0, 180, 0, 256]).reshape(H_BINS, S_BINS)
    # Neighbouring hue bins are the same colour to the eye, and red wraps
    # around, so spread each bin a little into its neighbours.
    hs = 0.5 * hs + 0.25 * np.roll(hs, 1, axis=0) + 0.25 * np.roll(hs, -1, axis=0)
    v = cv2.calcHist([hsv], [2], None, [V_BINS], [0, 256]).flatten()
    hs = hs.flatten() / max(hs.sum(), 1e-9)
    v = v / max(v.sum(), 1e-9)
    return np.concatenate([hs, v]).astype(np.float32)


def similarity(a, b):
    n = H_BINS * S_BINS
    return 0.7 * float(np.minimum(a[:n], b[:n]).sum()) + 0.3 * float(np.minimum(a[n:], b[n:]).sum())


def reach_note(height_cm, cap_height_cm):
    """Can the arm actually grip this bottle's cap? Uses the planner in the
    markos package, with the robot's base BASE_HEIGHT_CM above the surface."""
    try:
        from markos import reach
    except ImportError:
        return ""
    cap_mm = (height_cm - cap_height_cm / 2 - BASE_HEIGHT_CM) * 10
    ok = [r for r in range(240, 430, 5) if reach.plan_target(90, float(r), cap_mm) is not None]
    if not ok:
        return f"the cap is {cap_mm / 10:.0f} cm up -- the arm can't reach that height (needs to be about 25 cm or more)"
    return f"the cap is {cap_mm / 10:.0f} cm up -- the arm can grip it when the bottle is {min(ok)}-{max(ok)} mm from its base"


class BottleLibrary:
    """The bottle types you've enrolled (name, real height, cap height, and a
    few colour signatures each), plus signatures of things that were wrongly
    taken for a bottle. path=None keeps it in memory only."""

    def __init__(self, path=LIBRARY_PATH):
        self.path = path
        self.types = []
        self.negatives = []
        self.load()

    def load(self):
        if self.path is None or not os.path.exists(self.path):
            return
        with open(self.path) as f:
            data = json.load(f)
        self.types = [
            {**t, "samples": [{"aspect": s["aspect"], "desc": np.array(s["desc"], np.float32)} for s in t["samples"]]}
            for t in data.get("types", [])
        ]
        self.negatives = [np.array(n, np.float32) for n in data.get("negatives", [])]

    def save(self):
        if self.path is None:
            return
        data = {
            "types": [
                {**t, "samples": [{"aspect": s["aspect"], "desc": s["desc"].tolist()} for s in t["samples"]]}
                for t in self.types
            ],
            "negatives": [n.tolist() for n in self.negatives],
        }
        with open(ensure_parent(self.path), "w") as f:
            json.dump(data, f)

    @property
    def has_types(self):
        return bool(self.types)

    def add_sample(self, name, height_cm, cap_height_cm, box, desc, silhouette_aspect=None, neck_ratio=None, diameter_cm=None):
        """Adds one look at a bottle to its type, creating the type if it's
        new. Enrol the same name a few times, turning the bottle, so the
        label doesn't have to be facing the camera. silhouette_aspect is the
        bottle's height over body width from a clean cut-out; it's averaged
        across looks and used later to place the cap when the cap won't
        separate from the background. diameter_cm is the width of its foot,
        where it stands on the desk (the camera sees the near edge of that
        foot and the arm wants the axis); left out, what the type had stays."""
        t = next((t for t in self.types if t["name"] == name), None)
        if t is None:
            t = {"name": name, "height_cm": float(height_cm), "cap_height_cm": float(cap_height_cm),
                 "silhouette_aspect": None, "diameter_cm": None, "samples": []}
            self.types.append(t)
        t["height_cm"], t["cap_height_cm"] = float(height_cm), float(cap_height_cm)
        if diameter_cm is not None:
            t["diameter_cm"] = float(diameter_cm)
        for key, value in (("silhouette_aspect", silhouette_aspect), ("neck_ratio", neck_ratio)):
            if value is not None:
                old, n = t.get(key), len(t["samples"])
                t[key] = value if old is None else (old * n + value) / (n + 1)
        t["samples"].append({"aspect": box[3] / box[2], "desc": desc})
        self.save()
        return t

    def add_negative(self, desc):
        self.negatives.append(desc)
        self.save()

    def classify(self, desc, box):
        """{'verdict', 'type', 'sim', 'neg_sim'}. Verdict is 'ok' for a
        match with an enrolled type, 'unknown bottle' for something that
        isn't (yet) one of them -- 'unverified' while no types exist at all --
        and 'known non-bottle' for a false alarm you've pointed out. Being
        unknown is not a rejection."""
        neg_sim = max((similarity(desc, n) for n in self.negatives), default=0.0) if desc is not None else 0.0
        if not self.types:
            # Nothing to match against yet, but a false alarm you've already
            # pointed out should still be turned down.
            verdict = "known non-bottle" if neg_sim >= NEG_ONLY_SIM else "unverified"
            return {"verdict": verdict, "type": None, "sim": 0.0, "neg_sim": neg_sim}
        if desc is None:
            return {"verdict": "unknown bottle", "type": None, "sim": 0.0, "neg_sim": 0.0}
        aspect = box[3] / box[2]
        best_sim, best_type = 0.0, None
        for t in self.types:
            for s in t["samples"]:
                if abs(np.log(aspect / s["aspect"])) > ASPECT_TOL:
                    continue
                sim = similarity(desc, s["desc"])
                if sim > best_sim:
                    best_sim, best_type = sim, t
        if best_type is not None and best_sim >= ACCEPT_SIM:
            verdict = "known non-bottle" if neg_sim > best_sim - NEG_MARGIN else "ok"
        else:
            # A bottle that isn't one of the enrolled types is still a bottle;
            # it just isn't known, so there's no height to work from yet.
            verdict = "known non-bottle" if neg_sim >= NEG_ONLY_SIM else "unknown bottle"
        return {"verdict": verdict, "type": best_type if verdict == "ok" else None, "sim": best_sim, "neg_sim": neg_sim}
