"""Locates a bottle's cap from its outline.

``analyse_mask`` measures a bottle's silhouette (height, body width, neck-to-body ratio) and places the cap at its top;
``find_cap`` cuts the silhouette out with GrabCut when only a bounding box is available. Both fall back to an estimate from
the box when the cut-out looks wrong.
"""

import cv2
import numpy as np

DEFAULT_CAP_FRAC = 0.09  # cap height as a share of the bottle's height, when the bottle type doesn't say
GRABCUT_ITERS = 4
MAX_ROI_H = 260  # GrabCut gets slow on big crops, and a bottle's outline doesn't need more resolution than this
SANE_HEIGHT_RATIO = (0.6, 1.4)  # silhouette height vs the detector's box; outside this the cut-out went wrong
MAX_CAP_TO_BODY = 0.85  # a cap as wide as the whole body means the cut-out merged with something


def _box_estimate(box, cap_frac):
    x, y, w, h = box
    cap_h = max(3.0, cap_frac * h)
    return {"cap": (x + w / 2, y + cap_h / 2), "cap_w": 0.5 * w, "cap_h": cap_h, "top_y": y,
            "bottom_y": y + h, "height_px": h, "body_w": w, "neck_ratio": None, "method": "box"}


def neck_ratio(row_widths):
    """How much narrower the top of the bottle is than its body: the width
    across the neck and shoulder region (10-30% of the way down) over the
    width of the body (widest part of the lower half). Real bottles sit
    around 0.3 to 0.5. A straight column -- a robot arm, a can, a box -- is
    about 1.0, which is the whole point of measuring it."""
    n = len(row_widths)
    if n < 20:
        return None
    neck = np.median(row_widths[int(0.10 * n): int(0.30 * n) + 1])
    body = np.percentile(row_widths[int(0.45 * n):], 90)
    return float(neck / body) if body > 0 else None


def silhouette(frame_bgr, box):
    """The bottle's outline cut out of the detector's box with GrabCut, in
    full-frame coordinates: (mask of the bottle, x offset, y offset, scale),
    or None. The detector's box is loose -- fine for "where is the bottle",
    but too sloppy at the top to say where a 3 cm cap is."""
    fh, fw = frame_bgr.shape[:2]
    # A detector's box can run off the edge of the frame; only the part that's
    # actually in the picture can be cut out.
    x0c, y0c = max(0.0, float(box[0])), max(0.0, float(box[1]))
    x1c, y1c = min(float(fw), float(box[0] + box[2])), min(float(fh), float(box[1] + box[3]))
    x, y, w, h = x0c, y0c, x1c - x0c, y1c - y0c
    if w < 4 or h < 10:
        return None
    pad_x, pad_top, pad_bot = 0.30 * w, 0.15 * h, 0.05 * h
    x0, x1 = int(max(0, x - pad_x)), int(min(fw, x + w + pad_x))
    y0, y1 = int(max(0, y - pad_top)), int(min(fh, y + h + pad_bot))
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.shape[0] < 30 or roi.shape[1] < 12:
        return None
    scale = min(1.0, MAX_ROI_H / roi.shape[0])
    if scale < 1.0:
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    rh, rw = roi.shape[:2]

    def to_roi(px, py):
        return (min(max(int((px - x0) * scale), 0), rw - 1), min(max(int((py - y0) * scale), 0), rh - 1))

    mask = np.full(roi.shape[:2], cv2.GC_BGD, np.uint8)        # the margins are certainly not the bottle
    bx0, by0 = to_roi(x, y)
    bx1, by1 = to_roi(x + w, y + h)
    mask[: by0, :] = cv2.GC_PR_BGD                              # a loose box may leave the cap sticking out the top
    mask[by0:by1, bx0:bx1] = cv2.GC_PR_BGD
    inner0, inner1 = to_roi(x + 0.10 * w, y)[0], to_roi(x + 0.90 * w, y)[0]
    mask[by0:by1, inner0:inner1] = cv2.GC_PR_FGD
    ax0, ay0 = to_roi(x + 0.44 * w, y + 0.20 * h)               # the bottle's axis, below the cap, is certainly bottle
    ax1, ay1 = to_roi(x + 0.56 * w, y + 0.90 * h)
    mask[ay0:ay1, ax0:ax1] = cv2.GC_FGD

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, mask, None, bgd, fgd, GRABCUT_ITERS, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Keep only the piece connected to the bottle's axis.
    _, labels = cv2.connectedComponents(fg)
    seed = labels[min((ay0 + ay1) // 2, rh - 1), min((ax0 + ax1) // 2, rw - 1)]
    if seed == 0:
        return None
    return labels == seed, x0, y0, scale


def find_cap(frame_bgr, box, cap_frac=DEFAULT_CAP_FRAC, prior_aspect=None):
    """Where the cap is, as a dict: cap (x, y centre), cap_w, cap_h, top_y,
    bottom_y, height_px (the whole bottle), body_w, neck_ratio, method
    ('grabcut' when the outline was cut out cleanly, 'grabcut+height prior'
    when it stopped short and the enrolled height/width ratio filled in the
    rest, else 'box' -- a rougher estimate from the detector's box alone).

    prior_aspect is the bottle type's height over body width, measured when
    it was enrolled. A dark cap against a dark desk won't cut out, but the
    body always does, and the body width plus that ratio says where the top
    must be."""
    fallback = _box_estimate(box, cap_frac)
    try:
        cut = silhouette(frame_bgr, box)
    except (cv2.error, IndexError, ValueError):
        cut = None                     # the cut-out is a refinement; never let it take the whole loop down
    if cut is None:
        return fallback
    mask, x0, y0, scale = cut
    return analyse_mask(mask, x0, y0, scale, box, cap_frac, prior_aspect, "grabcut")


def analyse_mask(mask, x0, y0, scale, box, cap_frac=DEFAULT_CAP_FRAC, prior_aspect=None, method="mask"):
    """Cap and neck-ratio measurements from a bottle's outline, however it
    was obtained (GrabCut here, or a segmentation model's mask). mask is a
    boolean image covering the region at `scale` times the original size with
    its top-left at (x0, y0) in the original frame."""
    fallback = _box_estimate(box, cap_frac)
    if mask is None:
        return fallback
    ys, xs = np.nonzero(mask)
    if len(ys) < 50:
        return fallback

    top, bottom = ys.min(), ys.max()
    height_px = (bottom - top + 1) / scale
    ratio = height_px / box[3]
    if not SANE_HEIGHT_RATIO[0] <= ratio <= SANE_HEIGHT_RATIO[1]:
        return fallback

    cap_rows = max(2.0, cap_frac * (bottom - top + 1))
    in_cap = ys <= top + cap_rows
    cap_xs = xs[in_cap]
    row_widths = np.bincount(ys - top)
    body_w = float(row_widths.max()) / scale                  # widest row, in px of the original frame
    widths = [np.count_nonzero(mask[r]) for r in range(top, int(top + cap_rows) + 1)]
    cap_w = float(np.median(widths)) / scale
    if body_w > 0 and cap_w / body_w > MAX_CAP_TO_BODY and cap_frac < 0.3:
        # No believable cap here -- a straight column has none. The outline is
        # still good enough to say how bottle-shaped it is, though.
        return {**fallback, "neck_ratio": neck_ratio(row_widths)}

    result = {
        "cap": (x0 + float(cap_xs.mean()) / scale, y0 + (top + cap_rows / 2) / scale),
        "cap_w": cap_w,
        "cap_h": cap_rows / scale,
        "top_y": y0 + top / scale,
        "bottom_y": y0 + bottom / scale,
        "height_px": height_px,
        "body_w": body_w,
        "neck_ratio": neck_ratio(row_widths),
        "method": method,
    }

    if prior_aspect and body_w > 0:
        expected = prior_aspect * body_w
        missing = expected - height_px
        if missing > 0.08 * expected:
            # The cut-out stops short of where this bottle's top should be:
            # the cap didn't separate from the background. Put it where the
            # ratio says, on the same axis.
            cap_h = cap_frac * expected
            top_y = result["top_y"] - missing
            result.update(cap=(result["cap"][0], top_y + cap_h / 2), cap_h=cap_h, top_y=top_y,
                          height_px=expected, method=f"{method}+height prior")
    return result
