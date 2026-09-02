"""The desk calibration tool: teach the camera where the desk is, with a bottle and a tape measure.

Stand the bottle on 6-10 spots around where it will be picked up, press ``c`` and type each spot's distances from a fixed
corner of the robot's box. From those the tool fits the mapping from a picture pixel to a position around the robot's
base axis (``geometry.desk_calibration``), reports how well the points agree with each other, and can then learn the
two red stars and the camera's height (keys ``s`` and ``h``) so that the laptop can be moved afterwards. Key ``d`` sets the
calibration bottle's diameter. The on-screen instructions (printed at start) describe the measuring frame.

Run: python -m markos_vision.apps.calibrate_desk       (close the hover window first: both use the camera)
"""

import math
import time
import traceback
from collections import deque

import cv2
import numpy as np

from markos_vision.geometry import camera_model as cm
from markos_vision.geometry import desk_markers as dm
from markos_vision.geometry.desk_calibration import (
    CALIB_PATH,
    RECOMMENDED_POINTS,
    ContactFilter,
    DeskCalibration,
    base_contact,
)
from markos_vision.perception.bottle_yolo import YoloBottleFinder
from markos_vision.perception.camera import open_camera
from markos_vision.perception.star_tracking import StarTracker, load_focal_length, load_templates, star_side_order

SAMPLE_FRAMES = 15  # frames of a still bottle averaged into one calibration point
STEADY_PX = 3.0  # the (filtered) contact point may wander this far, peak to peak, over those frames and still count as still
STAR_SMOOTH = 0.05
CONFIRM_RESET_S = 3.0
MAX_COORD_CM = 150.0  # a typed coordinate beyond this is almost certainly mm typed instead of cm
MIN_POINTS_FOR_MARKERS = 4
MIN_POINTS_FOR_STARS = 6  # the camera is fitted to the tape spots as well; a handful spread out pins it down
BOX_HEIGHT_CM = 7.05  # height of the box's end face, from the Thor CAD (70.5 mm); the stars are measured down from its top
STAR_FRAMES = 40  # frames of the two stars averaged when fitting the camera at calibration

# Where the middle of the turntable is, measured from the RIGHT-hand corner of the robot box's end face as seen from
# the laptop, with u counting to the RIGHT and v positive toward the laptop: the axis is centred on the box's 18 cm
# width, so 9.0 cm to the LEFT of that corner (u = -9.0), and 21.9 cm behind the end face, away from the laptop
# (v = -21.9). From the Thor CAD (Base.dae), tape-checked on the author's own box.
BOX_AXIS_UV = (-9.0, -21.9)

INSTRUCTIONS = """
DESK CALIBRATION -- teach the camera where the desk is, so a bottle's position comes from where it stands
and not from guessing its distance.

You measure each bottle from ONE fixed spot: the RIGHT-hand corner of the robot box's end face, as seen from
the laptop. The robot's axis is already known (it is built in), so all you type is the bottle's two distances.

        +----------------------+         the turntable and arm are behind the box, away from the laptop
        |    the box (Arduino) |         end face = the side facing the laptop
        +----------------------O ------> u
                               |
                               |
                               v

  O = the RIGHT corner of the box's end face, on the desk, as you look from the laptop.
  u = how far to the RIGHT of O                   (to the left of O = negative)
  v = how far toward the laptop from the end face (positive); level with the end face = 0;
      further back beside the box, away from the laptop = negative
  The bottle stands to the RIGHT of the box, so u is positive and v is around 0 or negative.
  Measure to the CENTRE of the bottle's foot, with a set square against the box.

  1. Stand the bottle on 6-8 spots spread over where you'll want it picked up (31-37 cm from the base axis, to the
     right of the box): some level with the end face, some further back beside the box, some nearer to the box and
     some farther from it.
  2. For each spot: wait until the bottle's foot dot is cyan and it says STILL, press c, then type
     the two distances as  u v   (e.g.  22 -22  is 22 cm right of the corner, level with the turntable).
  3. Do NOT move the laptop or the robot base while you do this.
  4. Use the SAME bottle for every point, and tell it how wide the bottle's foot is (key d, or enrol the bottle with its
     diameter): you measure to the centre of the foot, the camera sees its near edge, and the difference is half the width.

IF THE LAPTOP WILL BE PUT IN A DIFFERENT PLACE EACH TIME (using the two red stars on the robot):
after 6 or more bottle spots, with both stars in view, press s. It asks once for the two stars' positions
(how far left of the box's top-right corner and how far down from the top edge, to the millimetre), fits the
camera, and prints its estimated height above the desk --
check that with a ruler, it tells you the star numbers are right. From then on the hover window works out where the
laptop has been put from the stars. Keep the laptop at the same height and lid angle (a small tilt change is fine).
(Printed desk markers, key m, are more accurate if you ever want them: python -m markos_vision.geometry.desk_markers
writes the printable pages.)

Keys: c add a point   s learn the stars   h lens height   d bottle diameter   k forget the star positions
      m learn desk markers   u undo   t list   r reset (press twice)   q quit
"""


def parse_xy(text):
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError
    return float(parts[0]), float(parts[1])


def ask_axis(cal):
    """Where the base axis is in the measuring frame. Enter keeps what's there (or 'the origin is the axis')."""
    current = f"now {cal.axis[0]:+.1f} {cal.axis[1]:+.1f}" if cal.axis else "now: the origin IS the axis"
    try:
        text = input(f"  Middle of the turntable, measured from your corner O: 'u v' in cm, v NEGATIVE behind the front "
                     f"face (e.g. 10 -8) ({current}; Enter to keep): ").strip()
        if not text:
            return
        u, v = parse_xy(text)
    except ValueError:
        print("Type two numbers, like  10 -8  (u then v, in cm) -- axis unchanged.")
        return
    cal.set_axis(u, v)
    print(f"Base axis is at u {u:+.1f}, v {v:+.1f}.")


def ask_bottle_diameter(cal):
    """The width of the calibration bottle's foot. Enter keeps what's there."""
    current = f"now {cal.bottle_diameter_cm:g} cm" if cal.bottle_diameter_cm else "not set"
    try:
        text = input("  Diameter of the calibration bottle's foot, where it stands on the desk, in cm "
                     f"({current}; Enter to keep): ").strip()
        diameter = float(text) if text else None
    except ValueError:
        print("Type one number, like  5.5  -- nothing changed.")
        return
    if diameter is None:
        return
    if not 1.0 <= diameter <= 20.0:
        print("That doesn't look like a bottle's diameter in cm. Nothing changed.")
        return
    cal.set_bottle_diameter(diameter)
    print(f"Calibration bottle: {diameter:g} cm across." + (" Press s to refit the camera with it." if cal.star_cam else ""))


def check_bottle_diameter(cal, found):
    """Every point has to be measured with the same bottle, as the tape marks the centre of the foot and the camera sees
    its near edge. Takes the diameter from the enrolled bottle when the calibration doesn't have one yet."""
    bottle_type = found.get("type") if found is not None else None
    diameter = bottle_type.get("diameter_cm") if bottle_type else None
    if diameter is None:
        if cal.bottle_diameter_cm is None:
            print("  The bottle's foot diameter isn't known: press d and type it (the camera sees the near edge of the foot, "
                  "the tape marks its centre).")
    elif cal.bottle_diameter_cm is None:
        cal.set_bottle_diameter(diameter)
        print(f"  Calibration bottle: {diameter:g} cm across (from the enrolled '{bottle_type['name']}').")
    elif abs(diameter - cal.bottle_diameter_cm) > 0.3:
        print(f"  !! This bottle is {diameter:g} cm across but the calibration was measured with {cal.bottle_diameter_cm:g} cm: "
              "use one bottle for every point, or press d to change it.")


def ask_box_height():
    """Height of the box's end face above the desk: the stars are measured down from its top edge."""
    while True:
        text = input(f"  Height of the box's END FACE above the desk, in cm (Enter = {BOX_HEIGHT_CM}, the CAD value -- "
                     "measure yours): ").strip()
        if not text:
            return BOX_HEIGHT_CM
        try:
            h = float(text)
            if 2.0 < h < 30.0:
                return h
        except ValueError:
            pass
        print("  Type the height in cm, like  7.0")


def ask_star_geometry(cal):
    """Where the two red stars are on the robot: in the same u, v as the bottle spots, plus how high each is above
    the desk. Everything the moved-laptop placing knows about the robot comes from these six numbers, so they are
    worth measuring carefully (millimetres). Stars stuck on the box's end face are entered the way you measure them:
    how far LEFT of the top-right corner and how far DOWN from the top edge."""
    print("""
Where are the two red stars? They are stuck on the box's END FACE, so all you measure is where on that face.
For each star, measure the CENTRE of the star from the TOP-RIGHT corner of the end face (the corner straight above
your corner O, as you look from the laptop):

      how far to the LEFT of that corner,  how far DOWN from the top edge      in cm, e.g.   5.3 2.4

The 'left' star is the one on the LEFT in the camera's picture when you look from the laptop. Millimetres matter:
a 2 mm slip in a star position costs about 1 cm of accuracy after the laptop is moved.
(A star not on the end face: type three numbers instead,  u v height-above-the-desk,  e.g.  -5.3 4.0 4.6 .)""")
    stars = []
    box_h = None
    for name in ("LEFT star (left in the camera's picture)", "RIGHT star"):
        while True:
            text = input(f"  {name}: left down   (cm, Enter to cancel): ").strip()
            if not text:
                print("Cancelled.")
                return False
            try:
                parts = [float(p) for p in text.replace(",", " ").split()]
                if len(parts) == 2:
                    left, down = parts
                    if left < 0 or down < 0:
                        print("  'left' and 'down' are distances: type them as positive numbers, like  5.3 2.4")
                        continue
                    if box_h is None:
                        box_h = ask_box_height()
                    height = box_h - down
                    if not 0.2 < height < box_h + 0.5:
                        print(f"  That puts the star {height:.1f} cm above the desk, but the face is only {box_h:.1f} cm tall "
                              "-- check 'down'.")
                        continue
                    star = [-left, 0.0, height]
                    print(f"    -> u {star[0]:+.2f}, v 0, height {height:.2f} cm above the desk")
                elif len(parts) == 3:
                    star = parts
                else:
                    raise ValueError
                stars.append(star)
                break
            except ValueError:
                print("  Type two numbers, like  5.3 2.4   (left, down)")
    sep = math.hypot(stars[0][0] - stars[1][0], stars[0][1] - stars[1][1])
    print(f"  The stars are {sep:.1f} cm apart on the desk plane (you measured 7.9 cm between them before"
          + ("" if abs(sep - 7.9) < 0.4 else "  <- that doesn't match: check the numbers") + ").")
    cal.set_star_geometry(stars)
    return True


def laptop_displaced(star_lock):
    """True if the star lock has found the laptop somewhere other than where the tape points were measured."""
    if star_lock is None or not star_lock.locked or star_lock.camera is None:
        return False
    shift = math.hypot(*(star_lock.camera.pos[:2] - star_lock.cam_a.pos[:2]))
    turn = abs(math.degrees(star_lock.camera.yaw - star_lock.cam_a.yaw))
    return shift > 0.7 or turn > 0.5


def calibration_pixel(star_lock, px, py):
    """Where the camera as it was at calibration would have seen the desk point this pixel shows now. The tape points are
    all recorded in that camera's pixels, so a point measured with the laptop somewhere else has to be converted. The pixel
    is the near edge of the bottle's foot as this camera sees it; the point that gets converted is the bottle's axis, and
    the calibration camera sees the axis's own near edge."""
    gx, gy = star_lock.camera.ground(px, py)
    r = star_lock.calib.star_camera_radius_cm
    if r is not None:
        dx, dy = cm.edge_to_centre(star_lock.cam_a, star_lock.camera, gx, gy, r, r)
        gx, gy = gx + dx, gy + dy
    return tuple(float(v) for v in star_lock.cam_a.project((gx, gy, 0.0)))


def remembered_star_pixels(cal, star_lock):
    """The stars' pixels as they were when the camera was fitted: the model's projection plus the disagreement that was
    remembered then. Available whether or not the laptop is still where it was."""
    stars = [(*cal._to_frame(u, v), z) for u, v, z in cal.star3d]
    return np.array([star_lock.cam_a.project(s) + off for s, off in zip(stars, star_lock.offsets)], float)


def learn_stars(cal, cap, tracker, focal_px, star_lock=None):
    """Fit the camera as it stands now (the tape spots plus the two stars) and remember how the stars' pixels disagree
    with it. Do this once, with the laptop where the tape calibration was done; afterwards the laptop can be moved and
    the hover window works out where it went from the stars. If the laptop is somewhere else now but the stars are
    locked, the stars' pixels from calibration are rebuilt from what was remembered, so this can still be done."""
    if not cal.ready or len(cal.points) < MIN_POINTS_FOR_STARS:
        print(f"Measure at least {MIN_POINTS_FOR_STARS} bottle spots first (spread out), then press s with both stars in view.")
        return
    if tracker is None:
        print("No star templates: run markos_vision.apps.select_stars first.")
        return
    if cal.star3d is None and not ask_star_geometry(cal):
        return
    if laptop_displaced(star_lock):
        star_px = remembered_star_pixels(cal, star_lock)
        print("The laptop is not where the tape points were measured, so the camera is re-fitted from the stars' remembered "
              "pixels from then, not the live ones.")
    else:
        print(f"Watching the stars for {STAR_FRAMES} frames -- keep the laptop still and both stars in view...")
        samples = []
        tries = 0
        while len(samples) < STAR_FRAMES and tries < STAR_FRAMES * 6:
            tries += 1
            ok, frame = cap.read()
            if not ok:
                continue
            det = tracker.match(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if len(det) >= 2 and all(d[4] for d in det):
                order = star_side_order(det)
                samples.append([det[order[0]][:2], det[order[-1]][:2]])
        if len(samples) < STAR_FRAMES:
            print(f"Only saw both stars steadily in {len(samples)} of {tries} frames. Make sure both are in view and lit evenly.")
            return
        stack = np.array(samples, float)
        if stack.std(axis=0).max() > 1.0:
            print(f"The stars wobbled {stack.std(axis=0).max():.1f} px -- hold the laptop still and try again.")
            return
        star_px = stack.mean(axis=0)
    fit_star_camera(cal, star_px, focal_px)


def fit_star_camera(cal, star_px, focal_px):
    """Fit the camera to the tape spots and the stars' pixels, keep it in the calibration and report it. The bottle's foot
    is modelled when its diameter is known: the pixels are its near edge, the tape spots its axis."""
    tape = [(p["px"], p["py"], *cal._to_frame(p["x_cm"], p["y_cm"])) for p in cal.points]
    stars = [(*cal._to_frame(u, v), z) for u, v, z in cal.star3d]
    try:
        cam, rms, offsets = cm.fit_calibration_pose(tape, stars, star_px, f0=focal_px or 600.0, height=cal.lens_height_cm,
                                                    foot_radius_cm=cal.bottle_radius_cm)
    except ValueError as exc:
        print(f"Couldn't fit a camera: {exc}")
        return
    cal.set_star_camera(cam.params, offsets, rms, cal.bottle_radius_cm)
    if cal.bottle_radius_cm is None:
        print("  (The bottle's diameter isn't set, so its near edge is treated as its centre: press d, then s again.)")
    print(f"Camera fitted (rms {rms:.2f} px over {len(tape)} bottle spots + 2 stars):")
    if cal.lens_height_cm is not None:
        print(f"  height above the desk  {cam.h:5.1f} cm     (you measured {cal.lens_height_cm:.1f} cm with a ruler: pinned to that)")
    else:
        print(f"  height above the desk  {cam.h:5.1f} cm     <- CHECK THIS with a ruler (the camera lens, above the desk), "
              "then press h to tell me")
    print(f"  distance from the axis {math.hypot(*cam.pos[:2]):5.1f} cm")
    print(f"  tilted down            {math.degrees(cam.pitch):5.1f} deg")
    print(f"  focal length           {cam.f:5.0f} px" + (f"   (the star calibration said {focal_px:.0f})" if focal_px else ""))
    print("  disagreement of the stars' pixels with this fit (mostly the ruler error on the star positions): "
          + ", ".join(f"{np.hypot(*o):.1f} px" for o in offsets))
    if rms > 1.5:
        print("  !! The fit is loose: recheck the tape measurements, or the star positions.")
    print("From now on the hover window works out where the laptop is from the stars, wherever it is put. "
          "Keep the laptop at the same HEIGHT and lid angle as now (a small tilt change is allowed).")


def learn_markers(cal, cap):
    """Record where the printed desk markers are, using the tape calibration as it stands. Once, with the camera
    where the calibration was done; afterwards the laptop can go anywhere."""
    if not cal.ready:
        print(f"Measure at least {MIN_POINTS_FOR_MARKERS} bottle spots first, then press m with the desk markers in view.")
        return
    print(f"Learning the desk markers -- keep the laptop and the markers still ({dm.LEARN_FRAMES} frames)...")
    acc = dm.Accumulator()
    while acc.frames < dm.LEARN_FRAMES:
        ok, frame = cap.read()
        if ok:
            acc.add(dm.detect(frame))
    learned = dm.learn(cal, acc)
    if len(learned) < dm.MIN_MARKERS:
        seen = sorted(acc.mean()[0])
        print(f"Only {len(learned)} steady marker(s) learned (seen: {seen}); need {dm.MIN_MARKERS} or more. "
              "Tape more markers flat where the camera can see them, or hold everything still, and press m again.")
        return
    print(f"Learned {len(learned)} desk markers {sorted(learned)}. From now on the hover window rebuilds the mapping "
          "from them wherever the laptop is put -- no tape needed.")
    means = acc.mean()[0]
    for mid, corners in sorted(learned.items()):
        centre = np.mean(np.array(corners), axis=0)
        x, y = cal._to_frame(*centre)
        note = ""
        if not cal.inside(*np.mean(means[mid], axis=0)):
            note = "   <- outside the area you measured with the tape: less accurate; measure a bottle spot near it"
        print(f"  marker {mid}: centre {np.hypot(x, y) * 10:4.0f} mm from the axis (u {centre[0]:+.1f}, v {centre[1]:+.1f} cm){note}")
    print("Don't move the markers after this. If you ever do, press m again.")


def report(cal):
    print(f"\n{len(cal.points)} point(s); base axis "
          + (f"at u {cal.axis[0]:+.1f}, v {cal.axis[1]:+.1f}" if cal.axis else "NOT SET (press a) -- assuming your origin is the axis")
          + ("; your u, v were mirrored compared with the camera's picture, so the tool flipped u" if cal.mirrored else "") + ":")
    res = cal.residuals_mm() if cal.ready else []
    loo = cal.leave_one_out_mm()
    for i, p in enumerate(cal.points):
        extra = ""
        if res:
            extra += f"  fit error {res[i]:4.1f} mm"
        if loo:
            extra += f"  leave-one-out {loo[i]:5.1f} mm"
        around = cal._to_frame(p["x_cm"], p["y_cm"])
        print(f"  #{i + 1}: u {p['x_cm']:+6.1f}  v {p['y_cm']:+6.1f} cm  -> {math.hypot(*around) * 10:4.0f} mm from the axis"
              f"   (pixel {p['px']:.0f}, {p['py']:.0f}){extra}")
    if loo:
        worst = int(np.argmax(loo))
        print(f"  worst leave-one-out: {loo[worst]:.1f} mm at #{worst + 1}"
              + ("  <- recheck that tape measurement" if loo[worst] > 15 else ""))
    print("  calibration bottle: " + (f"{cal.bottle_diameter_cm:g} cm across" if cal.bottle_diameter_cm else
                                     "diameter NOT SET (press d): the near edge the camera sees is not corrected to the bottle's centre"))
    for note in cal.notes():
        print(f"  note: {note}")
    if cal.ready and len(cal.points) >= RECOMMENDED_POINTS and not cal.notes():
        print("  looks good.")
    print()


def main():
    print(INSTRUCTIONS)
    print("Loading the YOLO model...")
    finder = YoloBottleFinder()
    templates = load_templates()
    tracker = StarTracker(templates) if templates is not None else None
    cal = DeskCalibration()
    cap = open_camera()
    if not cap.isOpened():
        print("FAILED: could not open camera")
        return
    if cal.points:
        print(f"Loaded {len(cal.points)} saved point(s) -- adding to them. (r twice to start over.)")
        report(cal)
    if tracker is None:
        print("(No star templates: the 'camera moved' check is off.)")
    # With the stars' geometry learned, the laptop can be anywhere: points measured from a different spot are converted
    # into the calibration's own camera view through the stars, instead of being refused.
    star_lock = cm.StarLock(cal) if (tracker is not None and cal.star_cam and cal.star3d) else None
    if cal.axis is None:
        cal.set_axis(*BOX_AXIS_UV)
        print(f"Robot axis: u {BOX_AXIS_UV[0]:+.1f}, v {BOX_AXIS_UV[1]:+.1f} cm from the box's right end corner "
              "(from the Thor CAD). Just measure the bottle.\n")

    win = "desk calibration | c=add point  d=bottle diameter  m=learn desk markers  u=undo  t=list  r=reset  q=quit"
    history = deque(maxlen=SAMPLE_FRAMES)
    contact_filter = ContactFilter()
    stars_mid = stars_sep = None
    last_error, reset_asked_at = None, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            try:
                found = finder.find(frame)
            except Exception as exc:  # noqa: BLE001 -- one bad frame shouldn't end the session
                found = None
                if repr(exc) != last_error:
                    last_error = repr(exc)
                    print(f"[error on a frame, carrying on] {last_error}")
                    traceback.print_exc()

            star_pixels = None
            if tracker is not None:
                detections = tracker.match(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if len(detections) >= 2 and all(d[4] for d in detections):
                    order = star_side_order(detections)
                    a, b = detections[order[0]], detections[order[-1]]
                    star_pixels = [a[:2], b[:2]]
                    mid, sep = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), float(np.hypot(b[0] - a[0], b[1] - a[1]))
                    if stars_mid is None:
                        stars_mid, stars_sep = mid, sep
                    else:
                        stars_mid = tuple(o + STAR_SMOOTH * (n - o) for o, n in zip(stars_mid, mid))
                        stars_sep += STAR_SMOOTH * (sep - stars_sep)
                if star_lock is not None:
                    star_lock.update(star_pixels)

            display = frame.copy()
            marker_dets = dm.detect(frame)
            for mid, c in marker_dets.items():
                cv2.polylines(display, [np.round(c).astype(np.int32)], True, (255, 200, 0), 1)
                cv2.putText(display, f"marker {mid}", (int(c[0][0]), int(c[0][1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 200, 0), 1)
            contact = None
            if found is not None and found["missed"] == 0:
                raw_contact = base_contact(found["polygon"], found["box"])
                contact = contact_filter.update(raw_contact)
                history.append(contact)
                if found["polygon"] is not None:
                    cv2.polylines(display, [np.round(found["polygon"]).astype(np.int32)], True, (0, 255, 0), 2)
                cv2.circle(display, (int(raw_contact[0]), int(raw_contact[1])), 2, (0, 0, 255), -1)   # the raw, noisy point
            else:
                history.clear()
                contact_filter.reset()

            steady = False
            if len(history) == SAMPLE_FRAMES:
                h = np.array(history)
                steady = bool(np.ptp(h[:, 0]) < STEADY_PX and np.ptp(h[:, 1]) < STEADY_PX)
            if contact is not None:   # the big dot is the filtered point, the one that gets used
                cv2.circle(display, (int(contact[0]), int(contact[1])), 6, (255, 255, 0) if steady else (0, 165, 255), -1)

            for i, p in enumerate(cal.points):
                cv2.circle(display, (int(p["px"]), int(p["py"])), 4, (255, 0, 255), -1)
                cv2.putText(display, f"{i + 1}", (int(p["px"]) + 6, int(p["py"]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 255), 1)

            lines = [f"{len(cal.points)} point(s)" + (" -- calibrated" if cal.ready else f" -- need {4 - len(cal.points)} more"
                                                       if len(cal.points) < 4 else "")]
            if contact is None:
                lines.append("no bottle in view")
            else:
                lines.append("STILL -- press c" if steady else "hold the bottle still...")
                if cal.ready:
                    x, y = cal.to_desk(*contact)
                    lines.append(f"from the axis: x {x:+.1f}  y {y:+.1f} cm   radius {math.hypot(x, y) * 10:.0f} mm  "
                                 f"bearing {math.degrees(math.atan2(x, y)):+.1f} deg"
                                 + ("" if cal.inside(*contact) else "   (outside the marked area)"))
            if marker_dets or cal.markers:
                lines.append(f"desk markers in view: {sorted(marker_dets)}   learned: {sorted(cal.markers) or 'none yet (press m)'}")
            if cal.ready and stars_mid is not None:
                moved = cal.camera_moved(stars_mid, stars_sep)
                if star_lock is not None and laptop_displaced(star_lock):
                    lines.append("laptop moved: new points are converted through the stars  (" + star_lock.detail + ")")
                elif moved:
                    why = f"  ({star_lock.detail})" if star_lock is not None else ""
                    lines.append("!! " + moved + why)
            for i, line in enumerate(lines):
                cv2.putText(display, line, (10, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                if not steady:
                    print("Get the bottle outlined (green) and holding still first (cyan dot, 'STILL').")
                    continue
                sample = np.mean(np.array(history), axis=0)
                if laptop_displaced(star_lock):
                    sample = np.array(calibration_pixel(star_lock, *sample))
                    print("  (The laptop isn't where the tape points were measured, so this point is converted into that "
                          "camera's view through the stars.)")
                elif cal.stars is not None and stars_mid is not None:
                    moved = cal.camera_moved(stars_mid, stars_sep)
                    if moved:
                        why = f" The stars can't place the laptop either: {star_lock.detail}." if star_lock is not None else ""
                        print(f"Not adding a point: {moved}.{why} Put the camera back, or reset (r twice) and start over.")
                        continue
                try:
                    text = input("  Measured position of the centre of the bottle's foot in your measuring frame, "
                                 "'u v' in cm (Enter to cancel): ").strip()
                    if not text:
                        print("Cancelled.")
                        continue
                    x_cm, y_cm = parse_xy(text)
                except ValueError:
                    print("Type two numbers, like  -8.5 32  -- nothing added.")
                    continue
                if max(abs(x_cm), abs(y_cm)) > MAX_COORD_CM:
                    print(f"A coordinate over {MAX_COORD_CM:.0f} cm -- check the numbers (cm, not mm). Nothing added.")
                    continue
                cal.add(sample[0], sample[1], x_cm, y_cm)
                if cal.stars is None and stars_mid is not None:
                    cal.set_star_reference(stars_mid, stars_sep)
                check_bottle_diameter(cal, found)
                history.clear()
                report(cal)
            elif key == ord("d"):
                ask_bottle_diameter(cal)
            elif key == ord("m"):
                learn_markers(cal, cap)
            elif key == ord("s"):
                learn_stars(cal, cap, tracker, load_focal_length(), star_lock)
                # the camera may have been refitted: lock on the stars again, against the new fit
                star_lock = cm.StarLock(cal) if (tracker is not None and cal.star_cam and cal.star3d) else None
            elif key == ord("h"):
                try:
                    text = input("  Height of the camera lens above the desk, measured with a ruler, in cm (Enter to cancel): ").strip()
                    height = float(text) if text else None
                except ValueError:
                    print("Type one number, like  24  -- nothing changed.")
                    height = None
                if height is not None and not 5.0 <= height <= 90.0:
                    print("That doesn't look like a height in cm. Nothing changed.")
                elif height is not None:
                    cal.lens_height_cm = height
                    cal.save()
                    learn_stars(cal, cap, tracker, load_focal_length(), star_lock)
                    star_lock = cm.StarLock(cal) if (tracker is not None and cal.star_cam and cal.star3d) else None
            elif key == ord("k"):
                cal.star3d = None
                cal.star_cam = None
                cal.save()
                print("Forgot the star positions -- press s to enter them again.")
            elif key == ord("a"):
                ask_axis(cal)
                report(cal)
            elif key == ord("u"):
                cal.undo()
                print("Removed the last point.")
                report(cal)
            elif key == ord("t"):
                report(cal)
            elif key == ord("r"):
                if time.time() - reset_asked_at < CONFIRM_RESET_S:
                    cal.reset()
                    cal.set_axis(*BOX_AXIS_UV)   # reset wipes the axis too; the built-in one has to come straight back
                    history.clear()
                    contact_filter.reset()
                    print("Calibration cleared (the robot axis stays at the built-in position). Measure again from the first spot.")
                    reset_asked_at = 0.0
                else:
                    reset_asked_at = time.time()
                    print(f"Press r again within {CONFIRM_RESET_S:.0f} s to erase all {len(cal.points)} point(s).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if cal.points:
            print(f"Saved to {CALIB_PATH}.")
            report(cal)


if __name__ == "__main__":
    main()
