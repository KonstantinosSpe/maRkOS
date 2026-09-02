"""The hover window: the arm follows a bottle it sees, staying a set height above its cap.

Watches the camera, finds the enrolled bottle, works out where it stands on the desk (from the desk calibration, with the
laptop placed from the two red stars if it has been moved), plans the joint angles that put the gripper above the cap
and publishes them to the bridge (``/joint_command``). Nothing moves until the bridge is armed *and* ``g`` is on in this
window and the bottle is recognised and steady.

Keys: g go / stop, p arm / disarm the bridge, h twice = home the arm, w / s hover height up / down (33 mm), , / . fine
height (5 mm), [ ] and { } turn the base zero by 1 / 10 degrees, < > base-turn gain, - = radius offset (5 mm),
e flip the base direction, ; ' base height, a enrol a bottle, x mark a false alarm, t list types, q quit.

Run: python -m markos_vision.apps.hover          (needs the ROS 2 environment and a running thor_bridge)
     python -m markos_vision.apps.hover --simulate 30,335 --sweep 20   (no camera: a made-up bottle, for RViz + mock bridge)
"""

import argparse
import math
import os
import time
import traceback

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker

from markos_vision.config import HOVER_LOG, PROBE_LOG
from markos_vision.geometry import camera_model as cm
from markos_vision.geometry import desk_markers as dm
from markos_vision.geometry.desk_calibration import ContactFilter, DeskCalibration, base_contact
from markos_vision.geometry.pick_geometry import PickConfig, bottle_radius_cm, hover_plan, locate_cap, locate_on_desk, urdf_xyz

MIN_STAR_SAMPLES = 25  # the stars don't move, so their position is averaged; this many frames before it's trusted
STAR_SMOOTH = 0.05
STEADY_FRAMES = 8
STEADY_BEARING_DEG = 2.0
STEADY_RADIUS_MM = 10.0
TARGET_SMOOTH = 0.3
PUBLISH_FRAME = "base_link"
LOG_EVERY_S = 2.0
HEIGHT_COARSE_MM = 33.0     # w / s: six presses are about 20 cm; , and . are the fine 5 mm steps
HEIGHT_FLOOR_MM = 30.0      # the hover clearance is never taken below this, from the middle of the cap
HOME_CONFIRM_S = 3.0        # h has to be pressed twice within this long, since homing moves the arm
ARM_POLL_S = 1.5            # how often the window asks the bridge whether it is armed


class StarBase:
    """The robot's base as the camera sees it: the two stars' midpoint and
    spacing, averaged over time. Their spacing is the noisiest number in the
    whole chain (7.9 cm apart, so one pixel of error is ~15 mm of distance),
    and they never move, which is exactly when averaging pays."""

    def __init__(self):
        self.mid = None
        self.sep = None
        self.n = 0

    def update(self, left, right):
        mid = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
        sep = math.hypot(right[0] - left[0], right[1] - left[1])
        if self.mid is None:
            self.mid, self.sep = mid, sep
        else:
            self.mid = tuple(o + STAR_SMOOTH * (n - o) for o, n in zip(self.mid, mid))
            self.sep += STAR_SMOOTH * (sep - self.sep)
        self.n += 1

    @property
    def ready(self):
        return self.n >= MIN_STAR_SAMPLES


class TargetFilter:
    """Smooths the bottle's measured position and says whether it has held
    still long enough to be worth sending the arm toward."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.bearing = self.radius = None
        self.recent = []

    def update(self, loc):
        if self.bearing is None:
            self.bearing, self.radius = loc["bearing_deg"], loc["radius_mm"]
        else:
            self.bearing += TARGET_SMOOTH * (loc["bearing_deg"] - self.bearing)
            self.radius += TARGET_SMOOTH * (loc["radius_mm"] - self.radius)
        self.recent = (self.recent + [(self.bearing, self.radius)])[-STEADY_FRAMES:]

    @property
    def steady(self):
        if len(self.recent) < STEADY_FRAMES:
            return False
        b, r = zip(*self.recent)
        return max(b) - min(b) < STEADY_BEARING_DEG and max(r) - min(r) < STEADY_RADIUS_MM


class Ros:
    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node("hover_bottle")
        self.joints = self.node.create_publisher(JointState, "/joint_command", 10)
        self.gripper = self.node.create_publisher(Float32, "/gripper_command", 10)
        self.marker = self.node.create_publisher(Marker, "/bottle_target", 10)
        self.command_gripper = False   # off unless asked (--gripper): the servo's current draw is a suspect for a dropped USB link

    def send_joints(self, plan):
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = ["E", "D", "BC"]
        msg.position = [math.radians(plan["E"]), math.radians(plan["D"]), math.radians(plan["B"])]
        self.joints.publish(msg)
        if self.command_gripper:
            self.gripper.publish(Float32(data=1.0))

    def send_markers(self, info, status, base_id=0, ns="bottle"):
        """Green sphere on the cap where the bottle really is; the other one
        is where the gripper is headed: yellow if that's the wanted hover
        point, orange if the arm had to settle for somewhere else, red if
        there's nowhere it can go."""
        hover_colour = {"ok": (1.0, 0.85, 0.1), "raised": (1.0, 0.5, 0.0), "unreachable": (1.0, 0.2, 0.2)}[status]
        spheres = (
            (info["cap_azimuth_deg"], info["cap_radius_mm"], info["cap_mm"], (0.1, 0.9, 0.2)),
            (info["azimuth_deg"], info["radius_mm"], info["height_mm"], hover_colour),
        )
        for marker_id, (azimuth, radius_mm, height, colour) in enumerate(spheres):
            m = Marker()
            m.header.frame_id = PUBLISH_FRAME
            m.header.stamp = self.node.get_clock().now().to_msg()
            m.ns, m.id, m.type, m.action = ns, base_id + marker_id, Marker.SPHERE, Marker.ADD
            m.pose.position = Point(**dict(zip("xyz", urdf_xyz(azimuth, radius_mm, height))))
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.03
            m.color.r, m.color.g, m.color.b, m.color.a = (*colour, 0.9)
            m.lifetime.nanosec = 500_000_000
            self.marker.publish(m)

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()


class BridgeControl:
    """The bridge's arming gate and its homing, from inside this window: p arms and disarms, h homes. It never blocks the
    camera loop: requests go out and their answers are picked up by spin() on later frames."""

    def __init__(self, node):
        self.node = node
        self.set_client = node.create_client(SetParameters, "/thor_bridge/set_parameters")
        self.get_client = node.create_client(GetParameters, "/thor_bridge/get_parameters")
        self.home_client = node.create_client(Trigger, "/home")
        node.create_subscription(JointState, "/joint_states", self._on_state, 10)
        self.armed = None            # None until the bridge has answered
        self.complete = False        # the bridge is reporting E, D and B: the arm is homed and its position is known
        self.homing = False
        self.homing_started = 0.0
        self.note, self.note_t = "", 0.0
        self._poll_t = 0.0

    def say(self, text):
        self.note, self.note_t = text, time.time()
        print(text)

    def _on_state(self, msg):
        self.complete = {"E", "D", "BC"} <= set(msg.name)

    def spin(self):
        for _ in range(20):
            rclpy.spin_once(self.node, timeout_sec=0.0)
        if self.homing and self.complete and time.time() - self.homing_started > 3.0:
            self.homing = False
            self.say("Homing complete. The bridge is not armed: press p to arm it.")
        if self.homing and time.time() - self.homing_started > 150.0:
            self.homing = False
            self.say("No sign that homing finished after 150 s: look at the bridge window.")

    def poll(self):
        """Ask the bridge if it is armed, every so often; the answer arrives later."""
        if time.time() - self._poll_t < ARM_POLL_S:
            return
        self._poll_t = time.time()
        if not self.get_client.service_is_ready():
            self.armed = None
            return
        req = GetParameters.Request()
        req.names = ["armed"]
        self.get_client.call_async(req).add_done_callback(self._got_armed)

    def _got_armed(self, fut):
        try:
            self.armed = bool(fut.result().values[0].bool_value)
        except Exception:  # noqa: BLE001 -- no answer just means "unknown"
            self.armed = None

    def set_armed(self, value):
        if not self.set_client.service_is_ready():
            self.say("The bridge isn't answering: is it running?")
            return
        req = SetParameters.Request()
        p = ParameterMsg()
        p.name = "armed"
        p.value = ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=bool(value))
        req.parameters = [p]
        self.set_client.call_async(req).add_done_callback(lambda f, v=bool(value): self._armed_result(f, v))

    def _armed_result(self, fut, value):
        try:
            result = fut.result().results[0]
        except Exception as exc:  # noqa: BLE001
            self.say(f"No answer from the bridge: {exc}")
            return
        if result.successful:
            self.armed = value
            self.say("ARMED: the arm now follows the hover while g is on." if value else "Disarmed: the arm holds where it is.")
        else:
            self.say(f"The bridge refused: {result.reason}")

    def home(self):
        if not self.home_client.service_is_ready():
            self.say("The bridge isn't answering: is it running?")
            return
        self.home_client.call_async(Trigger.Request()).add_done_callback(self._home_result)

    def _home_result(self, fut):
        try:
            r = fut.result()
        except Exception as exc:  # noqa: BLE001
            self.say(f"No answer from the bridge: {exc}")
            return
        if r.success:
            self.homing, self.homing_started, self.armed, self.complete = True, time.time(), False, False
            self.say("HOMING: the bridge disarmed itself and is moving B, then D, then E. Keep clear.")
        else:
            self.say(f"Not homing: {r.message}")

    def line(self):
        if self.homing:
            return "ARM: HOMING -- the arm is moving, keep clear"
        if self.armed is None:
            return "ARM: the bridge is not answering"
        return "ARM: ARMED   (p = disarm,  h h = home)" if self.armed else "ARM: not armed   (p = arm,  h h = home)"


def waiting_on(found, stars, need_stars, camera_moved, steady, plan, go, listeners, usable, marker_wait=None):
    """Every reason the arm isn't being sent anywhere right now. The stars
    are only needed to place the bottle when there's no desk calibration;
    with one, they're just watched to catch the camera being moved (or, with
    desk markers, the markers do that and the mapping waits for them)."""
    reasons = []
    if found is None or found["missed"] != 0:
        reasons.append("no bottle in view")
    elif not found["verified"]:
        reasons.append("bottle not enrolled (a)")
    if need_stars and not stars.ready:
        reasons.append(f"stars locking {stars.n}/{MIN_STAR_SAMPLES}")
    if marker_wait:
        reasons.append(marker_wait)
    if camera_moved:
        reasons.append("camera moved: redo calibrate_desk.py")
    if not steady:
        reasons.append("bottle still moving")
    if usable and plan is None:
        reasons.append("nowhere reachable")
    if not go:
        reasons.append("press g")
    if listeners == 0:
        reasons.append("no bridge listening on /joint_command")
    return reasons


def log_gate(waiting, loc, info):
    """One text line every couple of seconds saying what the arm is being
    sent, or every reason it isn't. Numbers only."""
    parts = [time.strftime("%H:%M:%S"), "SENDING" if not waiting else "waiting: " + ", ".join(waiting)]
    if loc is not None:
        parts.append(f"bearing {loc['bearing_deg']:.1f} deg, radius {loc['radius_mm']:.0f} mm"
                     + (f", u {loc['u_cm']:+.1f} v {loc['v_cm']:+.1f}" if loc.get("u_cm") is not None else ""))
    if info is not None:
        parts.append(f"wanted {info['wanted_mm']:.0f} mm up, heading {info['height_mm']:.0f} mm up at {info['radius_mm']:.0f} mm"
                     + (f", {info['lean']}" if info.get("lean") else "")
                     + (f" [{info['degraded']}]" if info.get("degraded") else "")
                     + (f" [{info['why']}]" if info.get("why") and info.get("degraded") is None else ""))
    try:
        os.makedirs(os.path.dirname(HOVER_LOG), exist_ok=True)
        with open(HOVER_LOG, "a") as f:
            f.write(" | ".join(parts) + "\n")
    except OSError:
        pass


def change_clearance(cfg, delta_mm):
    """Raise or lower how far above the cap the gripper hovers, never below HEIGHT_FLOOR_MM. Returns the new clearance."""
    current = cfg["hover_clearance_mm"]
    wanted = max(HEIGHT_FLOOR_MM, current + delta_mm)
    if wanted != current + delta_mm:
        print(f"The hover height is held at {HEIGHT_FLOOR_MM:.0f} mm above the middle of the cap at the lowest.")
    cfg.nudge("hover_clearance_mm", wanted - current)
    return cfg["hover_clearance_mm"]


def log_probe(x_cm, y_cm, verified, calib):
    """One line: where the bottle stands on the desk, in cm right of and toward the laptop from the arm's assumed axis,
    and the same in the tape's u, v. Numbers only."""
    u_cm, v_cm = calib.from_desk(x_cm, y_cm)
    try:
        os.makedirs(os.path.dirname(PROBE_LOG), exist_ok=True)
        with open(PROBE_LOG, "a") as f:
            f.write(f"{time.time():.2f} x {x_cm:.2f} y {y_cm:.2f} u {u_cm:.2f} v {v_cm:.2f} verified {int(bool(verified))}\n")
    except OSError:
        pass


def status_lines(cfg, go, stars_ready, steady, loc, azimuth, plan, info, waiting, position_note, notes):
    lines = [position_note] + notes
    if loc is not None:
        lines.append(f"bearing {loc['bearing_deg']:+.1f} deg  radius {loc['radius_mm']:.0f} mm  -> E azimuth {azimuth:+.1f}")
        if loc.get("u_cm") is not None:
            lines.append(f"bottle at  u {loc['u_cm']:+.1f}  v {loc['v_cm']:+.1f} cm   (the same u v you measure with the tape)")
    if plan is not None:
        pose = {"toward": "facing the bottle", "away": "base turned away, arm leaning back"}.get(info.get("lean"), "")
        lines.append(f"hover {info['height_mm']:.0f} mm up: E {plan['E']:+.0f}  D {plan['D']:+.0f}  B {plan['B']:+.0f}   {pose}")
        if info.get("degraded"):
            lines.append(f"NOTE: {info['degraded']}")
    elif info is not None:
        lines.append(f"CAN'T REACH: {info['why']}")
    lines.append("SENDING to the arm" if not waiting else "waiting: " + ", ".join(waiting))
    lines.append(f"e_zero {cfg['e_zero_deg']:+.0f}  e_gain {cfg['e_gain']:.2f}  e_sign {cfg['e_sign']:+d}  "
                 f"radius {cfg['radius_offset_mm']:+.0f} mm  clearance {cfg['hover_clearance_mm']:.0f} mm  "
                 f"riser {-cfg['base_height_cm']:.1f} cm")
    lines.append("keys: p arm  g go  h h home  w s height 3 cm  , . height 5 mm  [ ] { } e_zero  < > turn gain  - = radius")
    return lines


def run_camera(args):
    from markos_vision.apps.find_bottle import draw, enrol, list_types, smooth_cap
    from markos_vision.perception.bottle_yolo import YoloBottleFinder
    from markos_vision.perception.camera import open_camera
    from markos_vision.perception.star_tracking import StarTracker, load_focal_length, load_templates, star_side_order

    templates = load_templates()
    focal_px = load_focal_length()
    if templates is None or focal_px is None:
        print("Needs the star templates and the focal-length calibration from the star work "
              "(markos_vision.apps.select_stars: box both stars, then press 'f' a few times).")
        return
    print("Loading the YOLO model...")
    finder = YoloBottleFinder()
    tracker = StarTracker(templates)
    cfg = PickConfig()
    calib = DeskCalibration()
    cap = open_camera()
    if not cap.isOpened():
        print("FAILED: could not open camera")
        return
    ros = Ros()
    ros.command_gripper = bool(getattr(args, "gripper", False))
    control = BridgeControl(ros.node) if hasattr(ros, "node") else None     # arm / home from this window
    home_asked_at = 0.0
    stars, filt, contact_filter = StarBase(), TargetFilter(), ContactFilter()
    probe_filter, probe_t = ContactFilter(), 0.0
    marker_lock = dm.MarkerLock(calib) if (calib.ready and calib.markers) else None
    star_lock = cm.StarLock(calib) if (calib.ready and calib.star_cam and calib.star3d and marker_lock is None) else None
    if marker_lock is not None:
        print(f"Position comes from the desk calibration, rebuilt from the printed desk markers {sorted(calib.markers)} "
              "wherever the laptop is: keep three or more in view. The arm waits until they're locked.")
    elif star_lock is not None:
        print("Position comes from the desk calibration, moved to wherever the laptop is by the two red stars: keep both "
              "in view, and the laptop at the same height and lid angle as when you calibrated. The arm waits until it's locked.")
    elif calib.ready:
        print(f"Position comes from the desk calibration ({len(calib.points)} points), for the laptop where it was "
              "calibrated. The stars only watch for the camera moving. (Learn desk markers with calibrate_desk.py to move the laptop.)")
    else:
        print("NO DESK CALIBRATION: falling back to the rough star-based estimate. Run calibrate_desk.py for a real one.")

    win = "hover over the bottle | p=arm  g=go  h h=home  w s=height  a=add bottle  x=not a bottle  q=quit"
    print("Both stars and an ENROLLED bottle (press 'a') have to be in view. Nothing is sent to the arm until the bridge is armed")
    print("(press 'p') AND you press 'g'. Press g again to stop it, p to disarm, h twice to home the arm, q to quit (which disarms).")
    go, cap_state, last_error, last_log_t = False, None, None, 0.0
    lean = None   # the pose the arm was last sent into: facing the bottle, or turned away with the arm leaning back
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

            detections = tracker.match(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            star_pixels = None
            if len(detections) >= 2 and all(d[4] for d in detections):
                order = star_side_order(detections)
                stars.update(detections[order[0]][:2], detections[order[-1]][:2])
                star_pixels = [detections[order[0]][:2], detections[order[-1]][:2]]
            display = frame.copy()
            for (cx, cy, score, (x, y, w, h), live) in detections:
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0) if live else (0, 165, 255), 1)

            if found is None:
                cap_state = None
            elif found["missed"] == 0:
                cap_state = smooth_cap(cap_state, found["cap"])

            desk_mode = calib.ready
            notes = []
            marker_wait = None
            lock = marker_lock or star_lock
            if marker_lock is not None:
                marker_lock.update(frame)
                for c in marker_lock.seen.values():
                    cv2.polylines(display, [np.round(c).astype(np.int32)], True, (255, 200, 0), 1)
            elif star_lock is not None:
                star_lock.update(star_pixels)
            if lock is not None:
                if not lock.locked:
                    marker_wait = lock.detail
                    notes.append("!! " + lock.detail)
                moved_note = None   # the markers / stars are what catch a moved camera now
            else:
                moved_note = calib.camera_moved(stars.mid, stars.sep) if (desk_mode and stars.ready) else None
                if moved_note:
                    notes.append("!! " + moved_note)
                elif desk_mode and calib.stars is not None and not stars.ready:
                    notes.append("stars not seen: can't check the camera hasn't moved")

            loc = azimuth = plan = info = None
            fresh_verified = found is not None and found["missed"] == 0 and found["verified"]
            usable = (fresh_verified and moved_note is None and marker_wait is None) if desk_mode else \
                (fresh_verified and cap_state is not None and stars.ready)
            if usable:
                bt = found["type"]
                if desk_mode:
                    base_px = contact_filter.update(base_contact(found["polygon"], found["box"]))
                    loc = locate_on_desk(base_px, calib, cfg, bottle_radius_cm(bt))
                    if not calib.inside(*base_px):
                        notes.append("outside the calibrated area: less accurate here")
                    if calib.bottle_diameter_cm is None:
                        notes.append("the desk calibration doesn't know its bottle's diameter: the near edge is not corrected to the axis")
                    elif bottle_radius_cm(bt) is None:
                        notes.append(f"'{bt['name']}' has no diameter: taken as {calib.bottle_diameter_cm:g} cm "
                                     "like the calibration bottle")
                else:
                    loc = locate_cap(cap_state["cap"], cap_state["height_px"], stars.mid, stars.sep, focal_px,
                                     bt["height_cm"], cfg)
                filt.update(loc)
                loc = {**loc, "bearing_deg": filt.bearing, "radius_mm": filt.radius}
                if desk_mode:   # the same position in the u, v the tape measures, so it can be checked against a ruler
                    raw_mm = filt.radius - cfg["radius_offset_mm"]      # without the radius nudge, which isn't where the bottle is
                    loc["u_cm"], loc["v_cm"] = calib.from_desk(raw_mm / 10.0 * math.sin(math.radians(filt.bearing)),
                                                               raw_mm / 10.0 * math.cos(math.radians(filt.bearing)))
                plan, info = hover_plan(filt.bearing, filt.radius, bt["height_cm"], bt["cap_height_cm"], cfg, lean=lean)
                azimuth = info["cap_azimuth_deg"]
                ros.send_markers(info, "unreachable" if plan is None else ("raised" if info["degraded"] else "ok"))
                if go and plan is not None and filt.steady:
                    if lean is not None and info["lean"] != lean:
                        print("Turning the base around: only the other pose gets the gripper this close to the bottle.")
                    lean = info["lean"]
                    ros.send_joints(plan)
            else:
                filt.reset()
                contact_filter.reset()

            # Where the camera puts the bottle, logged whether or not it's recognised as an enrolled type: an arm
            # hovering over it changes its colours, and pose_probe.py needs the position just the same.
            if desk_mode and found is not None and found["missed"] == 0 and moved_note is None and marker_wait is None:
                probe_px = probe_filter.update(base_contact(found["polygon"], found["box"]))
                if time.time() - probe_t >= 0.5:
                    probe_t = time.time()
                    log_probe(*calib.to_desk(*probe_px), found["verified"], calib)
            else:
                probe_filter.reset()

            waiting = waiting_on(found, stars, not desk_mode, moved_note is not None, filt.steady, plan, go,
                                 ros.joints.get_subscription_count(), usable, marker_wait)
            if time.time() - last_log_t >= LOG_EVERY_S:
                last_log_t = time.time()
                log_gate(waiting, loc, info)

            draw(display, finder, found, cap_state, focal_px)
            if found is not None and found["missed"] == 0 and not found["verified"]:
                cv2.putText(display, "NOT ENROLLED: the arm needs this bottle's height -- press a",
                            (10, display.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            if marker_lock is not None:
                position_note = f"position: desk markers -- {marker_lock.detail}"
            elif star_lock is not None:
                position_note = f"position: from the stars -- {star_lock.detail}"
            elif desk_mode:
                position_note = f"position: desk calibration ({len(calib.points)} points), laptop must stay put"
            else:
                position_note = "position: ROUGH STAR ESTIMATE -- run calibrate_desk.py"
            lines = status_lines(cfg, go, stars.ready, filt.steady, loc, azimuth, plan, info, waiting, position_note, notes)
            if control is not None:
                control.spin()
                control.poll()
                lines.insert(0, control.line())
                if control.note and time.time() - control.note_t < 6.0:
                    lines.insert(1, ">> " + control.note)
            for i, line in enumerate(lines):
                cv2.putText(display, line, (10, 50 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                go = not go
                if go and control is not None and control.homing:
                    go = False
                    control.say("Homing is still going: wait for it to finish before pressing g.")
                elif go:
                    print("GO ON -- it will now command the arm whenever the bridge is armed and the bottle is recognised, "
                          "steady and reachable." + ("" if control is None or control.armed
                                                      else "  (The bridge is NOT armed yet: press p.)"))
                else:
                    print("GO off -- no more commands (the bridge stops the arm within half a second).")
            elif key == ord('p') and control is not None:
                if control.homing:
                    control.say("Homing is still going: wait for it to finish.")
                elif control.armed:
                    control.set_armed(False)
                elif go:
                    control.say("Turn g off first, arm with p, then press g: arming with g on would move the arm at once.")
                else:
                    control.set_armed(True)
            elif key == ord('h') and control is not None:
                if time.time() - home_asked_at < HOME_CONFIRM_S:
                    home_asked_at = 0.0
                    go = False
                    control.home()
                else:
                    home_asked_at = time.time()
                    control.say(f"Press h again within {HOME_CONFIRM_S:.0f} s to HOME the arm: it disarms and moves B, D, then E.")
            elif key in (ord('w'), ord('s')):
                change_clearance(cfg, HEIGHT_COARSE_MM if key == ord('w') else -HEIGHT_COARSE_MM)
            elif key == ord('['):
                cfg.nudge("e_zero_deg", -1.0)
            elif key == ord(']'):
                cfg.nudge("e_zero_deg", 1.0)
            elif key == ord('{'):
                cfg.nudge("e_zero_deg", -10.0)
            elif key == ord('}'):
                cfg.nudge("e_zero_deg", 10.0)
            elif key == ord('<'):
                cfg.nudge("e_gain", -0.02)
            elif key == ord('>'):
                cfg.nudge("e_gain", 0.02)
            elif key == ord('-'):
                cfg.nudge("radius_offset_mm", -5.0)
            elif key == ord('='):
                cfg.nudge("radius_offset_mm", 5.0)
            elif key == ord(','):
                change_clearance(cfg, -5.0)
            elif key == ord('.'):
                change_clearance(cfg, 5.0)
            elif key == ord('e'):
                cfg.flip_e()
            elif key == ord(';'):
                cfg.nudge("base_height_cm", -0.5)   # bottle stands 0.5 cm higher relative to the robot's base
            elif key == ord("'"):
                cfg.nudge("base_height_cm", 0.5)
            elif key == ord('a'):
                enrol(finder, found)
            elif key == ord('x'):
                print("Remembered that as NOT a bottle." if finder.mark_not_a_bottle() else "Nothing tracked right now.")
            elif key == ord('t'):
                list_types(finder)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if control is not None:      # closing this window leaves the arm holding still, not able to be commanded
            control.set_armed(False)
            end = time.time() + 0.5
            while time.time() < end:
                control.spin()
                time.sleep(0.02)
        ros.close()


def run_simulated(args):
    """No camera: a made-up bottle at a given bearing and radius, so the arm
    model can be watched in RViz (with the mock bridge) before anything real
    is involved. Markers are always sent; joint commands only with --go."""
    cfg = PickConfig(path=None)
    ros = Ros()
    bearing0, radius = (float(v) for v in args.simulate.split(","))
    print(f"Simulated bottle: {args.bottle_height} cm tall, cap {args.cap_height} cm, at bearing {bearing0} deg, "
          f"{radius} mm from the base" + (f", swinging +/-{args.sweep} deg" if args.sweep else "")
          + ("  -- sending joint commands" if args.go else "  -- markers only (add --go to command the arm)"))
    t0, last_note, lean = time.time(), None, None
    try:
        while time.time() - t0 < args.seconds:
            bearing = bearing0 + (args.sweep * math.sin((time.time() - t0) / 4.0) if args.sweep else 0.0)
            plan, info = hover_plan(bearing, radius, args.bottle_height, args.cap_height, cfg, lean=lean)
            # its own namespace, so a simulated bottle and a real one can be shown side by side
            ros.send_markers(info, "unreachable" if plan is None else ("raised" if info["degraded"] else "ok"),
                             base_id=10, ns="sim")
            if plan is None:
                print(f"bearing {bearing:+.0f}: can't reach -- {info['why']}")
            else:
                if info["degraded"] and info["degraded"] != last_note:
                    last_note = info["degraded"]
                    print(f"note: {last_note}")
                if args.go:
                    lean = info["lean"]
                    ros.send_joints(plan)
            time.sleep(0.05)
    finally:
        ros.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--simulate", metavar="BEARING,RADIUS_MM", help="no camera: a made-up bottle, e.g. 30,335")
    p.add_argument("--sweep", type=float, default=0.0, help="with --simulate, swing the bearing +/- this many degrees")
    p.add_argument("--go", action="store_true", help="with --simulate, also send joint commands")
    p.add_argument("--gripper", action="store_true", help="also command the gripper open while hovering (off by default)")
    p.add_argument("--bottle-height", type=float, default=32.0)
    p.add_argument("--cap-height", type=float, default=3.0)
    p.add_argument("--seconds", type=float, default=60.0)
    args = p.parse_args()
    (run_simulated if args.simulate else run_camera)(args)


if __name__ == "__main__":
    main()
