"""Hand-following demo: the arm in RViz reaches toward your palm, located from the two red stars and one webcam.

Publishes ``/joint_states`` (RViz only) or, with ``--command``, ``/joint_command`` for a bridge that is armed.
Run with a Python that has MediaPipe, next to a ROS 2 environment (``ros2 launch thor_urdf follow_palm.launch.py``).
"""

import math
import sys

import cv2
import mediapipe as mp
import rclpy
from arm_ik import HEIGHT_CEIL, HEIGHT_FLOOR, SHOULDER_H, clamp_reach, solve_reach_ik
from sensor_msgs.msg import JointState

from markos_vision.config import HAND_LANDMARKER as MODEL_PATH
from markos_vision.perception.camera import open_camera
from markos_vision.perception.star_tracking import (
    StarTracker,
    load_focal_length,
    load_templates,
    star_frame,
    star_side_order,
    to_local_cm,
)

PALM_LANDMARK = 9  # middle finger MCP -- same stable "hand center" point gesture_math.py's hand_scale() uses

# All revolute joints in thor.urdf.xacro. E (base rotation) and reach (via
# D/BC and arm_ik's 2-link solve) both come from the same 2D top-down
# triangulation in target_geometry(); height comes from target_height_mm().
# A and the gripper joints just hold a neutral pose.
JOINT_NAMES = [
    "E", "D", "A", "BC",
    "gripperbase_to_armgearright", "gripperbase_to_armgearleft",
    "gripperbase_to_armsimpleright", "gripperbase_to_armsimpleleft",
    "armgearright_to_fingerright", "armgearleft_to_fingerleft",
]
E_LIMIT = 2.967  # radians, matches the URDF's <limit> on E
BC_LIMIT = math.pi / 2  # URDF's BC limit is +-90deg, tighter than arm_ik's own J5 range of +-150deg

DEFAULT_REACH_MM = 280.0  # used only if camera distance isn't calibrated at all
HAND_LENGTH_CM = 9.0  # assumed wrist-to-middle-knuckle length (0->9) for an average adult hand
PALM_WIDTH_CM = 8.0  # assumed index-knuckle-to-pinky-knuckle width (5->17) for an average adult hand
JOINT_SMOOTH = 0.15  # EMA factor on target joint angles, same smoothing style gesture_math.py uses elsewhere
DEPTH_SMOOTH = 0.08  # slower EMA specifically on hand distance, since it's the noisiest input
FALLBACK_RAD_PER_CM = math.radians(3.0)  # used only if camera distance isn't calibrated
MM_PER_CM = 10.0


def hand_distance_cm(lm, w_img, h_img, focal_length_px):
    """Absolute distance from the camera to the hand, via the same
    known-size trick used for the star markers. Averages two independent
    measurements -- wrist-to-middle-knuckle 'length' (0->9) and
    index-to-pinky-knuckle 'width' (5->17) -- since they run in roughly
    perpendicular directions across the hand, which damps down (but
    doesn't eliminate) the error a hand rotation introduces by
    foreshortening one of them."""
    def px_dist(i, j):
        xi, yi = lm[i].x * w_img, lm[i].y * h_img
        xj, yj = lm[j].x * w_img, lm[j].y * h_img
        return math.hypot(xi - xj, yi - yj)

    estimates = []
    length_px = px_dist(0, 9)
    if length_px > 1e-6:
        estimates.append(HAND_LENGTH_CM * focal_length_px / length_px)
    width_px = px_dist(5, 17)
    if width_px > 1e-6:
        estimates.append(PALM_WIDTH_CM * focal_length_px / width_px)
    if not estimates:
        return None
    return sum(estimates) / len(estimates)


def target_geometry(dx_cm, cam_dist_cm, hand_dist_cm):
    """Proper top-down triangulation instead of treating dx_cm (lateral)
    and depth as independent numbers. Picture looking straight down at the
    desk: the markers sit at the origin, the camera looks along its own Z
    axis toward them from cam_dist_cm away, and the hand sits at lateral
    offset dx_cm with its own distance hand_dist_cm from the camera. The
    vector FROM the robot TO the hand in that view is
    (dx_cm, depth_diff_cm), where depth_diff_cm = cam_dist_cm - hand_dist_cm
    is how much closer to the camera (and so further from the robot) the
    hand is than the markers are. From there it's just 2D vector math:
    atan2 gives the angle E must rotate to face that vector, and its length
    (hypot) gives how far out reach must extend once E is facing that way.
    Falls back to cruder approximations if depth isn't measurable."""
    if cam_dist_cm is None:
        return dx_cm * FALLBACK_RAD_PER_CM, DEFAULT_REACH_MM
    depth_diff_cm = cam_dist_cm - hand_dist_cm if hand_dist_cm is not None else cam_dist_cm
    angle = math.atan2(dx_cm, depth_diff_cm)
    reach_mm = math.hypot(dx_cm, depth_diff_cm) * MM_PER_CM
    return angle, reach_mm


def target_height_mm(dy_cm):
    """Maps the palm's vertical offset from the star markers to a target
    height in the arm's own coordinate frame (measured from the base twist,
    same origin arm_ik's SHOULDER_H uses). dy_cm is positive when the palm
    is BELOW the markers on screen, so it's subtracted, not added, to make
    "palm up" raise the target height. SHOULDER_H is used as the dy_cm=0
    baseline -- an arbitrary but reasonable default, since the markers and
    the arm's own shoulder pivot aren't at a precisely known common height."""
    return max(HEIGHT_FLOOR, min(HEIGHT_CEIL, SHOULDER_H - dy_cm * MM_PER_CM))


def main():
    templates = load_templates()
    if templates is None:
        print("No saved star templates found -- run markos_vision.apps.select_stars first, "
              "box both stars, and press 'c' to lock + save them.")
        sys.exit(1)
    tracker = StarTracker(templates)
    focal_length_px = load_focal_length()
    if focal_length_px is None:
        print("No camera-distance calibration found -- falling back to a flat "
              f"{math.degrees(FALLBACK_RAD_PER_CM):.1f} deg/cm mapping. Run markos_vision.apps.select_stars "
              "and press 'f' a few times for a real angle instead.")

    base_opts = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    hand_opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(hand_opts)

    # Default is the original simulation: this script owns /joint_states and
    # RViz shows it directly. With --command it sends targets to the bridge
    # (mock or real) instead, and RViz shows the bridge's confirmed state.
    command_mode = "--command" in sys.argv[1:]

    rclpy.init()
    node = rclpy.create_node("follow_palm_sim")
    publisher = node.create_publisher(JointState, "/joint_command" if command_mode else "/joint_states", 10)

    cap = open_camera()
    if not cap.isOpened():
        print("FAILED: could not open camera")
        return

    topic_note = "/joint_command (bridge must be armed)" if command_mode else "/joint_states"
    print(f"Loaded {len(templates)} star template(s). Publishing {topic_note} -- "
          "open RViz with follow_palm.launch.py to watch it move.")
    print("Press 'q' (with the preview window focused) to quit.")

    frame_idx = 0
    smoothed_e = 0.0
    smoothed_d = 0.0
    smoothed_bc = 0.0
    smoothed_hand_dist = None
    try:
        while rclpy.ok():
            ok, frame = cap.read()
            if not ok:
                continue
            display = frame.copy()
            h_img, w_img = frame.shape[:2]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = tracker.match(gray)
            order = star_side_order(detections)
            for (cx, cy, score, (x, y, w, h), live) in detections:
                color = (0, 255, 0) if live else (0, 165, 255)
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

            frame_geom = star_frame(detections, order) if len(detections) >= 2 else None
            cam_dist_cm = None
            if frame_geom is not None:
                ox, oy = frame_geom[0]
                cv2.circle(display, (int(ox), int(oy)), 5, (0, 255, 255), -1)
                if focal_length_px is not None:
                    cm_per_px = frame_geom[3]
                    cam_dist_cm = focal_length_px * cm_per_px

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * (1000 / 30))
            frame_idx += 1
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            status = "no hand"
            tracking = False
            if result.hand_landmarks and frame_geom is not None:
                tracking = True
                lm = result.hand_landmarks[0]
                px, py = lm[PALM_LANDMARK].x * w_img, lm[PALM_LANDMARK].y * h_img
                cv2.circle(display, (int(px), int(py)), 6, (255, 0, 255), -1)

                dx_cm, dy_cm = to_local_cm((px, py), frame_geom)

                raw_hand_dist_cm = (hand_distance_cm(lm, w_img, h_img, focal_length_px)
                                     if focal_length_px is not None else None)
                if raw_hand_dist_cm is not None:
                    if smoothed_hand_dist is None:
                        smoothed_hand_dist = raw_hand_dist_cm
                    smoothed_hand_dist += DEPTH_SMOOTH * (raw_hand_dist_cm - smoothed_hand_dist)
                hand_dist_cm = smoothed_hand_dist

                target_e_raw, reach_mm = target_geometry(dx_cm, cam_dist_cm, hand_dist_cm)
                target_e = max(-E_LIMIT, min(E_LIMIT, target_e_raw))
                smoothed_e += JOINT_SMOOTH * (target_e - smoothed_e)

                height_mm = target_height_mm(dy_cm)
                reach_mm = clamp_reach(height_mm, reach_mm)
                j3_deg, j5_deg = solve_reach_ik(height_mm, reach_mm)
                target_d = math.radians(j3_deg)
                target_bc = max(-BC_LIMIT, min(BC_LIMIT, math.radians(j5_deg)))
                smoothed_d += JOINT_SMOOTH * (target_d - smoothed_d)
                smoothed_bc += JOINT_SMOOTH * (target_bc - smoothed_bc)

                hand_note = f"hand_dist={hand_dist_cm:.0f}cm" if hand_dist_cm is not None else "no depth"
                status = (f"palm offset: x={dx_cm:+.1f}cm y={dy_cm:+.1f}cm  {hand_note}  reach={reach_mm:.0f}mm  "
                          f"E={math.degrees(smoothed_e):+.1f} D={math.degrees(smoothed_d):+.1f} "
                          f"BC={math.degrees(smoothed_bc):+.1f}deg")
            elif result.hand_landmarks:
                status = "hand detected, need both stars for offset"

            cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow("follow palm (sim)", display)

            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            if command_mode:
                # Going quiet when the hand or the stars are lost is the stop
                # signal: the bridge drops a target that stops refreshing, so
                # the arm halts instead of chasing the last frame's pose.
                if tracking:
                    msg.name = ["E", "D", "BC"]
                    msg.position = [smoothed_e, smoothed_d, smoothed_bc]
                    publisher.publish(msg)
            else:
                msg.name = JOINT_NAMES
                msg.position = [smoothed_e, smoothed_d, 0.0, smoothed_bc] + [0.0] * (len(JOINT_NAMES) - 4)
                publisher.publish(msg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
