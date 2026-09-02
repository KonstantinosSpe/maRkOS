"""Where does the gripper REALLY go?

The planner turns a bottle's position on the desk into joint angles, using the arm's centre from the drawing, a guessed turn
scale and the drawing's link lengths. If any of those is off, the gripper lands somewhere else. Instead of guessing, this
measures it, with the bottle and the camera (which your tape calibrated) as the ruler:

  1. It moves the arm to a handful of poses, high above the desk (about 40 cm), one at a time.
  2. At each one you put the bottle directly under the gripper's fingers. The camera says where the bottle is.
  3. Now for each pose it knows where the planner meant the gripper to go AND where it really went. It fits the
     arm's true centre, the offset and scale of its turn, and a reach error, and says what to change.

Needs: the bridge running and homed, and the hover window open with g OFF and the bottle recognisable (it logs the
bottle's position to probe.log even when it doesn't recognise the type, e.g. under the arm's shadow).
The arm is disarmed whenever you are placing the bottle, and re-armed only for each move.

    scripts/wsl/aim_probe.sh            # run the measurement
    scripts/wsl/aim_probe.sh --apply    # write the last fit into the hover settings and the calibration
"""
import json
import math
import shutil
import subprocess
import sys
import time

import numpy as np

from markos_vision.config import AIM_FIT as FIT_PATH
from markos_vision.config import AIM_PROBE_DATA as DATA_PATH
from markos_vision.config import DESK_CALIBRATION, PICK_CONFIG, PROBE_LOG, ensure_parent
from markos_vision.geometry import pick_geometry as pg

# (bearing deg, radius mm) the planner aims at: inside the arm's working zone (26-34 cm out to the right of the box), spread
# in bearing and radius so the fit can tell the arm's centre, its turn offset and scale and its reach apart (checked in
# simulation: about 3 mm typical error at places it never saw, with 5 mm camera noise)
POINTS = [(55, 335), (70, 285), (85, 310), (100, 275), (112, 335), (125, 300)]
HEIGHT_MM = 420.0           # the fingertips' planned height above base_link: well clear of a 22 cm bottle
BOTTLE_CM, CAP_CM = 22.0, 1.8
NAMES = ["E", "D", "BC"]


# ── the model and its fit (no ROS here, so it can be tested on its own) ──────────────────────────────────────────────

def predict(p, e_cmd, radius_p):
    """Where the gripper really is (mm right of, and toward the laptop from, the ASSUMED axis) when the planner asked
    for radius_p and turned the base by e_cmd. p = [axis shift x, y (mm), turn offset (deg), turn scale, reach error (mm)]."""
    dax, day, th0, k, rho0 = p
    b = math.radians(90.0 + k * e_cmd + th0)
    rho = radius_p + rho0
    return dax + rho * math.sin(b), day + rho * math.cos(b)


def fit(data):
    """Least squares (Levenberg-Marquardt) for the five numbers in predict()."""
    scale = np.array([1.0, 1.0, 0.05, 0.002, 1.0])
    # Sliding the axis sideways and lengthening the reach look alike from poses at similar radii, so a weak prior keeps the
    # poorly determined direction from wandering: a 1-sigma departure costs about what a 5 mm miss does.
    prior_sigma = np.array([40.0, 40.0, 6.0, 0.25, 40.0])
    prior_mean = np.array([0.0, 0.0, 0.0, 1.0, 0.0])

    def resid(p):
        out = []
        for d in data:
            px, py = predict(p, d["e_cmd"], d["radius_p"])
            out += [px - d["x_mm"], py - d["y_mm"]]
        return np.concatenate([np.array(out), 5.0 * (p - prior_mean) / prior_sigma])

    x = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    r = resid(x)
    cost, lam = float(r @ r), 1e-2
    for _ in range(300):
        J = np.column_stack([(resid(x + np.eye(5)[j] * scale[j]) - r) / scale[j] for j in range(5)])
        A = J.T @ J
        step = np.linalg.solve(A + lam * np.diag(np.diag(A)) + 1e-9 * np.eye(5), -J.T @ r)
        cand = x + step
        rc = resid(cand)
        cc = float(rc @ rc)
        if cc < cost:
            x, r, cost, lam = cand, rc, cc, max(lam / 3.0, 1e-9)
            if np.abs(step / scale).max() < 1e-5:
                break
        else:
            lam *= 4.0
            if lam > 1e9:
                break
    return x, r[:2 * len(data)].reshape(-1, 2)      # the misses only, without the prior's terms


def recommend(p):
    """What to change so the planner's commands land where it means them to. The hover computes
    E = e_gain * (bearing + e_zero_deg), and radius = measured + radius_offset_mm."""
    dax, day, th0, k, rho0 = p
    return {"axis_shift_mm": [float(dax), float(day)], "e_zero_deg": -90.0 - float(th0), "e_gain": 1.0 / float(k),
            "radius_offset_mm": -float(rho0)}


def plans():
    cfg = pg.PickConfig(path=None)
    cfg.values.update(e_sign=1, e_zero_deg=-90.0, e_gain=1.0, radius_offset_mm=0.0, base_height_cm=0.0,
                      hover_clearance_mm=HEIGHT_MM - (BOTTLE_CM - CAP_CM / 2) * 10.0)
    out = []
    for bearing, radius in POINTS:
        plan, info = pg.hover_plan(bearing, radius, BOTTLE_CM, CAP_CM, cfg)
        if plan is None or info["degraded"]:
            print(f"  (skipping bearing {bearing}, radius {radius}: the arm can't hold {HEIGHT_MM:.0f} mm exactly there)")
            continue
        out.append({"bearing": bearing, "radius_p": info["radius_mm"], "E": plan["E"], "D": plan["D"], "B": plan["B"]})
    return out


# ── the arm ─────────────────────────────────────────────────────────────────────────────────────────────────────────

class Arm:
    def __init__(self):
        import rclpy
        from sensor_msgs.msg import JointState
        rclpy.init()
        self._rclpy, self._JointState = rclpy, JointState
        self.node = rclpy.create_node("pose_probe")
        self.pub = self.node.create_publisher(JointState, "/joint_command", 10)
        self.pos = {}
        self.node.create_subscription(JointState, "/joint_states", lambda m: self.pos.update(zip(m.name, m.position)), 10)

    def spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            self._rclpy.spin_once(self.node, timeout_sec=0.02)

    def deg(self, joint):
        return math.degrees(self.pos[joint])

    def send(self, e, d, b):
        msg = self._JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = NAMES
        msg.position = [math.radians(e), math.radians(d), math.radians(b)]
        self.pub.publish(msg)

    @staticmethod
    def armed(value):
        out = subprocess.run(["ros2", "param", "set", "/thor_bridge", "armed", "true" if value else "false"],
                             capture_output=True, text=True, timeout=25)
        return "successful" in out.stdout

    def go(self, e, d, b, timeout=90.0):
        t0, last_progress, best, settled = time.time(), time.time(), 1e9, 0
        while time.time() - t0 < timeout:
            self.send(e, d, b)
            self.spin(0.05)
            gap = max(abs(self.deg("E") - e), abs(self.deg("D") - d), abs(self.deg("BC") - b))
            if gap < best - 0.5:
                best, last_progress = gap, time.time()
            settled = settled + 1 if gap < 1.5 else 0
            if settled >= 10:
                return True
            if time.time() - last_progress > 5.0:
                return False
        return False

    def close(self):
        self.node.destroy_node()
        self._rclpy.shutdown()


def read_probe(since, need=6):
    """The bottle's position (mm, x right / y toward the laptop from the assumed axis) once it has held still."""
    try:
        lines = open(PROBE_LOG).read().splitlines()[-80:]
    except OSError:
        return None
    rows = []
    for ln in lines:
        p = ln.split()
        if len(p) >= 5 and float(p[0]) >= since:
            rows.append((float(p[2]), float(p[4])))
    if len(rows) < need:
        return None
    xs, ys = zip(*rows[-need:])
    if max(xs) - min(xs) > 1.0 or max(ys) - min(ys) > 1.0:
        return None
    return float(np.median(xs)) * 10.0, float(np.median(ys)) * 10.0


def report(data, p, resid):
    print("\n== What the arm actually did ==")
    print("  aimed at (bearing, radius)  ->  the camera says (bearing, radius)   miss after the fit")
    for d, r in zip(data, resid):
        bm = math.degrees(math.atan2(d["x_mm"], d["y_mm"]))
        rm = math.hypot(d["x_mm"], d["y_mm"])
        print(f"  {d['bearing']:5.0f} deg, {d['radius_p']:4.0f} mm  ->  {bm:6.1f} deg, {rm:4.0f} mm      {math.hypot(*r):4.0f} mm")
    rec = recommend(p)
    print("\n== The fit ==")
    print(f"  the arm's centre is {p[0]:+.0f} mm to the right and {p[1]:+.0f} mm toward the laptop of where I assumed")
    print(f"  its turn is {p[2]:+.1f} deg off, and the base turns {p[3]:.2f}x what it's told")
    print(f"  the gripper's reach is {p[4]:+.0f} mm off")
    print(f"  the typical miss left over: {math.sqrt((resid ** 2).sum() / len(resid)):.0f} mm")
    print(f"\n  To fix it: e_zero {rec['e_zero_deg']:+.1f}, e_gain {rec['e_gain']:.3f}, radius offset {rec['radius_offset_mm']:+.0f} mm,"
          f" and move the arm's centre by ({p[0] / 10:+.1f}, {p[1] / 10:+.1f}) cm.")
    return rec


def apply_fit():
    fit_ = json.load(open(FIT_PATH))
    rec = fit_["recommend"]
    for f in (PICK_CONFIG, DESK_CALIBRATION):
        shutil.copy(f, f.replace(".json", ".before_aim_fit.json"))
    cfg = json.load(open(PICK_CONFIG))
    cfg.update(e_zero_deg=round(rec["e_zero_deg"], 2), e_gain=round(rec["e_gain"], 4), radius_offset_mm=round(rec["radius_offset_mm"], 1))
    json.dump(cfg, open(PICK_CONFIG, "w"), indent=2)
    from markos_vision.geometry.desk_calibration import DeskCalibration
    cal = DeskCalibration()
    old = list(cal.axis)
    new_u, new_v = cal.from_desk(rec["axis_shift_mm"][0] / 10.0, rec["axis_shift_mm"][1] / 10.0)
    cal.set_axis(new_u, new_v)
    print(f"hover settings: e_zero {cfg['e_zero_deg']}, e_gain {cfg['e_gain']}, radius offset {cfg['radius_offset_mm']} mm")
    print(f"the arm's centre in the tape's u, v: {old[0]:+.1f}, {old[1]:+.1f}  ->  {new_u:+.1f}, {new_v:+.1f}")
    print("(backups: *.before_aim_fit.json). Restart the hover window for it to take effect.")


# ── the measurement ─────────────────────────────────────────────────────────────────────────────────────────────────

def main():
    if "--apply" in sys.argv:
        apply_fit()
        return
    poses = plans()
    if len(poses) < 4:
        print("Too few reachable poses to fit anything. Nothing moved.")
        return
    print(f"\nI'll move the arm to {len(poses)} poses about {HEIGHT_MM / 10:.0f} cm above the desk. At each one:")
    print("  - I stop the arm and DISARM it, so it's safe to put your hand under it;")
    print("  - you put the bottle on the desk directly under the middle of the gripper's fingers, let go, press Enter;")
    print("  - then you take the bottle away before it moves to the next pose.")
    print("The hover window must be open with g OFF and the bottle in view, and the area around the arm clear.")
    input("Enter to start (Ctrl+C to cancel): ")

    arm = Arm()
    data = []
    try:
        arm.spin(3.0)
        if not all(n in arm.pos for n in NAMES):
            print("No joint states from the bridge: is it running and homed? Nothing moved.")
            return
        for i, pose in enumerate(poses, 1):
            print(f"\n-- Pose {i}/{len(poses)}: the planner aims the gripper at bearing {pose['bearing']} deg, "
                  f"radius {pose['radius_p']:.0f} mm")
            input("   Bottle out of the way and hands clear? Enter to move the arm: ")
            if not arm.armed(True):
                print("   could not arm the bridge (is it homed?)")
                return
            ok = arm.go(pose["E"], pose["D"], pose["B"])
            arm.armed(False)
            if not ok:
                print("   the arm did not reach the pose; stopping here (it is disarmed).")
                return
            print("   The arm is holding, DISARMED. Put the bottle directly under the gripper's fingers (the cap under their middle).")
            for attempt in range(4):
                input("   Let go of it and press Enter: ")
                since = time.time()
                reading = None
                while time.time() - since < 10.0 and reading is None:
                    reading = read_probe(since)
                    time.sleep(0.5)
                if reading:
                    break
                print("   The camera doesn't have a steady reading of the bottle yet (is the hover window open and seeing it?).")
            else:
                print("   Giving up on this pose.")
                continue
            print(f"   The camera puts the bottle at x {reading[0] / 10:+.1f}, y {reading[1] / 10:+.1f} cm from the assumed axis.")
            data.append({"bearing": pose["bearing"], "radius_p": pose["radius_p"], "e_cmd": pose["E"], "d": pose["D"], "b": pose["B"],
                         "x_mm": reading[0], "y_mm": reading[1]})
            print("   Take the bottle away.")
        json.dump(data, open(ensure_parent(DATA_PATH), "w"), indent=1)
        if len(data) < 4:
            print("Fewer than 4 measured poses: not enough to fit. What there is was saved to", DATA_PATH)
            return
        p, resid = fit(data)
        rec = report(data, p, resid)
        json.dump({"params": list(map(float, p)), "recommend": rec, "n": len(data)}, open(ensure_parent(FIT_PATH), "w"), indent=1)
        print("\nSaved. To apply it: scripts/wsl/aim_probe.sh --apply   (then restart the hover window).")
        input("\nEnter to bring the arm home: ")
        arm.armed(True)
        arm.go(0.0, 0.0, 0.0)
    finally:
        try:
            arm.armed(False)
        except Exception:  # noqa: BLE001 -- best effort; the bridge also stops chasing after half a second
            pass
        arm.close()
        print("The bridge is disarmed.")


if __name__ == "__main__":
    main()
