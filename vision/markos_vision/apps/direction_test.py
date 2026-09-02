"""Which way does the real arm move when the bridge says E, D or B goes positive? Moves each a few degrees, one at a time, back
to the start each time, and asks what the real arm did, in terms of the room (you, the laptop, the bottle side). Needs the
bridge up and homed. Arms the bridge for the moves and disarms it again when it finishes, when you press Ctrl-C, or if
anything goes wrong.

The questions assume the arm at home faces the bottle side (right in the camera's picture), where its gripper sticks out.

    python -m markos_vision.apps.direction_test          # real arm: watch it, answer the questions
    python -m markos_vision.apps.direction_test --auto   # no questions, for the mock arm
"""
import argparse
import json
import math
import subprocess
import time

import rclpy
from sensor_msgs.msg import JointState

from markos_vision.config import DIRECTION_RESULT as RESULT_PATH
from markos_vision.config import ensure_parent

# (joint, degrees away from home, question, the answer that means it moves the way the model does)
MOVES = [
    ("E", 15.0, "The gripper end of the arm swung: toward YOU (the laptop) [t], or AWAY from you [a]?", "a"),
    ("D", 5.0, "The whole arm tipped: to the RIGHT, toward the bottle side [r], or to the LEFT [l]?", "r"),
    ("BC", -10.0, "The part above the elbow (forearm and gripper) tipped: to the RIGHT [r], or to the LEFT [l]?", "l"),
]
NAMES = ["E", "D", "BC"]
SETTLE_TOLERANCE_DEG = 1.0
MOVE_TIMEOUT_S = 12.0
PUBLISH_HZ = 20.0


class Rig:
    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node("direction_test")
        self.pub = self.node.create_publisher(JointState, "/joint_command", 10)
        self.pos = {}
        self.node.create_subscription(JointState, "/joint_states", self._on_state, 10)

    def _on_state(self, msg):
        self.pos.update(zip(msg.name, msg.position))

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self.node, timeout_sec=0.02)

    def deg(self, joint):
        return math.degrees(self.pos[joint]) if joint in self.pos else None

    def send(self, targets_deg):
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = NAMES
        msg.position = [math.radians(targets_deg.get(n, 0.0)) for n in NAMES]
        self.pub.publish(msg)

    def go(self, joint, degrees):
        """Chase one target until the bridge reports the joint there. Returns where it ended up."""
        targets = {joint: degrees}
        settled, end = 0, time.time() + MOVE_TIMEOUT_S
        while time.time() < end and settled < 10:
            self.send(targets)
            self.spin_for(1.0 / PUBLISH_HZ)
            here = self.deg(joint)
            settled = settled + 1 if here is not None and abs(here - degrees) < SETTLE_TOLERANCE_DEG else 0
        return self.deg(joint)

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()


def set_armed(value):
    out = subprocess.run(["ros2", "param", "set", "/thor_bridge", "armed", "true" if value else "false"],
                         capture_output=True, text=True, timeout=20)
    return "successful" in out.stdout, (out.stdout + out.stderr).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--auto", action="store_true", help="no questions (for the mock arm)")
    args = p.parse_args()

    rig = Rig()
    answers = {}
    try:
        rig.spin_for(3.0)
        if not all(n in rig.pos for n in NAMES):
            print("No joint states from the bridge: is it running, and has it finished homing? Nothing moved.")
            return
        print("Bridge is up. Where it thinks the arm is (degrees from home): "
              + "  ".join(f"{n} {rig.deg(n):+.1f}" for n in NAMES))
        if not args.auto:
            print("\nEach move is a few degrees and then back to the start. Keep a hand by the arm's power switch.")
            print("Answer from where you sit at the laptop, looking at the arm the way the camera does.")
            if input("Press Enter to arm the bridge and begin (q to quit): ").strip().lower() == "q":
                return
        ok, text = set_armed(True)
        if not ok:
            print("Could not arm the bridge:", text)
            return

        for joint, degrees, question, expected in MOVES:
            if not args.auto:
                answer = input(f"\nNext: {joint} {degrees:+.0f} deg. Enter to move, s to skip, q to stop: ").strip().lower()
                if answer == "q":
                    break
                if answer == "s":
                    continue
            before = rig.deg(joint)
            reached = rig.go(joint, degrees)
            print(f"  the bridge counted {joint} going from {before:+.1f} to {reached:+.1f} deg (asked for {degrees:+.0f})")
            if reached is None or abs(reached - before) < 0.3:
                print("  it did not move: is the arm powered, and did the bridge say why?")
            elif not args.auto:
                answers[joint] = input("  " + question + " ").strip().lower()[:1]
            rig.go(joint, 0.0)

        if answers:
            flips = [j for j, a in answers.items() if a and a != next(e for jj, _, _, e in MOVES if jj == j)]
            with open(ensure_parent(RESULT_PATH), "w") as f:
                json.dump({"answers": answers, "flips": flips}, f)
            print(f"\nyour answers: {answers}   (saved to {RESULT_PATH})")
            print("These need the sign flipped on the bridge: " + (", ".join(flips) if flips else "none"))
    finally:
        try:
            set_armed(False)
        except Exception as exc:  # noqa: BLE001 -- disarming is best effort; the bridge also stops chasing after 0.5 s
            print("could not disarm:", exc)
        rig.close()
        print("Bridge disarmed.")


if __name__ == "__main__":
    main()
