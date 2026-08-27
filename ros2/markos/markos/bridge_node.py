#!/usr/bin/env python3
"""
bridge_node.py — the one thing standing between ROS2 and the Arduino
============================================================================
Wraps the same serial protocol mark1os.py already speaks to the firmware:
HOME/MOVE/G<n> out, "[fw] ... done" acknowledgments back.

It homes on startup, then publishes the arm's confirmed position to
/joint_states (what lets RViz2 show the real robot). It will also chase
targets sent to /joint_command and /gripper_command — but only while the
`armed` parameter is true, and it starts out false on purpose: bringing the
node up should never be able to move the arm by itself.

  ros2 param set /thor_bridge armed true     # let it move
  ros2 param set /thor_bridge armed false    # stop, instantly
  ros2 service call /home std_srvs/srv/Trigger   # re-home after a fault

A fault (a lost ack, a firmware direction that doesn't match hw_config.py)
disarms it and it stays refused until it's been re-homed and re-armed.

Requires: pyserial (already available to this system's python3)
Run:      ros2 run markos thor_bridge
      or: ros2 run markos thor_bridge --ros-args -p port:=/dev/ttyACM0
      or: ros2 run markos thor_bridge --ros-args -p mock:=true   # no arm
"""

import math
import queue
import re
import time

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from std_srvs.srv import Trigger

from .axis_controller import AxisController
from .hw_config import HOMING_ORDER, HOMING_STEP_TIMEOUT, SIGN
from .serial_link import HW_BOOT_DELAY_MS, FirmwareLink
from .sim_firmware import SimFirmware

# The firmware's axis letters, mapped to this project's actual URDF joint
# names (thor_urdf/urdf/thor.urdf.xacro) — E/D/A/BC, not Thor's original
# joint_1/joint_3/joint_4/joint_5 numbering.
AXIS_TO_JOINT = {"E": "E", "D": "D", "A": "A", "B": "BC"}
JOINT_TO_AXIS = {joint: axis for axis, joint in AXIS_TO_JOINT.items()}

# Movable joints this bridge neither drives nor measures. robot_state_publisher
# leaves out everything downstream of a movable joint it has no state for,
# which cuts the rest of the arm out of the model, so these are reported at
# zero, the URDF's rest pose. (A is here because it has no beacon to be homed
# against; the gripper joints because the URDF's finger angles aren't mapped
# to the servo yet.)
UNDRIVEN_JOINTS = [
    "A",
    "gripperbase_to_armgearright", "gripperbase_to_armgearleft",
    "gripperbase_to_armsimpleright", "gripperbase_to_armsimpleleft",
    "armgearright_to_fingerright", "armgearleft_to_fingerleft",
]

HOME_RESULT_RE = re.compile(r"^\[fw\] HOME (\w) (found|already at home|FAILED)")


class ThorBridge(Node):
    def __init__(self) -> None:
        super().__init__("thor_bridge")

        self.declare_parameter("port", "/dev/ttyACM0")
        self.declare_parameter("mock", False)
        self.declare_parameter("armed", False)
        # Flip one to -1 if RViz shows that joint turning the opposite way
        # from the real arm. Applied to targets going out and positions
        # coming back, so the two stay consistent. The defaults are the
        # measured ones in hw_config, which the planner's limits follow too.
        for joint in ("E", "D", "BC"):
            self.declare_parameter(f"sign_{joint}", SIGN[joint])

        self.ctl = AxisController()
        self.homing_queue = []
        self.homing_axis = None
        self.homing_sent_t = 0.0
        self.grip_openness = None
        self._armed = False

        port = self.get_parameter("port").get_parameter_value().string_value
        if self.get_parameter("mock").get_parameter_value().bool_value:
            self.get_logger().warning("MOCK MODE — talking to a simulated firmware, not the arm")
            self.link = SimFirmware()
            self.link.open(port)
        else:
            self.link = FirmwareLink()
            self.get_logger().info(f"Opening {port} at boot...")
            try:
                self.link.open(port)
            except Exception as exc:
                self.get_logger().error(
                    f"Could not open {port}: {exc!r} — check the port name (try "
                    "'ros2 run markos thor_bridge --ros-args -p port:=/dev/ttyXXX') "
                    "and that the Arduino is actually attached into WSL via usbipd."
                )
                raise

        self.joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self.create_subscription(JointState, "joint_command", self._on_joint_command, 10)
        self.create_subscription(Float32, "gripper_command", self._on_gripper_command, 10)
        self.create_service(Trigger, "home", self._on_home_request)
        self.add_on_set_parameters_callback(self._on_parameters)

        # Fires once, after the firmware's had time to boot (same
        # HW_BOOT_DELAY_MS wait mark1os.py does), then cancels itself.
        self._boot_timer = self.create_timer(HW_BOOT_DELAY_MS / 1000.0, self._start_homing)
        self.create_timer(0.02, self._tick)

    # ── homing ─────────────────────────────────────────────────────────────

    def _start_homing(self) -> None:
        self._boot_timer.cancel()
        self.link.reset_buffers()
        self.homing_queue = list(HOMING_ORDER)
        self.get_logger().info(f"Homing: {' -> '.join(HOMING_ORDER)}")
        self._advance_homing()

    def _advance_homing(self) -> None:
        if not self.homing_queue:
            self.homing_axis = None
            self.get_logger().info("Homing complete")
            return
        self.homing_axis = self.homing_queue.pop(0)
        self.homing_sent_t = time.monotonic()
        self.link.send_line(f"HOME {self.homing_axis}")

    def _homing_busy(self) -> bool:
        return self.homing_axis is not None or bool(self.homing_queue)

    def _on_home_request(self, _request, response):
        if self._homing_busy() or self.ctl.in_flight:
            response.success = False
            response.message = "arm is busy — try again once it's idle"
            return response
        self._set_armed(False)
        self.ctl.invalidate()
        self.ctl.clear_fault()
        self._start_homing()
        response.success = True
        response.message = "homing started"
        return response

    # ── arming ─────────────────────────────────────────────────────────────

    def _on_parameters(self, params) -> SetParametersResult:
        for p in params:
            if p.name != "armed":
                continue
            if p.value and not self._armed:
                problem = self._why_not_armable()
                if problem:
                    return SetParametersResult(successful=False, reason=problem)
            self._armed = bool(p.value)
            if not self._armed:
                self._stand_down()
        return SetParametersResult(successful=True)

    def _why_not_armable(self):
        if self.ctl.fault is not None:
            return f"latched fault: {self.ctl.fault} — call /home first"
        if self._homing_busy():
            return "still homing"
        unhomed = [a for a in HOMING_ORDER if self.ctl.pos[a] is None]
        if unhomed:
            return f"not homed: {', '.join(unhomed)}"
        return None

    def _set_armed(self, value: bool) -> None:
        self.set_parameters([Parameter("armed", Parameter.Type.BOOL, value)])

    def _stand_down(self) -> None:
        self.ctl.cancel_targets()
        self.grip_openness = None

    # ── commands in ────────────────────────────────────────────────────────

    def _sign(self, joint: str) -> int:
        if not self.has_parameter(f"sign_{joint}"):
            return 1
        return -1 if self.get_parameter(f"sign_{joint}").get_parameter_value().integer_value < 0 else 1

    def _on_joint_command(self, msg: JointState) -> None:
        if not self._armed:
            self.get_logger().warning(
                "joint_command ignored — not armed (ros2 param set /thor_bridge armed true)",
                throttle_duration_sec=5.0,
            )
            return
        now = time.monotonic()
        for joint, rad in zip(msg.name, msg.position):
            axis = JOINT_TO_AXIS.get(joint)
            # A (and the fixed/gripper joints) can show up in a full-robot
            # JointState; they're just not this bridge's to drive.
            if axis is None or axis not in self.ctl.target:
                continue
            deg = math.degrees(rad) * self._sign(joint)
            if self.ctl.set_target_deg(axis, deg, now):
                self.get_logger().info(
                    f"{joint} target {math.degrees(rad):.1f} deg is outside its range — clamped",
                    throttle_duration_sec=2.0,
                )

    def _on_gripper_command(self, msg: Float32) -> None:
        if not self._armed:
            self.get_logger().warning(
                "gripper_command ignored — not armed", throttle_duration_sec=5.0
            )
            return
        self.grip_openness = float(msg.data)

    # ── the loop ───────────────────────────────────────────────────────────

    def _tick(self) -> None:
        while True:
            try:
                kind, payload = self.link.rx.get_nowait()
            except queue.Empty:
                break
            if kind == "error":
                self.get_logger().warning(payload)
                continue

            self.ctl.on_ack(payload)

            hm = HOME_RESULT_RE.match(payload)
            if hm:
                axis, result = hm.group(1), hm.group(2)
                if result == "FAILED":
                    self.get_logger().warning(f"HOME {axis} failed")
                else:
                    self.ctl.mark_homed(axis)
                if self.homing_axis == axis:
                    self.homing_axis = None
                    self._advance_homing()

        if self.homing_axis is not None and time.monotonic() - self.homing_sent_t > HOMING_STEP_TIMEOUT:
            self.get_logger().error(f"HOME {self.homing_axis} got no answer — giving up on homing")
            self.homing_axis = None
            self.homing_queue = []

        if self._armed:
            self._drive()
        self._report_fault()
        self._publish_joint_states()

    def _drive(self) -> None:
        now = time.monotonic()
        line = self.ctl.next_move(now)
        if line:
            self.link.send_line(line)
        if self.grip_openness is not None:
            grip = self.ctl.gripper_line(self.grip_openness, now)
            if grip:
                self.link.send_line(grip)

    def _report_fault(self) -> None:
        fault = self.ctl.take_fault()
        if fault:
            self.get_logger().error(f"FAULT — disarming: {fault}")
            self._set_armed(False)

    def _publish_joint_states(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for axis, joint in AXIS_TO_JOINT.items():
            rad = self.ctl.position_rad(axis)
            if rad is None:
                continue
            msg.name.append(joint)
            msg.position.append(rad * self._sign(joint))
        if msg.name:
            for joint in UNDRIVEN_JOINTS:
                if joint not in msg.name:
                    msg.name.append(joint)
                    msg.position.append(0.0)
            self.joint_pub.publish(msg)

    def destroy_node(self) -> None:
        self.link.close()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = ThorBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
