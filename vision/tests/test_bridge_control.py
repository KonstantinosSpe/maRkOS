"""p arms and disarms, h homes: ``BridgeControl`` against a MOCK bridge on an isolated ROS network.

Needs a ROS 2 installation with the markos package built and sourced (skipped otherwise). The mock bridge runs on its own
ROS domain, so this can never reach the real arm's bridge.
"""

import os
import subprocess
import time

import pytest

pytestmark = pytest.mark.ros


@pytest.fixture(scope="module")
def control():
    os.environ["ROS_DOMAIN_ID"] = "77"  # an isolated ROS network: nothing here can reach the real bridge
    import rclpy

    from markos_vision.apps import hover

    bridge = subprocess.Popen(["ros2", "run", "markos", "thor_bridge", "--ros-args", "-p", "mock:=true"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=os.environ)
    rclpy.init()
    node = rclpy.create_node("bridge_control_test")
    yield hover.BridgeControl(node)
    bridge.terminate()
    try:
        bridge.wait(timeout=5)
    except subprocess.TimeoutExpired:
        bridge.kill()
    node.destroy_node()
    rclpy.shutdown()


def until(control, condition, seconds, what):
    t0 = time.time()
    while time.time() - t0 < seconds:
        control.spin()
        control.poll()
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}; state: armed={control.armed} complete={control.complete} "
                         f"homing={control.homing} note={control.note!r}")


def test_p_arms_and_disarms_and_h_homes(control):
    until(control, lambda: control.complete and control.armed is False, 40, "the mock bridge to finish homing and report not armed")

    control.set_armed(True)
    until(control, lambda: control.armed is True, 8, "arming")
    control.set_armed(False)
    until(control, lambda: control.armed is False, 8, "disarming")

    control.set_armed(True)
    until(control, lambda: control.armed is True, 8, "arming again")
    control.home()
    until(control, lambda: control.homing, 8, "homing to start")
    assert control.armed is False, "homing must disarm"
    until(control, lambda: not control.homing, 60, "homing to finish")
    assert control.complete and control.armed is False
