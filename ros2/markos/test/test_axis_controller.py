"""
Drives AxisController against SimFirmware the way bridge_node does, minus
ROS: acks in, next_move out, target refreshed every loop like a publisher
would. Runs with `python3 -m pytest test/test_axis_controller.py` from the
package directory.
"""

import re
import time

from markos import hw_config as cfg
from markos.axis_controller import STEPS_PER_DEG, AxisController
from markos.sim_firmware import SimFirmware

HOME_RE = re.compile(r"^\[fw\] HOME (\w) found")


def make(**sim_kwargs):
    sim = SimFirmware(speedup=1000.0, **sim_kwargs)
    sim.open("sim")
    return AxisController(), sim


def drain(ctl, sim):
    while not sim.rx.empty():
        _, line = sim.rx.get_nowait()
        ctl.on_ack(line)
        m = HOME_RE.match(line)
        if m:
            ctl.mark_homed(m.group(1))


def home_all(ctl, sim):
    for axis in cfg.HOMING_ORDER:
        sim.send_line(f"HOME {axis}")
    deadline = time.monotonic() + 3.0
    while any(ctl.pos[a] is None for a in cfg.HOMING_ORDER):
        assert time.monotonic() < deadline, "sim never finished homing"
        drain(ctl, sim)
        time.sleep(0.001)


def run(ctl, sim, targets_deg, seconds=0.6, until=None):
    """Refreshes the targets every loop, like a node publishing at 30 Hz."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        now = time.monotonic()
        for axis, deg in targets_deg.items():
            ctl.set_target_deg(axis, deg, now)
        drain(ctl, sim)
        line = ctl.next_move(now)
        if line:
            sim.send_line(line)
        if until and until():
            return
        time.sleep(0.001)


def moves_for(sim, axis):
    return [line for line in sim.log if line.startswith("MOVE") and re.search(rf"\b{axis}-?\d+", line)]


def test_homing_zeroes_counters_from_off_home():
    ctl, sim = make(start_pos={"B": -80, "D": 900, "E": 300})
    home_all(ctl, sim)
    assert [ctl.pos[a] for a in "BDE"] == [0, 0, 0]
    assert ctl.pos["A"] is None


def test_converges_to_target_and_tracks_flipped_axis():
    ctl, sim = make()
    home_all(ctl, sim)
    want = round(30 * STEPS_PER_DEG["E"])
    run(ctl, sim, {"E": 30.0}, until=lambda: ctl.pos["E"] == want)
    assert ctl.pos["E"] == want
    # E is flipped in the firmware: the physical counter went the other way
    # while the bridge's own counter stayed in the command frame.
    assert sim.pos["E"] == -want
    assert ctl.fault is None


def test_each_move_respects_the_rate_cap():
    ctl, sim = make()
    home_all(ctl, sim)
    run(ctl, sim, {"E": 60.0, "D": 20.0, "B": -30.0}, seconds=0.8)
    for line in filter(lambda l: l.startswith("MOVE"), sim.log):
        for axis, n in re.findall(r"([EDB])(-?\d+)", line):
            assert abs(int(n)) <= cfg.MAX_MOVE_STEPS[axis], line


def test_targets_past_the_limit_are_clamped_not_chased():
    ctl, sim = make()
    home_all(ctl, sim)
    assert ctl.set_target_deg("D", 500.0, time.monotonic()) is True
    lo, hi = cfg.STEP_LIMITS["D"]
    run(ctl, sim, {"D": 500.0}, seconds=1.5, until=lambda: ctl.pos["D"] == hi)
    assert ctl.pos["D"] == hi
    assert not any("SOFT LIMIT" in text for _, text in list(sim.rx.queue))


def test_blocked_direction_isnt_retried_until_target_reverses():
    ctl, sim = make(walls={"D": (-50, 50)})
    home_all(ctl, sim)
    run(ctl, sim, {"D": 10.0}, seconds=0.6)          # 10 deg = ~206 steps, wall at 50
    assert ctl.pos["D"] == 50
    sent = len(moves_for(sim, "D"))
    run(ctl, sim, {"D": 10.0}, seconds=0.4)
    assert len(moves_for(sim, "D")) == sent          # no retry into the wall

    run(ctl, sim, {"D": -2.0}, seconds=0.6, until=lambda: ctl.pos["D"] < 0)
    assert ctl.pos["D"] < 50                         # reversing works again


def test_wrong_direction_flag_faults_and_stops_motion():
    ctl, sim = make(flips={"E": False})               # firmware disagrees with hw_config
    home_all(ctl, sim)
    run(ctl, sim, {"E": 30.0}, seconds=0.3, until=lambda: ctl.fault is not None)
    assert ctl.fault and "FW_DIR_FLIP" in ctl.fault
    assert ctl.next_move(time.monotonic()) is None
    assert ctl.take_fault() == ctl.fault
    assert ctl.take_fault() is None                   # reported once...
    assert ctl.next_move(time.monotonic()) is None    # ...but still refused
    ctl.clear_fault()


def test_lost_ack_faults_and_forgets_position():
    ctl = AxisController()
    for axis in "BDE":
        ctl.mark_homed(axis)
    ctl.set_target_deg("E", 30.0, 0.0)
    assert ctl.next_move(0.0).startswith("MOVE E")
    assert ctl.next_move(0.1) is None                 # still waiting on the ack
    ctl.set_target_deg("E", 30.0, cfg.MOVE_ACK_TIMEOUT + 0.05)
    assert ctl.next_move(cfg.MOVE_ACK_TIMEOUT + 0.1) is None
    assert ctl.fault and "no ack" in ctl.fault
    assert all(ctl.pos[a] is None for a in "BDE")


def test_stale_target_is_dropped():
    ctl = AxisController()
    for axis in "BDE":
        ctl.mark_homed(axis)
    ctl.set_target_deg("E", 30.0, 0.0)
    assert ctl.next_move(cfg.COMMAND_TIMEOUT + 0.1) is None
    assert ctl.target["E"] is None


def test_unhomed_axes_and_A_are_never_driven():
    ctl = AxisController()
    ctl.set_target_deg("E", 30.0, 0.0)
    assert ctl.next_move(0.0) is None                 # E not homed yet
    assert ctl.set_target_deg("A", 30.0, 0.0) is False
    assert "A" not in ctl.target


def test_gripper_maps_openness_to_servo_angle_once():
    ctl = AxisController()
    assert ctl.gripper_line(1.0, 0.0) == f"G{cfg.GRIP_OPEN}"
    assert ctl.gripper_line(1.0, 1.0) is None         # unchanged
    assert ctl.gripper_line(0.0, 2.0) == f"G{cfg.GRIP_CLOSED}"
    assert ctl.gripper_line(-3.0, 3.0) is None        # clamps to closed, already there
    assert ctl.gripper_line(9.0, 4.0) == f"G{cfg.GRIP_OPEN}"
    assert ctl.gripper_line(0.0, 4.02) is None        # too soon after the last one
