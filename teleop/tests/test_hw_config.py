"""hw_config: the desktop app's safety and calibration state. A change to these numbers should be a deliberate one."""

import hw_config as cfg


def test_the_homing_order_starts_with_the_axis_that_has_a_hardware_limit_switch():
    assert cfg.HOMING_ORDER == ["B", "D", "E"]


def test_axis_a_has_no_beacon_so_it_is_neither_homed_nor_trusted_for_automatic_motion():
    assert "A" not in cfg.HOMING_ORDER
    assert cfg.HW_CALIBRATED_A is False


def test_the_gesture_speed_ceiling_is_cut_further_for_the_two_hinges():
    assert cfg.HW_ELBOW_B_SCALE < 1.0 and cfg.HW_SHOULDER_D_SCALE < 1.0
    assert cfg.HW_MAX_SPEED > 0


def test_every_axis_has_a_positive_steps_per_degree():
    for name in ("A", "B", "D", "E"):
        assert getattr(cfg, f"stepsPerDeg{name}") > 0


def test_homing_gets_more_time_than_the_worst_case_search():
    assert cfg.HOMING_STEP_TIMEOUT >= 60.0
