"""The hover window's height keys: w / s move the hover in 33 mm steps (six presses are about 20 cm), , and . in 5 mm steps."""

from markos_vision.apps import hover
from markos_vision.geometry.pick_geometry import PickConfig


def test_coarse_steps_walk_down_to_the_floor_and_stop_there():
    cfg = PickConfig(path=None)
    cfg.values["hover_clearance_mm"] = 100.0
    assert hover.change_clearance(cfg, -hover.HEIGHT_COARSE_MM) == 67.0
    assert hover.change_clearance(cfg, -hover.HEIGHT_COARSE_MM) == 34.0
    assert hover.change_clearance(cfg, -hover.HEIGHT_COARSE_MM) == hover.HEIGHT_FLOOR_MM  # held at the floor, not below
    assert hover.change_clearance(cfg, -5.0) == hover.HEIGHT_FLOOR_MM


def test_fine_steps_move_by_five_millimetres():
    cfg = PickConfig(path=None)
    cfg.values["hover_clearance_mm"] = hover.HEIGHT_FLOOR_MM
    assert hover.change_clearance(cfg, 5.0) == hover.HEIGHT_FLOOR_MM + 5.0


def test_six_coarse_presses_are_about_twenty_centimetres():
    assert 6 * hover.HEIGHT_COARSE_MM == 198.0
