"""The prompts in ``calibrate_desk`` for where the two red stars are on the robot."""

import builtins
import io
from contextlib import redirect_stdout

import pytest

from markos_vision.apps import calibrate_desk
from markos_vision.geometry.desk_calibration import DeskCalibration


@pytest.fixture
def run(monkeypatch):
    def _run(answers):
        it = iter(answers)
        monkeypatch.setattr(builtins, "input", lambda prompt="": next(it))
        out = io.StringIO()
        cal = DeskCalibration(path=None)
        with redirect_stdout(out):
            ok = calibrate_desk.ask_star_geometry(cal)
        return ok, cal, out.getvalue()

    return _run


# The way the stars are measured: LEFT of the top-right corner, DOWN from the top edge; the box height is left at the CAD
# default. The stars are 7.9 cm apart, both 3.5 cm down from the top of a 7.05 cm face.

def test_two_number_entry_gives_positions_on_the_end_face(run):
    ok, cal, text = run(["12.97 3.5", "", "5.07 3.55"])
    assert ok and cal.star3d == [[-12.97, 0.0, 7.05 - 3.5], [-5.07, 0.0, 7.05 - 3.55]]
    assert "7.9 cm apart" in text and "doesn't match" not in text


def test_a_measured_box_height_is_used(run):
    _, cal, _ = run(["12.97 3.5", "6.8", "5.07 3.55"])
    assert cal.star3d[0][2] == 6.8 - 3.5 and cal.star3d[1][2] == 6.8 - 3.55


def test_negative_impossible_and_non_numeric_entries_are_refused_then_the_corrected_one_accepted(run):
    # a negative distance, letters, a 'down' bigger than the face is tall (the box-height question comes here, answered with
    # Enter for the default), then the corrected entry, then the right star
    ok, cal, text = run(["-12.97 3.5", "abc", "12.97 8.0", "", "12.97 3.5", "5.07 3.55"])
    assert ok and cal.star3d[0] == [-12.97, 0.0, 7.05 - 3.5]
    assert "positive" in text and "check 'down'" in text and "Type two numbers" in text


def test_the_three_number_form_still_works_for_a_star_off_the_end_face(run):
    ok, cal, _ = run(["-12.97 0 3.55", "-5.07 0 3.5"])
    assert ok and cal.star3d == [[-12.97, 0.0, 3.55], [-5.07, 0.0, 3.5]]


def test_stars_entered_the_wrong_distance_apart_are_flagged(run):
    _, _, text = run(["12.97 3.5", "", "2.0 3.5"])
    assert "doesn't match" in text


def test_enter_cancels(run):
    ok, cal, _ = run([""])
    assert not ok and cal.star3d is None
