"""gesture_math: the pure geometry on MediaPipe hand landmarks (no camera needed, made-up landmarks are enough)."""

from types import SimpleNamespace

import gesture_math as gm
import pytest


def hand(**points):
    """21 landmarks, all at the origin except the ones given (index -> (x, y))."""
    landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(21)]
    for index, (x, y) in points.items():
        landmarks[int(index.lstrip("p"))] = SimpleNamespace(x=x, y=y)
    return landmarks


def open_hand(scale=1.0):
    """A flat open hand: wrist at the origin, fingers pointing away. `scale` mimics the hand nearer to or farther from the camera."""
    lm = hand()
    lm[9] = SimpleNamespace(x=0.0, y=0.2 * scale)  # middle finger base knuckle
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        lm[pip] = SimpleNamespace(x=0.02 * tip / 8 * scale, y=0.25 * scale)
        lm[tip] = SimpleNamespace(x=0.02 * tip / 8 * scale, y=0.45 * scale)  # tip beyond its own knuckle
    return lm


def test_pinch_ratio_means_the_same_thing_near_and_far():
    def pinch(scale):
        lm = open_hand(scale)
        lm[4] = SimpleNamespace(x=0.10 * scale, y=0.30 * scale)
        lm[8] = SimpleNamespace(x=0.10 * scale + 0.04 * scale, y=0.30 * scale)
        return lm

    near, far = gm.pinch_ratio(pinch(1.0)), gm.pinch_ratio(pinch(0.4))
    assert near == pytest.approx(far)
    assert near < gm.PINCH_ON, "a 4 cm gap on a 20 cm-tall palm reads as a closed pinch"


def test_pinch_ratio_does_not_divide_by_zero_for_a_degenerate_hand():
    assert gm.pinch_ratio(hand()) == 1.0


def test_the_pinch_has_hysteresis_so_the_toggle_does_not_chatter():
    assert gm.PINCH_ON < gm.PINCH_OFF
    assert gm.GRIP_PINCH_ON < gm.GRIP_PINCH_OFF


def test_hand_scale_and_pinch_distance_are_euclidean():
    lm = hand(p0=(0.0, 0.0), p9=(0.3, 0.4), p4=(0.1, 0.1), p8=(0.4, 0.5))
    assert gm.hand_scale(lm) == pytest.approx(0.5)
    assert gm.pinch_distance(lm) == pytest.approx(0.5)


def test_thumb_to_pinky_distance_is_kept_separate_from_the_thumb_index_pinch():
    lm = hand(p4=(0.0, 0.0), p8=(0.05, 0.0), p20=(0.3, 0.4))
    assert gm.thumb_pinky_distance(lm) == pytest.approx(0.5)
    assert gm.pinch_distance(lm) == pytest.approx(0.05)


def test_an_open_hand_is_not_a_fist():
    assert not gm.is_fist(open_hand())


def test_a_curled_hand_is_a_fist():
    lm = open_hand()
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        lm[tip] = SimpleNamespace(x=lm[pip].x, y=0.15)  # the tip folds back closer to the wrist than its own knuckle
    assert gm.is_fist(lm)


def test_three_curled_fingers_are_not_enough_for_a_fist():
    lm = open_hand()
    for tip, pip in [(8, 6), (12, 10), (16, 14)]:
        lm[tip] = SimpleNamespace(x=lm[pip].x, y=0.15)
    assert not gm.is_fist(lm)
