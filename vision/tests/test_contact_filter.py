"""ContactFilter: the steadied point where the bottle meets the desk.

The segmentation outline wobbles a pixel or two every frame and now and then jumps by several. The filter has to be steady
on a still bottle, ignore a single spike, and still follow a real move quickly.
"""

import numpy as np
import pytest

from markos_vision.geometry.desk_calibration import ContactFilter


def noisy(rng, true_xy, sigma=2.0, spike_p=0.06, spike=22.0):
    """A wobbling outline: ~2 px of jitter every frame, and now and then a jump of a couple dozen pixels."""
    x = true_xy[0] + rng.normal(0, sigma)
    y = true_xy[1] + rng.normal(0, sigma)
    if rng.random() < spike_p:
        x += rng.choice([-1, 1]) * spike * rng.uniform(0.6, 1)
        y += rng.choice([-1, 1]) * spike * rng.uniform(0.6, 1)
    return x, y


@pytest.fixture
def rng():
    return np.random.default_rng(8)


def test_still_bottle_is_much_steadier_than_the_raw_point(rng):
    true = (320.0, 400.0)
    f = ContactFilter()
    raw, out = [], []
    for _ in range(400):
        r = noisy(rng, true)
        raw.append(r)
        out.append(f.update(r))
    raw, out = np.array(raw[30:]), np.array(out[30:])
    raw_dev = np.hypot(*(raw - true).T)
    out_dev = np.hypot(*(out - true).T)
    assert out_dev.mean() < 0.4 * raw_dev.mean()
    assert out_dev.max() < 6.0
    assert np.hypot(*np.diff(out, axis=0).T).mean() < np.hypot(*np.diff(raw, axis=0).T).mean()


def test_a_single_spike_does_not_drag_the_point():
    f = ContactFilter()
    for _ in range(20):
        f.update((320.0, 400.0))
    before = f.value
    after = f.update((350.0, 375.0))  # one wild frame
    assert np.hypot(after[0] - before[0], after[1] - before[1]) < 1.5


def test_a_real_move_arrives_quickly(rng):
    f = ContactFilter()
    for _ in range(20):
        f.update(noisy(rng, (320.0, 400.0), spike_p=0))
    frames_to_arrive = None
    for i in range(1, 40):
        v = f.update(noisy(rng, (180.0, 360.0), spike_p=0))
        if np.hypot(v[0] - 180, v[1] - 360) < 5:
            frames_to_arrive = i
            break
    assert frames_to_arrive is not None and frames_to_arrive <= 8


def test_a_slow_drift_is_followed_without_snapping(rng):
    f = ContactFilter()
    lag = []
    for i in range(120):
        truth = (300.0 + 0.5 * i, 400.0)  # 7.5 px per second at 15 fps
        v = f.update(noisy(rng, truth, spike_p=0.03))
        lag.append(np.hypot(v[0] - truth[0], v[1] - truth[1]))
    assert np.mean(lag[20:]) < 4.0


def test_reset_forgets_everything():
    f = ContactFilter()
    f.update((5.0, 5.0))
    f.reset()
    assert f.value is None
    assert f.update((10.0, 10.0)) == (10.0, 10.0)
