import numpy as np

from climate_optimizer.dispatch import FanDispatcher
from climate_optimizer.model.layout import default_layout, default_params
from climate_optimizer.model.thermal import level_heat

MIN = 60_000


def test_levels_average_to_the_target():
    P = default_params(default_layout())
    heats = level_heat(P, "office", np.arange(4), 1, 7.0)
    target = 0.4 * heats[1] + 0.6 * heats[2]
    d = FanDispatcher(hold_min=0)
    got = [d.level(P, "office", "heat", target, 7.0, (0, 3), k * MIN) for k in range(30)]
    assert set(got) <= {1, 2}
    assert abs(np.mean(heats[got]) - target) < 0.05 * target


def test_same_minute_gives_same_answer_and_bounds_hold():
    P = default_params(default_layout())
    d = FanDispatcher()
    a = d.level(P, "office", "heat", 99.0, 7.0, (0, 2), 5 * MIN + 1000)
    b = d.level(P, "office", "heat", 0.0, 7.0, (0, 2), 5 * MIN + 30_000)
    assert a == b == 2
    c = d.level(P, "office", "heat", 0.0, 7.0, (1, 2), 9 * MIN)
    assert c == 1


def test_hold_limits_flapping():
    P = default_params(default_layout())
    heats = level_heat(P, "living", np.arange(4), 1, 7.0)
    target = 0.5 * (heats[0] + heats[1])
    d = FanDispatcher(hold_min=3)
    got = [d.level(P, "living", "heat", target, 7.0, (0, 3), k * MIN) for k in range(30)]
    changes = sum(1 for x, y in zip(got, got[1:]) if x != y)
    assert changes <= 10
    assert abs(np.mean(heats[got]) - target) < 0.1 * target
