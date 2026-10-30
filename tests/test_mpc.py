from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from climate_optimizer import mpc
from climate_optimizer.comfort import ComfortFunction, Side, default_functions
from climate_optimizer.model.layout import default_layout, default_params
from climate_optimizer.model.thermal import DIST, build

TZ = ZoneInfo("Europe/Berlin")
MIN = 60_000


def ms(*a):
    return int(datetime(*a, tzinfo=TZ).timestamp() * 1000)


def steady_x0(params, layout, Ta, tout):
    Tm, Tw = [], []
    for z, ta in zip(layout.zone_ids, Ta):
        p = params.zones[z]
        M = np.array([[-(p["Ham"] + p["Hmw"]), p["Hmw"]], [p["Hmw"], -(p["Hmw"] + p["Hwo"] + p["Hwn"])]])
        tm, tw = np.linalg.solve(M, -np.array([p["Ham"] * ta, p["Hwo"] * tout + p["Hwn"] * 20.0]))
        Tm.append(tm)
        Tw.append(tw)
    return np.array(list(Ta) + Tm + Tw)


def problem(price_fn, tout=2.0, Ta=(20.5, 19.0, 20.5, 21.5, 19.5), settings=None, functions=None, units=None,
            comfort=None, t0=None):
    L = default_layout()
    P = default_params(L)
    S = build(L, P)
    st = settings or mpc.Settings()
    t0 = t0 or ms(2026, 1, 14, 13, 2)
    day0 = ms(2026, 1, 14)
    iv = [(day0 + i * 900_000, day0 + (i + 1) * 900_000, float(price_fn((i % 96) / 4))) for i in range(2 * 96)]
    t = mpc.step_grid(t0, iv[-1][1], st)
    N = len(t) - 1
    dist = np.zeros((N, len(DIST)))
    dist[:, DIST.index("tout")] = tout
    dist[:, DIST.index("one")] = 1.0
    comfort = comfort or {"office": 21.0, "bedroom": 19.0, "living": 21.5, "bath": 22.0}
    fns = functions or {z: default_functions()[0] for z in comfort}
    units = units or [mpc.UnitSpec(u.id, u.kind, u.zone, levels=(0, u.levels - 1)) for u in L.units]
    return mpc.Problem(S, t, dist, np.zeros((N, S.nw)), mpc.price_steps(t, iv), steady_x0(P, L, Ta, tout),
                       {z: np.full(N, v) for z, v in comfort.items()}, {z: [fns[z]] * N for z in comfort}, units, st)


def test_step_grid():
    s = mpc.Settings()
    t0 = ms(2026, 1, 14, 13, 2, 30)
    t = mpc.step_grid(t0, ms(2026, 1, 16), s)
    d = np.diff(t) / MIN
    assert t[0] == t0 and t[1] == ms(2026, 1, 14, 13, 5)
    assert np.all(t[1:] % (5 * MIN) == 0)
    assert set(d[1:24]) == {5}
    assert 15 in d and d[-1] <= 30
    assert (t[-1] - t0) / 3.6e6 == pytest.approx(34.96, abs=0.01)  # prices end, under 36 h
    t = mpc.step_grid(t0, ms(2026, 1, 14, 18), s)
    assert (t[-1] - t0) / 3.6e6 >= 12  # never shorter than the minimum horizon
    # just before a 5-minute mark the first step runs to the next one
    t = mpc.step_grid(ms(2026, 1, 14, 13, 4, 30), ms(2026, 1, 16), s)
    assert (t[1] - t[0]) / MIN == 5.5


def test_price_steps_fill_with_yesterday():
    day0 = ms(2026, 1, 14)
    iv = [(day0 + h * 3_600_000, day0 + (h + 1) * 3_600_000, 0.2 + 0.01 * h) for h in range(24)]
    t = np.array([day0 + 10 * 3_600_000, day0 + 11 * 3_600_000, day0 + 34 * 3_600_000, day0 + 35 * 3_600_000], dtype=np.int64)
    p = mpc.price_steps(t, iv)
    assert p[0] == pytest.approx(0.30)
    assert p[2] == pytest.approx(0.30)  # tomorrow 10:00 = today 10:00


def test_preheats_before_the_expensive_evening():
    # cheap until 15:00, very expensive 16-21
    cheap = problem(lambda h: 0.15 if h < 15 else 0.60 if h < 21 else 0.30)
    flat = problem(lambda h: 0.30)
    pc = mpc.solve(cheap)
    pf = mpc.solve(flat)
    assert pc.status == "optimal" and pc.mode == "heat"
    k15 = int(np.searchsorted(pc.t, ms(2026, 1, 14, 15)))
    k18 = int(np.searchsorted(pc.t, ms(2026, 1, 14, 18)))
    # warmer at 15:00 than with flat prices, and heats less in the evening
    assert pc.temp["living"][k15] > pf.temp["living"][k15] + 0.3
    evening = lambda p: sum(float(np.sum(p.heat[u][k15:k18] * np.diff(p.t)[k15:k18])) for u in p.heat)
    assert evening(pc) < evening(pf)
    assert pc.seconds < 15


def test_lockout_and_min_run():
    pr = problem(lambda h: 0.3, Ta=(19.0, 17.0, 19.0, 21.0, 19.0))
    N = pr.N
    allowed = np.ones(N, bool)
    allowed[:24] = False  # bedroom must not run for two hours
    pr.units[1].allowed = allowed
    pr.units[0].on, pr.units[0].since_min = True, 3.0  # office just started
    plan = mpc.solve(pr)
    assert not plan.on["bedroom"][:24].any()
    assert plan.on["office"][:2].all()  # must keep running for its minimum run time
    assert plan.on["bedroom"][24:40].any()  # catches up once allowed


def test_min_off_respected():
    pr = problem(lambda h: 0.3, Ta=(18.0, 19.0, 21.0, 21.5, 19.5))
    pr.units[0].on, pr.units[0].since_min = False, 2.0  # office stopped two minutes ago
    plan = mpc.solve(pr)
    starts = (plan.t[:-1] - plan.t[0]) / MIN
    assert not plan.on["office"][starts < 8].any()
    assert plan.on["office"][(starts >= 8) & (starts < 30)].any()


def test_shared_capacity():
    pr = problem(lambda h: 0.3, tout=-8.0, Ta=(17.0, 17.0, 17.0, 21.0, 18.0))
    plan = mpc.solve(pr)
    total = sum(plan.heat[u] for u in ("office", "bedroom", "living"))
    capf = np.clip(1 - 0.02 * (7 - (-8.0)), 0.4, 1.2)
    assert total.max() <= 7.0 * capf + 1e-6


def test_cooling_when_too_warm():
    pr = problem(lambda h: 0.3, tout=30.0, Ta=(26.0, 26.0, 26.0, 24.0, 25.0),
                 comfort={"office": 23.0, "bedroom": 22.0, "living": 23.0, "bath": 22.0})
    plan = mpc.solve(pr)
    assert plan.mode == "cool"
    assert (plan.heat["living"][:12] < 0).any()
    assert not (plan.heat["office"] > 0).any()
    assert plan.temp["living"][24] < 26.0


def test_relaxed_function_saves_money():
    strict = default_functions()[2]
    relaxed = default_functions()[1]
    zones = ["office", "bedroom", "living", "bath"]
    a = mpc.solve(problem(lambda h: 0.3, functions={z: strict for z in zones}))
    b = mpc.solve(problem(lambda h: 0.3, functions={z: relaxed for z in zones}))
    assert b.cost["energy"] < a.cost["energy"]
    assert b.temp["living"][-1] < a.temp["living"][-1]


def test_hard_limit_holds_even_when_expensive():
    f = ComfortFunction("x", "x", below=Side(deadband=0.5, linear=0.001, limit=1.0), above=Side(deadband=2))
    zones = ["office", "bedroom", "living", "bath"]
    plan = mpc.solve(problem(lambda h: 2.0, Ta=(21.0, 19.0, 21.5, 22.0, 20.0), functions={z: f for z in zones}))
    assert plan.temp["office"][1:].min() >= 21.0 - 1.0 - 0.05


def test_window_open_input_cools_the_room():
    pr = problem(lambda h: 0.3)
    T0 = mpc.free_run(pr)
    pr.window[:2, 0] = 1.0  # office window open for the first steps
    T1 = mpc.free_run(pr)
    assert T1[3, 0] < T0[3, 0] - 0.1


def airing_for(pr, starts, per_group=None, gap_h=4.0):
    """Ten-minute airings at the given starts (ms), grouped by day."""
    L = pr.sys.layout
    days = {}
    for i, s in enumerate(starts):
        days.setdefault(datetime.fromtimestamp(s / 1000, TZ).date(), []).append(i)
    groups = [(idx, (per_group or {}).get(d, 1)) for d, idx in sorted(days.items())]
    return mpc.Airing(cand=[(s, s + 10 * MIN) for s in starts], groups=groups, min_gap_ms=gap_h * 3.6e6,
                      tref={z: pr.comfort[z] for z in L.window_zones})


def half_hours(a, b):
    return list(range(a, b + 1, 30 * MIN))


def test_airing_goes_where_it_costs_least():
    pr = problem(lambda h: 0.3)
    tout = pr.dist[:, DIST.index("tout")]
    mid = pr.t[:-1]
    tout[(mid >= ms(2026, 1, 14, 15)) & (mid < ms(2026, 1, 14, 16))] = 12.0  # a mild hour today
    starts = half_hours(ms(2026, 1, 14, 13, 30), ms(2026, 1, 14, 21, 30)) + half_hours(ms(2026, 1, 15, 7), ms(2026, 1, 15, 21, 30))
    pr.airing = airing_for(pr, starts)
    plan = mpc.solve(pr)
    assert len(plan.airing) == 2  # one each day
    today = plan.airing[0]
    assert ms(2026, 1, 14, 15) <= today[0] and today[1] <= ms(2026, 1, 14, 16)
    assert plan.airing_cost and plan.airing_cost > 0
    # the units where the windows are stay off while they are open
    for a, b in plan.airing:
        k = int(np.searchsorted(plan.t, a, side="right")) - 1
        for u in pr.units:
            if u.zone in pr.sys.layout.window_zones:
                assert plan.share[u.id][k] < 1.0
    # and it is cheaper there than at the coldest time
    pr.airing = airing_for(pr, [ms(2026, 1, 14, 19), ms(2026, 1, 15, 12)])
    forced = mpc.solve(pr)
    assert forced.airing_cost > plan.airing_cost


def test_airing_that_cannot_fit_is_left_out():
    pr = problem(lambda h: 0.3)
    s = [ms(2026, 1, 14, 14), ms(2026, 1, 14, 15)]
    pr.airing = airing_for(pr, s, per_group={datetime(2026, 1, 14).date(): 2})  # an hour apart, not four
    plan = mpc.solve(pr)
    assert plan.airing == [] and plan.airing_cost is None
