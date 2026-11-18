import json

import numpy as np
import pytest

from climate_optimizer.model import fit, synthetic
from climate_optimizer.model.dataset import build_grid, parse_rows
from climate_optimizer.model.layout import Layout, Params, default_layout, default_params
from climate_optimizer.model.thermal import build, zone_summary


@pytest.fixture(scope="module")
def trained():
    layout = default_layout()
    truth = default_params(layout)
    truth.zones["office"].update(Ca=0.35, Cm=1.4, Cw=4.0, Ham=0.25, Hmw=0.10, Hwo=0.020, H=0.008)
    truth.units["living"]["heat"] = [1.6, 2.6, 3.6, 4.4]
    truth.plant.update(eta=0.36)
    r = synthetic.run(days=8, seed=3, params=truth, realism=False)
    parsed = parse_rows(json.loads(json.dumps(synthetic.to_export(r))))  # through JSON, as from the store
    grid = build_grid(parsed, layout, synthetic.mapping_for(layout))
    old = dict(fit.FIT)
    fit.FIT.update(max_nfev=25, burn_in_h=24)
    try:
        res = fit.train(layout, grid)
    finally:
        fit.FIT.clear()
        fit.FIT.update(old)
    return layout, truth, grid, res, r


def test_layout_and_params_roundtrip():
    L = default_layout()
    assert Layout.from_dict(json.loads(json.dumps(L.to_dict()))) == L
    P = default_params(L)
    assert Params.from_dict(json.loads(json.dumps(P.to_dict()))).to_dict() == P.to_dict()
    with pytest.raises(ValueError):
        Layout.from_dict({**L.to_dict(), "links": [["office", "cellar"]]})


def test_discretization_is_exact_for_constant_inputs():
    L = default_layout()
    S = build(L, default_params(L))
    A1, B1 = S.discretize(1.0)
    A5, B5 = S.discretize(5 / 60)
    x = np.full(S.n, 20.0)
    u = np.zeros(S.m)
    u[S.nu] = 0.0  # tout
    u[S.nu + 7] = 1.0  # constant gains
    y1 = A1 @ x + B1 @ u
    y = x
    for _ in range(12):
        y = A5 @ y + B5 @ u
    assert np.allclose(y, y1, atol=1e-9)


def test_training_recovers_the_apartment(trained):
    layout, truth, grid, res, r = trained
    rep = res["report"]
    assert max(rep["rmse_val"]) < 0.35
    for z in layout.zone_ids:
        a, b = zone_summary(truth, z)["tau_h"], rep["zones"][z]["tau_h"]
        assert 0.5 * a < b < 2.0 * a, z
    m = res["params"]
    assert 0.25 < m.plant["eta"] < 0.5
    lv = m.units["living"]["heat"]
    assert all(x < y for x, y in zip(lv, lv[1:]))
    true_e = sum(synthetic.true_unit_energy(r, d, d + 864e5) for d in rep["days"]["val"])
    assert abs(rep["energy"]["sim"] - true_e) < 0.3 * true_e
    # checked on the held-back days, then learned once more with them
    rf = rep["refit"]
    assert rf["used"] and rf["days"] == len(rep["days"]["train"]) + len(rep["days"]["val"])
    assert rf["rmse_after"] < 0.35 and res["checked"] is not m


def test_a_refit_that_does_not_hold_up_is_not_taken(trained):
    layout, truth, grid, _, _ = trained
    old = dict(fit.FIT)
    fit.FIT.update(max_nfev=6, refit_nfev=4, burn_in_h=24, refit_jump_factor=1.0)  # any change is too much
    try:
        res = fit.train(layout, grid)
    finally:
        fit.FIT.clear()
        fit.FIT.update(old)
    rf = res["report"]["refit"]
    assert not rf["used"] and "would change by a factor of" in rf["reason"]
    assert res["params"] is res["checked"]
    assert any(w.startswith("Not learned again") for w in res["report"]["warnings"])


def test_robust_residual_is_huber():
    from climate_optimizer.model.fit import robust

    r = np.linspace(-6, 6, 241)
    c = 1.5
    huber = np.where(np.abs(r) <= c, r ** 2 / 2, c * np.abs(r) - c ** 2 / 2)
    assert np.allclose(robust(r, c) ** 2 / 2, huber)
    assert np.all(np.sign(robust(r, c)) == np.sign(r))


def test_stuck_thermometer_is_masked():
    from climate_optimizer.model.dataset import mask_stuck

    T = np.column_stack([20 + 0.01 * np.arange(100), np.full(100, 21.0)])
    T[:10, 1] = np.linspace(20, 20.9, 10)
    M, share = mask_stuck(T, 5, keep_h=1.0, min_h=3.0)
    assert np.isfinite(M[:, 0]).all() and share[0] == 0
    assert np.isfinite(M[:22, 1]).all() and np.isnan(M[22:, 1]).all()  # an hour of the stretch stays
    assert share[1] == pytest.approx(78 / 100)


def test_facades_limit_the_sun():
    from climate_optimizer.model.fit import param_list

    d = default_layout().to_dict()
    for z in d["zones"]:
        if z["id"] == "office":
            z["facades"] = ["N"]
        if z["id"] == "living":
            z["facades"] = ["S", "W"]
    L = Layout.from_dict(json.loads(json.dumps(d)))
    assert Layout.from_dict(L.to_dict()) == L
    P = default_params(L)
    assert P.zones["office"]["sS"] == 0 and P.zones["office"]["sD"] > 0
    assert P.zones["living"]["sE"] == 0 and P.zones["living"]["sS"] > 0
    names = {p.path[-1] for p in param_list(L, P) if p.path[:2] == ("zones", "office")}
    assert "sD" in names and not {"sE", "sS", "sW"} & names
    with pytest.raises(ValueError):
        Layout.from_dict({**d, "zones": [{**d["zones"][0], "facades": ["X"]}] + d["zones"][1:]})


def test_retraining_keeps_physics_as_the_prior():
    """A bad previous model is a start, not a prior: retraining moves away from it."""
    from climate_optimizer.model import fit as F

    layout = default_layout()
    truth = default_params(layout)
    r = synthetic.run(days=6, seed=5, params=truth, realism=False)
    grid = build_grid(parse_rows(json.loads(json.dumps(synthetic.to_export(r)))), layout, synthetic.mapping_for(layout))
    bad = default_params(layout)
    bad.zones["living"]["Cw"] = 60.0
    bad.building["Tnb"] = 15.0
    old = dict(F.FIT)
    F.FIT.update(max_nfev=15, burn_in_h=24)
    try:
        res = F.train(layout, grid, bad)
    finally:
        F.FIT.clear()
        F.FIT.update(old)
    m = res["params"]
    assert m.zones["living"]["Cw"] < 30 and m.building["Tnb"] > 17.5


def test_power_from_an_energy_meter():
    """A cycling load, its meter read now and then: the 5-minute means come out exact,
    a reset leaves its step unknown."""
    from climate_optimizer.model.dataset import counter_power

    rng = np.random.default_rng(2)
    sec = np.arange(0, 3 * 3600)  # three hours, by the second
    p = np.where((sec // 97) % 3 == 0, 1.8, 0.05) + 0.1 * rng.random(len(sec))  # kW, cycling
    e = np.cumsum(p) / 3600.0  # kWh
    reports = np.sort(rng.choice(len(sec), 400, replace=False))
    t = sec[reports] * 1000.0
    v = e[reports].copy()
    v[300:] -= 100.0  # the meter is reset once
    got = counter_power(t, v, 0.0, 300_000.0, 36, 1.0)
    true = p[: 36 * 300].reshape(36, 300).mean(1)
    ok = np.isfinite(got)
    assert ok[1:-1].sum() == 33  # unknown: only the step with the reset (and the ends, outside the reports)
    # each step's ends fall between reports: interpolated there, the energy itself exact
    assert np.abs(got[ok] - true[ok]).max() < 0.2 and abs(got[ok].mean() - true[ok].mean()) < 0.01


def test_grid_takes_power_from_a_meter():
    """The phase given as a kWh meter gives the same kW as its power sensor."""
    layout = default_layout()
    r = synthetic.run(days=2, seed=4, realism=False)
    ex = synthetic.to_export(r)
    rows = ex["stats_5min"]["sensor.phase_l2_leistung"]
    kwh, acc = [], 0.0
    for t, w in rows:  # the meter's reading at the end of each 5-minute mean
        acc += w / 1000 / 12
        kwh.append([t + 300_000, round(acc, 4)])
    ex["history"]["sensor.phase_l2_energie"] = kwh
    ex["meta"]["sensor.phase_l2_energie"] = {"unit": "kWh", "device_class": "energy"}
    m = synthetic.mapping_for(layout)
    p = parse_rows(json.loads(json.dumps(ex)))
    a = build_grid(p, layout, m).phase
    b = build_grid(p, layout, {**m, "phase": "sensor.phase_l2_energie"}).phase
    ok = np.isfinite(a) & np.isfinite(b)
    assert ok.mean() > 0.95 and np.abs(a[ok] - b[ok]).max() < 0.01


def test_start_energy_shows_at_the_starts(trained):
    layout, truth, grid, _, _ = trained
    p = default_params(layout)
    p.plant["Estart"] = 0.0
    _, a = fit.simulate_windows(layout, p, grid, np.array([0]), grid.K - 1)
    p.plant["Estart"] = 0.05
    _, b = fit.simulate_windows(layout, p, grid, np.array([0]), grid.K - 1)
    ac = [u.id for u in layout.ac_units]
    run = np.max([grid.unit_running(u) for u in ac], axis=0)
    starts = np.clip(run[1:] - run[:-1], 0, 1) * (run[:-1] < 0.5)
    extra = (b - a)[0].sum() * grid.dt_h
    assert starts.sum() > 3 and extra == pytest.approx(0.05 * starts[: grid.K - 2].sum(), rel=0.05)


def test_setpoint_power_control_levels():
    """Under setpoint power control the levels are setpoints past the device's own
    reading: training reads them from the proxy's level, else from that gap, and the
    days before with the fan modes then."""
    from climate_optimizer.model.dataset import _ts
    from climate_optimizer.model.synthetic import FAN_MODES

    layout = default_layout()
    r = synthetic.run(days=2, seed=4, realism=False)
    ex = json.loads(json.dumps(synthetic.to_export(r)))
    m = synthetic.mapping_for(layout)
    base = build_grid(parse_rows(ex), layout, m).ac_frac["living"]
    src, prx = m["units"]["living"]["source"], m["units"]["living"]["proxy"]
    rows = ex["history"][src]
    since = _ts(rows[len(rows) // 2][0])
    offs, labels = [1.0, 2.0, 3.0, 4.0], ["+1 K", "+2 K", "+3 K", "+4 K"]
    new, prows = [], []
    for t, st, sp, cur, act, fan in rows:
        if _ts(t) >= since and st in ("heat", "cool"):
            lvl = FAN_MODES.index(fan)
            sp = cur + offs[lvl] + 0.5 if st == "heat" else cur - offs[lvl] - 0.5  # rounded away from the room
            prows.append([t, st, 21.0, 20.0, act, "medium", labels[lvl]])
            fan = "auto"
        new.append([t, st, sp, cur, act, fan])
    ex["history"][src] = new
    unit = {**m["units"]["living"], "power_control": "setpoint", "setpoint_offsets": offs, "power_since": since,
            "fan_modes": labels, "fan_modes_before": FAN_MODES}
    m2 = {**m, "units": {**m["units"], "living": unit}}
    ex["history"][prx] = prows
    got = build_grid(parse_rows(ex), layout, m2).ac_frac["living"]
    assert base.sum() > 5 and np.abs(got - base).max() < 1e-9  # from the proxy's level
    ex["history"][prx] = []
    gap = build_grid(parse_rows(ex), layout, m2).ac_frac["living"]
    assert np.abs(gap - base).max() < 1e-9  # from the gap: 0.5 K past the offset is still that level
    wrong = build_grid(parse_rows(ex), layout, m).ac_frac["living"]  # neither the proxy's level nor the switch known
    assert np.abs(wrong - base).max() > 0.5
    ex["history"][prx] = prows  # back on fan speeds now: the stretch under setpoint control still reads as such
    back = build_grid(parse_rows(ex), layout, m).ac_frac["living"]
    assert np.abs(back - base).max() < 1e-9


def test_heat_is_learned_along_the_setpoint_offsets():
    from climate_optimizer.model.dataset import _sp_offset, offset_weights, regrid_heat
    from climate_optimizer.planner import matching_params

    assert _sp_offset("+2 K") == 2.0 and _sp_offset("+2.5 K low") == 2.5 and _sp_offset("low") is None
    assert offset_weights(3.0, [2.0, 4.0]) == [(0, 0.5), (1, 0.5)]  # between two levels: half each
    assert offset_weights(1.0, [2.0, 4.0]) == [(0, 1.0)] and offset_weights(6.0, [2.0, 4.0]) == [(1, 1.0)]
    assert offset_weights(4.0, [1.0, 2.0, 4.0]) == [(2, 1.0)]
    assert regrid_heat([1.0, 2.0, 3.0], [1, 2, 3], [1.5, 3, 4]) == [1.5, 3.0, 4.0]  # along, and straight on past the end
    layout = default_layout()
    p = default_params(layout)
    p.units["living"]["heat"] = [1.0, 2.0, 3.0, 4.0]
    p.units["living"]["offsets"] = [1.0, 2.0, 3.0, 4.0]
    m = matching_params(p, layout, {"living": [2.0, 4.0, 6.0, 8.0]})
    assert m.units["living"]["heat"] == [2.0, 4.0, 6.0, 8.0] and m.units["living"]["offsets"] == [2.0, 4.0, 6.0, 8.0]
    assert p.units["living"]["heat"] == [1.0, 2.0, 3.0, 4.0]  # the trained model itself is left as it is


def test_metered_loads_come_off_the_phase():
    """A fridge on the outdoor unit's phase with a plug of its own: taken off, the phase is the units again."""
    layout = default_layout()
    r = synthetic.run(days=2, seed=4, realism=False)
    ex = json.loads(json.dumps(synthetic.to_export(r)))
    m = synthetic.mapping_for(layout)
    base = build_grid(parse_rows(ex), layout, m).phase
    rows = ex["stats_5min"][m["phase"]]
    fridge = [[t, 150.0 if (i // 6) % 2 else 0.0] for i, (t, _) in enumerate(rows)]  # W, half an hour on, half off
    ex["stats_5min"][m["phase"]] = [[t, w + f[1]] for (t, w), f in zip(rows, fridge)]
    ex["stats_5min"]["sensor.fridge_power"] = fridge
    ex["meta"]["sensor.fridge_power"] = {"unit": "W", "device_class": "power"}
    with_it = build_grid(parse_rows(ex), layout, m).phase
    off = build_grid(parse_rows(ex), layout, {**m, "phase_subtract": ["sensor.fridge_power"]}).phase
    ok = np.isfinite(base) & np.isfinite(off)
    assert np.abs(with_it[ok] - base[ok]).max() > 0.1 and np.abs(off[ok] - base[ok]).max() < 1e-6


def test_an_outdoor_gap_and_a_room_without_thermometer_do_not_stop_the_model():
    from climate_optimizer.model.fit import simulate_windows
    from climate_optimizer.model.thermal import build, estimate_hidden

    layout = default_layout()
    r = synthetic.run(days=3, seed=4, realism=False)
    ex = json.loads(json.dumps(synthetic.to_export(r)))
    m = synthetic.mapping_for(layout)
    rows = ex["stats_5min"][m["outdoor"]]
    a, b = rows[0][0] + 30 * 3600e3, rows[0][0] + 42 * 3600e3  # twelve hours without the outdoor sensor
    ex["stats_5min"][m["outdoor"]] = [x for x in rows if not a <= x[0] < b]
    m = {**m, "weather": ""}
    hall = m["zones"]["hall"]["thermometer"]
    m["zones"] = {**m["zones"], "hall": {**m["zones"]["hall"], "thermometer": ""}}  # and no thermometer in the hall
    g = build_grid(parse_rows(ex), layout, m)
    assert np.isfinite(g.tout).all()
    gap = (g.t >= a + 4 * 3600e3) & (g.t < b - 1)
    assert not g.valid[gap].any() and g.valid[~gap].mean() > 0.9  # not checked in the gap, the rest is
    p = default_params(layout)
    Tm, Tw = estimate_hidden(build(layout, p), g.T, g.tout, g.sol, g.dt_h)
    assert np.isfinite(Tm).all() and np.isfinite(Tw).all()
    Ta, Pel = simulate_windows(layout, p, g, np.array([0, len(g.t) // 2]), 288)
    assert np.isfinite(Ta).all() and np.isfinite(Pel).all()
    assert hall not in [z.get("thermometer") for z in m["zones"].values()]


def test_a_stopped_thermometer_is_not_checked_against_but_keeps_the_day():
    layout = default_layout()
    r = synthetic.run(days=2, seed=4, realism=False)
    ex = json.loads(json.dumps(synthetic.to_export(r)))
    m = synthetic.mapping_for(layout)
    th = m["zones"]["office"]["thermometer"]
    rows = ex["stats_5min"][th]
    for x in rows[100:200]:  # eight hours of the same reading
        x[1] = rows[100][1]
    g = build_grid(parse_rows(ex), layout, m)
    k = int(np.searchsorted(g.t, rows[150][0]))
    zi = layout.zone_index("office")
    assert np.isnan(g.T[k, zi]) and g.valid[k]  # masked for the check, the day still counts


def _grid_and_hidden(truth, layout, seed=4, days=3):
    from climate_optimizer.model.thermal import build, estimate_hidden

    r = synthetic.run(days=days, seed=seed, realism=False, layout=layout, params=truth)
    g = build_grid(parse_rows(json.loads(json.dumps(synthetic.to_export(r)))), layout, synthetic.mapping_for(layout))
    return g, estimate_hidden(build(layout, default_params(layout)), g.T, g.tout, g.sol, g.dt_h)


def test_the_correction_learns_what_the_model_misses():
    """The living room loses 0.3 kW more than the model knows: run over each past hour with
    what the units really did, the misses lead the correction there; the other rooms stay."""
    from climate_optimizer.model.correction import hour_misses, hour_response, step, with_correction

    layout = default_layout()
    model = default_params(layout)
    g, hidden = _grid_and_hidden(with_correction(model, {"living": -0.3}), layout)
    kw = {z: 0.0 for z in layout.zone_ids}
    r = {z: hour_response(layout, model, z) for z in layout.zone_ids}
    for k1 in range(30 * 12, 60 * 12):
        miss = hour_misses(layout, with_correction(model, kw), g, hidden, k1)
        for z, e in miss.items():
            if e is not None:
                kw[z] = step(kw[z], e, r[z], 0.05, 0.5 ** (5 / 1440), 0.5)
    assert -0.4 < kw["living"] < -0.2
    assert all(abs(kw[z]) < 0.1 for z in ("office", "bedroom", "bath"))


def test_the_correction_stays_put_without_a_model_error():
    """No model error: whatever the units did (the synthetic thermostat, not any plan), nothing to correct."""
    from climate_optimizer.model.correction import hour_misses

    layout = default_layout()
    g, hidden = _grid_and_hidden(default_params(layout), layout, seed=7)
    misses = [e for k1 in range(30 * 12, 48 * 12, 6) for e in hour_misses(layout, default_params(layout), g, hidden, k1).values()
              if e is not None]
    assert misses and abs(float(np.mean(misses))) < 0.05


def test_no_correction_over_an_hour_with_a_window_open():
    from climate_optimizer.model.correction import hour_misses

    layout = default_layout()
    g, hidden = _grid_and_hidden(default_params(layout), layout)
    g.open[30 * 12 - 3, 0] = 1.0
    assert hour_misses(layout, default_params(layout), g, hidden, 30 * 12) == {}


def _plan_like(g, layout, k0, rise, edge_above=0.1, comfort=21.0, z="living"):
    """A plan made at step k0 that expects room z to warm by rise over the hour (its own
    temperatures just under its edge, so held at it), with what plans keep for the check."""
    t = [int(g.t[k0] + 60_000 * 5 * i) for i in range(14)]
    T0 = float(np.nan_to_num(g.T[k0, layout.zone_index(z)], nan=20.0))
    temps = [T0 + rise * i / 12 for i in range(14)]
    edge = max(temps) - edge_above
    return {"t": t[0], "times": t, "temp": {z: temps},
            "learn": {"t": t[:13], "comfort": {z: [comfort] * 13}, "edge": {z: [edge] * 13},
                      "override": {z: False}, "forced": {z: False}, "auto": {z: True}}}


def test_the_gap_between_plan_and_doing():
    """The plan expected the room to warm 0.3 K more than what was then done gives; a plan
    expecting just what was done has no gap; a comfort changed meanwhile or a window keeps
    the hour out; a change by schedule (in every plan alike) does not."""
    from climate_optimizer.model.correction import exec_gaps
    from climate_optimizer.model.fit import simulate_windows

    layout = default_layout()
    model = default_params(layout)
    g, hidden = _grid_and_hidden(model, layout)
    zi = layout.zone_index("living")
    k0 = 36 * 12
    while g.open[k0:k0 + 13].any() or not g.valid[k0:k0 + 13].all():
        k0 += 1
    Ta, _ = simulate_windows(layout, model, g, np.array([k0]), 12, hidden=hidden)
    done = float(Ta[0, 12, zi] - Ta[0, 0, zi])
    ref = _plan_like(g, layout, k0, done + 0.3)
    gap = exec_gaps(layout, model, g, hidden, ref, [])["living"]
    assert abs(gap - 0.3) < 0.03
    assert abs(exec_gaps(layout, model, g, hidden, _plan_like(g, layout, k0, done), [])["living"]) < 0.03
    later = _plan_like(g, layout, k0 + 2, 0.0, comfort=22.0)  # the comfort temperature raised meanwhile
    assert "living" not in exec_gaps(layout, model, g, hidden, ref, [later])
    same = _plan_like(g, layout, k0 + 2, 0.0)  # nothing new: counts
    assert "living" in exec_gaps(layout, model, g, hidden, ref, [same])
    sched_ref, sched_later = _plan_like(g, layout, k0, done + 0.3), _plan_like(g, layout, k0 + 2, 0.0)
    sched_ref["learn"]["comfort"]["living"] = [21.0] * 6 + [19.0] * 7  # a rule from half past: both plans have it
    sched_later["learn"]["comfort"]["living"] = [21.0] * 4 + [19.0] * 9
    assert "living" in exec_gaps(layout, model, g, hidden, sched_ref, [sched_later])
    away = _plan_like(g, layout, k0, done + 0.3, edge_above=1.0)  # the plan let it rise past its edge: not held there
    assert "living" not in exec_gaps(layout, model, g, hidden, away, [])
    g.open[k0 + 5, 0] = 1.0
    assert exec_gaps(layout, model, g, hidden, ref, []) == {}
