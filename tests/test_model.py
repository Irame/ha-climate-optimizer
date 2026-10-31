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
