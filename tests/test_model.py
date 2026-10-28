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
