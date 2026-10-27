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
