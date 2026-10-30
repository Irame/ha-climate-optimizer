"""Training the model on recorded history.

The criterion is what the optimizer needs the model for: from the measured
temperatures (and the reconstructed hidden ones), how well does it predict
the next six hours of every zone, given what the units, the heater, the
windows, the sun and the weather did? Electricity on the outdoor unit's
phase is matched as well, with a robust loss so other loads on that phase
do not drag the fit. Priors keep parameters plausible where the data says
little, and the last days are held back to check the result.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from scipy.optimize import least_squares

from .dataset import Grid
from .layout import Layout, Params, default_params, link_key
from .thermal import (DIST, _ffill, build, capacity_factor, carnot_heat, electric, estimate_hidden, outdoor_unit,
                      plant_base, zone_summary)

SOLAR_AIR = 0.3  # share of the sun through a window that warms the air at once, before training
FIT = dict(window_h=6.0, stride_h=3.0, every=3, sigma_t=0.25, sigma_p=0.3, burn_in_h=36.0, max_nfev=60,
           huber=1.5, prior_weight=10.0, eta_measured=0.5)


def robust(r: np.ndarray, c: float) -> np.ndarray:
    """Huber's loss as a residual: sum(robust(r)**2) / 2 is the Huber cost of r.
    Only the data gets it; the priors stay quadratic, or the fit could leave
    them behind at the price of a straight line."""
    a = np.abs(r)
    return np.where(a <= c, r, np.sign(r) * np.sqrt(np.maximum(0.0, 2 * c * a - c * c)))


# ---- parameters ---------------------------------------------------------------
@dataclass
class P:
    path: tuple
    kind: str  # "log" | "lin" | "inc" (increment on the previous level, log) | "unit" (a share, logit)
    sigma: float
    floor: float = 1e-6
    measured: float = 0.0  # 1: pinned down by the power measurement, its prior is not weighted up; between: partly


def param_list(layout: Layout, params: Params) -> list[P]:
    out: list[P] = []
    for z in layout.zones:
        zp = ("zones", z.id)
        for k, s in (("Ca", 0.8), ("Cm", 0.7), ("Cw", 0.7), ("Ham", 0.8), ("Hmw", 0.8)):
            out.append(P(zp + (k,), "log", s))
        out += [P(zp + ("Hwo",), "log", 0.9, 1e-3), P(zp + ("Hwn",), "log", 1.0, 1e-3), P(zp + ("H",), "log", 1.0, 1e-3),
                P(zp + ("q",), "log", 1.2, 5e-3), P(zp + ("qc",), "lin", 0.05), P(zp + ("qs",), "lin", 0.05)]
        if z.windows:
            out.append(P(zp + ("Hw",), "log", 1.2, 1e-2))
        if z.solar:
            out += [P(zp + ("s" + f,), "log", 1.5, 5e-2) for f in "DESW" if f == "D" or z.sun_from(f)]
    for a, b in layout.links:
        out.append(P(("links", link_key(a, b)), "log", 0.9, 3e-3))
    for u in layout.ac_units:
        for i in range(u.levels):
            out.append(P(("units", u.id, "heat", i), "log" if i == 0 else "inc", 0.5, 0.05, measured=1.0))
    out += [P(("plant", "Qmax"), "log", 0.5), P(("plant", "eta"), "log", 0.15, measured=FIT["eta_measured"]),
            P(("plant", "cap_slope"), "lin", 0.02),
            # the outdoor unit: its base shows in the power at low load, its limit where the power stops rising
            P(("plant", "base"), "log", 0.8, 0.02, measured=0.5), P(("plant", "kq"), "log", 1.0, 1e-4),
            P(("plant", "Pmax"), "log", 0.3, 0.3, measured=1.0),
            P(("building", "Tnb"), "lin", 2.0)]
    if any(z.solar for z in layout.zones):
        out.append(P(("building", "solar_air"), "unit", 1.0))
    # Standby stays at its preset: other loads on the same phase that run
    # whenever the units do cannot be told apart from it.
    return out


def _get(d, path):
    for k in path:
        d = d[k]
    return d


def _set(d, path, v):
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = v


def to_theta(params: Params, plist: list[P]) -> np.ndarray:
    d = params.to_dict()
    out = []
    for p in plist:
        v = _get(d, p.path)
        if p.kind == "inc":
            prev = _get(d, p.path[:-1] + (p.path[-1] - 1,))
            v = max(v - prev, p.floor)
        if p.kind == "unit":
            v = min(max(v, 0.01), 0.99)
            out.append(np.log(v / (1 - v)))
            continue
        out.append(np.log(max(v, p.floor)) if p.kind in ("log", "inc") else v)
    return np.array(out, dtype=float)


def from_theta(theta: np.ndarray, plist: list[P], base: Params) -> Params:
    d = copy.deepcopy(base.to_dict())
    for th, p in zip(theta, plist):
        if p.kind == "lin":
            _set(d, p.path, float(th))
        elif p.kind == "log":
            _set(d, p.path, float(np.exp(th)))
        elif p.kind == "unit":
            _set(d, p.path, float(1 / (1 + np.exp(-th))))
        else:
            prev = _get(d, p.path[:-1] + (p.path[-1] - 1,))
            _set(d, p.path, float(prev + np.exp(th)))
    return Params.from_dict(d)


# ---- what the data offers --------------------------------------------------------
def power_observation(grid: Grid) -> dict:
    """The outdoor unit's electricity: its phase minus the other loads on it."""
    K = grid.K
    obs = np.full(K, np.nan)
    notes = []
    if not np.isfinite(grid.phase).any():
        return {"obs": obs, "ok": np.zeros(K, bool), "notes": ["No phase power assigned: efficiency and output come from the priors."]}
    adj = grid.phase.copy()
    car_mask = np.zeros(K, bool)
    if grid.car is not None:
        adj = adj - np.nan_to_num(grid.car) / 3.0
        notes.append("Car charging subtracted on the units' phase (a third of the charger's power).")
    elif grid.others:
        p10 = [np.nanpercentile(o, 10) for o in grid.others]
        pp = np.nanpercentile(grid.phase, 10)
        thr = 1.0 if len(grid.others) > 1 else 2.5
        up = np.all(np.vstack([np.nan_to_num(o - p) > thr for o, p in zip(grid.others, p10)]), axis=0)
        mask = up & (np.nan_to_num(grid.phase - pp) > 1.0)
        for k in np.where(mask)[0]:
            car_mask[max(0, k - 3):k + 4] = True
        if mask.any():
            notes.append(f"{round(mask.sum() * grid.step_min / 60)} h of what looks like three-phase charging left out of the power fit.")
    running = np.zeros(K)
    for uid in grid.ac_frac:
        running += grid.unit_running(uid)
    tz = ZoneInfo(grid.tz)
    hours = np.array([datetime.fromtimestamp(t / 1000, tz).hour for t in grid.t])
    off = (running == 0) & ~car_mask & np.isfinite(adj)
    overall = np.nanmedian(adj[off]) if off.any() else np.nanpercentile(adj, 10)
    base = np.array([np.nanmedian(adj[off & (hours == h)]) if (off & (hours == h)).sum() >= 6 else overall for h in range(24)])
    if off.sum() < 72:
        notes.append("Few periods with all units off, so the other loads on that phase are only roughly known.")
    ok = np.isfinite(adj) & ~car_mask
    obs[ok] = adj[ok] - base[hours[ok]]
    return {"obs": obs, "ok": ok, "notes": notes, "base": base.tolist()}


def split_days(grid: Grid, val_share=0.25):
    per_day = int(round(24 * 60 / grid.step_min))
    tz = ZoneInfo(grid.tz)
    days = []
    for k in range(0, grid.K - per_day + 1):
        d = datetime.fromtimestamp(grid.t[k] / 1000, tz)
        if d.hour or d.minute:
            continue
        if grid.valid[k:k + per_day].mean() >= 0.95:
            days.append(k)
    if len(days) < 2:
        return days, [], per_day
    n_val = max(1, round(len(days) * val_share))
    return days[:-n_val], days[-n_val:], per_day


def windows_for(grid: Grid, days: list[int], per_day: int) -> list[int]:
    L = int(FIT["window_h"] * 60 / grid.step_min)
    stride = int(FIT["stride_h"] * 60 / grid.step_min)
    burn = int(FIT["burn_in_h"] * 60 / grid.step_min)
    out = []
    for d in days:
        for k0 in range(max(d, burn), d + per_day - L + 1, stride):
            if k0 + L < grid.K and grid.valid[k0:k0 + L + 1].all():
                out.append(k0)
    return out


def heater_power(grid: Grid) -> dict[str, float]:
    """A heater's power when on: the top of what its plug measured (5-minute
    means are diluted wherever it switched within the step)."""
    out = {}
    for uid, kw in grid.heater_kw.items():
        on = kw[kw > 0.1]
        if len(on) > 20:
            out[uid] = float(np.quantile(on, 0.95))
    return out


def _merged(base: Params, other: Params) -> Params:
    """base with every value other has for the same thing (a model trained
    on another layout still lends what fits)."""
    d = base.to_dict()
    o = other.to_dict()
    for z, zp in d["zones"].items():
        oz = o["zones"].get(z, {})
        for k in zp:
            if k in oz and not (k in ("sD", "sE", "sS", "sW") and zp[k] == 0.0):  # no sun stays no sun
                zp[k] = oz[k]
    for k in d["links"]:
        if k in o["links"]:
            d["links"][k] = o["links"][k]
    for u, up in d["units"].items():
        ou = o["units"].get(u) or {}
        if ("heat" in up and len(ou.get("heat", [])) == len(up["heat"])) or ("kw" in up and "kw" in ou):
            d["units"][u] = dict(ou)
    for sect in ("plant", "building"):
        for k in d[sect]:
            if k in o[sect]:
                d[sect][k] = o[sect][k]
    return Params.from_dict(d)


def initial_params(layout: Layout, grid: Grid) -> Params:
    p = default_params(layout)
    for uid, kw in heater_power(grid).items():
        p.units[uid]["kw"] = kw
    # The outdoor unit's limit: start well above the most its phase ever showed
    # while units ran. The fit only brings it down where the power stops rising
    # although more is asked; data that never reached the limit leave it above.
    pw = power_observation(grid)
    ac = [u.id for u in layout.ac_units]
    if ac:
        running = np.max([grid.unit_running(u) for u in ac], axis=0) > 0.5
        seen = pw["obs"][pw["ok"] & running]
        if len(seen) >= 50:
            p.plant["Pmax"] = float(max(1.5, 1.3 * np.quantile(seen, 0.995)))
    return p


# ---- simulation over windows --------------------------------------------------------
def unit_heat(params: Params, layout: Layout, grid: Grid, ks: np.ndarray) -> np.ndarray:
    """Heat (kW) each unit delivered at steps ks, from what it was recorded doing."""
    out = np.zeros(ks.shape + (len(layout.units),))
    capf = capacity_factor(params, grid.tout[ks])
    for j, u in enumerate(layout.units):
        if u.kind == "ac":
            lv = np.asarray(params.units[u.id]["heat"])
            fr = grid.ac_frac[u.id][ks]  # (..., 2, L)
            out[..., j] = (fr[..., 0, :] @ lv) * capf - (fr[..., 1, :] @ lv) * params.units[u.id].get("cool_ratio", 0.9)
        else:
            out[..., j] = grid.heater_kw[u.id][ks]
    return out


def simulate_windows(layout: Layout, params: Params, grid: Grid, k0s: np.ndarray, L: int, sysm=None, hidden=None):
    """Free-run every window from its start; returns air temperatures (nw, L+1, nz) and electricity (nw, L)."""
    sysm = sysm or build(layout, params)
    Ad, Bd = sysm.discretize(grid.dt_h)
    nz = sysm.nz
    if hidden is None:
        hidden = estimate_hidden(sysm, grid.T, grid.tout, grid.sol, grid.dt_h)
    Tm, Tw = hidden
    Tf = np.column_stack([_ffill(grid.T[:, i]) for i in range(nz)]) if nz else grid.T  # a masked start: the last reading
    X = np.concatenate([Tf[k0s], Tm[k0s], Tw[k0s]], axis=1)  # (nw, n)
    nw = len(k0s)
    Ta = np.empty((nw, L + 1, nz))
    Pel = np.empty((nw, L))
    Ta[:, 0] = X[:, :nz]
    ac_idx = [j for j, u in enumerate(layout.units) if u.kind == "ac"]
    ac_zone = [layout.zone_index(layout.units[j].zone) for j in ac_idx]
    win_zone = [layout.zone_index(z) for z in layout.window_zones]
    running = np.max([grid.unit_running(layout.units[i].id) for i in ac_idx], axis=0) if ac_idx else None
    for j in range(L):
        ks = k0s + j
        q = unit_heat(params, layout, grid, ks)  # (nw, nu), as asked
        dist = np.column_stack([grid.tout[ks], grid.sol[ks, 0], grid.sol[ks, 1], grid.sol[ks, 2], grid.sol[ks, 3],
                                grid.hc[ks], grid.hs[ks], np.ones(nw)])
        wv = grid.open[ks] * (grid.tout[ks][:, None] - X[:, win_zone]) if win_zone else np.zeros((nw, 0))
        if ac_idx:
            qd, pe = outdoor_unit(params, q[:, ac_idx], X[:, ac_zone], grid.tout[ks], running[ks])
            q[:, ac_idx] = qd  # what the outdoor unit could deliver
        else:
            pe = np.zeros(nw)
        V = np.concatenate([q, dist, wv], axis=1)
        Pel[:, j] = pe
        X = X @ Ad.T + V @ Bd.T
        Ta[:, j + 1] = X[:, :nz]
    return Ta, Pel


# ---- training ------------------------------------------------------------------------
def train(layout: Layout, grid: Grid, initial: Params | None = None, progress=None, val_share=0.25) -> dict:
    t_start = time.time()
    pw = power_observation(grid)
    train_days, val_days, per_day = split_days(grid, val_share)
    warnings = list(grid.notes) + list(pw["notes"])
    if not train_days:
        raise ValueError("not enough complete days of room temperatures to train on")
    if not val_days:
        warnings.append("Only one complete day: the check uses the same day the model was trained on.")
    if not any((kw > 0.1).any() for kw in grid.heater_kw.values()):
        warnings.append("No heater power: absolute heat capacities lean on the priors.")
    if not any(f.sum() > 0 for f in grid.ac_frac.values()):
        warnings.append("No unit was ever recorded heating or cooling.")
    k0s = np.array(windows_for(grid, train_days, per_day))
    if len(k0s) < 4:
        raise ValueError("too few gap-free six-hour stretches to train on")
    L = int(FIT["window_h"] * 60 / grid.step_min)
    e = FIT["every"]
    # The priors are physics (a plausible apartment, the heater's measured
    # power), always; a previous model is only where the search starts.
    # Centred on the previous model, strong priors would freeze it.
    base = initial_params(layout, grid)
    base.building.setdefault("solar_air", SOLAR_AIR)
    for z in layout.zones:  # facades the zone's windows don't face get no direct sun
        for f in "ESW":
            if not z.sun_from(f):
                base.zones[z.id]["s" + f] = 0.0
    plist = param_list(layout, base)
    theta0 = to_theta(base, plist)
    start = to_theta(_merged(base, initial), plist) if initial else theta0
    # Temperatures five minutes apart, in windows that overlap, are far from
    # independent: taken one by one they would outvote the priors and push
    # what the data cannot tell apart to implausible values. Weighting the
    # priors up keeps those near physics; what the phase measurement shows
    # (the units' heat, the efficiency) keeps its own say.
    sig = np.array([p.sigma / FIT["prior_weight"] ** (1.0 - p.measured) for p in plist])
    obs_ok, obs = pw["ok"], pw["obs"]
    sel = np.arange(e, L + 1, e)

    def residuals(theta):
        params = from_theta(theta, plist, base)
        Ta, Pel = simulate_windows(layout, params, grid, k0s, L)
        meas = grid.T[k0s[:, None] + sel[None, :]]  # (nw, ns, nz)
        rt = ((Ta[:, sel] - meas) / FIT["sigma_t"]).ravel()
        # power, averaged over the same 15-minute blocks
        idx = k0s[:, None] + np.arange(L)[None, :]
        ok = obs_ok[idx]
        nb = L // e
        okb = ok[:, : nb * e].reshape(len(k0s), nb, e).all(axis=2)
        ps = Pel[:, : nb * e].reshape(len(k0s), nb, e).mean(axis=2)
        po = np.nan_to_num(obs[idx][:, : nb * e]).reshape(len(k0s), nb, e).mean(axis=2)
        rp = np.where(okb, (ps - po) / FIT["sigma_p"], 0.0).ravel()
        rpr = (theta - theta0) / sig
        c = FIT["huber"]
        return np.concatenate([robust(np.nan_to_num(rt), c), robust(rp, c), rpr])

    calls = {"n": 0}
    # least_squares counts at most max_nfev rounds, each one Jacobian (one
    # evaluation per parameter) and a step: an upper bound, often not reached
    budget = (len(plist) + 1) * FIT["max_nfev"]

    def fun(theta):
        calls["n"] += 1
        if progress and calls["n"] % 25 == 0:
            progress({"stage": "fit", "evaluations": calls["n"], "budget": budget})
        return residuals(theta)

    if progress:
        progress({"stage": "fit", "evaluations": 0, "budget": budget})

    res = least_squares(fun, start, method="trf", loss="linear", max_nfev=FIT["max_nfev"], x_scale="jac")
    model = from_theta(res.x, plist, base)
    if progress:
        progress({"stage": "check"})
    for uid, kw in heater_power(grid).items():
        model.units[uid]["kw"] = kw
    report = check(layout, model, grid, pw, train_days, val_days, per_day)
    report.update(warnings=warnings, windows=int(len(k0s)), cost=float(res.cost), evaluations=calls["n"],
                  seconds=round(time.time() - t_start, 1), trained_at=time.time(),
                  range=[float(grid.t[0]), float(grid.t[-1])],
                  days={"train": [float(grid.t[k]) for k in train_days], "val": [float(grid.t[k]) for k in val_days]},
                  zones={z.id: zone_summary(model, z.id) for z in layout.zones},
                  cop7=float(model.plant["eta"] * carnot_heat(21.0, 7.0)),
                  plant=plant_curve(layout, model, grid, pw))
    return {"params": model, "report": report}


def plant_curve(layout: Layout, params: Params, grid: Grid, pw: dict, n_points: int = 400) -> dict | None:
    """The outdoor unit's electricity against the heat it delivers, at 7 °C outside
    and 21 °C inside, with what the phase measured in heating steps for comparison."""
    ac = [j for j, u in enumerate(layout.units) if u.kind == "ac"]
    if not ac:
        return None
    qs = np.linspace(0.0, 10.0, 81)
    one = np.zeros((len(qs), 1))
    one[:, 0] = qs
    _, p = outdoor_unit(params, one, np.full_like(one, 21.0), np.full(len(qs), 7.0), np.ones(len(qs)))
    curve_q, curve_p = [], []
    for q, pe in zip(qs, p):
        if params.plant.get("Pmax") and pe >= params.plant["Pmax"] - 1e-9:
            break
        curve_q.append(float(q))
        curve_p.append(float(pe))
    # measured: steps with only heating at the AC units, no heater, power known
    ks = np.arange(grid.K)
    q = unit_heat(params, layout, grid, ks)[:, ac]
    share = np.max([grid.unit_running(layout.units[j].id) for j in ac], axis=0)
    heater = sum(grid.heater_kw.values()) > 0.05 if grid.heater_kw else np.zeros(grid.K, bool)
    room = grid.T[:, [layout.zone_index(layout.units[j].zone) for j in ac]]
    qd, pel = outdoor_unit(params, q, np.nan_to_num(room, nan=21.0), grid.tout, share)
    at_limit = bool(params.plant.get("Pmax")) & (pel >= params.plant.get("Pmax", np.inf) - 1e-6) & (np.abs(qd).sum(1) < np.abs(q).sum(1) - 1e-6)
    sel = pw["ok"] & (share > 0.99) & ~heater & (q.min(axis=1) >= 0)
    idx = np.flatnonzero(sel)
    if len(idx) > n_points:
        idx = idx[np.linspace(0, len(idx) - 1, n_points).astype(int)]
    return {"q": curve_q, "p": curve_p, "pmax": params.plant.get("Pmax"), "base": plant_base(params),
            "points": [[round(float(qd[k].sum()), 3), round(float(pw["obs"][k]), 3)] for k in idx],
            "hours_at_limit": round(float(at_limit.sum()) * grid.dt_h, 1),
            "tout_points": [round(float(grid.tout[k]), 1) for k in idx]}


def check(layout: Layout, params: Params, grid: Grid, pw: dict, train_days, val_days, per_day) -> dict:
    """Predict whole days from midnight with nothing but the recorded inputs."""
    sysm = build(layout, params)
    hidden = estimate_hidden(sysm, grid.T, grid.tout, grid.sol, grid.dt_h)

    def run(days):
        days = [k for k in days if k + per_day < grid.K]
        if not days:
            return None
        k0s = np.array(days)
        Ta, Pel = simulate_windows(layout, params, grid, k0s, per_day, sysm, hidden)
        meas = grid.T[k0s[:, None] + np.arange(per_day + 1)[None, :]]
        err = Ta - meas
        rmse = np.sqrt(np.nanmean(err ** 2, axis=(0, 1)))
        idx = k0s[:, None] + np.arange(per_day)[None, :]
        ok = pw["ok"][idx]
        e_sim = float((Pel * ok).sum() * grid.dt_h)
        e_obs = float((np.nan_to_num(pw["obs"][idx]) * ok).sum() * grid.dt_h)
        return {"rmse": rmse.tolist(), "bias": np.nanmean(err, axis=(0, 1)).tolist(), "energy_sim": e_sim, "energy_obs": e_obs,
                "days": [{"t0": float(grid.t[k]), "pred": Ta[i].tolist(), "meas": np.where(np.isfinite(meas[i]), meas[i], None).tolist()}
                         for i, k in enumerate(days)]}

    tr = run(train_days)
    va = run(val_days) or run(train_days[-1:])
    return {"rmse_train": tr["rmse"] if tr else None, "rmse_val": va["rmse"], "bias_val": va["bias"],
            "energy": {"sim": va["energy_sim"], "obs": va["energy_obs"], "observed": bool(pw["ok"].any())},
            "validation": va["days"]}
