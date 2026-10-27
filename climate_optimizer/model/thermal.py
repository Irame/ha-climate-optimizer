"""The apartment as a linear thermal network.

Every zone has three nodes: its air (what the thermometer reads, warmed
directly by units and heaters), its surfaces and furniture (charged by the
air within hours, warmed by the sun), and the structure behind them (walls,
floor slab: days, leaking to the outside and to the neighbours). Zones
exchange heat through their air along the layout's links.

    Ca dTa/dt = H (Tout - Ta) + Hw open (Tout - Ta) + Ham (Tm - Ta)
              + sum L (Ta_other - Ta) + Q_units + q + qc cos(h) + qs sin(h)
    Cm dTm/dt = Ham (Ta - Tm) + Hmw (Tw - Tm) + sD D + sE E + sS S + sW W
    Cw dTw/dt = Hmw (Tm - Tw) + Hwo (Tout - Tw) + Hwn (Tnb - Tw)

Everything is linear in the temperatures, so a step of any length is exact
(zero-order hold through the matrix exponential), which the optimizer needs.
The window term is bilinear; it enters as an input computed from the air
temperature at the start of each step.

Inputs, in this order: one heat input per unit (kW into its zone's air),
the disturbances DIST, and one window input per window zone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.signal import lfilter

from .layout import Layout, Params, link_key

DIST = ["tout", "D", "E", "S", "W", "hc", "hs", "one"]
FACADES = ["D", "E", "S", "W"]


@dataclass
class System:
    layout: Layout
    params: Params
    A: np.ndarray  # continuous, n x n
    B: np.ndarray  # continuous, n x m
    n: int
    nz: int
    nu: int
    nw: int

    @property
    def m(self) -> int:
        return self.B.shape[1]

    def cols(self):
        """Column slices of B: units, disturbances, windows."""
        return slice(0, self.nu), slice(self.nu, self.nu + len(DIST)), slice(self.nu + len(DIST), self.m)

    def discretize(self, dt_h: float) -> tuple[np.ndarray, np.ndarray]:
        n, m = self.n, self.m
        M = np.zeros((n + m, n + m))
        M[:n, :n] = self.A * dt_h
        M[:n, n:] = self.B * dt_h
        E = expm(M)
        return E[:n, :n], E[:n, n:]


def build(layout: Layout, params: Params) -> System:
    zids = layout.zone_ids
    nz = len(zids)
    n = 3 * nz
    units = list(layout.units)
    nu = len(units)
    wz = layout.window_zones
    nw = len(wz)
    m = nu + len(DIST) + nw
    A = np.zeros((n, n))
    B = np.zeros((n, m))
    Tnb = params.building.get("Tnb", 20.0)
    for i, z in enumerate(zids):
        p = params.zones[z]
        a, mm, w = i, nz + i, 2 * nz + i
        Ca, Cm, Cw = p["Ca"], p["Cm"], p["Cw"]
        A[a, a] -= (p["H"] + p["Ham"]) / Ca
        A[a, mm] += p["Ham"] / Ca
        A[mm, a] += p["Ham"] / Cm
        A[mm, mm] -= (p["Ham"] + p["Hmw"]) / Cm
        A[mm, w] += p["Hmw"] / Cm
        A[w, mm] += p["Hmw"] / Cw
        A[w, w] -= (p["Hmw"] + p["Hwo"] + p["Hwn"]) / Cw
        d0 = nu
        B[a, d0 + DIST.index("tout")] += p["H"] / Ca
        B[w, d0 + DIST.index("tout")] += p["Hwo"] / Cw
        for f in FACADES:
            B[mm, d0 + DIST.index(f)] += p["s" + f] / Cm
        B[a, d0 + DIST.index("hc")] += p.get("qc", 0.0) / Ca
        B[a, d0 + DIST.index("hs")] += p.get("qs", 0.0) / Ca
        B[a, d0 + DIST.index("one")] += p["q"] / Ca
        B[w, d0 + DIST.index("one")] += p["Hwn"] * Tnb / Cw
    for x, y in layout.links:
        L = params.links[link_key(x, y)]
        i, j = layout.zone_index(x), layout.zone_index(y)
        A[i, i] -= L / params.zones[x]["Ca"]
        A[i, j] += L / params.zones[x]["Ca"]
        A[j, j] -= L / params.zones[y]["Ca"]
        A[j, i] += L / params.zones[y]["Ca"]
    for k, u in enumerate(units):
        i = layout.zone_index(u.zone)
        B[i, k] = 1.0 / params.zones[u.zone]["Ca"]
    for k, z in enumerate(wz):
        i = layout.zone_index(z)
        B[i, nu + len(DIST) + k] = params.zones[z]["Hw"] / params.zones[z]["Ca"]
    return System(layout, params, A, B, n, nz, nu, nw)


# ---- heat pump ------------------------------------------------------------------
def carnot_heat(tin, tout):
    th = np.asarray(tin) + 273.15 + 12.0
    tc = np.asarray(tout) + 273.15 - 6.0
    return th / np.maximum(8.0, th - tc)


def carnot_cool(tin, tout):
    tc = np.asarray(tin) + 273.15 - 10.0
    th = np.asarray(tout) + 273.15 + 10.0
    return tc / np.maximum(8.0, th - tc)


def capacity_factor(params: Params, tout):
    """Heating output falls as it gets colder outside."""
    return np.clip(1.0 - params.plant.get("cap_slope", 0.02) * (7.0 - np.asarray(tout)), 0.4, 1.2)


def level_heat(params: Params, unit_id: str, level, mode, tout):
    """Heat (kW, negative when cooling) of an AC unit at a fan level.
    mode: 1 heat, 2 cool, 0 off; arrays broadcast."""
    lv = np.asarray(params.units[unit_id]["heat"], dtype=float)
    level = np.clip(np.asarray(level, dtype=int), 0, len(lv) - 1)
    q = lv[level]
    mode = np.asarray(mode)
    heat = q * capacity_factor(params, tout)
    cool = -q * params.units[unit_id].get("cool_ratio", 0.9)
    return np.where(mode == 1, heat, np.where(mode == 2, cool, 0.0))


def electric(params: Params, q, tin, tout):
    """Electric kW for an AC unit's heat q at the given temperatures."""
    q = np.asarray(q, dtype=float)
    eta = params.plant["eta"]
    with np.errstate(divide="ignore", invalid="ignore"):
        heat = np.where(q > 0, q / (eta * carnot_heat(tin, tout)), 0.0)
        cool = np.where(q < 0, -q / (eta * carnot_cool(tin, tout)), 0.0)
    return heat + cool


# ---- hidden states --------------------------------------------------------------
def estimate_hidden(sys: System, T: np.ndarray, tout: np.ndarray, sol: np.ndarray, dt_h: float):
    """Surfaces and structure are never measured. Driving them with the
    measured air temperatures reconstructs them; the error dies out within
    a day or two, so after a burn-in this is a good estimate at every step.

    T: (K, nz) air, NaN allowed; tout: (K,); sol: (K, 4) in FACADES order.
    Returns (K, nz) surfaces and (K, nz) structure, the state at each step's start.
    """
    K, nz = T.shape
    params, zids = sys.params, sys.layout.zone_ids
    Tm = np.empty((K, nz))
    Tw = np.empty((K, nz))
    Tnb = params.building.get("Tnb", 20.0)
    for i, z in enumerate(zids):
        p = params.zones[z]
        ta = _ffill(T[:, i])
        first = min(K, int(round(24 / dt_h)))
        T0 = float(np.nanmean(ta[:first])) if np.isfinite(ta[:first]).any() else 20.0
        Ac = np.array([[-(p["Ham"] + p["Hmw"]) / p["Cm"], p["Hmw"] / p["Cm"]],
                       [p["Hmw"] / p["Cw"], -(p["Hmw"] + p["Hwo"] + p["Hwn"]) / p["Cw"]]])
        Bc = np.array([[p["Ham"] / p["Cm"], 0.0, 1.0 / p["Cm"], 0.0],
                       [0.0, p["Hwo"] / p["Cw"], 0.0, p["Hwn"] / p["Cw"]]])
        M = np.zeros((6, 6))
        M[:2, :2] = Ac * dt_h
        M[:2, 2:] = Bc * dt_h
        E = expm(M)
        Phi, Gam = E[:2, :2], E[:2, 2:]
        s = sol @ np.array([p["s" + f] for f in FACADES])
        u = np.vstack([ta - T0, tout - T0, s, np.full(K, Tnb - T0)])  # deviations from T0
        lam, V = np.linalg.eig(Phi)
        lam, V = lam.real, V.real
        g = np.linalg.solve(V, Gam @ u)  # (2, K)
        z_ = np.vstack([lfilter([0.0, 1.0], [1.0, -lam[k]], g[k]) for k in range(2)])
        x = V @ z_
        Tm[:, i] = x[0] + T0
        Tw[:, i] = x[1] + T0
    return Tm, Tw


def _ffill(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float)
    ok = np.isfinite(a)
    if not ok.any():
        return np.full_like(a, 20.0)
    idx = np.where(ok, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    first = np.argmax(ok)
    out[:first] = a[first]
    return out


def hour_wave(ts_ms: np.ndarray, tz) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin of the local hour of day, for the internal-gain rhythm."""
    import datetime as dt
    hours = np.array([(lambda d: d.hour + d.minute / 60)(dt.datetime.fromtimestamp(t / 1000, tz)) for t in ts_ms])
    return np.cos(2 * np.pi * hours / 24), np.sin(2 * np.pi * hours / 24)


def zone_summary(params: Params, z: str) -> dict:
    p = params.zones[z]
    C = p["Ca"] + p["Cm"] + p["Cw"]
    H = p["H"] + 1.0 / (1.0 / p["Ham"] + 1.0 / p["Hmw"] + 1.0 / max(1e-9, p["Hwo"] + p["Hwn"]))
    return {"C": C, "Hloss": H, "tau_h": C / max(1e-6, H)}
