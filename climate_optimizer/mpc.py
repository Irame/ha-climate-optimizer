"""Planning the next hours: when each unit runs and how hard.

The plan minimises, over a horizon reaching to the end of the known prices,

    electricity × price  +  comfort cost of every zone  +  starts × start cost
    +  the value of heat missing from the rooms at the end

subject to the learned thermal model, each unit's lowest and highest fan
level, the shared outdoor unit's capacity, minimum run and off times, and
the times a unit must not run. On/off are binary, the heat between the
lowest and highest level is continuous (the dispatcher turns it into fan
levels minute by minute), and the comfort costs are convex pieces. That
makes a mixed-integer linear programme, solved with HiGHS.

Steps are 5 minutes for the first two hours, 15 minutes to hour six and 30
minutes after that: what happens now has to be right; what happens
tomorrow only has to be about right, it gets planned again. Only the
first quarter hour decides on or off as such (the minimum run time makes
that the decision that counts); after that "on" is the share of a step a
unit runs, which keeps the programme small and fast.

Heating and cooling: all AC units share one outdoor unit, so they run in the
same mode. When the rooms would drift both too cold and too warm, both
modes are planned and the cheaper plan wins.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .comfort import ComfortFunction, Side
from .model.layout import Layout, Params
from .model.thermal import DIST, System, build, capacity_factor, carnot_cool, carnot_heat, level_heat

MIN = 60_000


@dataclass
class Settings:
    horizon_min_h: float = 12.0
    horizon_max_h: float = 36.0
    fine_h: float = 2.0  # 5-minute steps up to here
    binary_h: float = 0.25  # on/off decided as such up to here, as a share of each step after
    mid_h: float = 6.0  # 15-minute steps up to here, 30 after
    start_cost: float = 0.03  # € per compressor start
    heater_start_cost: float = 0.002  # € per heater switch-on
    min_run_min: float = 15.0
    min_off_min: float = 10.0
    terminal_quantile: float = 0.25  # missing heat at the end is valued at this price quantile
    window_assume_open_min: float = 10.0  # an open window is assumed closed after this
    season: str = "auto"  # auto | heat | cool
    heat_below_outdoor: float = 20.0  # auto: heating only if it gets colder than this within a day
    cool_above_outdoor: float = 20.0  # auto: cooling only if it gets warmer than this within a day
    time_limit_s: float = 20.0
    mip_gap: float = 0.01

    @staticmethod
    def from_dict(d: dict) -> Settings:
        return Settings(**{k: v for k, v in d.items() if k in Settings.__dataclass_fields__})


@dataclass
class UnitSpec:
    """What a unit may do over the horizon."""

    id: str
    kind: str  # ac | heater
    zone: str
    levels: tuple[int, int] = (0, 3)  # lowest and highest fan level allowed
    allowed: np.ndarray | None = None  # (N,) bool; False = must be off
    on: bool = False  # running now
    since_min: float = 1e9  # minutes in the current on/off state
    modes: tuple[str, ...] = ("heat", "cool")


@dataclass
class Problem:
    sys: System
    t: np.ndarray  # (N+1,) step boundaries, ms
    dist: np.ndarray  # (N, len(DIST))
    window: np.ndarray  # (N, nw) open fraction of each step
    price: np.ndarray  # (N,) €/kWh
    x0: np.ndarray  # (n,) state now
    comfort: dict[str, np.ndarray]  # zone -> (N,) comfort temperature
    functions: dict[str, list[ComfortFunction]]  # zone -> (N,) function per step
    units: list[UnitSpec]
    settings: Settings = field(default_factory=Settings)

    @property
    def N(self) -> int:
        return len(self.t) - 1

    @property
    def dt_h(self) -> np.ndarray:
        return np.diff(self.t) / 3.6e6


@dataclass
class Plan:
    mode: str  # heat | cool: the AC units' mode
    t: np.ndarray  # (N+1,)
    on: dict[str, np.ndarray]  # unit -> (N,) bool: runs in (some of) the step
    share: dict[str, np.ndarray]  # unit -> (N,) share of the step running (0/1 in 5-minute steps)
    heat: dict[str, np.ndarray]  # unit -> (N,) kW into the zone (negative when cooling)
    temp: dict[str, np.ndarray]  # zone -> (N+1,) predicted air temperature
    comfort: dict[str, np.ndarray]  # zone -> (N,)
    electric: np.ndarray  # (N,) kW of all units
    price: np.ndarray  # (N,)
    cost: dict[str, float]  # energy, comfort, starts, terminal, total (€)
    status: str
    seconds: float
    gap: float | None = None
    alternatives: dict[str, float] = field(default_factory=dict)  # mode -> objective of the others tried

    def step_at(self, t_ms: float) -> int | None:
        if self.t[0] - 5 * MIN <= t_ms < self.t[0]:
            return 0  # a clock a little behind the planner's
        k = int(np.searchsorted(self.t, t_ms, side="right")) - 1
        return k if 0 <= k < len(self.t) - 1 else None

    def to_dict(self) -> dict:
        r = lambda a, d=3: [None if not np.isfinite(v) else round(float(v), d) for v in a]
        return {
            "mode": self.mode, "t": [int(v) for v in self.t],
            "on": {u: [bool(v) for v in a] for u, a in self.on.items()},
            "share": {u: r(a, 2) for u, a in self.share.items()},
            "heat": {u: r(a) for u, a in self.heat.items()},
            "temp": {z: r(a, 2) for z, a in self.temp.items()},
            "comfort": {z: r(a, 2) for z, a in self.comfort.items()},
            "electric": r(self.electric), "price": r(self.price, 4),
            "cost": {k: round(v, 4) for k, v in self.cost.items()},
            "status": self.status, "seconds": round(self.seconds, 2), "gap": self.gap,
            "alternatives": {k: round(v, 4) for k, v in self.alternatives.items()},
        }


# ---- the step grid --------------------------------------------------------------------
def step_grid(t0_ms: int, price_end_ms: int | None, s: Settings) -> np.ndarray:
    """Boundaries from now to the horizon. The first step ends at the next
    5-minute mark (at least a minute away); later ones sit on their own grid."""
    h = 3.6e6
    end = t0_ms + s.horizon_max_h * h
    if price_end_ms is not None:
        end = min(end, max(price_end_ms, t0_ms + s.horizon_min_h * h))
    else:
        end = t0_ms + s.horizon_min_h * h
    t = [t0_ms, math.ceil((t0_ms + MIN) / (5 * MIN)) * 5 * MIN]
    for span, step in ((s.fine_h, 5), (s.mid_h, 15), (None, 30)):
        grid = step * MIN
        # the last span runs to the first mark at or after the end
        lim = math.ceil(end / grid) * grid if span is None else min(end, t0_ms + span * h)
        while t[-1] < lim:
            nxt = math.ceil((t[-1] + 1) / grid) * grid
            if nxt > lim:
                break
            t.append(nxt)
        if t[-1] >= end:
            break
    return np.array(t, dtype=np.int64)


def step_means(t: np.ndarray, times: np.ndarray, values: np.ndarray, kind: str = "linear") -> np.ndarray:
    """Average of a forecast over each step, sampled every 5 minutes.
    kind 'linear' interpolates points, 'hold' holds each value until the next."""
    out = np.empty(len(t) - 1)
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    for k in range(len(t) - 1):
        n = max(1, int(round((t[k + 1] - t[k]) / (5 * MIN))))
        s = t[k] + (np.arange(n) + 0.5) * (t[k + 1] - t[k]) / n
        if kind == "hold":
            idx = np.clip(np.searchsorted(times, s, side="right") - 1, 0, len(times) - 1)
            out[k] = np.mean(values[idx])
        else:
            out[k] = np.mean(np.interp(s, times, values))
    return out


def price_steps(t: np.ndarray, intervals: list[tuple[int, int, float]]) -> np.ndarray:
    """Time-weighted price of each step. Beyond the known prices a step takes
    the price from the same time a day earlier, else the last known one."""
    iv = sorted(intervals)
    if not iv:
        raise ValueError("no prices")
    starts = np.array([a for a, _, _ in iv], dtype=float)
    ends = np.array([b for _, b, _ in iv], dtype=float)
    vals = np.array([p for _, _, p in iv], dtype=float)
    last_end = ends.max()

    def at(ts):
        ts = np.asarray(ts, dtype=float)
        day = 864e5
        shifted = ts.copy()
        while True:
            late = shifted >= last_end
            if not late.any() or (shifted[late] - day < starts[0]).all():
                break
            shifted = np.where(late & (shifted - day >= starts[0]), shifted - day, shifted)
        i = np.clip(np.searchsorted(starts, shifted, side="right") - 1, 0, len(iv) - 1)
        inside = (shifted >= starts[i]) & (shifted < ends[i])
        return np.where(inside, vals[i], vals[np.argmax(ends)])

    out = np.empty(len(t) - 1)
    for k in range(len(t) - 1):
        n = max(1, int(round((t[k + 1] - t[k]) / MIN)))
        s = t[k] + (np.arange(n) + 0.5) * (t[k + 1] - t[k]) / n
        out[k] = at(s).mean()
    return out


# ---- solving -------------------------------------------------------------------------
class _Vars:
    def __init__(self):
        self.n = 0
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.c: list[float] = []
        self.integ: list[int] = []

    def add(self, count, lb=0.0, ub=np.inf, c=0.0, integer=False) -> np.ndarray:
        idx = np.arange(self.n, self.n + count)
        self.n += count
        self.lb += list(np.broadcast_to(lb, (count,)).astype(float))
        self.ub += list(np.broadcast_to(ub, (count,)).astype(float))
        self.c += list(np.broadcast_to(c, (count,)).astype(float))
        self.integ += [1 if integer else 0] * count
        return idx


class _Rows:
    def __init__(self):
        self.r: list[int] = []
        self.c: list[int] = []
        self.v: list[float] = []
        self.lo: list[float] = []
        self.hi: list[float] = []

    def add(self, cols, vals, lo=-np.inf, hi=np.inf):
        row = len(self.lo)
        self.r += [row] * len(cols)
        self.c += [int(c) for c in cols]
        self.v += [float(v) for v in vals]
        self.lo.append(lo)
        self.hi.append(hi)


def _discretize_cached(sys: System, dts: np.ndarray):
    cache = {}
    out = []
    for dt in dts:
        key = int(round(dt * 3600))
        if key not in cache:
            cache[key] = sys.discretize(key / 3600)
        out.append(cache[key])
    return out


def unit_bounds(params: Params, u: UnitSpec, mode: str, tout: np.ndarray, dts: np.ndarray, s: Settings):
    """Lowest and highest heat (kW, a magnitude) per step while running, and
    electricity per kW of heat."""
    N = len(tout)
    if u.kind == "heater":
        kw = params.units[u.id]["kw"]
        return np.full(N, kw), np.full(N, kw), np.ones(N)
    lo_l, hi_l = u.levels
    m = 1 if mode == "heat" else 2
    lo = np.abs(level_heat(params, u.id, lo_l, m, tout))
    hi = np.abs(level_heat(params, u.id, hi_l, m, tout))
    return lo, hi, None


def cop(params: Params, mode: str, tin: np.ndarray, tout: np.ndarray) -> np.ndarray:
    eta = params.plant["eta"]
    return eta * (carnot_heat(tin, tout) if mode == "heat" else carnot_cool(tin, tout))


def steady_store(params: Params, z: str, Ta: float, tout: float) -> float:
    """Heat (kWh above 0 °C) a zone holds when its air is kept at Ta for long."""
    p = params.zones[z]
    Tnb = params.building.get("Tnb", 20.0)
    M = np.array([[-(p["Ham"] + p["Hmw"]), p["Hmw"]], [p["Hmw"], -(p["Hmw"] + p["Hwo"] + p["Hwn"])]])
    rhs = -np.array([p["Ham"] * Ta, p["Hwo"] * tout + p["Hwn"] * Tnb])
    Tm, Tw = np.linalg.solve(M, rhs)
    return p["Ca"] * Ta + p["Cm"] * Tm + p["Cw"] * Tw


_stdout_lock = threading.Lock()


@contextlib.contextmanager
def _quiet_stdout():
    """HiGHS prints progress notes to the process's stdout even when told not to."""
    with _stdout_lock:
        try:
            fd = sys.stdout.fileno()
        except (AttributeError, OSError, ValueError):
            yield
            return
        sys.stdout.flush()
        saved = os.dup(fd)
        try:
            with open(os.devnull, "w") as null:
                os.dup2(null.fileno(), fd)
                yield
        finally:
            os.dup2(saved, fd)
            os.close(saved)


def solve_mode(pr: Problem, mode: str) -> tuple[Plan | None, float]:
    """One MILP with the AC units in `mode`. Returns the plan and its objective."""
    t_start = time.time()
    s = pr.settings
    sys = pr.sys
    params, layout = sys.params, sys.layout
    N, n, nz = pr.N, sys.n, sys.nz
    dts = pr.dt_h
    tout = pr.dist[:, DIST.index("tout")]
    mats = _discretize_cached(sys, dts)
    U = pr.units
    nu = len(U)
    fine = (pr.t[:-1] - pr.t[0]) < s.binary_h * 3.6e6  # binary on/off; later: share of the step running
    sl_u, sl_d, sl_w = sys.cols()
    unit_col = {u.id: j for j, u in enumerate(layout.units)}
    zid = layout.zone_ids
    V, R = _Vars(), _Rows()

    # states
    X = V.add((N + 1) * n, lb=-np.inf, ub=np.inf).reshape(N + 1, n)
    for i in range(n):
        V.lb[X[0, i]] = V.ub[X[0, i]] = float(pr.x0[i])

    # units
    Q, ON, ST, sign, info = [], [], [], [], []
    for u in U:
        heats = u.kind == "heater" or mode == "heat"
        sgn = 1.0 if heats else -1.0
        lo, hi, el = unit_bounds(params, u, mode, tout, dts, s)
        tin = pr.comfort.get(u.zone, np.full(N, 21.0))
        el = 1.0 / cop(params, mode, tin, tout) if el is None else el
        allowed = np.ones(N, bool) if u.allowed is None else np.asarray(u.allowed, bool).copy()
        if u.kind == "ac" and mode not in u.modes:
            allowed[:] = False
        q = V.add(N, lb=0.0, ub=hi, c=pr.price * dts * el)
        on_ub = allowed.astype(float)
        on_lb = np.zeros(N)
        starts = (pr.t[:-1] - pr.t[0]) / MIN  # minutes from now to each step's start
        if u.on and u.since_min < s.min_run_min:
            keep = starts < s.min_run_min - u.since_min
            blocked = np.flatnonzero(~allowed)
            if len(blocked):
                keep[blocked[0]:] = False  # a lockout ends the minimum run early
            on_lb[keep] = 1.0
        if not u.on and u.since_min < s.min_off_min:
            on_ub[starts < s.min_off_min - u.since_min] = 0.0
        on = np.concatenate([V.add(int(fine.sum()), lb=on_lb[fine], ub=on_ub[fine], integer=True),
                             V.add(int((~fine).sum()), lb=on_lb[~fine], ub=on_ub[~fine])])
        st = V.add(N, lb=0.0, ub=1.0, c=s.start_cost if u.kind == "ac" else s.heater_start_cost)
        for k in range(N):
            R.add([q[k], on[k]], [1.0, -hi[k]], hi=0.0)
            R.add([q[k], on[k]], [1.0, -lo[k]], lo=0.0)
            prev = None if k == 0 else on[k - 1]
            if prev is None:
                R.add([st[k], on[k]], [1.0, -1.0], lo=-float(u.on))
            else:
                R.add([st[k], on[k], prev], [1.0, -1.0, 1.0], lo=0.0)
            # minimum run and off time after a switch at k
            for j in range(k + 1, N):
                if starts[j] - starts[k] >= s.min_run_min:
                    break
                if prev is None:
                    R.add([on[j], on[k]], [1.0, -1.0], lo=-float(u.on))
                else:
                    R.add([on[j], on[k], prev], [1.0, -1.0, 1.0], lo=0.0)
            for j in range(k + 1, N):
                if starts[j] - starts[k] >= s.min_off_min:
                    break
                if prev is None:
                    R.add([on[k], on[j]], [-1.0, 1.0], hi=1.0 - float(u.on))
                else:
                    R.add([prev, on[k], on[j]], [1.0, -1.0, 1.0], hi=1.0)
        Q.append(q)
        ON.append(on)
        ST.append(st)
        sign.append(sgn)
        info.append((u, lo, hi, el))

    # the outdoor unit: its capacity is shared, and it draws standby power whenever any AC runs
    ac = [j for j, u in enumerate(U) if u.kind == "ac"]
    if ac:
        qmax = params.plant.get("Qmax", 7.0)
        cap = qmax * (capacity_factor(params, tout) if mode == "heat" else np.full(N, 0.9))
        standby = params.plant.get("standby", 0.03)
        anyon = V.add(N, lb=0.0, ub=1.0, c=standby * pr.price * dts)
        for k in range(N):
            R.add([Q[j][k] for j in ac], [1.0] * len(ac), hi=float(cap[k]))
            for j in ac:
                R.add([anyon[k], ON[j][k]], [1.0, -1.0], lo=0.0)

    # dynamics
    for k in range(N):
        Ad, Bd = mats[k]
        d = pr.dist[k]
        free = Bd[:, sl_d] @ d
        if sys.nw:
            wz = [layout.zone_index(z) for z in layout.window_zones]
            free = free + Bd[:, sl_w] @ (pr.window[k] * (tout[k] - pr.x0[wz]))
        for i in range(n):
            cols = [X[k + 1, i]] + list(X[k])
            vals = [1.0] + list(-Ad[i])
            for j, u in enumerate(U):
                b = Bd[i, sl_u.start + unit_col[u.id]]
                if b:
                    cols.append(Q[j][k])
                    vals.append(-b * sign[j])
            R.add(cols, vals, lo=float(free[i]), hi=float(free[i]))

    # comfort
    seg_vars = []
    for z, Tc in pr.comfort.items():
        i = zid.index(z)
        for k in range(N):
            f = pr.functions[z][k]
            for side, sgn in ((f.below, 1.0), (f.above, -1.0)):
                segs = side.segments()
                cols, vals = [X[k + 1, i]], [sgn]
                for length, slope in segs:
                    v = V.add(1, lb=0.0, ub=np.inf if length is None else length, c=slope * dts[k])[0]
                    cols.append(v)
                    vals.append(1.0)
                    seg_vars.append(v)
                # below: Ta + sum >= Tc ; above: -Ta + sum >= -Tc
                R.add(cols, vals, lo=float(sgn * Tc[k]))

    # heat missing at the end, valued at a cheap price
    q_price = float(np.quantile(pr.price, s.terminal_quantile))
    term_vars = []
    tail = tout[-min(N, 24):].mean()
    for z, Tc in pr.comfort.items():
        i = zid.index(z)
        p = params.zones[z]
        has_ac = any(u.zone == z and u.kind == "ac" for u in U)
        per_kwh = q_price / (float(cop(params, mode, np.array(Tc[-1]), np.array(tout[-1]))) if has_ac else 1.0)
        if mode == "cool" and not has_ac:
            continue
        ref = steady_store(params, z, float(Tc[-1]), float(tail))
        D = V.add(1, lb=0.0, c=per_kwh)[0]
        term_vars.append(D)
        sgn = 1.0 if mode == "heat" else -1.0
        # heat: D >= ref - store ; cool: D >= store - ref
        R.add([D, X[N, i], X[N, nz + i], X[N, 2 * nz + i]], [1.0, sgn * p["Ca"], sgn * p["Cm"], sgn * p["Cw"]],
              lo=sgn * ref)

    vals = np.array(R.v)
    vals[np.abs(vals) < 1e-9] = 0.0  # far couplings in the step matrices: numerically nothing
    A = coo_matrix((vals, (R.r, R.c)), shape=(len(R.lo), V.n)).tocsr()
    A.eliminate_zeros()
    with _quiet_stdout():
        res = milp(c=np.array(V.c), integrality=np.array(V.integ), bounds=Bounds(np.array(V.lb), np.array(V.ub)),
                   constraints=LinearConstraint(A, np.array(R.lo), np.array(R.hi)),
                   options={"time_limit": s.time_limit_s, "mip_rel_gap": s.mip_gap, "disp": False})
    seconds = time.time() - t_start
    if res.x is None:
        return None, math.inf
    x = res.x
    temps = x[X[:, :nz].ravel()].reshape(N + 1, nz)
    share = {u.id: np.clip(x[ON[j]], 0.0, 1.0) for j, u in enumerate(U)}
    on = {u.id: share[u.id] > 0.01 for u in U}
    heat = {u.id: sign[j] * x[Q[j]] * on[u.id] for j, u in enumerate(U)}
    electric = np.zeros(N)
    energy = 0.0
    for j, (u, lo, hi, el) in enumerate(info):
        e = x[Q[j]] * el * on[u.id]
        electric += e
        energy += float(np.sum(e * pr.price * dts))
    if ac:
        sb = params.plant.get("standby", 0.03) * np.max([share[U[j].id] for j in ac], axis=0)
        electric += sb
        energy += float(np.sum(sb * pr.price * dts))
    starts_cost = float(sum(np.dot(x[ST[j]], V.c[ST[j][0]:ST[j][-1] + 1]) for j in range(nu)))
    comfort_cost = float(np.dot(x[seg_vars], np.array(V.c)[seg_vars])) if seg_vars else 0.0
    term_cost = float(np.dot(x[term_vars], np.array(V.c)[term_vars])) if term_vars else 0.0
    status = {0: "optimal", 1: "time limit"}.get(res.status, res.message)
    plan = Plan(
        mode=mode, t=pr.t, on=on, share=share, heat=heat,
        temp={z: temps[:, i] for i, z in enumerate(zid)},
        comfort={z: np.asarray(v, float) for z, v in pr.comfort.items()},
        electric=electric, price=pr.price,
        cost={"energy": energy, "comfort": comfort_cost, "starts": starts_cost, "terminal": term_cost,
              "total": energy + comfort_cost + starts_cost + term_cost},
        status=status, seconds=seconds, gap=getattr(res, "mip_gap", None),
    )
    return plan, float(res.fun)


def free_run(pr: Problem) -> np.ndarray:
    """Air temperatures (N+1, nz) with every unit off."""
    sys = pr.sys
    mats = _discretize_cached(sys, pr.dt_h)
    sl_u, sl_d, sl_w = sys.cols()
    tout = pr.dist[:, DIST.index("tout")]
    wz = [sys.layout.zone_index(z) for z in sys.layout.window_zones]
    x = pr.x0.astype(float).copy()
    out = [x[: sys.nz].copy()]
    for k in range(pr.N):
        Ad, Bd = mats[k]
        u = Bd[:, sl_d] @ pr.dist[k]
        if sys.nw:
            u = u + Bd[:, sl_w] @ (pr.window[k] * (tout[k] - pr.x0[wz]))
        x = Ad @ x + u
        out.append(x[: sys.nz].copy())
    return np.array(out)


def candidate_modes(pr: Problem) -> list[str]:
    s = pr.settings.season
    if s in ("heat", "cool"):
        return [s]
    T = free_run(pr)
    zid = pr.sys.layout.zone_ids
    ac_zones = {u.zone for u in pr.units if u.kind == "ac"}
    cold = warm = 0.0
    for z, Tc in pr.comfort.items():
        if z not in ac_zones:
            continue
        i = zid.index(z)
        for k in range(pr.N):
            f = pr.functions[z][k]
            dev = Tc[k] - T[k + 1, i]
            cold += f.below.cost(dev) * pr.dt_h[k]
            warm += f.above.cost(-dev) * pr.dt_h[k]
    tout = pr.dist[:, DIST.index("tout")]
    day = (pr.t[:-1] - pr.t[0]) < 24 * 3.6e6
    season_ok = {"heat": tout[day].min() < pr.settings.heat_below_outdoor,
                 "cool": tout[day].max() > pr.settings.cool_above_outdoor}
    modes = [m for m, c in (("heat", cold), ("cool", warm)) if c > 1e-6 and season_ok[m]]
    allowed = set().union(*[set(u.modes) for u in pr.units if u.kind == "ac"]) if ac_zones else {"heat"}
    modes = [m for m in modes if m in allowed]
    if not modes:
        modes = ["cool" if season_ok["cool"] and not season_ok["heat"] else "heat"]
        modes = [m for m in modes if m in allowed] or sorted(allowed)[:1]
    return modes


def solve(pr: Problem) -> Plan:
    best, best_obj, objs = None, math.inf, {}
    for mode in candidate_modes(pr):
        plan, obj = solve_mode(pr, mode)
        objs[mode] = obj
        if plan is not None and obj < best_obj:
            best, best_obj = plan, obj
    if best is None:
        raise RuntimeError("no feasible plan")
    best.alternatives = {m: o for m, o in objs.items() if m != best.mode}
    return best


def comfort_cost_of(temp: np.ndarray, Tc: np.ndarray, fns: list[ComfortFunction], dts: np.ndarray) -> float:
    """What the comfort part of the objective says about a temperature path
    (temp: (N+1,), using each step's end)."""
    c = 0.0
    for k in range(len(dts)):
        dev = Tc[k] - temp[k + 1]
        c += (fns[k].below.cost(dev) + fns[k].above.cost(-dev)) * dts[k]
    return c


def side_from(d: dict) -> Side:
    return Side(**d)
