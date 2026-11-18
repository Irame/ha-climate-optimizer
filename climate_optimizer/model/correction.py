"""Correcting the plans by what the model gets wrong.

Every few minutes the model is run again over the last hour, from the state an hour
ago, with what the units really did, the weather as it was and the windows as they
were. Where it ends away from what the thermometers say, the model misses something
about that room (it loses heat faster than the model thinks, a thermometer sits in a
unit's warm air). What it keeps missing by goes into the plans as heat the room gains
or loses beyond the model.

What was planned or decided in that hour plays no part: a comfort temperature raised,
a run by a button, a rule changed, a proxy off. Only what happened is reckoned with.
"""

from __future__ import annotations

import json

import numpy as np

from .dataset import Grid
from .fit import simulate_windows
from .layout import Layout, Params
from .thermal import DIST, build

STEPS = 12  # an hour of five-minute steps
TAIL = 2  # the last ten minutes, averaged (a reading between two steps is noisy)


def hour_misses(layout: Layout, params: Params, grid: Grid, hidden, k1: int, steps: int = STEPS) -> dict[str, float]:
    """Per room, how much warmer the model says it is at step k1 than its thermometer,
    run from k1 - steps on what was recorded. None for a room without a usable reading;
    nothing at all when the hour is not whole or a window was open in it."""
    k0 = k1 - steps
    if k0 < 1 or k1 >= grid.K:
        return {}
    if not grid.valid[k0:k1 + 1].all():
        return {}
    if grid.open.size and grid.open[k0:k1 + 1].any():
        return {}  # a window open: what it let out is the window's, not the model's
    Ta, _ = simulate_windows(layout, params, grid, np.array([k0]), steps, hidden=hidden)
    out: dict[str, float] = {}
    for i, z in enumerate(layout.zone_ids):
        meas = grid.T[k1 - TAIL + 1:k1 + 1, i]
        start = grid.T[k0, i]
        if not np.isfinite(meas).all() or not np.isfinite(start):
            out[z] = None  # no reading (or one standing still): nothing to hold it against
            continue
        out[z] = float(Ta[0, steps - TAIL + 1:steps + 1, i].mean() - meas.mean())
    return out


def hour_response(layout: Layout, params: Params, z: str) -> float:
    """How much warmer a room's air is after an hour of 1 kW more into it (K)."""
    p = Params.from_dict(json.loads(json.dumps(params.to_dict())))
    for zz in p.zones:
        p.zones[zz].update(q=1.0 if zz == z else 0.0, qc=0.0, qs=0.0)
    sysm = build(layout, p)
    Ad, Bd = sysm.discretize(5 / 60)
    u = np.zeros(Bd.shape[1])
    u[len(layout.units) + DIST.index("one")] = 1.0
    x = np.zeros(Ad.shape[0])
    for _ in range(STEPS):
        x = Ad @ x + Bd @ u
    return float(x[layout.zone_index(z)])


def with_correction(params: Params, kw: dict[str, float]) -> Params:
    """A copy of the model with each room's extra heat (kW) on its constant gain."""
    p = Params.from_dict(json.loads(json.dumps(params.to_dict())))
    for z, v in kw.items():
        if z in p.zones and v:
            p.zones[z]["q"] = p.zones[z].get("q", 0.0) + float(v)
    return p


def step(kw: float, err: float, r: float, gain: float, leak: float, lim: float) -> float:
    """One step of the correction: towards what cancels the miss, a little at a time."""
    if r <= 1e-6:
        return kw * leak
    return float(np.clip(kw * leak - gain * err / r, -lim, lim))


# ---- the gap between a plan and what was then done -------------------------------------

def value_at(times, values, t):
    """A step function's value at t (each value holds from its time on); None outside."""
    i = int(np.searchsorted(np.asarray(times, float), t, side="right")) - 1
    if i < 0 or i >= len(values):
        return None
    return values[i]


def _changed(ref: dict, later: dict, z: str, t1: float, tol: float = 0.05) -> bool:
    """Whether a later plan has the room's comfort (or a run, an override, a proxy out of
    auto) otherwise than ref had it for the same times: decided anew after ref was made.
    A change by schedule (a rule from 08:00) both had alike, so it is no change here."""
    a, b = ref["learn"], later["learn"]
    if b.get("forced", {}).get(z) or b.get("override", {}).get(z) or not b.get("auto", {}).get(z, True):
        return True
    for t, v in zip(b["t"], b.get("comfort", {}).get(z) or []):
        if t > t1:
            break
        w = value_at(a["t"], a["comfort"].get(z) or [], t)
        if v is None or w is None:
            continue
        if abs(float(v) - float(w)) > tol:
            return True
    return False


def exec_gaps(layout: Layout, params: Params, grid: Grid, hidden, ref: dict, later: list[dict],
              steps: int = STEPS, near: float = 0.3) -> dict[str, float]:
    """Per room, how much warmer the plan ref expected it to get over its first hour than
    the model run over that hour with what was then really done (its own error taken out,
    the start alike for both): what the doing fell short of the planning by. Only rooms
    the plan held at the edge of their comfort, with nothing decided anew meanwhile, no
    window open, its units in auto and none asked to run."""
    learn = ref.get("learn") or {}
    if not learn.get("t"):
        return {}
    t0 = float(ref["t"])
    k0 = int(np.searchsorted(grid.t, t0, side="right")) - 1
    k1 = k0 + steps
    if k0 < 1 or k1 >= grid.K or not grid.valid[k0:k1 + 1].all():
        return {}
    if grid.open.size and grid.open[k0:k1 + 1].any():
        return {}
    t1 = float(grid.t[k1])
    Ta, _ = simulate_windows(layout, params, grid, np.array([k0]), steps, hidden=hidden)
    out: dict[str, float] = {}
    for z in learn.get("comfort", {}):
        if z not in ref.get("temp", {}) or z not in layout.zone_ids:
            continue
        if not learn.get("auto", {}).get(z) or learn.get("forced", {}).get(z) or learn.get("override", {}).get(z):
            continue
        if any(_changed(ref, p, z, t1) for p in later):
            continue
        times = np.asarray(ref["times"], float)
        temps = np.asarray(ref["temp"][z], float)
        sel = (times >= t0) & (times <= t1)
        edges = [value_at(learn["t"], learn["edge"].get(z) or [], x) for x in times[sel]]
        if any(e is None or temp > float(e) + near for temp, e in zip(temps[sel], edges)):
            continue  # cooling down, heating ahead, or no edge: not where the gap is made
        zi = layout.zone_index(z)
        planned = float(np.interp(t1, times, temps) - np.interp(float(grid.t[k0]), times, temps))
        done = float(Ta[0, steps, zi] - Ta[0, 0, zi])
        out[z] = planned - done
    return out
