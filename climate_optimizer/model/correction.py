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
