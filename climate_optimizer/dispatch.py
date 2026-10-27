"""From the plan to what a proxy is told, once a minute.

The plan says, per 5-minute step, whether a unit runs and how much heat it
should deliver. A fan level delivers fixed heat, so a target between two
levels is met by switching between them minute by minute, carrying what was
delivered too much or too little into the next minute (error diffusion).
Over a step the heat comes out right and the level changes at most a few
times.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model.layout import Params
from .model.thermal import level_heat

MIN = 60_000


@dataclass
class _UnitState:
    owed: float = 0.0  # kW·min delivered too little (+) or too much (-)
    level: int | None = None
    since: float = 0.0  # ms the level was set
    minute: int | None = None  # minute the last decision was for


class FanDispatcher:
    def __init__(self, hold_min: float = 2.0):
        self.hold_min = hold_min
        self.units: dict[str, _UnitState] = {}

    def reset(self, unit: str) -> None:
        self.units.pop(unit, None)

    def level(self, params: Params, unit: str, mode: str, target_kw: float, tout: float,
              allowed: tuple[int, int], t_ms: float) -> int:
        """Fan level for this minute so the heat follows target_kw (a magnitude)."""
        st = self.units.setdefault(unit, _UnitState())
        lo, hi = allowed
        m = 1 if mode == "heat" else 2
        heats = np.abs(level_heat(params, unit, np.arange(lo, hi + 1), m, tout))
        minute = int(t_ms // MIN)
        if st.minute == minute and st.level is not None:
            return st.level  # asked again within the same minute
        target = float(np.clip(target_kw, heats[0], heats[-1]))
        # the two levels around the target
        j = int(np.searchsorted(heats, target, side="right")) - 1
        j = max(0, min(j, len(heats) - 2)) if len(heats) > 1 else 0
        a, b = (j, j + 1) if len(heats) > 1 else (0, 0)
        want = st.owed + target
        pick = b if want >= (heats[a] + heats[b]) / 2 else a
        level = lo + pick
        held = st.level is not None and (t_ms - st.since) < self.hold_min * MIN
        if held and lo <= st.level <= hi and abs(st.level - level) <= 1 and st.level in (lo + a, lo + b):
            level = st.level  # don't flap: keep a fresh level for a while if it is one of the two
        delivered = heats[level - lo]
        st.owed = float(np.clip(want - delivered, -heats[-1], heats[-1]))
        if level != st.level:
            st.since = t_ms
        st.level = level
        st.minute = minute
        return level
