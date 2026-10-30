"""Keeps the store filled: backfills from the recorder on connect, then
records state changes as they come. Numeric sensors are kept as one-minute
means, so a power sensor reporting every second stays a modest table."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone

from .ha import HomeAssistant
from .model.dataset import is_counter
from .store import Store

log = logging.getLogger(__name__)
MIN = 60_000


def tracked_entities(mapping: dict, extra: list[str] = ()) -> list[str]:
    out = []
    for z in (mapping.get("zones") or {}).values():
        out.append(z.get("thermometer"))
        out += z.get("windows") or []
    for u in (mapping.get("units") or {}).values():
        out += [u.get("source"), u.get("proxy"), u.get("power")]
    out += [mapping.get(k) for k in ("outdoor", "weather", "phase", "car", "price")]
    out += mapping.get("phases_other") or []
    out += list(extra)
    seen, res = set(), []
    for e in out:
        if e and e not in seen:
            seen.add(e)
            res.append(e)
    return res


def kind_of(eid: str) -> str:
    dom = eid.split(".")[0]
    if dom == "climate":
        return "climate"
    if dom == "weather":
        return "weather"
    if dom in ("binary_sensor", "switch", "input_boolean"):
        return "binary"
    return "numeric"


def _ms(v) -> int:
    if isinstance(v, (int, float)):
        return int(v * 1000) if v < 1e11 else int(v)
    d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1000)


def _iso(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def _float(s) -> float:
    try:
        v = float(s)
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def row_for(eid: str, t: int, state: str, attrs: dict) -> list:
    k = kind_of(eid)
    if k == "climate":
        return [t, state, attrs.get("temperature"), attrs.get("current_temperature"), attrs.get("hvac_action"), attrs.get("fan_mode")]
    if k == "weather":
        return [t, state, attrs.get("temperature"), attrs.get("cloud_coverage")]
    if k == "binary":
        return [t, state]
    v = _float(state)
    return [t, None if math.isnan(v) else v]


class MinuteMean:
    """Time-weighted mean of a step signal per whole minute."""

    def __init__(self):
        self.v = math.nan
        self.t: int | None = None
        self.acc = 0.0
        self.dur = 0.0

    def add(self, t: int, v: float) -> list[list]:
        out = self.advance(t)
        self.v = v
        return out

    def advance(self, t: int) -> list[list]:
        out = []
        if self.t is None:
            self.t = t
            return out
        a = self.t
        while a < t:
            m_end = (a // MIN + 1) * MIN
            b = min(m_end, t)
            if math.isfinite(self.v):
                self.acc += self.v * (b - a)
                self.dur += b - a
            if b == m_end:
                if self.dur > 0:
                    out.append([m_end - MIN, round(self.acc / self.dur, 4)])
                self.acc = self.dur = 0.0
            a = b
        self.t = max(self.t, t)
        return out


class Collector:
    def __init__(self, ha: HomeAssistant, store: Store):
        self.ha = ha
        self.store = store
        self.entities: list[str] = []
        self.states: dict[str, dict] = {}  # live: entity -> {"state", "attributes", "last_changed", ...}
        self.meta: dict[str, dict] = store.get("meta", {})
        self.means: dict[str, MinuteMean] = {}
        self.backfill_status: dict = {}

    def track(self, entities: list[str]) -> None:
        self.entities = list(entities)

    async def on_connect(self, history_days: int) -> None:
        states = await self.ha.states()
        for s in states:
            self.states[s["entity_id"]] = s
            if s["entity_id"] in self.entities:
                self._meta(s)
        self.store.put("meta", self.meta)
        await self.ha.subscribe("state_changed", self._event)
        await self.backfill(history_days)

    def _meta(self, s: dict) -> None:
        a = s.get("attributes") or {}
        self.meta[s["entity_id"]] = {
            "name": a.get("friendly_name"), "unit": a.get("unit_of_measurement"), "device_class": a.get("device_class"),
            **({"fan_modes": a.get("fan_modes"), "hvac_modes": a.get("hvac_modes")} if s["entity_id"].startswith("climate.") else {}),
        }

    def _event(self, ev: dict) -> None:
        d = ev.get("data") or {}
        new = d.get("new_state")
        eid = d.get("entity_id")
        if not eid:
            return
        if new is None:
            self.states.pop(eid, None)
            return
        self.states[eid] = new
        if eid not in self.entities:
            return
        t = _ms(new.get("last_updated") or new.get("last_changed") or time.time())
        if kind_of(eid) == "numeric":
            rows = self.means.setdefault(eid, MinuteMean()).add(t, _float(new.get("state")))
            self.store.add_rows(eid, rows)
        else:
            self.store.add_rows(eid, [row_for(eid, t, new.get("state"), new.get("attributes") or {})])

    def tick(self, now_ms: int | None = None) -> None:
        """Close finished minutes of numeric sensors that did not change."""
        now_ms = now_ms or int(time.time() * 1000)
        for eid in self.entities:
            if kind_of(eid) != "numeric":
                continue
            mm = self.means.get(eid)
            if mm is None:
                s = self.states.get(eid)
                if s is None:
                    continue
                mm = self.means[eid] = MinuteMean()
                mm.add(now_ms, _float(s.get("state")))
                continue
            self.store.add_rows(eid, mm.advance(now_ms))

    async def backfill(self, history_days: int) -> None:
        """Fetch what the store lacks, from the last stored row (or history_days ago) to now."""
        now = int(time.time() * 1000)
        oldest = now - history_days * 864e5
        groups: dict[str, list[tuple[str, int]]] = {"numeric": [], "climate": [], "weather": [], "binary": []}
        for eid in self.entities:
            last = self.store.last_time(eid)
            start = int(max(oldest, (last + 1) if last else oldest))
            if now - start > 2 * MIN:
                groups[kind_of(eid)].append((eid, start))
        self.backfill_status = {"running": True, "entities": sum(len(v) for v in groups.values())}
        try:
            if groups["numeric"]:
                await self._numeric(groups["numeric"], now)
            for k in ("climate", "weather"):
                for eid, start in groups[k]:
                    res = await self.ha.history(_iso(start), _iso(now), [eid], attributes=True)
                    self.store.add_rows(eid, [row_for(eid, _ms(s.get("lu") or s.get("lc")), s.get("s"), s.get("a") or {})
                                              for s in res.get(eid, [])])
            for eid, start in groups["binary"]:
                res = await self.ha.history(_iso(start), _iso(now), [eid], attributes=False)
                self.store.add_rows(eid, [[_ms(s.get("lu") or s.get("lc")), s.get("s")] for s in res.get(eid, [])])
        finally:
            self.backfill_status = {"running": False, "done": now}
        log.info("backfill done")

    async def _numeric(self, items: list[tuple[str, int]], now: int) -> None:
        start = min(s for _, s in items)
        ids = [e for e, _ in items]
        stats = await self.ha.statistics(_iso(start), _iso(now), ids, "5minute", ("mean", "state"))
        missing = []
        for eid, s0 in items:
            if is_counter(self.meta, eid):  # a meter: its reading at the end of each five minutes
                rows = [[_ms(r.get("end") or (_ms(r["start"]) + 5 * MIN)), r.get("state")] for r in stats.get(eid, [])
                        if r.get("state") is not None]
            else:
                rows = [[_ms(r["start"]), r.get("mean")] for r in stats.get(eid, []) if r.get("mean") is not None]
            rows = [r for r in rows if r[0] >= s0]
            if rows:
                self.store.add_rows(eid, rows)
                # the minutes since the last full 5-minute statistic come from the history
                s_last = rows[-1][0] + 5 * MIN
                if now - s_last > MIN:
                    missing.append((eid, s_last))
            else:
                missing.append((eid, s0))
        for eid, s0 in missing:
            res = await self.ha.history(_iso(s0), _iso(now), [eid], attributes=False)
            mm = MinuteMean()
            rows = []
            for s in res.get(eid, []):
                rows += mm.add(max(s0, _ms(s.get("lu") or s.get("lc"))), _float(s.get("s")))
            rows += mm.advance(now)
            self.store.add_rows(eid, rows)
            self.means[eid] = mm  # carry on from here with the live events
