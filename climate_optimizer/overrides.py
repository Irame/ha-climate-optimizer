"""Overrides: a room brought to a temperature, or a unit run for a while, and then
back to the plan.

A room override ("target") gives the room a temperature from a time on (now, or
"warm by 18:00"), with a comfort function steep enough that the plan reaches it
whatever electricity costs; the plan only chooses when to heat towards it. It
ends once reached (or a while after, or at a set time). A unit override ("run")
keeps a unit running for some minutes whatever the temperature, like a heater
switched on by a button for a quarter of an hour.

They live in the store, so a restart of the optimizer or of Home Assistant
does not lose one half way.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

import numpy as np

from .comfort import zone_lookup
from .model.layout import Layout

MIN = 60_000
FAR = 10 * 365 * 864e5  # "until reached": as far as the plan reaches
REACHED_K = 0.2  # within this of the target counts as reached
GIVE_UP_H = 12.0  # a target not reached in this long is dropped


@dataclass
class Override:
    id: str
    kind: str  # "target" | "run"
    created: int
    zone: str | None = None  # target
    target: float | None = None
    direction: str = "up"  # heat up to it, or cool down to it
    ready_at: int | None = None  # when it should be reached; None: as soon as it can
    hold_min: float = 0.0  # kept this long after it is reached (and ready_at has come)
    until: int | None = None  # or kept until then
    reached: int | None = None
    unit: str | None = None  # run
    end: int | None = None
    source: str = "ui"

    @property
    def start(self) -> int:
        return self.ready_at or self.created

    def ends_at(self) -> int | None:
        """When it ends, as far as known yet."""
        if self.kind == "run":
            return self.end
        if self.until is not None:
            return self.until
        if self.reached is not None:
            return int(max(self.start, self.reached) + self.hold_min * MIN)
        return None

    def to_dict(self) -> dict:
        return {**asdict(self), "ends_at": self.ends_at()}


def parse_time(v, now_ms: float, tz) -> int | None:
    """A point in time from the UI or Home Assistant: ms, an ISO timestamp, or
    "HH:MM" (today, or tomorrow when that has passed). Empty: none."""
    if v is None or v == "" or v == 0:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip()
    if s.isdigit():
        return int(s)
    now = datetime.fromtimestamp(now_ms / 1000, tz)
    if len(s) <= 5 and ":" in s:
        h, m = (int(x) for x in s.split(":"))
        d = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if d.timestamp() * 1000 <= now_ms:
            d += timedelta(days=1)
        return int(d.timestamp() * 1000)
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=tz)
    return int(d.timestamp() * 1000)


def find_zone(layout: Layout, mapping: dict, name: str) -> str | None:
    """A room by id, name, German name, alias, or an entity in it (its thermometer, a unit's proxy)."""
    key = str(name or "").strip().lower()
    hit = zone_lookup({z.id: z.name for z in layout.zones}, {z.id: list(z.aliases) for z in layout.zones}).get(key)
    if hit:
        return hit
    for zid, zm in (mapping.get("zones") or {}).items():
        if key == str(zm.get("thermometer") or "").lower():
            return zid
    u = find_unit(layout, mapping, name)
    return next((x.zone for x in layout.units if x.id == u), None)


def find_unit(layout: Layout, mapping: dict, name: str) -> str | None:
    """A unit by id, name, or one of its entities (proxy, the device under it, its plug)."""
    key = str(name or "").strip().lower()
    for u in layout.units:
        m = (mapping.get("units") or {}).get(u.id) or {}
        names = {u.id, u.name, m.get("proxy"), m.get("source"), m.get("power"), m.get("switch")}
        if key in {str(x).lower() for x in names if x}:
            return u.id
    return None


class Overrides:
    def __init__(self, store):
        self.store = store
        self.items: list[Override] = []
        for d in store.get("overrides") or []:
            try:
                self.items.append(Override(**{k: v for k, v in d.items() if k in Override.__dataclass_fields__}))
            except TypeError:
                pass

    def save(self) -> None:
        self.store.put("overrides", [asdict(o) for o in self.items])

    def list(self) -> list[dict]:
        return [o.to_dict() for o in self.items]

    def add_target(self, zone: str, target: float, now_ms: float, temp_now: float | None, ready_at: int | None = None,
                   hold_min: float = 0.0, until: int | None = None, source: str = "ui") -> Override:
        """One per room: a new one replaces the one before."""
        if ready_at is not None and ready_at <= now_ms:
            ready_at = None
        direction = "down" if temp_now is not None and np.isfinite(temp_now) and temp_now > target + REACHED_K else "up"
        o = Override(id=uuid.uuid4().hex[:8], kind="target", created=int(now_ms), zone=zone, target=float(target),
                     direction=direction, ready_at=ready_at, hold_min=float(hold_min or 0.0), until=until, source=source)
        self.items = [x for x in self.items if not (x.kind == "target" and x.zone == zone)] + [o]
        self.save()
        return o

    def add_run(self, unit: str, minutes: float, now_ms: float, source: str = "ui") -> Override:
        o = Override(id=uuid.uuid4().hex[:8], kind="run", created=int(now_ms), unit=unit,
                     end=int(now_ms + minutes * MIN), source=source)
        self.items = [x for x in self.items if not (x.kind == "run" and x.unit == unit)] + [o]
        self.save()
        return o

    def cancel(self, id: str | None = None, zone: str | None = None, unit: str | None = None) -> list[Override]:
        gone = [o for o in self.items if (id and o.id == id) or (zone and o.kind == "target" and o.zone == zone)
                or (unit and o.kind == "run" and o.unit == unit)]
        if gone:
            self.items = [o for o in self.items if o not in gone]
            self.save()
        return gone

    def run_for(self, unit: str, now_ms: float) -> Override | None:
        # a minute's slack at the start: a proxy's clock, or the demo's, may run a little behind
        return next((o for o in self.items if o.kind == "run" and o.unit == unit and o.created - MIN <= now_ms < (o.end or 0)), None)

    def tick(self, now_ms: float, temps: dict[str, float]) -> list[tuple[Override, str]]:
        """Note what was reached, and drop what is over; returns those, with why."""
        done, changed = [], False
        for o in list(self.items):
            if o.kind == "target":
                t = temps.get(o.zone)
                if o.reached is None and now_ms >= o.start and t is not None and np.isfinite(t):
                    if (o.direction == "up" and t >= o.target - REACHED_K) or (o.direction == "down" and t <= o.target + REACHED_K):
                        o.reached, changed = int(now_ms), True
                end = o.ends_at()
                if end is not None and now_ms >= end:
                    done.append((o, "held as long as asked" if o.until or o.hold_min else "reached"))
                elif o.reached is None and now_ms - o.start > GIVE_UP_H * 3.6e6:
                    done.append((o, f"not reached in {GIVE_UP_H:g} h"))
            elif now_ms >= (o.end or 0):
                done.append((o, "ran as long as asked"))
        if done:
            self.items = [o for o in self.items if o not in [d[0] for d in done]]
            changed = True
        if changed:
            self.save()
        return done

    def specs(self, now_ms: float, layout: Layout, tz) -> list[dict]:
        """The room overrides as rules for resolve(): above the calendar, and the
        room's units free of lockouts while they hold."""
        out = []
        for o in self.items:
            if o.kind != "target":
                continue
            end = o.ends_at()
            out.append({"action": "target", "zones": [o.zone], "target": o.target, "function": f"override-{o.direction}",
                        "units": [u.id for u in layout.units if u.zone == o.zone],
                        "start": datetime.fromtimestamp(max(o.start, now_ms - MIN) / 1000, tz),
                        "end": datetime.fromtimestamp((end if end is not None else now_ms + FAR) / 1000, tz),
                        "name": f"Override {o.target:g} °C", "priority": 20_000, "calendar": False})
        return out

    def forced(self, unit: str, t: np.ndarray) -> np.ndarray | None:
        """Per plan step, the share of it a run keeps the unit on."""
        runs = [o for o in self.items if o.kind == "run" and o.unit == unit]
        if not runs:
            return None
        a, b = t[:-1].astype(float), t[1:].astype(float)
        f = np.zeros(len(a))
        for o in runs:
            f = np.maximum(f, np.clip((np.minimum(b, o.end) - np.maximum(a, o.created)) / (b - a), 0.0, 1.0))
        return f if f.any() else None

    def next_change(self, now_ms: float) -> float | None:
        """The next moment an override ends by the clock (a run's end, an until)."""
        ts = [e for o in self.items if (e := o.ends_at()) is not None and e > now_ms]
        return min(ts) if ts else None


def now_ms() -> float:
    return time.time() * 1000
