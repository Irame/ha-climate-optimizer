"""What being off the comfort temperature costs, and when which rule applies.

A comfort function has two sides, too cold and too warm. Each side is a sum
of convex pieces: a deadband that costs nothing, a linear price per kelvin
and hour, a quadratic one, and a limit beyond which the room must not go.
Convex pieces keep the optimisation exact: the solver finds the best plan,
not just a good one.

Rules say which function applies when, or that a unit must not run, or that
the comfort temperature moves. They recur by an iCalendar RRULE (so "every
other week, Monday to Thursday, 18:00 to 23:00" is
FREQ=WEEKLY;INTERVAL=2;BYDAY=MO,TU,WE,TH) and the highest priority wins.
Events in a Home Assistant calendar override the rules for single occasions.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from dateutil.rrule import rrulestr

HARD = 50.0  # €/(K·h) beyond a limit: in practice never paid


@dataclass
class Side:
    deadband: float = 0.0  # K that cost nothing
    linear: float = 0.0  # € per K and hour
    quadratic: float = 0.0  # € per K² and hour
    limit: float | None = None  # K: not allowed further than this

    def cost(self, dev: float) -> float:
        """Exact cost rate (€/h) of being dev kelvin off on this side."""
        x = max(0.0, dev - self.deadband)
        c = self.linear * x + self.quadratic * x * x
        if self.limit is not None and dev > self.limit:
            c += HARD * (dev - self.limit)
        return c

    def segments(self) -> list[tuple[float | None, float]]:
        """cost() as convex pieces: [(length K, slope €/(K·h)), ...] with rising
        slopes, the last one unbounded (length None). The quadratic part is
        sampled at fixed steps past the deadband."""
        d = self.deadband
        pts = {0.0, d}
        if self.quadratic:
            pts.update(d + s for s in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))
        if self.limit is not None:
            pts = {p for p in pts if p < self.limit} | {self.limit}
        pts = sorted(pts)
        base = lambda x: self.linear * max(0.0, x - d) + self.quadratic * max(0.0, x - d) ** 2
        segs: list[tuple[float | None, float]] = [(b - a, (base(b) - base(a)) / (b - a)) for a, b in zip(pts, pts[1:])]
        last = pts[-1]
        tail = self.linear + self.quadratic * (2 * max(0.0, last - d) + 1.0) if last >= d else 0.0
        if self.limit is not None:
            tail += HARD
        segs.append((None, tail))
        for i in range(1, len(segs)):  # rounding must not break convexity
            if segs[i][1] < segs[i - 1][1]:
                segs[i] = (segs[i][0], segs[i - 1][1])
        return segs


@dataclass
class ComfortFunction:
    id: str
    name: str
    below: Side = field(default_factory=Side)  # too cold
    above: Side = field(default_factory=Side)  # too warm

    @staticmethod
    def from_dict(d: dict) -> ComfortFunction:
        return ComfortFunction(id=d["id"], name=d.get("name", d["id"]), below=Side(**d.get("below", {})), above=Side(**d.get("above", {})))

    def to_dict(self) -> dict:
        return asdict(self)


def default_functions() -> list[ComfortFunction]:
    return [
        ComfortFunction("standard", "Standard", below=Side(deadband=0.3, linear=0.10, quadratic=0.10),
                        above=Side(deadband=1.0, linear=0.05, quadratic=0.10, limit=3.0)),
        ComfortFunction("relaxed", "Relaxed", below=Side(deadband=1.5, linear=0.03, quadratic=0.03, limit=4.0),
                        above=Side(deadband=1.5, linear=0.05, quadratic=0.05, limit=3.0)),
        ComfortFunction("strict", "Strict", below=Side(deadband=0.2, linear=0.3, quadratic=0.3, limit=1.5),
                        above=Side(deadband=0.5, linear=0.1, quadratic=0.1, limit=2.0)),
        ComfortFunction("frost", "Frost protection only", below=Side(deadband=5.0, linear=0.05, limit=8.0),
                        above=Side(deadband=3.0, linear=0.02)),
    ]


@dataclass
class Rule:
    id: str
    name: str
    action: str  # "comfort" | "lockout" | "offset" | "no_airing"
    zones: list[str] = field(default_factory=list)  # zone ids; empty = all (for comfort / offset)
    units: list[str] = field(default_factory=list)  # unit ids (for lockout); empty = all
    function: str | None = None  # comfort function id (action comfort)
    offset: float = 0.0  # K added to the comfort temperature (action offset)
    rrule: str = "FREQ=DAILY"
    dtstart: str = "2026-01-05"  # first day the recurrence counts from (matters for INTERVAL>1)
    start: str = "00:00"
    end: str = "24:00"  # may be before start: runs past midnight
    priority: int = 10
    enabled: bool = True

    @staticmethod
    def from_dict(d: dict) -> Rule:
        r = Rule(**{k: v for k, v in d.items() if k in Rule.__dataclass_fields__})
        r.validate()
        return r

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.action not in ("comfort", "lockout", "offset", "no_airing"):
            raise ValueError(f"rule {self.name}: unknown action {self.action}")
        if self.action == "comfort" and not self.function:
            raise ValueError(f"rule {self.name}: which comfort function?")
        _hm(self.start)
        _hm(self.end)
        date.fromisoformat(self.dtstart)
        self._rr(ZoneInfo("UTC"))

    def _rr(self, tz):
        h, m = _hm(self.start)
        d0 = date.fromisoformat(self.dtstart)
        start = datetime.combine(d0, time(h % 24, m), tz)
        body = self.rrule.strip()
        if body.upper().startswith("RRULE:"):
            body = body[6:]
        return rrulestr("RRULE:" + (body or "FREQ=DAILY"), dtstart=start)

    def intervals(self, t0: datetime, t1: datetime) -> list[tuple[datetime, datetime]]:
        """Occurrences of this rule overlapping [t0, t1), as aware datetimes."""
        sh, sm = _hm(self.start)
        eh, em = _hm(self.end)
        dur = timedelta(hours=eh - sh, minutes=em - sm)
        if dur <= timedelta(0):
            dur += timedelta(days=1)
        rr = self._rr(t0.tzinfo)
        out = []
        for s in rr.between(t0 - dur - timedelta(days=1), t1, inc=True):
            e = s + dur
            if e > t0 and s < t1:
                out.append((s, e))
        return out


def _hm(s: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s.strip())
    if not m or int(m[1]) > 24 or int(m[2]) > 59:
        raise ValueError(f"time {s!r} is not HH:MM")
    return int(m[1]), int(m[2])


def night_no_airing() -> Rule:
    return Rule("night-no-airing", "No airing at night", "no_airing", start="22:00", end="07:00", priority=10)


def default_rules() -> list[Rule]:
    return [night_no_airing(),
        Rule("living-night", "Living room relaxed at night", "comfort", zones=["living"], function="relaxed",
             start="23:00", end="06:30", priority=20),
        Rule("bedroom-day", "Bedroom relaxed during the day", "comfort", zones=["bedroom"], function="relaxed",
             start="08:00", end="21:00", priority=20),
    ]


# ---- calendar ----------------------------------------------------------------------
OFF_WORDS = {"aus", "off", "sperre", "lock", "lockout", "gesperrt"}
ALL_WORDS = {"alle", "all", "wohnung", "apartment"}
AIRING_WORDS = {"lüften", "lueften", "luften", "airing", "air"}


# Names a calendar may use for the usual rooms and functions, besides their ids and names
ZONE_ALIASES = {"office": ("Büro", "Arbeitszimmer"), "bedroom": ("Schlafzimmer",),
                "living": ("Wohnzimmer", "Küche", "Wohnküche"), "bath": ("Bad", "Badezimmer"), "hall": ("Flur", "Diele")}
FUNCTION_ALIASES = {"standard": ("Normal",), "relaxed": ("Entspannt", "Locker"), "strict": ("Streng",),
                    "frost": ("Frostschutz",)}


def parse_event(summary: str, zones: dict[str, str], units: dict[str, tuple[str, str]],
                functions: dict[str, str], zone_aliases: dict[str, list[str]] | None = None) -> dict | None:
    """A calendar event's title as a rule: 'Wohnzimmer: Relaxed' picks a comfort
    function, 'Schlafzimmer: aus' locks the units there, 'Alle: +1' moves the
    comfort temperature. Names match case-insensitively on id, display name or
    alias. zones: id -> name; units: id -> (name, zone id); functions: id -> name;
    zone_aliases: id -> more names (the usual German ones are always known)."""
    if ":" not in summary:
        return None
    who, what = (" ".join(x.split()).lower() for x in summary.split(":", 1))
    znorm = {}
    for k, v in zones.items():
        for name in (k, v, *ZONE_ALIASES.get(k, ()), *((zone_aliases or {}).get(k) or ())):
            znorm.setdefault(str(name).strip().lower(), k)
    unorm = {**{k.lower(): k for k in units}, **{v[0].lower(): k for k, v in units.items()}}
    fnorm = {}
    for k, v in functions.items():
        for name in (k, v, *FUNCTION_ALIASES.get(k, ())):
            fnorm.setdefault(name.lower(), k)
    if who in AIRING_WORDS:  # 'Lüften: aus': no airing suggested then
        return {"action": "no_airing"} if what in OFF_WORDS | {"nein", "no", "nicht"} else None
    everyone = who in ALL_WORDS
    zone = [] if everyone else [znorm[who]] if who in znorm else None
    if what in OFF_WORDS:
        if everyone:
            return {"action": "lockout", "units": []}
        if who in unorm:
            return {"action": "lockout", "units": [unorm[who]]}
        if zone:
            hit = [u for u, (_, z) in units.items() if z == zone[0]]
            return {"action": "lockout", "units": hit} if hit else None
        return None
    if zone is None:
        return None
    m = re.fullmatch(r"([+-]\d+(?:[.,]\d+)?)\s*(k|°|°c)?", what)
    if m:
        return {"action": "offset", "zones": zone, "offset": float(m[1].replace(",", "."))}
    if what in fnorm:
        return {"action": "comfort", "zones": zone, "function": fnorm[what]}
    return None


# ---- resolving a horizon -------------------------------------------------------------
@dataclass
class Resolved:
    """Per step: which function each zone uses, its offset, and which units are locked."""
    function: dict[str, list[str]]
    offset: dict[str, list[float]]
    locked: dict[str, list[bool]]
    source: dict[str, list[str]]  # what decided the function, for the UI
    event_zone: dict[str, list[bool]] = field(default_factory=dict)  # a calendar event set the zone's function or offset
    event_unit: dict[str, list[bool]] = field(default_factory=dict)  # a calendar event keeps the unit off
    no_airing: list[bool] = field(default_factory=list)  # no airing to be suggested


def resolve(step_starts: list[datetime], step_ends: list[datetime], zones: list[str], units: list[str],
            defaults: dict[str, str], rules: list[Rule], events: list[dict] | None = None) -> Resolved:
    """events: [{"start": aware dt, "end": aware dt, **parse_event(...)}], beating every rule."""
    N = len(step_starts)
    t0, t1 = step_starts[0], step_ends[-1]
    active = []  # (priority, order, spec, intervals)
    for i, r in enumerate(rules):
        if r.enabled:
            active.append((r.priority, i, r.to_dict(), r.intervals(t0, t1)))
    for j, ev in enumerate(events or []):
        active.append((10_000, j, dict(ev, name=ev.get("name", "calendar"), calendar=True), [(ev["start"], ev["end"])]))
    active.sort(key=lambda a: (a[0], a[1]))  # later in the list wins on equal priority
    fn = {z: [defaults.get(z, "standard")] * N for z in zones}
    src = {z: ["default"] * N for z in zones}
    off = {z: [0.0] * N for z in zones}
    lock = {u: [False] * N for u in units}
    ev_z = {z: [False] * N for z in zones}
    ev_u = {u: [False] * N for u in units}
    no_air = [False] * N
    for _, _, spec, ivs in active:
        if not ivs:
            continue
        for k in range(N):
            mid = step_starts[k] + (step_ends[k] - step_starts[k]) / 2
            if not any(s <= mid < e for s, e in ivs):
                continue
            if spec["action"] == "comfort":
                for z in (spec.get("zones") or zones):
                    if z in fn:
                        fn[z][k] = spec["function"]
                        src[z][k] = spec.get("name", "")
                        ev_z[z][k] = bool(spec.get("calendar"))
            elif spec["action"] == "offset":
                for z in (spec.get("zones") or zones):
                    if z in off:
                        off[z][k] = spec.get("offset", 0.0)
                        ev_z[z][k] = ev_z[z][k] or bool(spec.get("calendar"))
            elif spec["action"] == "lockout":
                for u in (spec.get("units") or units):
                    if u in lock:
                        lock[u][k] = True
                        ev_u[u][k] = ev_u[u][k] or bool(spec.get("calendar"))
            elif spec["action"] == "no_airing":
                no_air[k] = True
    return Resolved(function=fn, offset=off, locked=lock, source=src, event_zone=ev_z, event_unit=ev_u, no_airing=no_air)
