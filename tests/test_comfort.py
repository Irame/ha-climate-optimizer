from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from climate_optimizer.comfort import ComfortFunction, Rule, Side, default_functions, parse_event, resolve

TZ = ZoneInfo("Europe/Berlin")


def _pwl(side: Side, dev: float) -> float:
    """Cost of a deviation through the segments, filling the cheap ones first."""
    c, left = 0.0, dev
    for length, slope in side.segments():
        take = left if length is None else min(left, length)
        c += take * slope
        left -= take
        if left <= 0:
            break
    return c


@pytest.mark.parametrize("side", [
    Side(),
    Side(linear=0.1),
    Side(deadband=0.5, linear=0.1),
    Side(deadband=0.3, linear=0.1, quadratic=0.2),
    Side(deadband=1.0, quadratic=0.1, limit=3.0),
    Side(limit=2.0),
])
def test_segments_are_convex_and_match_the_cost(side):
    segs = side.segments()
    slopes = [s for _, s in segs]
    assert slopes == sorted(slopes)
    assert segs[-1][0] is None
    for dev in (0.0, 0.2, 0.5, 1.0, 1.7, 2.5, 4.0):
        exact = side.cost(dev)
        approx = _pwl(side, dev)
        # the quadratic is sampled with chords: never below, close above
        assert approx >= exact - 1e-9
        assert approx <= exact + 0.03 + 0.02 * exact


def test_limit_makes_further_deviation_expensive():
    s = Side(deadband=1.0, linear=0.05, limit=2.0)
    assert s.cost(1.9) < 0.1
    assert s.cost(2.5) > 20


def test_rule_every_other_week_monday_to_thursday_evening():
    r = Rule("x", "Evenings", "comfort", zones=["living"], function="relaxed",
             rrule="FREQ=WEEKLY;INTERVAL=2;BYDAY=MO,TU,WE,TH", dtstart="2026-01-05", start="18:00", end="23:00")
    r.validate()
    t0 = datetime(2026, 1, 5, tzinfo=TZ)
    iv = r.intervals(t0, t0 + timedelta(days=14))
    days = sorted({s.date().isoformat() for s, _ in iv})
    assert days == ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    s, e = iv[0]
    assert (s.hour, e.hour) == (18, 23)
    # the week after the next is on again
    iv2 = r.intervals(t0 + timedelta(days=14), t0 + timedelta(days=21))
    assert len(iv2) == 4


def test_rule_over_midnight():
    r = Rule("n", "Night", "comfort", zones=["living"], function="relaxed", start="23:00", end="06:30")
    t0 = datetime(2026, 3, 2, 12, tzinfo=TZ)
    iv = r.intervals(t0, t0 + timedelta(hours=24))
    assert len(iv) >= 1
    s, e = [x for x in iv if x[0] > t0][0]
    assert (s.hour, s.minute, e.hour, e.minute) == (23, 0, 6, 30)
    assert e.date() > s.date()


def test_rule_validation():
    with pytest.raises(ValueError):
        Rule.from_dict({"id": "a", "name": "a", "action": "comfort"})
    with pytest.raises(ValueError):
        Rule.from_dict({"id": "a", "name": "a", "action": "lockout", "start": "25:00"})
    with pytest.raises(ValueError):
        Rule.from_dict({"id": "a", "name": "a", "action": "lockout", "rrule": "FREQ=SOMETIMES"})


def _steps(t0, n, minutes=30):
    s = [t0 + timedelta(minutes=minutes * k) for k in range(n)]
    return s, [x + timedelta(minutes=minutes) for x in s]


def test_resolve_priorities_and_calendar():
    t0 = datetime(2026, 1, 12, 0, tzinfo=TZ)  # a Monday
    starts, ends = _steps(t0, 48)
    rules = [
        Rule("a", "night relaxed", "comfort", zones=["living"], function="relaxed", start="22:00", end="07:00", priority=10),
        Rule("b", "strict sometimes", "comfort", zones=["living"], function="strict", start="06:00", end="08:00", priority=20),
        Rule("c", "no AC at night", "lockout", units=["bedroom"], start="00:00", end="06:00"),
        Rule("d", "warmer", "offset", zones=[], offset=1.0, start="18:00", end="20:00"),
        Rule("e", "off rule", "lockout", units=["office"], enabled=False),
    ]
    events = [{"start": t0 + timedelta(hours=12), "end": t0 + timedelta(hours=13), "action": "comfort",
               "zones": ["living"], "function": "frost", "name": "away"}]
    r = resolve(starts, ends, ["living", "bedroom"], ["living", "bedroom", "office"], {"bedroom": "relaxed"}, rules, events)
    fn = r.function["living"]
    assert fn[2] == "relaxed"  # 01:00
    assert fn[13] == "strict"  # 06:30: higher priority wins in the overlap
    assert fn[16] == "standard"  # 08:00
    assert fn[24] == "frost"  # 12:00 calendar
    assert r.source["living"][24] == "away"
    assert r.function["bedroom"][20] == "relaxed"  # its default
    assert all(r.locked["bedroom"][k] for k in range(12)) and not r.locked["bedroom"][12]
    assert not any(r.locked["office"])
    assert r.offset["bedroom"][37] == 1.0 and r.offset["living"][36] == 1.0 and r.offset["living"][40] == 0.0


def test_parse_event():
    zones = {"living": "Wohnzimmer", "bedroom": "Schlafzimmer"}
    units = {"living": ("Wohnzimmer Klima", "living"), "bedroom": ("Schlafzimmer Klima", "bedroom")}
    fns = {f.id: f.name for f in default_functions()}
    assert parse_event("Wohnzimmer: Relaxed", zones, units, fns) == {"action": "comfort", "zones": ["living"], "function": "relaxed"}
    assert parse_event("schlafzimmer: aus", zones, units, fns) == {"action": "lockout", "units": ["bedroom"]}
    assert parse_event("Alle: +1,5", zones, units, fns) == {"action": "offset", "zones": [], "offset": 1.5}
    assert parse_event("Alle: off", zones, units, fns) == {"action": "lockout", "units": []}
    assert parse_event("Zahnarzt", zones, units, fns) is None
    assert parse_event("Keller: aus", zones, units, fns) is None


def test_function_roundtrip():
    f = default_functions()[0]
    assert ComfortFunction.from_dict(f.to_dict()) == f
