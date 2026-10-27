"""What the plan needs from outside: weather forecast, prices, calendar events.

Prices come either from the tibber_prices integration's get_price action or
from a sensor listing the coming prices in an attribute (Nord Pool, EPEX
Spot, Tibber and others each name things differently; the parser looks for
any list of entries with a start time and a price).
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from .ha import HomeAssistant

log = logging.getLogger(__name__)

START_KEYS = ("startsAt", "start", "start_time", "starts_at", "from", "time", "datetime")
END_KEYS = ("endsAt", "end", "end_time", "ends_at", "till", "to")
PRICE_KEYS = ("total", "price", "value", "price_per_kwh", "marketprice", "price_eur_kwh")
# cloud cover when a forecast has none, by condition
CONDITION_CLOUD = {"sunny": 0.05, "clear-night": 0.05, "partlycloudy": 0.5, "cloudy": 0.95, "fog": 1.0, "rainy": 0.9,
                   "pouring": 1.0, "snowy": 0.95, "snowy-rainy": 0.95, "lightning": 0.9, "lightning-rainy": 0.95,
                   "hail": 0.95, "windy": 0.4, "windy-variant": 0.6, "exceptional": 0.7}


def _ms(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) * (1000.0 if v < 1e11 else 1.0)
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp() * 1000.0


def _price_lists(obj, depth=0):
    """Every list of dicts in obj that looks like price entries."""
    if depth > 5:
        return
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj[:3]):
            first = obj[0]
            if any(k in first for k in START_KEYS) and any(k in first for k in PRICE_KEYS):
                yield obj
                return
        for x in obj:
            yield from _price_lists(x, depth + 1)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _price_lists(v, depth + 1)


def parse_prices(obj, unit_hint: str = "") -> list[tuple[int, int, float]]:
    """[(start ms, end ms, €/kWh)] from whatever shape the source returns."""
    pts = {}
    ends = {}
    for lst in _price_lists(obj):
        for e in lst:
            s = next((_ms(e[k]) for k in START_KEYS if k in e), None)
            p = next((e[k] for k in PRICE_KEYS if k in e and e[k] is not None), None)
            if s is None or p is None:
                continue
            try:
                p = float(p)
            except (TypeError, ValueError):
                continue
            pts[int(s)] = p
            en = next((_ms(e[k]) for k in END_KEYS if k in e), None)
            if en:
                ends[int(s)] = int(en)
    if not pts:
        return []
    starts = sorted(pts)
    gaps = np.diff(starts)
    typical = int(np.median(gaps)) if len(gaps) else 3_600_000
    out = []
    for i, s in enumerate(starts):
        e = ends.get(s) or (starts[i + 1] if i + 1 < len(starts) and starts[i + 1] - s <= 2 * typical else s + typical)
        out.append((s, int(e), pts[s]))
    vals = np.array([p for _, _, p in out])
    cents = "ct" in unit_hint.lower() or "cent" in unit_hint.lower() or "/mwh" in unit_hint.lower()
    if "/mwh" in unit_hint.lower():
        vals = vals / 1000.0
    elif cents or np.median(np.abs(vals)) > 3.0:  # nobody pays 3 €/kWh: these are cents
        vals = vals / 100.0
    return [(s, e, float(v)) for (s, e, _), v in zip(out, vals)]


async def fetch_prices(ha: HomeAssistant, cfg, states: dict, now_ms: float) -> list[tuple[int, int, float]]:
    if cfg.source == "tibber_prices":
        start = datetime.fromtimestamp(now_ms / 1000, timezone.utc) - timedelta(days=1)
        data = {"start_time": start.isoformat(), "end_time": (start + timedelta(days=3)).isoformat()}
        if cfg.entry_id:
            data["entry_id"] = cfg.entry_id
        res = await ha.service("tibber_prices", "get_price", data, response=True)
        return parse_prices(res)
    s = states.get(cfg.sensor)
    if not s:
        raise ValueError(f"price sensor {cfg.sensor or '(none)'} not found")
    a = s.get("attributes") or {}
    return parse_prices(a, str(a.get("unit_of_measurement", "")))


def parse_forecast(items: list[dict]) -> dict:
    """{"t": ms, "temp": °C, "cloud": 0..1} arrays from weather.get_forecasts."""
    t, temp, cloud = [], [], []
    for f in items or []:
        ts = _ms(f.get("datetime"))
        tp = f.get("temperature")
        if ts is None or tp is None:
            continue
        cc = f.get("cloud_coverage")
        c = float(cc) / 100.0 if cc is not None else CONDITION_CLOUD.get(f.get("condition"), 0.6)
        t.append(ts)
        temp.append(float(tp))
        cloud.append(min(1.0, max(0.0, c)))
    order = np.argsort(t)
    return {"t": np.array(t)[order], "temp": np.array(temp)[order], "cloud": np.array(cloud)[order]}


async def fetch_forecast(ha: HomeAssistant, weather: str) -> dict:
    res = await ha.service("weather", "get_forecasts", {"type": "hourly"}, target={"entity_id": weather}, response=True)
    items = ((res or {}).get(weather) or {}).get("forecast") or []
    return parse_forecast(items)


def _event_dt(v, tz: ZoneInfo) -> datetime | None:
    if isinstance(v, dict):
        v = v.get("dateTime") or v.get("date")
    if v is None:
        return None
    s = str(v)
    if len(s) == 10:
        return datetime.combine(date.fromisoformat(s), datetime.min.time(), tz)
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return (d if d.tzinfo else d.replace(tzinfo=tz)).astimezone(tz)


async def fetch_events(ha: HomeAssistant, calendar: str, start: datetime, end: datetime, tz: ZoneInfo) -> list[dict]:
    res = await ha.service("calendar", "get_events", {"start_date_time": start.isoformat(), "end_date_time": end.isoformat()},
                           target={"entity_id": calendar}, response=True)
    out = []
    for e in ((res or {}).get(calendar) or {}).get("events") or []:
        a, b = _event_dt(e.get("start"), tz), _event_dt(e.get("end"), tz)
        if a and b and e.get("summary"):
            out.append({"start": a, "end": b, "summary": e["summary"]})
    return out


def blend_outdoor(fc_t: np.ndarray, fc_temp: np.ndarray, now_ms: float, measured: float | None,
                  times: np.ndarray, tau_h: float = 6.0) -> np.ndarray:
    """Forecast temperature at times, shifted toward the measured value now;
    the shift fades over a few hours (a balcony sensor is rarely the forecast's grid cell)."""
    if len(fc_t) == 0:
        base = np.full(len(times), measured if measured is not None and math.isfinite(measured) else 5.0)
        return base
    base = np.interp(times, fc_t, fc_temp)
    if measured is None or not math.isfinite(measured):
        return base
    off = measured - float(np.interp(now_ms, fc_t, fc_temp))
    off = float(np.clip(off, -6.0, 6.0))
    return base + off * np.exp(-np.maximum(0.0, times - now_ms) / (tau_h * 3.6e6))
