"""Recorded history laid onto one fixed-step grid.

Input is "rows" per entity, the format the Home Assistant export snippet
writes and the optimizer's own store keeps:

  numeric sensors  stats_5min {id: [[start, mean], ...]} or history {id: [[t, value], ...]}
  climate          history {id: [[t, state, setpoint, current_temperature, hvac_action, fan_mode], ...]}
  weather          history {id: [[t, condition, temperature, cloud_coverage], ...]}
  binary sensors   history {id: [[t, "on"|"off"], ...]}

The grid keeps, for every AC unit and step, the share of the step it spent
in each mode and fan level, so the heat it delivered follows from whatever
heat-per-level the model assumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from .layout import Layout
from .solar import irradiance

MIN = 60_000


def _ts(v) -> float:
    if isinstance(v, (int, float)):
        return float(v) if v > 1e11 else float(v) * 1000.0
    s = str(v).replace("Z", "+00:00")
    d = datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp() * 1000.0


def _num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) else float("nan")


@dataclass
class Parsed:
    meta: dict
    num: dict = field(default_factory=dict)  # id -> (t ndarray, v ndarray)
    kind: dict = field(default_factory=dict)  # id -> "mean5" | "state"
    climate: dict = field(default_factory=dict)  # id -> list of dict rows
    binary: dict = field(default_factory=dict)  # id -> (t, on)
    weather: dict = field(default_factory=dict)  # id -> list of dict rows
    location: dict | None = None


def parse_rows(export: dict) -> Parsed:
    if "meta" not in export:
        raise ValueError("not an export: no meta")
    p = Parsed(meta=export["meta"], location=export.get("location"))
    for eid, rows in (export.get("stats_5min") or {}).items():
        pts = [(_ts(t), _num(v)) for t, v in rows if v is not None]
        pts = [x for x in pts if np.isfinite(x[1])]
        if len(pts) > 1:
            pts.sort()
            p.num[eid] = (np.array([x[0] for x in pts]), np.array([x[1] for x in pts]))
            p.kind[eid] = "mean5"
    for eid, rows in (export.get("history") or {}).items():
        dom = eid.split(".")[0]
        if dom == "climate":
            out = []
            for r in rows:
                r = list(r) + [None] * 6
                out.append({"t": _ts(r[0]), "state": r[1], "set": _num(r[2]), "cur": _num(r[3]), "action": r[4], "fan": r[5]})
            out.sort(key=lambda x: x["t"])
            p.climate[eid] = out
        elif dom == "weather":
            out = []
            for r in rows:
                r = list(r) + [None] * 4
                cloud = _num(r[3])
                out.append({"t": _ts(r[0]), "temp": _num(r[2]), "cloud": cloud / 100.0 if np.isfinite(cloud) else float("nan")})
            out.sort(key=lambda x: x["t"])
            p.weather[eid] = out
        elif dom in ("binary_sensor", "switch", "input_boolean"):
            pts = sorted((_ts(t), 1.0 if s == "on" else 0.0 if s == "off" else float("nan")) for t, s in rows)
            if pts:
                p.binary[eid] = (np.array([x[0] for x in pts]), np.array([x[1] for x in pts]))
        elif eid not in p.num:
            pts = sorted((_ts(r[0]), _num(r[1])) for r in rows)
            if sum(np.isfinite(v) for _, v in pts) > 1:
                p.num[eid] = (np.array([x[0] for x in pts]), np.array([x[1] for x in pts]))
                p.kind[eid] = "state"
    return p


# ---- step-function helpers ---------------------------------------------------
def step_means(t, v, t0, step, K, max_age=6 * 3600e3):
    """Time-weighted mean of a step series over each grid interval."""
    out = np.full(K, np.nan)
    if len(t) == 0:
        return out
    # sample the step function on a fine sub-grid: exact enough at 30 s
    sub = 10
    fine = t0 + (np.arange(K * sub) + 0.5) * (step / sub)
    idx = np.searchsorted(t, fine, side="right") - 1
    ok = idx >= 0
    vals = np.full(fine.shape, np.nan)
    vals[ok] = v[idx[ok]]
    age = np.where(ok, fine - t[np.clip(idx, 0, None)], np.inf)
    vals[age > max_age] = np.nan
    vals = vals.reshape(K, sub)
    cnt = np.isfinite(vals).sum(1)
    with np.errstate(invalid="ignore"):
        m = np.nansum(vals, 1) / np.maximum(cnt, 1)
    out[cnt > sub / 2] = m[cnt > sub / 2]
    return out


def resample(parsed: Parsed, eid: str, t0, step, K, max_age=6 * 3600e3):
    if not eid or eid not in parsed.num:
        return np.full(K, np.nan)
    t, v = parsed.num[eid]
    if parsed.kind.get(eid) == "mean5" and abs(step - 5 * MIN) < 1:
        out = np.full(K, np.nan)
        k = np.round((t - t0) / step).astype(int)
        ok = (k >= 0) & (k < K) & (np.abs(t - (t0 + k * step)) < MIN)
        out[k[ok]] = v[ok]
        return out
    return step_means(t, v, t0, step, K, max_age)


def fill_gaps(a, max_steps):
    a = np.array(a, dtype=float)
    ok = np.where(np.isfinite(a))[0]
    for i, j in zip(ok[:-1], ok[1:]):
        if 1 < j - i <= max_steps + 1:
            a[i + 1:j] = np.interp(np.arange(i + 1, j), [i, j], [a[i], a[j]])
    return a


def _unit_w(meta: dict, eid: str) -> float:
    u = str((meta.get(eid) or {}).get("unit") or "").lower()
    return 1.0 if u == "kw" else 1e-3


@dataclass
class Grid:
    t: np.ndarray  # step starts, ms
    step_min: int
    T: np.ndarray  # (K, nz) measured air temperatures
    tout: np.ndarray
    open: np.ndarray  # (K, nw) share of the step with a window open
    sol: np.ndarray  # (K, 4) facade irradiance, kW/m²
    hc: np.ndarray
    hs: np.ndarray
    ac_frac: dict  # unit -> (K, 2 modes, levels) share of the step heating / cooling at each level
    heater_kw: dict  # unit -> (K,) measured or switched heater power
    phase: np.ndarray  # (K,) kW on the outdoor unit's phase, NaN if unknown
    others: list  # other phases, kW
    car: np.ndarray | None
    valid: np.ndarray  # (K,) all thermometers and the outdoor temperature present
    price: list  # [(t_ms, price)] as recorded
    tz: str = "UTC"

    @property
    def K(self) -> int:
        return len(self.t)

    @property
    def dt_h(self) -> float:
        return self.step_min / 60.0

    def unit_running(self, uid: str) -> np.ndarray:
        if uid in self.ac_frac:
            return self.ac_frac[uid].sum(axis=(1, 2))
        return (self.heater_kw[uid] > 0.05).astype(float)


def build_grid(parsed: Parsed, layout: Layout, mapping: dict, *, step_min=5, start=None, end=None,
               min_hours: float = 24) -> Grid:
    """mapping: {"zones": {zid: {"thermometer": eid, "windows": [eid]}},
                 "units": {uid: {"source": eid, "proxy": eid, "fan_modes": [...], "power": eid}},
                 "outdoor": eid, "weather": eid, "phase": eid, "phases_other": [...], "car": eid, "price": eid}"""
    step = step_min * MIN
    zones = mapping.get("zones", {})
    series = [parsed.num[zones[z]["thermometer"]][0] for z in layout.zone_ids
              if zones.get(z, {}).get("thermometer") in parsed.num]
    if not series:
        raise ValueError("no room thermometer has data")
    a = max(s[0] for s in series)
    b = min(s[-1] for s in series)
    if start:
        a = max(a, start)
    if end:
        b = min(b, end)
    t0 = np.ceil(a / step) * step
    K = int(max(0, np.floor((b - t0) / step)))
    if K < max(1, int(min_hours * 60 / step_min)):
        raise ValueError(f"less than {min_hours:g} h of overlapping room temperatures")
    t = t0 + step * np.arange(K)
    nz = len(layout.zone_ids)
    T = np.column_stack([fill_gaps(resample(parsed, zones.get(z, {}).get("thermometer", ""), t0, step, K), 6)
                         for z in layout.zone_ids]) if nz else np.zeros((K, 0))
    tout = resample(parsed, mapping.get("outdoor", ""), t0, step, K)
    wx = parsed.weather.get(mapping.get("weather", "")) or []
    cloud = np.full(K, np.nan)
    if wx:
        wt = np.array([r["t"] for r in wx])
        wtemp = step_means(wt, np.array([r["temp"] for r in wx]), t0, step, K, 3 * 3600e3)
        tout = np.where(np.isfinite(tout), tout, wtemp)
        cloud = step_means(wt, np.array([r["cloud"] for r in wx]), t0, step, K, 3 * 3600e3)
    tout = fill_gaps(tout, 36)
    wz = layout.window_zones
    opn = np.zeros((K, len(wz)))
    for j, z in enumerate(wz):
        for eid in zones.get(z, {}).get("windows", []) or []:
            if eid in parsed.binary:
                bt, bv = parsed.binary[eid]
                f = step_means(bt, bv, t0, step, K, 1e18)
                opn[:, j] = np.maximum(opn[:, j], np.nan_to_num(f))
    loc = parsed.location or {"lat": 51.0, "lon": 10.0}
    tzname = (parsed.location or {}).get("tz") or "UTC"
    sol = np.zeros((K, 4))
    for k in range(K):
        r = irradiance(t[k] + step / 2, loc["lat"], loc["lon"], cloud[k] if np.isfinite(cloud[k]) else None)
        sol[k] = (r["D"], r["E"], r["S"], r["W"])
    tz = ZoneInfo(tzname)
    hours = np.array([(lambda d: d.hour + d.minute / 60)(datetime.fromtimestamp((x + step / 2) / 1000, tz)) for x in t])
    hc, hs = np.cos(2 * np.pi * hours / 24), np.sin(2 * np.pi * hours / 24)

    units = mapping.get("units", {})
    ac_frac, heater_kw = {}, {}
    for u in layout.units:
        cfg = units.get(u.id, {})
        if u.kind == "ac":
            ac_frac[u.id] = _ac_fractions(parsed, cfg, u.levels, t0, step, K, T[:, layout.zone_index(u.zone)])
        else:
            pw = cfg.get("power")
            if pw and pw in parsed.num:
                heater_kw[u.id] = np.nan_to_num(resample(parsed, pw, t0, step, K, 1e18) * _unit_w(parsed.meta, pw))
            elif cfg.get("source") in parsed.binary:
                bt, bv = parsed.binary[cfg["source"]]
                heater_kw[u.id] = np.nan_to_num(step_means(bt, bv, t0, step, K, 1e18)) * float(cfg.get("rated_kw", 2.0))
            else:
                heater_kw[u.id] = np.zeros(K)

    def kw(eid):
        return resample(parsed, eid, t0, step, K, 1e18) * _unit_w(parsed.meta, eid) if eid else np.full(K, np.nan)

    phase = kw(mapping.get("phase", ""))
    others = [kw(e) for e in mapping.get("phases_other", []) if e and e != mapping.get("phase")]
    car = kw(mapping["car"]) if mapping.get("car") else None
    price = []
    if mapping.get("price") and mapping["price"] in parsed.num:
        pt, pv = parsed.num[mapping["price"]]
        price = [(float(a_), float(b_)) for a_, b_ in zip(pt, pv) if np.isfinite(b_)]
    valid = np.isfinite(T).all(axis=1) & np.isfinite(tout)
    return Grid(t=t, step_min=step_min, T=T, tout=tout, open=opn, sol=sol, hc=hc, hs=hs, ac_frac=ac_frac,
                heater_kw=heater_kw, phase=phase, others=others, car=car, valid=valid, price=price, tz=tzname)


def _ac_fractions(parsed: Parsed, cfg: dict, levels: int, t0, step, K, room) -> np.ndarray:
    """Share of each step heating/cooling at each fan level, from the real unit's history."""
    out = np.zeros((K, 2, levels))
    rows = parsed.climate.get(cfg.get("source", ""), [])
    if not rows:
        return out
    modes = list(cfg.get("fan_modes") or [])
    rt = np.array([r["t"] for r in rows])
    sub = 10
    fine = t0 + (np.arange(K * sub) + 0.5) * (step / sub)
    idx = np.searchsorted(rt, fine, side="right") - 1
    for n, i in enumerate(idx):
        if i < 0:
            continue
        r = rows[i]
        st = r["state"]
        if st not in ("heat", "cool"):
            continue
        k = n // sub
        sp = r["set"]
        # a unit held in heat at its lowest setpoint (idle_action: hold_setpoint) is idle
        if np.isfinite(sp) and np.isfinite(room[k]):
            if st == "heat" and sp < room[k] - 1.0:
                continue
            if st == "cool" and sp > room[k] + 1.0:
                continue
        fan = r.get("fan")
        lvl = modes.index(fan) if fan in modes else (levels - 1 if not modes else 0)
        lvl = min(max(lvl, 0), levels - 1)
        out[k, 0 if st == "heat" else 1, lvl] += 1.0 / sub
    return out
