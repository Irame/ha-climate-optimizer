"""Made-up apartment data in the format the Home Assistant export writes.

A known model is driven the way the apartment is run today: each AC unit
behind a proxy thermostat (deadband, fan level from demand, extreme setpoint
on the real unit), the bath heater behind a switch proxy, windows aired in
the morning, cooking, people, other loads and car charging on the phases.
It is then measured like Home Assistant would (quantised thermometers,
5-minute means, state changes only). Used by the tests and the demo mode.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from .layout import Layout, Params, default_layout, default_params
from .solar import irradiance
from .thermal import DIST, build, electric, level_heat

MIN = 60_000
HOUR = 3_600_000
FAN_MODES = ["quiet", "low", "mid", "high"]
TARGETS = {  # proxy targets (day, night) per zone
    "office": (21.0, 17.0), "bedroom": (19.5, 18.0), "living": (21.5, 18.0),
}
NAMES = {"office": "Büro", "bedroom": "Schlafzimmer", "living": "Wohnzimmer", "bath": "Bad", "hall": "Flur"}


def mapping_for(layout: Layout) -> dict:
    """The entity mapping that matches the synthetic export."""
    zones = {z.id: {"thermometer": f"sensor.{z.id}_temperatur", "windows": [f"binary_sensor.{z.id}_balkontuer"] if z.windows else []}
             for z in layout.zones}
    units = {}
    for u in layout.units:
        if u.kind == "ac":
            units[u.id] = {"source": f"climate.{u.id}_klima", "proxy": f"climate.{u.id}_proxy", "fan_modes": FAN_MODES}
        else:
            units[u.id] = {"source": f"switch.{u.id}_heizluefter", "proxy": f"climate.{u.id}_proxy", "power": f"sensor.{u.id}_heizluefter_leistung"}
    return {"zones": zones, "units": units, "outdoor": "sensor.balkon_temperatur", "weather": "weather.forecast_home",
            "phase": "sensor.phase_l2_leistung", "phases_other": ["sensor.phase_l1_leistung", "sensor.phase_l3_leistung"],
            "price": "sensor.zuhause_strompreis"}


def run(days=14, seed=1, start=None, layout: Layout | None = None, params: Params | None = None,
        lat=49.5, lon=8.5, tz="Europe/Berlin", mean_outdoor=7.0, realism=True):
    layout = layout or default_layout()
    params = params or default_params(layout)
    R = np.random.default_rng(seed)
    zone = ZoneInfo(tz)
    if start is None:
        start = int(datetime(2026, 10, 6, tzinfo=zone).timestamp() * 1000)
    K = days * 24 * 60
    sysm = build(layout, params)
    Ad, Bd = sysm.discretize(1 / 60)
    us, ds, ws = sysm.cols()
    nz, n = sysm.nz, sysm.n
    zids = layout.zone_ids

    # weather: hourly cloud as a Markov chain, the outdoor temperature following it
    H = days * 24 + 48
    cloud = np.empty(H)
    level = np.empty(days + 3)
    c = 0.6
    for h in range(H):
        if R.random() < 0.15:
            c = float(np.clip(c + (R.random() - 0.5) * 0.9, 0, 1))
        cloud[h] = c
    lv = mean_outdoor
    for d in range(days + 3):
        lv += R.normal() * 1.5 - (lv - mean_outdoor) * 0.3
        level[d] = lv

    def local(t):
        return datetime.fromtimestamp(t / 1000, zone)

    def tout_at(t):
        h = min(H - 1, max(0, int((t - start) // HOUR)))
        d = local(t)
        hod = d.hour + d.minute / 60
        x = (t - start) / HOUR / 24 - 0.5
        d0 = int(np.floor(x))
        f = x - d0
        base = level[max(0, d0)] + (level[max(0, d0 + 1)] - level[max(0, d0)]) * f
        return base + (2 + 3.5 * (1 - cloud[h])) * np.cos((hod - 15) / 24 * 2 * np.pi)

    # prices: two peaks, a night dip and a solar dip that depends on the clouds
    price = []
    for t in range(start - 24 * HOUR, start + (days + 2) * 24 * HOUR, 15 * MIN):
        d = local(t)
        hod = d.hour + d.minute / 60
        day = (t - start) // (24 * HOUR)
        g = lambda mu, s: np.exp(-0.5 * ((hod - mu) / s) ** 2)
        cl = cloud[int(min(H - 1, max(0, day * 24 + 13)))]
        p = 0.27 + 0.03 * np.sin(day * 1.7) + 0.07 * g(8, 1.3) + 0.11 * g(18.7, 1.6) - 0.03 * g(3, 2) - 0.09 * (1 - cl) * g(13, 2.2)
        price.append((t, round(float(p + 0.004 * R.normal()), 4)))

    plan = []
    for d in range(days):
        wd = local(start + d * 24 * HOUR).weekday()
        plan.append({
            "air": {"office": 7.5 + R.random() * 0.5 if R.random() < 0.8 else None,
                    "bedroom": 7 + R.random() * 0.5 if R.random() < 0.9 else None,
                    "living": 12 + R.random() if R.random() < 0.5 else None},
            "air_len": {"office": 8 + R.random() * 6, "bedroom": 10 + R.random() * 10, "living": 8 + R.random() * 10},
            "cook": 18.2 + R.random() * 0.8 if R.random() < 0.8 else None,
            "car": 19 + R.random() * 2 if R.random() < 0.3 else None, "car_len": 2 + R.random() * 2,
            "work": wd < 5,
        })

    x = np.array([19.5 if z == "bedroom" else 21.0 if z == "bath" else 20.5 for z in zids] * 3)
    ac = layout.ac_units
    active = {u.id: False for u in layout.units}
    base_level = {u.id: None for u in ac}
    rec = {k: [] for k in ("T", "tout", "open", "heater", "mode", "level", "Q", "pel", "cloud", "L1", "L2", "L3", "target", "car")}
    for k in range(K):
        t = start + k * MIN
        d = local(t)
        hod = d.hour + d.minute / 60
        dp = plan[(t - start) // (24 * HOUR)]
        night = hod < 6 or hod >= 22
        opn = [1.0 if dp["air"].get(z) is not None and dp["air"][z] <= hod < dp["air"][z] + dp["air_len"][z] / 60 else 0.0
               for z in layout.window_zones]
        tout = tout_at(t)
        h = min(H - 1, int((t - start) // HOUR))
        sol = irradiance(t + 30_000, lat, lon, cloud[h])
        u = np.zeros(sysm.nu)
        modes, levels, targets = [], [], []
        for j, unit in enumerate(layout.units):
            zi = layout.zone_index(unit.zone)
            room = round(x[zi] * 10) / 10
            if unit.kind == "ac":
                tgt = TARGETS.get(unit.zone, (21.0, 17.0))[1 if night else 0]
                demand = tgt - room
                if not active[unit.id] and demand >= 0.5:
                    active[unit.id] = True
                elif active[unit.id] and demand <= -0.5:
                    active[unit.id] = False
                if opn and unit.zone in layout.window_zones and opn[layout.window_zones.index(unit.zone)]:
                    active[unit.id] = False
                lvl = 0
                if active[unit.id]:
                    up = int(np.floor(max(demand, 0) / 0.5 + 1e-6))
                    down = int(np.floor(max(demand + 0.2, 0) / 0.5 + 1e-6))
                    b = base_level[unit.id]
                    b = up if b is None or up > b else (down if down < b else b)
                    lvl = base_level[unit.id] = int(np.clip(b, 0, unit.levels - 1))
                else:
                    base_level[unit.id] = None
                mode = 1 if active[unit.id] else 0
                q = float(level_heat(params, unit.id, lvl, mode, tout))
                # cycling at low load and a little noise in what the unit really delivers
                if realism and mode and lvl == 0 and (k // 7) % 3 == 0:
                    q *= 0.6
                u[j] = q
                modes.append(mode)
                levels.append(lvl)
                targets.append(tgt)
            else:
                tgt = 22.5 if (6 <= hod < 7.5 or 19 <= hod < 21.5) else 16.0
                demand = tgt - room
                if not active[unit.id] and demand >= 0.3:
                    active[unit.id] = True
                elif active[unit.id] and demand <= -0.3:
                    active[unit.id] = False
                u[j] = params.units[unit.id]["kw"] if active[unit.id] else 0.0
                modes.append(1 if active[unit.id] else 0)
                levels.append(0)
                targets.append(tgt)
        dist = np.zeros(len(DIST))
        dist[DIST.index("tout")] = tout
        for f in "DESW":
            dist[DIST.index(f)] = sol[f]
        dist[DIST.index("hc")] = np.cos(2 * np.pi * hod / 24)
        dist[DIST.index("hs")] = np.sin(2 * np.pi * hod / 24)
        dist[DIST.index("one")] = 1.0
        w = np.array([opn[i] * (tout - x[layout.zone_index(z)]) for i, z in enumerate(layout.window_zones)])
        v = np.concatenate([u, dist, w])
        x = Ad @ x + Bd @ v
        cooking = (dp["cook"] is not None and dp["cook"] <= hod < dp["cook"] + 0.6)
        if cooking and "living" in zids:
            x[layout.zone_index("living")] += 0.6 / 60 / params.zones["living"]["Ca"]
        if realism:
            people = {"bedroom": 0.12 if hod < 7 or hod >= 23 else 0.0,
                      "office": 0.1 if dp["work"] and 8.5 <= hod < 17.5 else 0.0,
                      "living": 0.15 if 17 <= hod < 23 else 0.03}
            for z, g in people.items():
                if z in zids:
                    x[layout.zone_index(z)] += g / 60 / params.zones[z]["Ca"]
        ac_q = [u[j] for j, unit in enumerate(layout.units) if unit.kind == "ac"]
        ac_t = [x[layout.zone_index(unit.zone)] for unit in ac]
        pel = float(np.sum(electric(params, np.array(ac_q), np.array(ac_t), tout))) + (params.plant["standby"] if any(m for m, un in zip(modes, layout.units) if un.kind == "ac") else 0.0)
        car = 11.0 if dp["car"] is not None and dp["car"] <= hod < dp["car"] + dp["car_len"] else 0.0
        office = 0.05 + (0.18 if dp["work"] and 8.5 <= hod < 17.5 else 0.0) + 0.02 * R.normal()
        stove = 1.6 if cooking else 0.0
        rec["T"].append(x[:nz].copy())
        rec["tout"].append(tout)
        rec["open"].append(opn)
        rec["heater"].append([u[j] for j, un in enumerate(layout.units) if un.kind == "heater"])
        rec["mode"].append(modes)
        rec["level"].append(levels)
        rec["target"].append(targets)
        rec["Q"].append(u.copy())
        rec["pel"].append(pel)
        rec["cloud"].append(cloud[h])
        rec["car"].append(car)
        rec["L1"].append(0.15 + 0.03 * R.normal() + stove * 0.5 + car / 3)
        rec["L2"].append(pel + office + 0.04 + 0.015 * R.normal() + stove * 0.5 + car / 3)
        rec["L3"].append(0.12 + 0.03 * R.normal() + car / 3)
    return {"rec": {k: np.array(v) for k, v in rec.items()}, "price": price, "params": params, "layout": layout,
            "start": start, "lat": lat, "lon": lon, "tz": tz, "seed": seed, "x_end": x.copy(),
            "level_end": float(level[min(days, len(level) - 1)]), "cloud_end": float(cloud[min(H - 1, days * 24)])}


def to_export(r: dict) -> dict:
    """Measure a run the way Home Assistant records it."""
    rec, layout, start = r["rec"], r["layout"], r["start"]
    R = np.random.default_rng(r["seed"] + 1000)
    K = len(rec["T"])
    ts = start + MIN * np.arange(K)
    iso = lambda t: datetime.fromtimestamp(t / 1000, timezone.utc).isoformat()
    meta, s5, hist = {}, {}, {}

    def mean5(eid, vals):
        m = vals[: (len(vals) // 5) * 5].reshape(-1, 5).mean(1)
        s5[eid] = [[int(ts[i * 5]), round(float(v), 3)] for i, v in enumerate(m)]

    q = lambda v, s: np.round(np.asarray(v) / s) * s
    for i, z in enumerate(layout.zones):
        eid = f"sensor.{z.id}_temperatur"
        meta[eid] = {"name": f"{NAMES.get(z.id, z.name)} Temperatur", "device_class": "temperature", "unit": "°C", "area": NAMES.get(z.id, z.name), "device": "dev_t_" + z.id}
        mean5(eid, q(rec["T"][:, i] + 0.05 * R.normal() + 0.03 * R.normal(size=K), 0.1))
    meta["sensor.balkon_temperatur"] = {"name": "Balkon Temperatur", "device_class": "temperature", "unit": "°C", "area": None, "device": "dev_out"}
    mean5("sensor.balkon_temperatur", q(rec["tout"] + 0.1 * R.normal(size=K), 0.1))
    for key in ("L1", "L2", "L3"):
        eid = f"sensor.phase_{key.lower()}_leistung"
        meta[eid] = {"name": f"Phase {key} Leistung", "device_class": "power", "unit": "W", "area": None, "device": "dev_meter"}
        mean5(eid, np.maximum(0, rec[key]) * 1000)
    ac_i = 0
    for j, u in enumerate(layout.units):
        zi = layout.zone_index(u.zone)
        proxy = f"climate.{u.id}_proxy"
        meta[proxy] = {"name": f"{NAMES.get(u.zone, u.name)} Proxy", "area": NAMES.get(u.zone, u.name), "device": None,
                       "hvac_modes": ["off", "heat", "cool"]}
        prow, last = [], None
        for k in range(K):
            running = rec["mode"][k][j] > 0
            row = ("heat", float(rec["target"][k][j]), round(float(rec["T"][k][zi]), 1), "heating" if running else "idle")
            if row != last:
                prow.append([iso(ts[k] + 1000), *row, "medium" if u.kind == "ac" else None])
                last = row
        hist[proxy] = prow
        if u.kind == "ac":
            src = f"climate.{u.id}_klima"
            meta[src] = {"name": f"{NAMES.get(u.zone, u.name)} Klima", "area": NAMES.get(u.zone, u.name), "device": "dev_ac_" + u.id,
                         "hvac_modes": ["off", "heat", "cool", "fan_only"], "fan_modes": FAN_MODES, "min_temp": 16, "max_temp": 30}
            srow, last = [], None
            for k in range(K):
                running = rec["mode"][k][j] > 0
                fan = FAN_MODES[int(rec["level"][k][j])] if running else FAN_MODES[0]
                row = ("heat" if running else "off", 30.0, float(q(rec["T"][k][zi] + 1.0, 0.5)), "heating" if running else "off", fan)
                if row != last:
                    srow.append([iso(ts[k] + 2000), *row])
                    last = row
            hist[src] = srow
            ac_i += 1
        else:
            sw = f"switch.{u.id}_heizluefter"
            meta[sw] = {"name": "Bad Heizlüfter", "area": NAMES.get(u.zone, u.name), "device": "dev_plug"}
            pw = f"sensor.{u.id}_heizluefter_leistung"
            meta[pw] = {"name": "Bad Heizlüfter Leistung", "device_class": "power", "unit": "W", "area": NAMES.get(u.zone, u.name), "device": "dev_plug"}
            on = rec["Q"][:, j] > 0
            mean5(pw, rec["Q"][:, j] * 1000 + on * 5 * R.normal(size=K))
            srow, last = [], None
            for k in range(K):
                s = "on" if on[k] else "off"
                if s != last:
                    srow.append([iso(ts[k]), s])
                    last = s
            hist[sw] = srow
    for wi, z in enumerate(layout.window_zones):
        eid = f"binary_sensor.{z}_balkontuer"
        meta[eid] = {"name": f"{NAMES.get(z, z)} Balkontür", "device_class": "door", "area": NAMES.get(z, z), "device": "dev_w_" + z}
        rows, last = [], None
        for k in range(K):
            s = "on" if rec["open"][k][wi] else "off"
            if s != last:
                rows.append([iso(ts[k]), s])
                last = s
        hist[eid] = rows
    meta["weather.forecast_home"] = {"name": "Forecast Home"}
    hist["weather.forecast_home"] = [[iso(ts[k]), "cloudy" if rec["cloud"][k] > 0.7 else "partlycloudy",
                                      round(float(rec["tout"][k]), 1), int(round(rec["cloud"][k] * 100))] for k in range(0, K, 60)]
    meta["sensor.zuhause_strompreis"] = {"name": "Zuhause Strompreis", "device_class": "monetary", "unit": "EUR/kWh", "device": "dev_tibber"}
    hist["sensor.zuhause_strompreis"] = [[iso(t), v] for t, v in r["price"]]
    return {"exported": iso(ts[-1]), "ha_version": "synthetic", "location": {"lat": r["lat"], "lon": r["lon"], "tz": r["tz"]},
            "meta": meta, "stats_5min": s5, "stats_hourly": {}, "history": hist}


def true_unit_energy(r: dict, t0: float, t1: float) -> float:
    rec = r["rec"]
    ts = r["start"] + MIN * np.arange(len(rec["pel"]))
    sel = (ts >= t0) & (ts < t1)
    return float(rec["pel"][sel].sum() / 60.0)


__all__ = ["run", "to_export", "mapping_for", "true_unit_energy", "FAN_MODES", "timedelta"]
