"""A simulated apartment standing in for Home Assistant, to try the optimizer.

DEMO=1 python -m climate_optimizer starts the service against it: a week of
made-up history is loaded into the store, then the apartment lives on in
real time. Its proxies ask the optimizer every minute, like the real ones,
and run their own thermostat when told not to follow. The demo keeps its
data in DATA_DIR/demo-data, apart from the real data.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .model import synthetic
from .model.layout import default_layout
from .model.solar import irradiance
from .model.thermal import DIST, build, electric, level_heat

MIN = 60_000
HOUR = 3_600_000


def _iso(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


class World:
    def __init__(self, days: int = 7, seed: int = 11, lat: float = 49.5, lon: float = 8.5, tz: str = "Europe/Berlin"):
        self.layout = default_layout()
        self.tz = ZoneInfo(tz)
        self.lat, self.lon = lat, lon
        now = int(time.time() * 1000) // MIN * MIN
        start = now - days * 24 * HOUR
        self.run = synthetic.run(days=days, seed=seed, start=start, lat=lat, lon=lon, tz=tz, mean_outdoor=6.0)
        self.params = self.run["params"]
        self.export = synthetic.to_export(self.run)
        self.mapping = synthetic.mapping_for(self.layout)
        self.t = start + days * 24 * HOUR
        self.x = self.run["x_end"].copy()
        self.level = self.run["level_end"]
        self.cloud = self.run["cloud_end"]
        self.R = np.random.default_rng(seed + 5)
        sysm = build(self.layout, self.params)
        self.Ad, self.Bd = sysm.discretize(1 / 60)
        self.units = {u.id: {"on": False, "level": 0, "mode": "heat", "since": self.t, "answer": None,
                             "proxy_mode": "auto", "target": synthetic.TARGETS.get(u.zone, (22.0, 22.0))[0]}
                      for u in self.layout.units}
        self.windows = {z: False for z in self.layout.window_zones}
        self.planner = None
        self.handlers = []
        self.last_pel = 0.0
        self.extra = {}  # states the optimizer sets itself
        self.pokes = []  # proxies told to ask now

    # ---- weather and prices ----
    def tout_at(self, t: float) -> float:
        d = datetime.fromtimestamp(t / 1000, self.tz)
        hod = d.hour + d.minute / 60
        return self.level + (2 + 3.5 * (1 - self.cloud)) * math.cos((hod - 15) / 24 * 2 * math.pi)

    def price_at(self, t: float) -> float:
        d = datetime.fromtimestamp(t / 1000, self.tz)
        hod = d.hour + d.minute / 60
        g = lambda mu, s: math.exp(-0.5 * ((hod - mu) / s) ** 2)
        return round(0.27 + 0.03 * math.sin(t / 864e5 * 1.7) + 0.07 * g(8, 1.3) + 0.11 * g(18.7, 1.6) - 0.03 * g(3, 2)
                     - 0.09 * (1 - self.cloud) * g(13, 2.2), 4)

    def price_info(self) -> list[dict]:
        today = datetime.fromtimestamp(self.t / 1000, self.tz).replace(hour=0, minute=0, second=0, microsecond=0)
        start = today - timedelta(days=1)
        hour_local = datetime.fromtimestamp(self.t / 1000, self.tz).hour
        days = 3 if hour_local >= 13 else 2  # tomorrow's prices arrive around 13:00
        out = []
        t = start
        while t < start + timedelta(days=days):
            ms = t.timestamp() * 1000
            out.append({"startsAt": t.isoformat(), "total": self.price_at(ms)})
            t += timedelta(minutes=15)
        return out

    def calendar_events(self) -> list[dict]:
        """A few events in the demo calendar, one of them not meant for the optimizer."""
        day = datetime.fromtimestamp(self.t / 1000, self.tz).replace(hour=0, minute=0, second=0, microsecond=0)
        at = lambda d, h: (day + timedelta(days=d, hours=h)).isoformat()
        return [
            {"start": at(1, 18), "end": at(1, 23), "summary": "Wohnzimmer: Strict"},
            {"start": at(3, 13), "end": at(3, 16), "summary": "Schlafzimmer: aus"},
            {"start": at(5, 0), "end": at(7, 0), "summary": "Alle: Frost protection only"},
            {"start": at(2, 9), "end": at(2, 10), "summary": "Zahnarzt"},
            {"start": at(2, 11), "end": at(2, 15), "summary": "Lüften: aus"},
            {"start": at(4, 7), "end": at(4, 9), "summary": "Büro: 22,5 °C"},
        ]

    def forecast(self) -> list[dict]:
        h0 = self.t // HOUR * HOUR
        return [{"datetime": _iso(h0 + i * HOUR), "temperature": round(self.tout_at(h0 + i * HOUR) + 0.3 * math.sin(i / 5), 1),
                 "cloud_coverage": int(self.cloud * 100), "condition": "cloudy" if self.cloud > 0.7 else "partlycloudy"}
                for i in range(48)]

    # ---- the apartment ----
    def request(self, uid: str) -> dict:
        u = next(x for x in self.layout.units if x.id == uid)
        st = self.units[uid]
        zi = self.layout.zone_index(u.zone)
        body = {"entity_id": f"climate.{uid}_proxy", "name": u.name, "source": self.mapping["units"][uid]["source"],
                "modes": ["heat", "cool"] if u.kind == "ac" else ["heat"], "current_temperature": round(float(self.x[zi]), 1),
                "target_temperature": st["target"], "running": st["on"], "run_mode": st["mode"],
                "last_switch": _iso(st["since"]), "min_cycle_s": 300, "control_interval_s": 60}
        if u.kind == "ac":
            body.update({"source_fan_modes": synthetic.FAN_MODES, "source_fan_mode": synthetic.FAN_MODES[st["level"]],
                         "fan_mode": "normal", "fan_mode_offset": 0, "fan_modes": ["quiet", "normal", "boost"], "idle_action": "turn_off"})
        return body

    def poke(self, entity_ids) -> None:
        """homeassistant.update_entity on proxies: they ask the optimizer now and do what it says."""
        self.pokes.append(list(entity_ids))
        for u in self.layout.units:
            st = self.units[u.id]
            if f"climate.{u.id}_proxy" not in entity_ids or not self.planner or st["proxy_mode"] != "auto":
                continue
            ans = self.planner.command(self.request(u.id), max(self.t, time.time() * 1000))
            st["answer"] = ans
            if ans.get("follow", True) and "hvac_mode" in ans:
                on = ans["hvac_mode"] != "off"
                if on != st["on"]:
                    st["since"] = self.t
                st.update(on=on, level=int(ans.get("fan_level", st["level"])), mode=ans["hvac_mode"] if on else st["mode"])

    def step(self) -> None:
        """One minute: ask the optimizer, run the units, move the temperatures."""
        t = self.t
        d = datetime.fromtimestamp(t / 1000, self.tz)
        hod = d.hour + d.minute / 60
        if d.minute == 0 and self.R.random() < 0.15:
            self.cloud = float(np.clip(self.cloud + (self.R.random() - 0.5) * 0.9, 0, 1))
        if d.hour == 0 and d.minute == 0:
            self.level += self.R.normal() * 1.2 - (self.level - 6.0) * 0.3
        air = self.extra.get("sensor.climate_optimizer_airing")
        a0 = a1 = 0
        if air and air[0] not in ("unknown", "unavailable"):  # the residents follow the suggestion
            a0 = datetime.fromisoformat(air[0]).timestamp() * 1000
            a1 = datetime.fromisoformat(air[1]["end"]).timestamp() * 1000
        for z in self.windows:
            if a0 <= t < a1:
                self.windows[z] = True
            elif a1 <= t < a1 + MIN:
                self.windows[z] = False
            elif self.windows[z]:
                if self.R.random() < 0.12:
                    self.windows[z] = False
            elif 7 <= hod < 22 and self.R.random() < 0.0015:
                self.windows[z] = True
        tout = self.tout_at(t)
        q = np.zeros(len(self.layout.units))
        for j, u in enumerate(self.layout.units):
            st = self.units[u.id]
            zi = self.layout.zone_index(u.zone)
            ans = self.planner.command(self.request(u.id), t) if self.planner and st["proxy_mode"] == "auto" else {"follow": False}
            st["answer"] = ans
            if ans.get("follow", True) and "hvac_mode" in ans:
                on = ans["hvac_mode"] != "off"
                lvl = int(ans.get("fan_level", st["level"]))
                mode = ans["hvac_mode"] if on else st["mode"]
            else:  # the proxy's own thermostat
                demand = st["target"] - round(float(self.x[zi]), 1)
                on = st["on"]
                if not on and demand >= 0.5:
                    on = True
                elif on and demand <= -0.3:
                    on = False
                lvl = int(np.clip(math.floor(max(demand, 0) / 0.5), 0, u.levels - 1))
                mode = "heat"
            if u.zone in self.windows and self.windows[u.zone]:
                on = False
            if on != st["on"]:
                st["since"] = t
            st.update(on=on, level=lvl, mode=mode)
            if u.kind == "ac":
                q[j] = float(level_heat(self.params, u.id, lvl, 1 if mode == "heat" else 2, tout)) if on else 0.0
            else:
                q[j] = self.params.units[u.id]["kw"] if on else 0.0
        sol = irradiance(t + 30_000, self.lat, self.lon, self.cloud)
        dist = np.zeros(len(DIST))
        dist[DIST.index("tout")] = tout
        for f in "DESW":
            dist[DIST.index(f)] = sol[f]
        dist[DIST.index("hc")] = math.cos(2 * math.pi * hod / 24)
        dist[DIST.index("hs")] = math.sin(2 * math.pi * hod / 24)
        dist[DIST.index("one")] = 1.0
        w = np.array([float(self.windows[z]) * (tout - self.x[self.layout.zone_index(z)]) for z in self.layout.window_zones])
        self.x = self.Ad @ self.x + self.Bd @ np.concatenate([q, dist, w])
        ac = [j for j, u in enumerate(self.layout.units) if u.kind == "ac"]
        self.last_pel = float(np.sum(electric(self.params, q[ac], self.x[[self.layout.zone_index(self.layout.units[j].zone) for j in ac]], tout)))
        self.last_q = q
        self.t = t + MIN

    def states(self) -> list[dict]:
        now = _iso(self.t)
        out = []

        def add(eid, state, **attrs):
            out.append({"entity_id": eid, "state": str(state), "attributes": attrs, "last_changed": now, "last_updated": now})

        for i, z in enumerate(self.layout.zones):
            add(f"sensor.{z.id}_temperatur", round(float(self.x[i]) + 0.05 * self.R.normal(), 1),
                friendly_name=f"{synthetic.NAMES.get(z.id, z.name)} Temperatur", unit_of_measurement="°C", device_class="temperature")
        add("sensor.balkon_temperatur", round(self.tout_at(self.t), 1), friendly_name="Balkon Temperatur",
            unit_of_measurement="°C", device_class="temperature")
        for z, o in self.windows.items():
            add(f"binary_sensor.{z}_balkontuer", "on" if o else "off", friendly_name=f"{synthetic.NAMES.get(z, z)} Balkontür", device_class="door")
        for u in self.layout.units:
            st = self.units[u.id]
            zi = self.layout.zone_index(u.zone)
            action = ("heating" if st["mode"] == "heat" else "cooling") if st["on"] else "idle"
            ans = st["answer"] or {}
            add(f"climate.{u.id}_proxy", st["proxy_mode"], friendly_name=f"{synthetic.NAMES.get(u.zone, u.name)} Proxy",
                hvac_modes=["off", "heat", "cool", "auto"] if u.kind == "ac" else ["off", "heat", "auto"],
                temperature=st["target"], current_temperature=round(float(self.x[zi]), 1), hvac_action=action,
                fan_mode="normal" if u.kind == "ac" else None, optimizer="following" if ans.get("follow", True) and "hvac_mode" in ans else "deferred",
                optimizer_reason=ans.get("reason"))
            if u.kind == "ac":
                add(f"climate.{u.id}_klima", st["mode"] if st["on"] else "off", friendly_name=f"{synthetic.NAMES.get(u.zone, u.name)} Klima",
                    temperature=30.0, current_temperature=round(float(self.x[zi]) + 1.0, 1), hvac_action=action if st["on"] else "off",
                    fan_mode=synthetic.FAN_MODES[st["level"]] if st["on"] else synthetic.FAN_MODES[0], fan_modes=synthetic.FAN_MODES,
                    hvac_modes=["off", "heat", "cool", "fan_only"])
            else:
                add(f"switch.{u.id}_heizluefter", "on" if st["on"] else "off", friendly_name="Bad Heizlüfter")
                add(f"sensor.{u.id}_heizluefter_leistung", round(self.params.units[u.id]["kw"] * 1000 if st["on"] else 0.0),
                    friendly_name="Bad Heizlüfter Leistung", unit_of_measurement="W", device_class="power")
        pel = self.last_pel + (0.03 if any(self.units[u.id]["on"] for u in self.layout.ac_units) else 0.0)
        add("sensor.phase_l1_leistung", round(150 + 30 * self.R.normal()), unit_of_measurement="W", device_class="power", friendly_name="Phase L1 Leistung")
        add("sensor.phase_l2_leistung", round((pel + 0.1) * 1000 + 15 * self.R.normal()), unit_of_measurement="W", device_class="power", friendly_name="Phase L2 Leistung")
        add("sensor.phase_l3_leistung", round(120 + 30 * self.R.normal()), unit_of_measurement="W", device_class="power", friendly_name="Phase L3 Leistung")
        add("weather.forecast_home", "cloudy" if self.cloud > 0.7 else "partlycloudy", temperature=round(self.tout_at(self.t), 1),
            cloud_coverage=int(self.cloud * 100), friendly_name="Forecast Home")
        add("sensor.zuhause_strompreis", self.price_at(self.t), unit_of_measurement="EUR/kWh", friendly_name="Zuhause Strompreis")
        add("calendar.klima", "off", friendly_name="Klima")
        for eid, (st_, at_) in self.extra.items():
            add(eid, st_, **at_)
        return out


class FakeHA:
    """The parts of HomeAssistant the service uses, answered by a World."""

    def __init__(self, world: World):
        self.world = world
        self.connected = asyncio.Event()
        self.error = None
        self.version = "demo"
        self.url = "demo"
        self.token = "demo"
        self.verify_ssl = True
        self._ws = None
        self.handlers = []

    @property
    def configured(self) -> bool:
        return True

    async def run(self, on_connect, stop: asyncio.Event) -> None:
        self.connected.set()
        await on_connect()
        while not stop.is_set():
            nxt = (time.time() // 60 + 1) * 60 + 0.5
            try:
                await asyncio.wait_for(stop.wait(), max(0.1, nxt - time.time()))
            except asyncio.TimeoutError:
                pass
            if stop.is_set():
                break
            while self.world.t <= time.time() * 1000 - MIN:
                await asyncio.to_thread(self.world.step)
            for s in self.world.states():
                for h in self.handlers:
                    h({"data": {"entity_id": s["entity_id"], "new_state": s}})

    async def states(self):
        return self.world.states()

    async def ha_config(self):
        return {"time_zone": str(self.world.tz), "latitude": self.world.lat, "longitude": self.world.lon, "version": "demo",
                "location_name": "Demo"}

    async def subscribe(self, event_type, handler):
        self.handlers.append(handler)

    async def service(self, domain, service, data=None, target=None, response=False):
        if (domain, service) == ("tibber_prices", "get_price"):
            return {"price_info": self.world.price_info()}
        if (domain, service) == ("weather", "get_forecasts"):
            return {target["entity_id"]: {"forecast": self.world.forecast()}}
        if (domain, service) == ("calendar", "get_events"):
            return {target["entity_id"]: {"events": self.world.calendar_events()}}
        if (domain, service) == ("homeassistant", "update_entity"):
            self.world.poke(target["entity_id"])
        return {}

    async def history(self, *a, **k):
        return {}

    async def set_state(self, entity_id, state, attributes):
        self.world.extra[entity_id] = (state, attributes)

    async def statistics(self, *a, **k):
        return {}


def demo_service(data: Path):
    """A Service wired to a simulated apartment with a week of history."""
    from .app import Service
    from .config import ConfigStore

    data.mkdir(parents=True, exist_ok=True)
    world = World()
    cs = ConfigStore(data / "config.json")
    if not (data / "config.json").exists():
        cs.update({"mapping": world.mapping, "prices": {"source": "tibber_prices", "sensor": "sensor.zuhause_strompreis"},
                   "shadow": False, "comfort": {"bath": 22.0}, "location": {"lat": world.lat, "lon": world.lon},
                   "calendar": "calendar.klima"})
    svc = Service(data, ha=FakeHA(world))
    world.planner = svc.planner
    # a week of history, as if the collector had recorded it
    ex = world.export
    for eid, rows in ex["history"].items():
        svc.store.add_rows(eid, [[int(datetime.fromisoformat(str(r[0])).timestamp() * 1000) if isinstance(r[0], str) else int(r[0])] + list(r[1:])
                                 for r in rows])
    for eid, rows in ex["stats_5min"].items():
        svc.store.add_rows(eid, [[int(a), b] for a, b in rows])
    svc.collector.meta.update(ex["meta"])
    svc.store.put("meta", svc.collector.meta)
    hist = {}
    t0 = world.t - 8 * 24 * HOUR
    for t in range(int(t0), int(world.t), 15 * MIN):
        hist[str(t)] = [t + 15 * MIN, world.price_at(t)]
    svc.store.put("price_history", hist)
    return svc


def demo_dir() -> Path:
    """Apart from the real data: the demo's made-up history must never be trained on."""
    return Path(os.environ.get("DATA_DIR", ".")) / "demo-data"


def demo_app():
    from .app import create_app

    return create_app(service=demo_service(demo_dir()))
