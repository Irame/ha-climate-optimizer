"""Gathers the inputs, plans every five minutes and answers the proxies."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from . import mpc
from .comfort import parse_event, resolve
from .config import Config
from .dispatch import FanDispatcher
from .inputs import blend_outdoor, fetch_events, fetch_forecast, fetch_prices
from .model.dataset import build_grid, parse_rows
from .model.layout import Layout, Params, default_params
from .model.solar import irradiance
from .model.thermal import DIST, build, estimate_hidden

log = logging.getLogger(__name__)
MIN = 60_000
OFF_STATES = ("off", "unavailable", "unknown")


def _num(v) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else math.nan
    except (TypeError, ValueError):
        return math.nan


def matching_params(params: Params | None, layout: Layout) -> Params:
    """The trained parameters if they describe this layout, else the defaults
    with whatever trained values still fit."""
    base = default_params(layout)
    if params is None:
        return base
    for z in base.zones:
        if z in params.zones:
            base.zones[z].update(params.zones[z])
    for k in base.links:
        if k in params.links:
            base.links[k] = params.links[k]
    for u, d in base.units.items():
        src = params.units.get(u)
        if src and (("heat" in d and len(src.get("heat", [])) == len(d["heat"])) or ("kw" in d and "kw" in src)):
            base.units[u] = dict(src)
    base.plant.update(params.plant)
    base.building.update(params.building)
    return base


@dataclass
class ProxyRequest:
    t: float
    body: dict


@dataclass
class PlanState:
    plan: mpc.Plan | None = None
    created: float = 0.0
    error: str | None = None
    info: dict = field(default_factory=dict)


class Planner:
    def __init__(self, get_config, store, collector, ha):
        self.get_config = get_config
        self.store = store
        self.collector = collector
        self.ha = ha
        self.state = PlanState()
        self.dispatcher = FanDispatcher()
        self.requests: dict[str, ProxyRequest] = {}
        self.lock = asyncio.Lock()
        p = store.get("params")
        self.params: Params | None = Params.from_dict(p) if p else None
        self.report: dict | None = store.get("report")

    # ---- context ----
    @property
    def cfg(self) -> Config:
        return self.get_config()

    def tz(self) -> ZoneInfo:
        name = (self.store.get("ha_config") or {}).get("time_zone") or "UTC"
        try:
            return ZoneInfo(name)
        except Exception:
            return ZoneInfo("UTC")

    def location(self) -> dict:
        if self.cfg.location:
            return self.cfg.location
        hc = self.store.get("ha_config") or {}
        return {"lat": hc.get("latitude", 51.0), "lon": hc.get("longitude", 10.0)}

    def model(self) -> tuple[Layout, Params]:
        layout = self.cfg.layout_obj()
        return layout, matching_params(self.params, layout)

    def unit_for_proxy(self, eid: str) -> str | None:
        for uid, m in (self.cfg.mapping.get("units") or {}).items():
            if m.get("proxy") == eid:
                return uid
        return None

    # ---- inputs ----
    async def fetch(self, now_ms: float) -> dict:
        cfg = self.cfg
        states = self.collector.states
        out: dict = {"errors": []}
        try:
            out["prices"] = await fetch_prices(self.ha, cfg.prices, states, now_ms)
        except Exception as e:
            out["errors"].append(f"prices: {e}")
            out["prices"] = self.store.get("last_prices") or []
        else:
            self.store.put("last_prices", out["prices"])
            hist = {int(k): v for k, v in (self.store.get("price_history") or {}).items()}
            for s_, e_, p_ in out["prices"]:
                if s_ <= now_ms + 3.6e6:
                    hist[int(s_)] = [int(e_), p_]
            cut = now_ms - self.cfg.history_days * 864e5
            self.store.put("price_history", {str(k): v for k, v in sorted(hist.items()) if k >= cut})
        weather = cfg.mapping.get("weather")
        out["forecast"] = {"t": np.array([]), "temp": np.array([]), "cloud": np.array([])}
        if weather:
            try:
                out["forecast"] = await fetch_forecast(self.ha, weather)
            except Exception as e:
                out["errors"].append(f"forecast: {e}")
        out["events"] = []
        if cfg.calendar:
            tz = self.tz()
            now = datetime.fromtimestamp(now_ms / 1000, tz)
            try:
                out["events"] = await fetch_events(self.ha, cfg.calendar, now - timedelta(hours=1), now + timedelta(hours=40), tz)
            except Exception as e:
                out["errors"].append(f"calendar: {e}")
        return out

    def hidden_state(self, layout: Layout, params: Params, now_ms: float) -> tuple[np.ndarray, np.ndarray, str]:
        """Surfaces and structure from the last days of air temperatures."""
        from .collector import tracked_entities
        cfg = self.cfg
        ents = tracked_entities(cfg.mapping)
        ex = self.store.export(ents, int(now_ms - 4 * 864e5), int(now_ms), self.collector.meta,
                               {**self.location(), "tz": str(self.tz())})
        try:
            grid = build_grid(parse_rows(ex), layout, cfg.mapping, min_hours=6)
        except ValueError as e:
            return None, None, str(e)
        sysm = build(layout, params)
        Tm, Tw = estimate_hidden(sysm, grid.T, grid.tout, grid.sol, grid.dt_h)
        return Tm[-1], Tw[-1], f"from {grid.K} steps"

    def live(self, eid: str | None) -> float:
        s = self.collector.states.get(eid or "")
        return _num(s.get("state")) if s else math.nan

    def build_problem(self, now_ms: float, inputs: dict) -> tuple[mpc.Problem, dict]:
        cfg = self.cfg
        layout, params = self.model()
        sysm = build(layout, params)
        st = cfg.settings_obj()
        tz = self.tz()
        loc = self.location()
        mapping = cfg.mapping
        zmap = mapping.get("zones") or {}
        umap = mapping.get("units") or {}
        info: dict = {"notes": list(inputs.get("errors", []))}

        prices = inputs.get("prices") or []
        if not prices:
            raise ValueError("no prices")
        price_end = max(e for _, e, _ in prices)
        t = mpc.step_grid(int(now_ms), price_end, st)
        N = len(t) - 1
        mid = (t[:-1] + t[1:]) / 2

        # disturbances, averaged over each step
        fc = inputs["forecast"]
        tout_now = self.live(mapping.get("outdoor"))
        dist = np.zeros((N, len(DIST)))
        cloud_t = fc["t"] if len(fc["t"]) else np.array([now_ms])
        cloud_v = fc["cloud"] if len(fc["t"]) else np.array([0.6])
        for k in range(N):
            n = max(1, int(round((t[k + 1] - t[k]) / (10 * MIN))))
            ts = t[k] + (np.arange(n) + 0.5) * (t[k + 1] - t[k]) / n
            to = blend_outdoor(fc["t"], fc["temp"], now_ms, tout_now, ts)
            cl = np.interp(ts, cloud_t, cloud_v)
            sol = np.array([[(r := irradiance(x, loc["lat"], loc["lon"], c))["D"], r["E"], r["S"], r["W"]]
                            for x, c in zip(ts, cl)]).mean(axis=0)
            hrs = np.array([(lambda d: d.hour + d.minute / 60)(datetime.fromtimestamp(x / 1000, tz)) for x in ts])
            dist[k] = [to.mean(), *sol, np.cos(2 * np.pi * hrs / 24).mean(), np.sin(2 * np.pi * hrs / 24).mean(), 1.0]
        if not math.isfinite(tout_now) and not len(fc["t"]):
            info["notes"].append("no outdoor temperature: assuming 5 °C")
        price = mpc.price_steps(t, prices)

        # state now
        zids = layout.zone_ids
        Ta = np.array([self.live(zmap.get(z, {}).get("thermometer")) for z in zids])
        Tm, Tw, how = self.hidden_state(layout, params, now_ms)
        if Tm is None:
            info["notes"].append(f"state estimate: {how}; assuming surfaces at room temperature")
        known = np.isfinite(Ta)
        if not known.any():
            raise ValueError("no room temperature available")
        Ta = np.where(known, Ta, np.nanmean(Ta))
        if Tm is None:
            Tm = Ta.copy()
            Tw = Ta - 1.5
        x0 = np.concatenate([Ta, Tm, Tw])

        # windows: open now -> open for a while, and the units there stay off
        win = np.zeros((N, sysm.nw))
        starts_min = (t[:-1] - now_ms) / MIN
        open_zone = set()
        for j, z in enumerate(layout.window_zones):
            if any((self.collector.states.get(w) or {}).get("state") == "on" for w in zmap.get(z, {}).get("windows", []) or []):
                win[starts_min < st.window_assume_open_min, j] = 1.0
                open_zone.add(z)

        # schedule
        step_starts = [datetime.fromtimestamp(x / 1000, tz) for x in t[:-1]]
        step_ends = [datetime.fromtimestamp(x / 1000, tz) for x in t[1:]]
        fns = self.cfg.function_objs()
        zone_names = {z.id: z.name for z in layout.zones}
        unit_names = {u.id: (u.name, u.zone) for u in layout.units}
        events = []
        for ev in inputs.get("events", []):
            spec = parse_event(ev["summary"], zone_names, unit_names, {f.id: f.name for f in fns.values()})
            if spec:
                events.append({**spec, "start": ev["start"], "end": ev["end"], "name": ev["summary"]})
            else:
                info["notes"].append(f"calendar event not understood: {ev['summary']}")
        res = resolve(step_starts, step_ends, zids, [u.id for u in layout.units], cfg.defaults, cfg.rule_objs(), events)

        # comfort temperatures: the proxy's target, else the configured one
        comfort, functions, proxies = {}, {}, {}
        for z in zids:
            base = math.nan
            for u in layout.units:
                if u.zone == z:
                    s = self.collector.states.get((umap.get(u.id) or {}).get("proxy", ""))
                    if s:
                        base = _num((s.get("attributes") or {}).get("temperature"))
                        if math.isfinite(base):
                            break
            if not math.isfinite(base):
                base = cfg.comfort.get(z, math.nan)
            if not math.isfinite(base):
                continue
            comfort[z] = base + np.array(res.offset[z])
            functions[z] = [fns.get(f) or next(iter(fns.values())) for f in res.function[z]]

        units = []
        for u in layout.units:
            m = umap.get(u.id) or {}
            ps = self.collector.states.get(m.get("proxy", ""))
            req = self.requests.get(m.get("proxy", ""))
            allowed = ~np.array(res.locked[u.id])
            mode_now = ps.get("state") if ps else None
            note = None
            if ps is None:
                note = "proxy not found"
            elif mode_now in OFF_STATES:
                allowed[:] = False
                note = "proxy is off"
            elif not cfg.shadow and mode_now != "auto":
                allowed[:] = False
                note = f"proxy in {mode_now}, not auto"
            if u.zone in open_zone:
                allowed &= starts_min >= st.window_assume_open_min
            attrs = (ps or {}).get("attributes") or {}
            modes = tuple(x for x in attrs.get("hvac_modes", ["heat", "cool"]) if x in ("heat", "cool")) or ("heat",)
            lo, hi = 0, max(0, u.levels - 1)
            if req and u.kind == "ac":
                off = int(req.body.get("fan_mode_offset") or 0)
                lo, hi = max(0, off), max(max(0, off), min(hi, hi + off))
            running = attrs.get("hvac_action") in ("heating", "cooling")
            since = 1e9
            last = (req.body.get("last_switch") if req else None) or (ps or {}).get("last_changed")
            if last:
                try:
                    since = (now_ms - datetime.fromisoformat(str(last).replace("Z", "+00:00")).timestamp() * 1000) / MIN
                except ValueError:
                    pass
            units.append(mpc.UnitSpec(u.id, u.kind, u.zone, levels=(lo, hi), allowed=allowed, on=running,
                                      since_min=max(0.0, since), modes=modes))
            proxies[u.id] = {"proxy": m.get("proxy"), "state": mode_now, "note": note, "levels": [lo, hi],
                             "running": running, "comfort": comfort.get(u.zone, [None])[0] if u.zone in comfort else None}

        pr = mpc.Problem(sysm, t, dist, win, price, x0, comfort, functions, units, st)
        info.update({"units": proxies, "schedule": {"function": res.function, "source": res.source, "offset": res.offset,
                                                    "locked": res.locked}, "x0": x0.tolist(), "trained": self.params is not None})
        return pr, info

    # ---- planning ----
    async def replan(self) -> None:
        if self.lock.locked():  # one is running: its plan will do
            async with self.lock:
                return
        async with self.lock:
            now = time.time() * 1000
            try:
                inputs = await self.fetch(now)
                pr, info = await asyncio.to_thread(self.build_problem, now, inputs)
                plan = await asyncio.to_thread(mpc.solve, pr)
            except Exception as e:
                log.exception("planning failed")
                self.state.error = str(e)
                return
            info["forecast"] = {"t": inputs["forecast"]["t"].tolist(), "temp": inputs["forecast"]["temp"].tolist(),
                                "cloud": inputs["forecast"]["cloud"].tolist()}
            info["tout"] = pr.dist[:, DIST.index("tout")].round(2).tolist()
            info["sun"] = pr.dist[:, 1:5].sum(axis=1).round(3).tolist()
            info["prices"] = inputs["prices"]
            self.state = PlanState(plan=plan, created=now, error=None, info=info)
            self.store.put("plan", {"plan": plan.to_dict(), "info": info, "created": now})
            k1 = min(len(plan.t) - 1, int(np.searchsorted(plan.t, now + 2 * 3.6e6)))
            self.store.add_plan(int(now), {
                "mode": plan.mode, "cost": plan.cost, "times": [int(x) for x in plan.t[:k1 + 1]],
                "temp": {z: [round(float(v), 2) for v in a[:k1 + 1]] for z, a in plan.temp.items()},
                "on": {u: bool(a[0]) for u, a in plan.on.items()},
            })
            log.info("plan: %s, %.2f € over %.0f h, %.1f s", plan.mode, plan.cost["total"], (plan.t[-1] - plan.t[0]) / 3.6e6, plan.seconds)

    # ---- answering the proxies ----
    def command(self, body: dict, now_ms: float | None = None) -> dict:
        now_ms = now_ms or time.time() * 1000
        eid = body.get("entity_id", "")
        self.requests[eid] = ProxyRequest(now_ms, body)
        uid = self.unit_for_proxy(eid)
        valid = datetime.fromtimestamp((now_ms + 3 * MIN) / 1000, timezone.utc).isoformat()
        if uid is None:
            return {"follow": False, "reason": "not a unit of the optimizer"}
        plan = self.state.plan
        if plan is None or now_ms - self.state.created > 15 * MIN:
            return {"follow": False, "reason": "no current plan" + (f": {self.state.error}" if self.state.error else "")}
        k = plan.step_at(now_ms)
        if k is None:
            return {"follow": False, "reason": "plan does not cover now"}
        layout, params = self.model()
        unit = next(u for u in layout.units if u.id == uid)
        share = float(plan.share[uid][k])
        on = share >= 0.5
        modes = [m for m in body.get("modes", ["heat", "cool"]) if m in ("heat", "cool")]
        mode = "heat" if unit.kind == "heater" else plan.mode
        price = float(plan.price[k]) * 100
        decision: dict = {"on": on, "mode": mode, "price_ct": round(price, 1)}
        if not on or mode not in modes:
            self.dispatcher.reset(uid)
            nxt = next((j for j in range(k + 1, len(plan.t) - 1) if plan.share[uid][j] >= 0.5), None)
            when = (" · next " + datetime.fromtimestamp(plan.t[nxt] / 1000, self.tz()).strftime("%H:%M")) if nxt else ""
            answer = {"hvac_mode": "off", "valid_until": valid, "reason": f"off · {price:.1f} ct/kWh{when}"}
        else:
            target = abs(float(plan.heat[uid][k])) / max(share, 1e-6)
            answer = {"hvac_mode": mode, "valid_until": valid}
            if unit.kind == "ac":
                info = self.state.info.get("units", {}).get(uid, {})
                lo, hi = info.get("levels", [0, unit.levels - 1])
                tout = float(self.state.info.get("tout", [5.0])[min(k, len(self.state.info.get("tout", [5.0])) - 1)])
                level = self.dispatcher.level(params, uid, mode, target, tout, (lo, hi), now_ms)
                answer["fan_level"] = level
                decision["level"] = level
            decision["target_kw"] = round(target, 2)
            answer["reason"] = f"{'heat' if mode == 'heat' else 'cool'} {target:.1f} kW · {price:.1f} ct/kWh"
        if self.cfg.shadow:
            answer = {"follow": False, "reason": "shadow: would " + answer.get("reason", "")}
        decision["follow"] = answer.get("follow", True)
        self.store.add_decision(int(now_ms // MIN * MIN), uid, decision)
        return answer
