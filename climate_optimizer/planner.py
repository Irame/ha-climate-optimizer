"""Gathers the inputs, plans every five minutes and answers the proxies."""

from __future__ import annotations

import asyncio
import json
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
    if "base" not in params.plant:  # trained before the outdoor unit had a base, an extra and a limit
        base.plant["base"] = params.plant.get("standby", 0.03)
        base.plant.pop("kq", None)
        base.plant.pop("Pmax", None)
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
        self.airing_now: tuple[int, int] | None = None  # the airing the sensor suggests
        self._airing_sent = False
        self.state = PlanState()
        self.dispatcher = FanDispatcher()
        self.requests: dict[str, ProxyRequest] = {}
        self._calendar: dict | None = None
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

    def mapping(self) -> dict:
        """The configured mapping, with each AC unit's fan modes filled in when
        left empty: as its proxy reports them, else as the device lists them."""
        m = json.loads(json.dumps(self.cfg.mapping))
        reported = self.store.get("fan_modes") or {}
        for u in (m.get("units") or {}).values():
            if not u.get("fan_modes"):
                fm = reported.get(u.get("proxy", "")) or (self.collector.meta.get(u.get("source", "")) or {}).get("fan_modes")
                if fm:
                    u["fan_modes"] = list(fm)
        return m

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
        cal = await self.calendar()
        out["events"] = cal["events"]
        if cal["error"]:
            out["errors"].append(f"calendar: {cal['error']}")
        return out

    async def calendar(self, refresh: bool = False) -> dict:
        """The calendar's events from yesterday to two weeks ahead, fetched at most
        every ten minutes (or now, with refresh) and shared by plan and preview."""
        cfg = self.cfg
        if not cfg.calendar:
            return {"calendar": "", "events": [], "fetched": None, "error": None}
        now = time.time() * 1000
        c = self._calendar
        if not refresh and c and c["calendar"] == cfg.calendar and now - c["fetched"] < 10 * MIN:
            return c
        tz = self.tz()
        start = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        try:
            events = await fetch_events(self.ha, cfg.calendar, start, start + timedelta(days=17), tz)
            c = {"calendar": cfg.calendar, "events": events, "fetched": now, "error": None}
        except Exception as e:
            kept = c["events"] if c and c["calendar"] == cfg.calendar else []
            c = {"calendar": cfg.calendar, "events": kept, "fetched": now, "error": str(e) or e.__class__.__name__}
        self._calendar = c
        return c

    def event_specs(self, events: list[dict]) -> tuple[list[dict], list[dict]]:
        """Calendar events as rules for resolve(), and each one with what it was taken to mean."""
        layout = self.cfg.layout_obj()
        fns = self.cfg.function_objs()
        zones = {z.id: z.name for z in layout.zones}
        units = {u.id: (u.name, u.zone) for u in layout.units}
        specs, listing = [], []
        for ev in events:
            spec = parse_event(ev["summary"], zones, units, {f.id: f.name for f in fns.values()},
                               {z.id: list(z.aliases) for z in layout.zones})
            item = {"start": int(ev["start"].timestamp() * 1000), "end": int(ev["end"].timestamp() * 1000),
                    "summary": ev["summary"], "understood": spec is not None}
            if spec:
                specs.append({**spec, "start": ev["start"], "end": ev["end"], "name": ev["summary"]})
                every = "every room" if spec["action"] != "lockout" else "every unit"
                if spec["action"] == "comfort":
                    item["effect"] = f"{fns[spec['function']].name} in " + (", ".join(zones[z] for z in spec["zones"]) or every)
                elif spec["action"] == "offset":
                    item["effect"] = f"comfort {spec['offset']:+g} K in " + (", ".join(zones[z] for z in spec["zones"]) or every)
                elif spec["action"] == "no_airing":
                    item["effect"] = "no airing suggested"
                else:
                    item["effect"] = "off: " + (", ".join(units[u][0] for u in spec["units"]) or every)
            listing.append(item)
        return specs, listing

    def hidden_state(self, layout: Layout, params: Params, now_ms: float) -> tuple[np.ndarray, np.ndarray, str]:
        """Surfaces and structure from the last days of air temperatures."""
        from .collector import tracked_entities
        cfg = self.cfg
        ents = tracked_entities(cfg.mapping)
        ex = self.store.export(ents, int(now_ms - 4 * 864e5), int(now_ms), self.collector.meta,
                               {**self.location(), "tz": str(self.tz())})
        try:
            grid = build_grid(parse_rows(ex), layout, self.mapping(), min_hours=6)
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
        ghi, cloud = np.zeros(N), np.zeros(N)  # for the page: sun on the ground, cloud cover
        cloud_t = fc["t"] if len(fc["t"]) else np.array([now_ms])
        cloud_v = fc["cloud"] if len(fc["t"]) else np.array([0.6])
        for k in range(N):
            n = max(1, int(round((t[k + 1] - t[k]) / (10 * MIN))))
            ts = t[k] + (np.arange(n) + 0.5) * (t[k + 1] - t[k]) / n
            to = blend_outdoor(fc["t"], fc["temp"], now_ms, tout_now, ts)
            cl = np.interp(ts, cloud_t, cloud_v)
            irr = np.array([[(r := irradiance(x, loc["lat"], loc["lon"], c))["D"], r["E"], r["S"], r["W"], r["ghi"]]
                            for x, c in zip(ts, cl)]).mean(axis=0)
            sol = irr[:4]
            ghi[k], cloud[k] = irr[4], cl.mean()
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
        events, listing = self.event_specs([e for e in inputs.get("events", [])
                                            if e["end"] > step_starts[0] and e["start"] < step_ends[-1]])
        for item in listing:
            if not item["understood"]:
                info["notes"].append(f"calendar event not understood: {item['summary']}")
        res = resolve(step_starts, step_ends, zids, [u.id for u in layout.units], cfg.defaults, cfg.rule_objs(), events)

        # comfort temperatures: the proxy's target, else the configured one
        comfort, functions, proxies = {}, {}, {}
        for z in zids:
            base = cfg.shadow_comfort.get(z, math.nan) if cfg.shadow else math.nan
            for u in layout.units:
                if math.isfinite(base):
                    break
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
        uncontrolled = set()  # units the optimizer cannot run at all: off, unknown, or run by their own thermostat
        for u in layout.units:
            m = umap.get(u.id) or {}
            ps = self.collector.states.get(m.get("proxy", ""))
            req = self.requests.get(m.get("proxy", ""))
            allowed = ~np.array(res.locked[u.id])
            mode_now = ps.get("state") if ps else None
            note = None
            anyway = cfg.shadow and u.id in cfg.shadow_units  # planned whatever the proxy does
            if anyway:
                pass
            elif ps is None:
                allowed[:] = False
                note = "proxy not found"
            elif mode_now in OFF_STATES:
                allowed[:] = False
                note = "proxy is off"
            elif not cfg.shadow and mode_now != "auto":
                allowed[:] = False
                note = f"proxy in {mode_now}, not auto"
            if note:
                uncontrolled.add(u.id)
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
            proxies[u.id] = {"proxy": m.get("proxy"), "state": mode_now, "note": note, "levels": [lo, hi], "anyway": anyway,
                             "running": running, "comfort": comfort.get(u.zone, [None])[0] if u.zone in comfort else None}

        # A room whose units are all off is not the plan's to keep warm: pricing its
        # comfort would only have the plan heat the neighbours to warm it through
        # the hallway. (A scheduled lockout is different: heating ahead of it is fair.)
        for z in list(comfort):
            here = [u.id for u in layout.units if u.zone == z]
            if here and all(uid in uncontrolled for uid in here):
                del comfort[z], functions[z]
                info.setdefault("no_comfort", {})[z] = proxies[here[0]]["note"]
                for uid in here:
                    proxies[uid]["comfort"] = None
        airing = None
        if cfg.airing.enabled and layout.window_zones:
            airing, info["airing"] = self.airing_spec(now_ms, t, comfort, x0, layout, tz, events)
        pr = mpc.Problem(sysm, t, dist, win, price, x0, comfort, functions, units, st, airing)
        info["ghi"] = (ghi * 1000).round(0).tolist()  # W/m²
        info["cloud"] = (cloud * 100).round(0).tolist()  # %
        info.update({"units": proxies, "schedule": {"function": res.function, "source": res.source, "offset": res.offset,
                                                    "locked": res.locked}, "x0": x0.tolist(), "trained": self.params is not None})
        return pr, info

    # ---- airing ----
    def airing_done_today(self, now_ms: float, tz) -> tuple[int, float | None]:
        """Airings since midnight, from the window sensors: stretches with windows open
        in at least half the window rooms, for long enough (one tilted window is not
        the apartment aired); and when the last one ended. A suggestion under way
        counts too, so the next plan does not suggest another before it is over."""
        a = self.cfg.airing
        midnight = datetime.fromtimestamp(now_ms / 1000, tz).replace(hour=0, minute=0, second=0, microsecond=0)
        t0 = int(midnight.timestamp() * 1000)
        spans = []
        zmap = self.cfg.mapping.get("zones") or {}
        rooms = [zid for zid in self.cfg.layout_obj().window_zones if (zmap.get(zid) or {}).get("windows")]
        for zid in rooms:
            for w in zmap[zid]["windows"]:
                rows = self.store.rows(w, t0, int(now_ms))
                for r, nxt in zip(rows, rows[1:] + [[now_ms, None]]):
                    if r[1] == "on":
                        spans.append((max(r[0], t0), nxt[0], zid))
        spans.sort()
        merged = []  # [start, end, rooms]
        for a0, b0, zid in spans:
            if merged and a0 <= merged[-1][1] + 2 * MIN:
                merged[-1][1] = max(merged[-1][1], b0)
                merged[-1][2].add(zid)
            else:
                merged.append([a0, b0, {zid}])
        enough = math.ceil(len(rooms) / 2)
        done = [m[:2] for m in merged if len(m[2]) >= enough and m[1] - m[0] >= max(3, a.duration_min / 2) * MIN]
        cur = self.airing_now
        if cur and cur[0] <= now_ms < cur[1]:
            if not any(m[0] < cur[1] and m[1] > cur[0] - 2 * MIN for m in done):
                done.append([cur[0], cur[1]])
            done[-1][1] = max(done[-1][1], cur[1])
        return len(done), (done[-1][1] if done else None)

    def airing_spec(self, now_ms: float, t: np.ndarray, comfort: dict, x0: np.ndarray, layout: Layout, tz,
                    events: list[dict]) -> tuple[mpc.Airing | None, dict]:
        """When the plan may air: on the half hours, outside the 'no airing' times,
        as many a day as set, and apart enough."""
        cfg = self.cfg
        a = cfg.airing
        D = int(a.duration_min * MIN)
        gap = a.min_gap_h * 3.6e6
        grid = 30 * MIN
        end = int(t[-1])
        done, last = self.airing_done_today(now_ms, tz)
        today = datetime.fromtimestamp(now_ms / 1000, tz).date()
        info = {"done_today": done, "per_day": a.per_day, "duration_min": a.duration_min}
        # every half hour from now to the end of the plan's last day, to see each day's allowed times whole
        last_day = datetime.fromtimestamp(end / 1000, tz).date()
        day_end = datetime.combine(last_day, datetime.min.time(), tz) + timedelta(days=1)
        starts = list(range(int(math.ceil(now_ms / grid) * grid), int(day_end.timestamp() * 1000), grid))
        if not starts:
            return None, info
        s_dt = [datetime.fromtimestamp(x / 1000, tz) for x in starts]
        e_dt = [datetime.fromtimestamp((x + D) / 1000, tz) for x in starts]
        blocked = resolve(s_dt, e_dt, [], [], {}, cfg.rule_objs(), events).no_airing
        cand, groups, by_day = [], [], {}
        for x, d, no in zip(starts, s_dt, blocked):
            if no or (last is not None and d.date() == today and x - last < gap):
                continue
            by_day.setdefault(d.date(), []).append(x)
        for day, xs in sorted(by_day.items()):
            due = a.per_day - (done if day == today else 0)
            if due <= 0 or xs[-1] + D > end:
                continue  # done, or the day's allowed times run past the plan: planned later
            idx = list(range(len(cand), len(cand) + len(xs)))
            cand += [(x, x + D) for x in xs]
            groups.append((idx, min(due, len(xs))))
        if not cand:
            info["none"] = "done today" if done >= a.per_day else "no allowed time in the plan"
            return None, info
        N = len(t) - 1
        tref = {}
        for z in layout.window_zones:
            i = layout.zone_index(z)
            tref[z] = np.asarray(comfort[z], float) if z in comfort else np.full(N, float(x0[i]))
        return mpc.Airing(cand=cand, groups=groups, min_gap_ms=gap, tref=tref), info

    async def publish_airing(self, plan: mpc.Plan, info: dict) -> None:
        """The next suggestion as a sensor in Home Assistant, for its own automations."""
        a = self.cfg.airing
        if not a.sensor or not hasattr(self.ha, "set_state") or not (a.enabled or self._airing_sent):
            return
        now = time.time() * 1000
        cur = self.airing_now
        if not a.enabled:
            nxt = None
        elif cur and cur[0] <= now < cur[1]:
            nxt = cur  # under way: it stays the suggestion until it is over
        else:
            nxt = next(((s0, e0) for s0, e0 in plan.airing if e0 > now), None)
        self.airing_now = nxt
        ai = info.setdefault("airing", {})
        ai["suggested"] = list(nxt) if nxt else None
        layout = self.cfg.layout_obj()
        attrs = {"friendly_name": "Climate optimizer: airing", "device_class": "timestamp", "icon": "mdi:window-open-variant",
                 "duration_min": a.duration_min, "rooms": [z.name for z in layout.zones if z.windows],
                 "done_today": ai.get("done_today")}
        if nxt:
            state = datetime.fromtimestamp(nxt[0] / 1000, timezone.utc).isoformat()
            attrs["end"] = datetime.fromtimestamp(nxt[1] / 1000, timezone.utc).isoformat()
            if tuple(nxt) in plan.airing and plan.airing_cost is not None:
                attrs["extra_cost_eur"] = round(plan.airing_cost, 3)  # all of the plan's airings together
        else:
            state = "unknown"
            attrs["reason"] = "airing suggestions are off" if not a.enabled else ai.get("none") or "none in the plan yet"
        try:
            await self.ha.set_state(a.sensor, state, attrs)
            self._airing_sent = a.enabled
        except Exception as e:  # the plan stands without it
            log.warning("airing sensor: %s", e)

    # ---- planning ----
    async def replan(self, fresh: bool = False) -> None:
        """Plan now. With one already running, its plan will do, unless fresh:
        then plan once more after it, from the state as it is now (a setting
        changed, the calendar was read, someone asked)."""
        if self.lock.locked() and not fresh:
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
            await self.publish_airing(plan, info)
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
        fm = body.get("source_fan_modes")
        if fm:
            known = self.store.get("fan_modes") or {}
            if known.get(eid) != fm:
                self.store.put("fan_modes", {**known, eid: list(fm)})
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
