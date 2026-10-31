"""The service: loops that collect, plan and train, and the HTTP API."""

from __future__ import annotations

import asyncio
import bisect
import contextlib
import gzip
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .collector import Collector, tracked_entities
from .comfort import resolve
from .config import DATA, ConfigStore
from .ha import HomeAssistant
from .model.dataset import build_grid, parse_rows
from .planner import Planner
from .store import Store
from .trainer import Trainer

log = logging.getLogger(__name__)
WEB = Path(__file__).parent / "web"
# when these files were last changed: on the Settings page, to see which version runs
BUILD = max(f.stat().st_mtime for f in Path(__file__).parent.rglob("*") if f.suffix in (".py", ".js", ".html", ".css")) * 1000
MIN = 60_000


class Service:
    def __init__(self, data_dir: Path = DATA, ha: HomeAssistant | None = None):
        self.cfgs = ConfigStore(data_dir / "config.json")
        self.store_path = data_dir / "optimizer.db"
        self.store = Store(self.store_path)
        c = self.cfgs.config
        self.ha = ha or HomeAssistant(c.ha.url, c.ha.token, c.ha.verify_ssl)
        self.collector = Collector(self.ha, self.store)
        self.collector.track(tracked_entities(c.mapping))
        self.planner = Planner(lambda: self.cfgs.config, self.store, self.collector, self.ha)
        self.trainer = Trainer(self.planner, str(data_dir / "training.json"))
        self.stop = asyncio.Event()
        self.tasks: list[asyncio.Task] = []
        self.started = time.time()

    async def on_connect(self) -> None:
        hc = await self.ha.ha_config()
        self.store.put("ha_config", {k: hc.get(k) for k in ("time_zone", "latitude", "longitude", "version", "location_name")})
        await self.collector.on_connect(self.cfgs.config.history_days)
        asyncio.create_task(self.planner.replan())
        if self.planner.params is None and not self.trainer.running and self.cfgs.config.mapping:
            asyncio.create_task(self.trainer.run())  # first start: learn the apartment from what was backfilled

    async def start(self) -> None:
        self.tasks = [
            asyncio.create_task(self.ha.run(self.on_connect, self.stop)),
            asyncio.create_task(self._every_minute()),
            asyncio.create_task(self._plan_loop()),
            asyncio.create_task(self._train_loop()),
        ]

    async def close(self) -> None:
        self.stop.set()
        for t in self.tasks:
            t.cancel()
        for t in self.tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self.trainer.shutdown()

    async def _every_minute(self) -> None:
        last_prune = 0.0
        while not self.stop.is_set():
            await _sleep_until(self.stop, 60, offset=1)
            self.collector.tick()
            if time.time() - last_prune > 6 * 3600:
                last_prune = time.time()
                self.store.prune(int((time.time() - (self.cfgs.config.history_days + 5) * 86400) * 1000))

    async def _plan_loop(self) -> None:
        while not self.stop.is_set():
            await _sleep_until(self.stop, 300, offset=8)
            if self.ha.connected.is_set():
                await self.planner.replan()

    async def _train_loop(self) -> None:
        while not self.stop.is_set():
            await _sleep_until(self.stop, 600, offset=30)
            cfg = self.cfgs.config
            rep = self.planner.report or {}
            if self.ha.connected.is_set() and retrain_due((rep.get("finished") or 0) / 1000, datetime.now(self.planner.tz()), cfg.retrain_hour):
                await self.trainer.run()

    async def config_changed(self, old, new) -> None:
        if (old.ha.url, old.ha.token, old.ha.verify_ssl) != (new.ha.url, new.ha.token, new.ha.verify_ssl):
            self.ha.url, self.ha.token, self.ha.verify_ssl = new.ha.url.rstrip("/"), new.ha.token, new.ha.verify_ssl
            if self.ha._ws is not None:
                await self.ha._ws.close()  # the run loop reconnects with the new settings
        if old.mapping != new.mapping:
            self.collector.track(tracked_entities(new.mapping))
            if self.ha.connected.is_set():
                states = await self.ha.states()
                for s in states:
                    if s["entity_id"] in self.collector.entities:
                        self.collector._meta(s)
                self.store.put("meta", self.collector.meta)
                asyncio.create_task(self.collector.backfill(new.history_days))
        if self.ha.connected.is_set():
            asyncio.create_task(self.planner.replan(fresh=True))


async def _sleep_until(stop: asyncio.Event, period: float, offset: float = 0.0) -> None:
    """Sleep to the next multiple of period (plus offset seconds), or until stop."""
    now = time.time()
    nxt = (now - offset) // period * period + period + offset
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(stop.wait(), max(0.5, nxt - now))


def create_app(service: Service | None = None, data_dir: Path | None = None) -> FastAPI:
    holder: dict = {}

    @contextlib.asynccontextmanager
    async def lifespan(app):
        svc = service or Service(data_dir or DATA)
        holder["svc"] = svc
        await svc.start()
        yield
        await svc.close()

    app = FastAPI(title="Climate optimizer", lifespan=lifespan)

    @app.middleware("http")
    async def fresh_ui(request: Request, call_next):
        """The page and its scripts are revalidated on every load (a cheap 304 when
        unchanged), so after an update the browser shows the new version at once,
        not one it cached by guesswork."""
        resp = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    def svc() -> Service:
        return holder["svc"]

    def auth(request: Request):
        token = holder["svc"].cfgs.config.api_token
        if not token:
            return
        h = request.headers.get("authorization", "")
        if h == f"Bearer {token}" or request.query_params.get("token") == token:
            return
        raise HTTPException(401, "token required")

    # ---- for the proxies ----
    @app.post("/api/v1/command", dependencies=[Depends(auth)])
    async def command(body: dict):
        return svc().planner.command(body)

    # ---- for the UI ----
    @app.get("/api/status", dependencies=[Depends(auth)])
    async def status():
        s = svc()
        p = s.planner
        cfg = s.cfgs.config
        now = time.time() * 1000
        layout, _ = p.model()
        units = {}
        for u in layout.units:
            m = (cfg.mapping.get("units") or {}).get(u.id) or {}
            req = p.requests.get(m.get("proxy", ""))
            dec = s.store.decisions(int(now - 10 * MIN), int(now) + 1).get(u.id, [])
            ps = s.collector.states.get(m.get("proxy", "")) or {}
            units[u.id] = {"name": u.name, "zone": u.zone, "kind": u.kind, "proxy": m.get("proxy"),
                           "proxy_state": ps.get("state"), "proxy_attrs": ps.get("attributes"),
                           "last_request": req.t if req else None, "decision": dec[-1] if dec else None}
        zones = {}
        for z in layout.zones:
            zm = (cfg.mapping.get("zones") or {}).get(z.id) or {}
            zones[z.id] = {"name": z.name, "temperature": p.live(zm.get("thermometer")),
                           "windows_open": [w for w in zm.get("windows", []) or [] if (s.collector.states.get(w) or {}).get("state") == "on"]}
        return _json({
            "ha": {"connected": s.ha.connected.is_set(), "error": s.ha.error, "version": s.ha.version, "url": s.ha.url},
            "shadow": cfg.shadow, "plan": {"created": p.state.created or None, "error": p.state.error,
                                           "mode": p.state.plan.mode if p.state.plan else None,
                                           "cost": p.state.plan.cost if p.state.plan else None},
            "model": {"trained": p.params is not None, "finished": (p.report or {}).get("finished"),
                      "training": s.trainer.running, "progress": s.trainer.progress(), "error": s.trainer.error},
            "backfill": s.collector.backfill_status, "units": units, "zones": zones,
            "outdoor": p.live(cfg.mapping.get("outdoor")), "uptime_s": time.time() - s.started, "tz": str(p.tz()),
            "build": BUILD, "overrides": p.overrides.list(),
        })

    # ---- overrides: a room to a temperature, a unit to run a while ----
    @app.get("/api/overrides", dependencies=[Depends(auth)])
    async def overrides():
        return _json(svc().planner.overrides.list())

    @app.post("/api/overrides", dependencies=[Depends(auth)])
    async def add_override(body: dict):
        """{zone, target, at?, hold_min?, until?}: the room at target °C, now or by 'at',
        held hold_min after or until 'until'. {unit, run_min, toggle?}: the unit runs
        that long whatever the temperature; with toggle, a run going on is ended instead.
        Rooms and units by id, name, German name, alias, or one of their entities."""
        from .overrides import find_unit, find_zone, parse_time
        s = svc()
        p = s.planner
        cfg = s.cfgs.config
        layout = cfg.layout_obj()
        now = time.time() * 1000
        tz = p.tz()
        try:
            if body.get("unit") is not None:
                uid = find_unit(layout, cfg.mapping, body["unit"])
                if uid is None:
                    raise ValueError(f"no unit {body['unit']!r}")
                if body.get("toggle") and p.overrides.run_for(uid, now):
                    await p.cancel_override(unit=uid)
                    return {"cancelled": True, "unit": uid}
                minutes = float(body.get("run_min") or 15)
                if not 1 <= minutes <= 240:
                    raise ValueError("run_min: 1 to 240 minutes")
                o = await p.add_override("run", now, unit=uid, minutes=minutes, source=body.get("source", "api"))
            else:
                zid = find_zone(layout, cfg.mapping, body.get("zone"))
                if zid is None:
                    raise ValueError(f"no room {body.get('zone')!r}")
                target = float(body["target"])
                if not 5 <= target <= 30:
                    raise ValueError("target: 5 to 30 °C")
                ready = parse_time(body.get("at"), now, tz)
                until = parse_time(body.get("until"), now, tz)
                hold = float(body.get("hold_min") or 0)
                if until is not None and until <= max(now, ready or 0):
                    raise ValueError("until must come after now and after 'at'")
                if not 0 <= hold <= 24 * 60:
                    raise ValueError("hold_min: 0 to 1440")
                o = await p.add_override("target", now, zone=zid, target=target, ready_at=ready, hold_min=hold,
                                         until=until, source=body.get("source", "api"))
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(422, str(e) if not isinstance(e, KeyError) else f"missing {e}")
        return _json(o.to_dict())

    @app.delete("/api/overrides", dependencies=[Depends(auth)])
    async def cancel_overrides(zone: str | None = None, unit: str | None = None, id: str | None = None):
        from .overrides import find_unit, find_zone
        s = svc()
        cfg = s.cfgs.config
        layout = cfg.layout_obj()
        kw = {"id": id}
        if zone is not None:
            kw["zone"] = find_zone(layout, cfg.mapping, zone) or "?"
        if unit is not None:
            kw["unit"] = find_unit(layout, cfg.mapping, unit) or "?"
        gone = await s.planner.cancel_override(**kw)
        return _json({"cancelled": [o.to_dict() for o in gone]})

    @app.get("/api/plan", dependencies=[Depends(auth)])
    async def plan():
        p = svc().planner
        if p.state.plan is None:
            saved = svc().store.get("plan")
            if saved:
                return _json({**saved, "stale": True})
            return _json({"plan": None, "error": p.state.error})
        return _json({"plan": p.state.plan.to_dict(), "info": p.state.info, "created": p.state.created, "error": p.state.error})

    @app.post("/api/plan", dependencies=[Depends(auth)])
    async def replan():
        await svc().planner.replan(fresh=True)
        return await plan()

    @app.get("/api/history", dependencies=[Depends(auth)])
    async def history(hours: float = 48):
        s = svc()
        cfg = s.cfgs.config
        p = s.planner
        hours = float(min(max(hours, 3), 24 * cfg.history_days))
        now = int(time.time() * 1000)
        start = int(now - hours * 3.6e6)
        layout, _ = p.model()
        ex = s.store.export(tracked_entities(cfg.mapping), start, now, s.collector.meta, {**p.location(), "tz": str(p.tz())})
        out: dict = {"start": start, "end": now}
        try:
            g = build_grid(parse_rows(ex), layout, p.mapping(), start=start, min_hours=1)
        except ValueError as e:
            out["error"] = str(e)
        else:
            out.update({
                "t": g.t.astype(np.int64).tolist(),
                "temp": {z: _r(g.T[:, i], 2) for i, z in enumerate(layout.zone_ids)},
                "tout": _r(g.tout, 2),
                "ghi": _r(g.ghi * 1000, 0), "cloud": _r(g.cloud * 100, 0),
                "units": {u: {"heat": _r(f[:, 0].sum(1), 2), "cool": _r(f[:, 1].sum(1), 2),
                              "level": _r((f.sum(1) * np.arange(f.shape[2])).sum(1) / np.maximum(1e-9, f.sum((1, 2))), 2)}
                          for u, f in g.ac_frac.items()},
                "heater": {u: _r(v, 3) for u, v in g.heater_kw.items()},
                "phase": _r(g.phase, 3),
                "open": {z: _r(g.open[:, j], 2) for j, z in enumerate(layout.window_zones)},
            })
        ph = s.store.get("price_history") or {}
        out["prices"] = [[int(k), v[0], v[1]] for k, v in sorted(ph.items(), key=lambda x: int(x[0])) if int(k) >= start - 3.6e6]
        out["decisions"] = s.store.decisions(start, now + 1)
        # what the plans predicted an hour ahead, against what came
        pred: dict = {}
        for rec in s.store.plans(start - 3_600_000, now):
            times = rec.get("times")
            if not isinstance(times, list) or len(times) < 2:
                continue
            k = int(np.searchsorted(times, times[0] + 3_600_000))
            if k >= len(times):
                continue
            for z, a in (rec.get("temp") or {}).items():
                pred.setdefault(z, []).append([times[k], a[k]])
        out["predicted_1h"] = pred
        return _json(out)

    @app.get("/api/energy/days", dependencies=[Depends(auth)])
    async def energy(days: int = 14):
        s = svc()
        days = max(1, min(days, 60))
        return _json(await asyncio.to_thread(energy_days, s, days))

    @app.get("/api/config", dependencies=[Depends(auth)])
    async def get_config():
        return svc().cfgs.config.public()

    @app.put("/api/config", dependencies=[Depends(auth)])
    async def put_config(patch: dict):
        s = svc()
        old = s.cfgs.config
        try:
            new = s.cfgs.update(patch)
        except Exception as e:
            raise HTTPException(422, str(e))
        await s.config_changed(old, new)
        return new.public()

    @app.get("/api/entities", dependencies=[Depends(auth)])
    async def entities():
        out = []
        for eid, st in sorted(svc().collector.states.items()):
            a = st.get("attributes") or {}
            e = {"id": eid, "name": a.get("friendly_name"), "state": st.get("state"),
                 "unit": a.get("unit_of_measurement"), "device_class": a.get("device_class")}
            if eid.startswith("climate."):
                e["fan_modes"] = a.get("fan_modes")
            out.append(e)
        return out

    @app.get("/api/model", dependencies=[Depends(auth)])
    async def model():
        s = svc()
        _, params = s.planner.model()
        return _json({"params": params.to_dict(), "trained": s.planner.params is not None, "report": s.planner.report,
                      "training": s.trainer.running, "started": s.trainer.started, "progress": s.trainer.progress(),
                      "error": s.trainer.error})

    @app.post("/api/model/train", dependencies=[Depends(auth)])
    async def train(fresh: bool = False):
        s = svc()
        if s.trainer.running:
            raise HTTPException(409, "training is running")
        asyncio.create_task(s.trainer.run(use_current=not fresh))
        return {"started": True}

    @app.get("/api/schedule/preview", dependencies=[Depends(auth)])
    async def schedule_preview(days: int = 14, step_min: int = 30, refresh: bool = False):
        """The saved rules and the calendar's events over the coming days; refresh fetches the calendar now."""
        s = svc()
        cfg = s.cfgs.config
        tz = s.planner.tz()
        layout = cfg.layout_obj()
        days = max(1, min(days, 14))  # the calendar is fetched for two weeks
        t0 = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
        n = days * 24 * 60 // step_min
        starts = [t0 + timedelta(minutes=step_min * k) for k in range(n)]
        ends = [x + timedelta(minutes=step_min) for x in starts]
        cal = await s.planner.calendar(refresh=refresh) if s.ha.connected.is_set() or not cfg.calendar else \
            {"calendar": cfg.calendar, "events": [], "fetched": None, "error": "Home Assistant is not connected"}
        inside = [e for e in cal["events"] if e["end"] > starts[0] and e["start"] < ends[-1]]
        specs, listing = s.planner.event_specs(inside)
        res = resolve(starts, ends, layout.zone_ids, [u.id for u in layout.units], cfg.defaults, cfg.rule_objs(), specs)
        if refresh and s.ha.connected.is_set():
            asyncio.create_task(s.planner.replan(fresh=True))  # the plan sees the calendar as it is now
        return _json({"start": int(t0.timestamp() * 1000), "step_min": step_min, "function": res.function,
                      "offset": res.offset, "locked": res.locked, "source": res.source,
                      "event_zone": res.event_zone, "event_unit": res.event_unit, "no_airing": res.no_airing,
                      "target": {z: [None if v != v else v for v in a] for z, a in res.target.items()},
                      "calendar": {"entity": cal["calendar"], "fetched": cal["fetched"], "error": cal["error"],
                                   "events": sorted(listing, key=lambda e: e["start"])}})

    @app.get("/api/diagnostics", dependencies=[Depends(auth)])
    async def diagnostics(days: float = 30):
        from .diagnostics import collect, to_gzip
        s = svc()
        days = max(1.0, min(days, 365.0))
        blob = await asyncio.to_thread(lambda: to_gzip(collect(s.cfgs.config, s.store, s.collector.meta, days)))
        name = datetime.now(s.planner.tz()).strftime("climate-optimizer-diagnostics-%Y%m%d-%H%M.json.gz")
        return Response(blob, media_type="application/gzip", headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.post("/api/import", dependencies=[Depends(auth)])
    async def import_export(request: Request):
        """Load an export (the browser snippet's, or a diagnostics file) into the store, e.g. before the first training."""
        raw = await request.body()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        try:
            export = json.loads(raw)
        except ValueError:
            raise HTTPException(422, "not a JSON export")
        s = svc()
        n = 0
        for eid, rows in (export.get("history") or {}).items():
            conv = []
            for r in rows:
                t = _to_ms(r[0])
                if t is not None:
                    conv.append([t] + list(r[1:]))
            s.store.add_rows(eid, conv)
            n += len(conv)
        for eid, rows in (export.get("stats_5min") or {}).items():
            conv = [[_to_ms(a), b] for a, b in rows if b is not None and _to_ms(a) is not None]
            s.store.add_rows(eid, conv)
            n += len(conv)
        meta = {**s.collector.meta, **(export.get("meta") or {})}
        s.collector.meta = meta
        s.store.put("meta", meta)
        return {"rows": n}

    @app.get("/healthz")
    async def health():
        return {"ok": True}

    if WEB.exists():
        app.mount("/static", StaticFiles(directory=WEB), name="static")

        @app.get("/")
        async def index():
            return FileResponse(WEB / "index.html")

    return app


def _to_ms(v):
    from .collector import _ms
    try:
        return _ms(v)
    except (TypeError, ValueError):
        return None


def _r(a, d=2):
    a = np.asarray(a, dtype=float)
    return [None if not np.isfinite(x) else round(float(x), d) for x in a]


def retrain_due(done_s: float, now: datetime, hour: int) -> bool:
    """In the retrain hour, unless trained since it began (a training by hand in the
    evening does not skip the night's: that one has the evening's data too)."""
    if now.hour != hour:
        return False
    return done_s < now.replace(minute=0, second=0, microsecond=0).timestamp()


def energy_days(s, days: int) -> list[dict]:
    """Per day: the units' electricity as the day's first plan expected it, and as
    measured (the outdoor unit's phase, less what it draws with every unit off, plus
    the heaters' plugs), hour by hour."""
    cfg = s.cfgs.config
    p = s.planner
    tz = p.tz()
    layout, _ = p.model()
    now = datetime.now(tz)
    day0 = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    start, end = int(day0.timestamp() * 1000), int(now.timestamp() * 1000)
    fc = s.store.get("forecasts") or {}
    ph = s.store.get("price_history") or {}
    prices = sorted((int(k), v[0], v[1]) for k, v in ph.items())
    # measured, from the 5-minute grid; other loads on the phase from all-off times, by hour of day
    meas, other_h = {}, {}
    try:
        ex = s.store.export(tracked_entities(cfg.mapping), int(start - 7 * 864e5), end, s.collector.meta, {**p.location(), "tz": str(tz)})
        g = build_grid(parse_rows(ex), layout, p.mapping(), min_hours=1)
    except ValueError:
        g = None
    if g is not None:
        ac = [u.id for u in layout.ac_units]
        anyon = np.max([g.unit_running(u) for u in ac], axis=0) if ac else np.zeros(g.K)
        heat = sum(g.heater_kw.values()) if g.heater_kw else np.zeros(g.K)
        hod = np.array([datetime.fromtimestamp(t / 1000, tz).hour for t in g.t])
        off = (anyon < 0.01) & np.isfinite(g.phase)
        overall = float(np.nanmean(g.phase[off])) if off.any() else 0.0
        prof = np.array([float(np.nanmean(g.phase[off & (hod == h)])) if (off & (hod == h)).sum() >= 12 else overall for h in range(24)])
        for k in range(g.K):
            if g.t[k] < start or not np.isfinite(g.phase[k]):
                continue
            a = datetime.fromtimestamp(g.t[k] / 1000, tz)
            h = int(a.replace(minute=0, second=0, microsecond=0).timestamp() * 1000)
            m = meas.setdefault(h, [0.0, 0.0, 0.0, 0])  # phase, heaters, other loads (kWh), steps
            m[0] += float(g.phase[k]) * g.dt_h
            m[1] += float(np.nan_to_num(heat[k])) * g.dt_h
            m[2] += prof[hod[k]] * g.dt_h
            m[3] += 1
        other_h = {h: v[2] for h, v in meas.items()}

    p_starts = [a for a, _, _ in prices]

    def price_at(t):
        i = bisect.bisect_right(p_starts, t) - 1
        return prices[i][2] if i >= 0 and prices[i][0] <= t < prices[i][1] else None

    out = []
    for i in range(days):
        d = (day0 + timedelta(days=i)).date()
        f = fc.get(d.isoformat())
        rows = []
        midnight = datetime.combine(d, datetime.min.time(), tz).timestamp()
        for hh in range(25):  # 23 to 25 hours, as the clocks go
            t = datetime.fromtimestamp(midnight + hh * 3600, tz)
            if t.date() != d:
                break
            h = int(t.timestamp() * 1000)
            m = meas.get(h)
            whole = m is not None and m[3] >= 0.8 * 3600 / (g.dt_h * 3600) and h + 3_600_000 <= end  # a finished, recorded hour
            units = max(0.0, m[0] - m[2]) + m[1] if whole else None
            planned = (f or {}).get("hours", {}).get(str(h))
            rows.append([h, planned, None if units is None else round(units, 4), price_at(h)])
        covered = [r for r in rows if r[1] is not None and r[2] is not None]
        measured = [r for r in rows if r[2] is not None]
        out.append({
            "date": d.isoformat(), "start": rows[0][0] if rows else None,
            "forecast_made": (f or {}).get("made"),
            "planned_kwh": round(sum(r[1] for r in covered), 3) if covered else None,  # the hours both have
            "measured_kwh": round(sum(r[2] for r in covered), 3) if covered else None,
            "planned_day_kwh": round(sum((f or {}).get("hours", {}).values()), 3) if f else None,
            "measured_day_kwh": round(sum(r[2] for r in measured), 3) if measured else None,
            "measured_hours": len(measured), "compared_hours": len(covered),
            "planned_eur": round(sum(r[1] * r[3] for r in covered if r[3] is not None), 3) if covered else None,
            "measured_eur": round(sum(r[2] * r[3] for r in measured if r[3] is not None), 3) if measured else None,
            "other_kwh": round(sum(other_h.get(r[0], 0.0) for r in rows), 3),
            "hours": rows,
        })
    return out


def _json(obj):
    def conv(x):
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, np.ndarray):
            return conv(x.tolist())
        if isinstance(x, (np.floating, float)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    return JSONResponse(conv(obj))
