"""The service: loops that collect, plan and train, and the HTTP API."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
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
MIN = 60_000


class Service:
    def __init__(self, data_dir: Path = DATA, ha: HomeAssistant | None = None):
        self.cfgs = ConfigStore(data_dir / "config.json")
        self.store = Store(data_dir / "optimizer.db")
        c = self.cfgs.config
        self.ha = ha or HomeAssistant(c.ha.url, c.ha.token, c.ha.verify_ssl)
        self.collector = Collector(self.ha, self.store)
        self.collector.track(tracked_entities(c.mapping))
        self.planner = Planner(lambda: self.cfgs.config, self.store, self.collector, self.ha)
        self.trainer = Trainer(self.planner)
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
            done = (rep.get("finished") or 0) / 1000
            hour = datetime.now(self.planner.tz()).hour
            if self.ha.connected.is_set() and hour == cfg.retrain_hour and time.time() - done > 20 * 3600:
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
            asyncio.create_task(self.planner.replan())


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
                      "training": s.trainer.running, "error": s.trainer.error},
            "backfill": s.collector.backfill_status, "units": units, "zones": zones,
            "outdoor": p.live(cfg.mapping.get("outdoor")), "uptime_s": time.time() - s.started, "tz": str(p.tz()),
        })

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
        await svc().planner.replan()
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
            g = build_grid(parse_rows(ex), layout, cfg.mapping, start=start, min_hours=1)
        except ValueError as e:
            out["error"] = str(e)
        else:
            out.update({
                "t": g.t.astype(np.int64).tolist(),
                "temp": {z: _r(g.T[:, i], 2) for i, z in enumerate(layout.zone_ids)},
                "tout": _r(g.tout, 2),
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
            out.append({"id": eid, "name": a.get("friendly_name"), "state": st.get("state"),
                        "unit": a.get("unit_of_measurement"), "device_class": a.get("device_class")})
        return out

    @app.get("/api/model", dependencies=[Depends(auth)])
    async def model():
        s = svc()
        _, params = s.planner.model()
        return _json({"params": params.to_dict(), "trained": s.planner.params is not None, "report": s.planner.report,
                      "training": s.trainer.running, "started": s.trainer.started, "error": s.trainer.error})

    @app.post("/api/model/train", dependencies=[Depends(auth)])
    async def train(fresh: bool = False):
        s = svc()
        if s.trainer.running:
            raise HTTPException(409, "training is running")
        asyncio.create_task(s.trainer.run(use_current=not fresh))
        return {"started": True}

    @app.get("/api/schedule/preview", dependencies=[Depends(auth)])
    async def schedule_preview(days: int = 14, step_min: int = 30):
        s = svc()
        cfg = s.cfgs.config
        tz = s.planner.tz()
        layout = cfg.layout_obj()
        days = max(1, min(days, 62))
        t0 = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
        n = days * 24 * 60 // step_min
        starts = [t0 + timedelta(minutes=step_min * k) for k in range(n)]
        ends = [x + timedelta(minutes=step_min) for x in starts]
        res = resolve(starts, ends, layout.zone_ids, [u.id for u in layout.units], cfg.defaults, cfg.rule_objs())
        return {"start": int(t0.timestamp() * 1000), "step_min": step_min, "function": res.function,
                "offset": res.offset, "locked": res.locked, "source": res.source}

    @app.post("/api/import", dependencies=[Depends(auth)])
    async def import_export(export: dict):
        """Load an export from the browser snippet into the store, e.g. before the first training."""
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
