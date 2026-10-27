"""Training in a separate process, so planning and the proxies' questions
are answered while it runs."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .collector import tracked_entities

log = logging.getLogger(__name__)


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def train_job(export: dict, layout: dict, mapping: dict, initial: dict | None) -> dict:
    from .model.dataset import build_grid, parse_rows
    from .model.fit import train
    from .model.layout import Layout, Params

    lay = Layout.from_dict(layout)
    grid = build_grid(parse_rows(export), lay, mapping)
    res = train(lay, grid, Params.from_dict(initial) if initial else None)
    return {"params": res["params"].to_dict(), "report": _plain(res["report"])}


class Trainer:
    def __init__(self, planner):
        self.planner = planner
        self.running = False
        self.started: float | None = None
        self.error: str | None = None
        self._pool: ProcessPoolExecutor | None = None

    async def run(self, use_current: bool = True) -> None:
        if self.running:
            return
        self.running, self.started, self.error = True, time.time(), None
        p = self.planner
        cfg = p.cfg
        try:
            now = int(time.time() * 1000)
            ents = tracked_entities(cfg.mapping)
            export = await asyncio.to_thread(p.store.export, ents, int(now - cfg.history_days * 864e5), now,
                                             p.collector.meta, {**p.location(), "tz": str(p.tz())})
            if self._pool is None:
                self._pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
            initial = p.params.to_dict() if (use_current and p.params) else None
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(self._pool, train_job, export, cfg.layout, cfg.mapping, initial)
            from .model.layout import Params
            p.params = Params.from_dict(res["params"])
            p.report = {**res["report"], "finished": time.time() * 1000}
            p.store.put("params", res["params"])
            p.store.put("report", p.report)
            log.info("trained: rmse %s", res["report"].get("rmse_val"))
        except Exception as e:
            log.exception("training failed")
            self.error = str(e)
        finally:
            self.running = False

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
