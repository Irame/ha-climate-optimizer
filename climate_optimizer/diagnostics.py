"""Everything needed to look at the model from outside, in one file.

The recorded history the training reads (numeric sensors as 5-minute means
to keep it small), the settings without secrets, the trained model, its
report, and what the last days' plans predicted. The file is an export in
the format model/dataset.py reads, so it can be trained on again as it is.

From the container: python -m climate_optimizer.diagnostics --out /data/diagnostics.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .collector import kind_of, tracked_entities
from .model.dataset import step_means

MIN = 60_000
STEP = 5 * MIN


def _round_location(loc: dict | None) -> dict | None:
    if not loc:
        return loc
    return {**loc, **{k: round(float(loc[k]), 1) for k in ("lat", "lon", "latitude", "longitude") if loc.get(k) is not None}}


def _five_minute(rows: list[list], start: int, end: int) -> list[list]:
    """A numeric step series as 5-minute means, the way recorder statistics look."""
    if not rows:
        return []
    t = np.array([r[0] for r in rows], dtype=float)
    v = np.array([np.nan if r[1] is None else float(r[1]) for r in rows], dtype=float)
    t0 = int(np.ceil(start / STEP) * STEP)
    K = int((end - t0) // STEP)
    if K <= 0:
        return []
    m = step_means(t, v, t0, STEP, K, max_age=6 * 3600e3)
    return [[t0 + k * STEP, round(float(x), 4)] for k, x in enumerate(m) if np.isfinite(x)]


def collect(config, store, meta: dict, days: float = 30, plans_days: float = 3) -> dict:
    now = int(time.time() * 1000)
    start = int(now - days * 864e5)
    cfg = config.model_dump()
    cfg["ha"] = {**cfg["ha"], "url": "(removed)" if cfg["ha"].get("url") else "", "token": ""}
    cfg["api_token"] = ""
    cfg["location"] = _round_location(cfg.get("location"))
    ha_config = store.get("ha_config") or {}
    location = cfg["location"] or _round_location({"lat": ha_config.get("latitude"), "lon": ha_config.get("longitude")})
    stats, history = {}, {}
    for eid in tracked_entities(config.mapping):
        rows = store.rows(eid, start, now)
        if not rows:
            continue
        if kind_of(eid) == "numeric":
            stats[eid] = _five_minute(rows, start, now)
        else:
            history[eid] = rows
    p0 = int(now - plans_days * 864e5)
    return {
        "kind": "climate-optimizer diagnostics", "version": 1, "exported": now,
        "location": {**(location or {}), "tz": ha_config.get("time_zone")},
        "meta": {e: meta.get(e, {}) for e in set(stats) | set(history)},
        "stats_5min": stats, "history": history,
        "config": cfg, "params": store.get("params"), "report": store.get("report"),
        "fan_modes": store.get("fan_modes"), "price_history": store.get("price_history"),
        "plans": store.plans(p0, now), "decisions": store.decisions(p0, now),
    }


def to_gzip(d: dict) -> bytes:
    return gzip.compress(json.dumps(d, separators=(",", ":")).encode(), compresslevel=6)


def main() -> None:
    from .config import DATA, ConfigStore
    from .store import Store

    ap = argparse.ArgumentParser(description="Write the optimizer's diagnostics file.")
    ap.add_argument("--days", type=float, default=30, help="days of history (default 30)")
    ap.add_argument("--out", default="-", help="file to write, - for stdout (default)")
    ap.add_argument("--data", default=str(DATA), help="the data directory (default DATA_DIR or /data)")
    a = ap.parse_args()
    data = Path(a.data)
    if not (data / "optimizer.db").exists():
        sys.exit(f"no optimizer.db in {data}")
    if os.geteuid() == 0:  # docker exec runs as root: become the data's owner, so nothing in it ends up root's
        st = data.stat()
        os.setgid(st.st_gid)
        os.setuid(st.st_uid)
    store = Store(data / "optimizer.db")
    blob = to_gzip(collect(ConfigStore(data / "config.json").config, store, store.get("meta", {}), a.days))
    if a.out == "-":
        sys.stdout.buffer.write(blob)
    else:
        Path(a.out).write_bytes(blob)
        print(f"wrote {a.out}, {len(blob) / 1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
