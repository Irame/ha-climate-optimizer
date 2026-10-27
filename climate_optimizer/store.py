"""What the optimizer remembers: recorded states, the model, its decisions.

States are kept in the export format (see model/dataset.py), so training
and the state estimate read the store exactly as they read an export.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path


class Store:
    def __init__(self, path: Path | str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.lock = threading.Lock()
        with self.lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS rows (eid TEXT NOT NULL, t INTEGER NOT NULL, data TEXT NOT NULL,
                                                 PRIMARY KEY (eid, t)) WITHOUT ROWID;
                CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decisions (t INTEGER NOT NULL, unit TEXT NOT NULL, data TEXT NOT NULL,
                                                      PRIMARY KEY (unit, t)) WITHOUT ROWID;
                CREATE TABLE IF NOT EXISTS plans (t INTEGER PRIMARY KEY, data TEXT NOT NULL);
            """)

    # ---- state rows: [t_ms, ...values] ----
    def add_rows(self, eid: str, rows: list[list]) -> None:
        if not rows:
            return
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO rows VALUES (?, ?, ?)",
                                [(eid, int(r[0]), json.dumps(r[1:])) for r in rows])

    def last_time(self, eid: str) -> int | None:
        with self.lock:
            r = self.db.execute("SELECT MAX(t) FROM rows WHERE eid = ?", (eid,)).fetchone()
        return r[0] if r and r[0] is not None else None

    def first_time(self, eid: str) -> int | None:
        with self.lock:
            r = self.db.execute("SELECT MIN(t) FROM rows WHERE eid = ?", (eid,)).fetchone()
        return r[0] if r and r[0] is not None else None

    def rows(self, eid: str, start: int, end: int, before: bool = True) -> list[list]:
        """Rows in [start, end), and the last one before start (the state at start)."""
        with self.lock:
            out = []
            if before:
                r = self.db.execute("SELECT t, data FROM rows WHERE eid = ? AND t < ? ORDER BY t DESC LIMIT 1",
                                    (eid, start)).fetchone()
                if r:
                    out.append(r)
            out += self.db.execute("SELECT t, data FROM rows WHERE eid = ? AND t >= ? AND t < ? ORDER BY t",
                                   (eid, start, end)).fetchall()
        return [[t] + json.loads(d) for t, d in out]

    def export(self, eids: list[str], start: int, end: int, meta: dict, location: dict | None) -> dict:
        history = {}
        for e in sorted(set(eids)):
            r = self.rows(e, start, end)
            if r:
                history[e] = r
        return {"meta": {e: meta.get(e, {}) for e in eids}, "location": location, "stats_5min": {}, "history": history}

    def prune(self, before: int) -> None:
        with self.lock:
            self.db.execute("DELETE FROM rows WHERE t < ?", (before,))
            self.db.execute("DELETE FROM decisions WHERE t < ?", (before,))
            self.db.execute("DELETE FROM plans WHERE t < ?", (before,))

    # ---- key/value ----
    def get(self, k: str, default=None):
        with self.lock:
            r = self.db.execute("SELECT v FROM kv WHERE k = ?", (k,)).fetchone()
        return json.loads(r[0]) if r else default

    def put(self, k: str, v) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (k, json.dumps(v)))

    # ---- decisions and plans ----
    def add_decision(self, t: int, unit: str, data: dict) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO decisions VALUES (?, ?, ?)", (int(t), unit, json.dumps(data)))

    def decisions(self, start: int, end: int) -> dict[str, list]:
        with self.lock:
            rows = self.db.execute("SELECT unit, t, data FROM decisions WHERE t >= ? AND t < ? ORDER BY t",
                                   (start, end)).fetchall()
        out: dict[str, list] = {}
        for u, t, d in rows:
            out.setdefault(u, []).append({"t": t, **json.loads(d)})
        return out

    def add_plan(self, t: int, data: dict) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO plans VALUES (?, ?)", (int(t), json.dumps(data)))

    def plans(self, start: int, end: int) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT t, data FROM plans WHERE t >= ? AND t < ? ORDER BY t", (start, end)).fetchall()
        return [{"t": t, **json.loads(d)} for t, d in rows]
