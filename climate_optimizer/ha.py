"""A small Home Assistant WebSocket client: commands, events, reconnects."""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp

log = logging.getLogger(__name__)


class HAError(Exception):
    pass


class HomeAssistant:
    def __init__(self, url: str, token: str, verify_ssl: bool = True):
        if not url and os.environ.get("SUPERVISOR_TOKEN"):  # running as an add-on
            url, token = "http://supervisor/core", os.environ["SUPERVISOR_TOKEN"]
        self.url = url.rstrip("/")
        self.token = token
        self.verify_ssl = verify_ssl
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._subs: dict[int, Callable[[dict], None]] = {}
        self.connected = asyncio.Event()
        self.error: str | None = None
        self.version: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.url and self.token)

    async def run(self, on_connect: Callable[[], Awaitable[None]], stop: asyncio.Event) -> None:
        """Stay connected until stop is set; on_connect runs after every (re)connect."""
        delay = 2.0
        while not stop.is_set():
            if not self.configured:
                self.error = "not configured"
                await _wait(stop, 5)
                continue
            try:
                await self._connect()
                self.error = None
                delay = 2.0
                reader = asyncio.create_task(self._read())
                try:
                    await on_connect()
                except Exception as e:  # keep the connection, report the problem
                    log.exception("after connecting")
                    self.error = f"setup after connecting failed: {e}"
                stopper = asyncio.create_task(stop.wait())
                await asyncio.wait([reader, stopper], return_when=asyncio.FIRST_COMPLETED)
                for t in (reader, stopper):
                    t.cancel()
            except (aiohttp.ClientError, OSError, HAError, asyncio.TimeoutError) as e:
                self.error = str(e) or e.__class__.__name__
                log.warning("Home Assistant: %s", self.error)
            finally:
                await self._close()
            if not stop.is_set():
                await _wait(stop, delay)
                delay = min(delay * 2, 60.0)

    async def _connect(self) -> None:
        self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=None if self.verify_ssl else False))
        ws_url = self.url.replace("http", "ws", 1) + "/api/websocket"
        self._ws = await self._session.ws_connect(ws_url, heartbeat=30, max_msg_size=0, timeout=aiohttp.ClientWSTimeout(ws_close=10))
        hello = await self._ws.receive_json(timeout=10)
        if hello.get("type") != "auth_required":
            raise HAError(f"unexpected greeting {hello.get('type')}")
        await self._ws.send_json({"type": "auth", "access_token": self.token})
        res = await self._ws.receive_json(timeout=10)
        if res.get("type") != "auth_ok":
            raise HAError("authentication failed: check the access token")
        self.version = res.get("ha_version")
        self.connected.set()
        log.info("connected to Home Assistant %s", self.version)

    async def _close(self) -> None:
        self.connected.clear()
        for f in self._pending.values():
            if not f.done():
                f.set_exception(HAError("connection lost"))
        self._pending.clear()
        self._subs.clear()
        if self._ws is not None:
            await self._ws.close()
        if self._session is not None:
            await self._session.close()
        self._ws = self._session = None

    async def _read(self) -> None:
        assert self._ws is not None
        async for msg in self._ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
                continue
            data = msg.json()
            for m in data if isinstance(data, list) else [data]:
                self._dispatch(m)

    def _dispatch(self, m: dict) -> None:
        mid = m.get("id")
        if m.get("type") == "event" and mid in self._subs:
            try:
                self._subs[mid](m["event"])
            except Exception:
                log.exception("event handler")
            return
        f = self._pending.pop(mid, None)
        if f is None or f.done():
            return
        if m.get("success", True):
            f.set_result(m.get("result"))
        else:
            err = m.get("error") or {}
            f.set_exception(HAError(f"{err.get('code')}: {err.get('message')}"))

    async def call(self, payload: dict, timeout: float = 60.0) -> Any:
        if self._ws is None or not self.connected.is_set():
            raise HAError("not connected")
        mid = next(self._ids)
        fut = asyncio.get_running_loop().create_future()
        self._pending[mid] = fut
        await self._ws.send_json({**payload, "id": mid})
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(mid, None)

    async def subscribe(self, event_type: str, handler: Callable[[dict], None]) -> None:
        if self._ws is None:
            raise HAError("not connected")
        mid = next(self._ids)
        fut = asyncio.get_running_loop().create_future()
        self._pending[mid] = fut
        self._subs[mid] = handler
        await self._ws.send_json({"id": mid, "type": "subscribe_events", "event_type": event_type})
        await asyncio.wait_for(fut, 30)

    # ---- the calls the optimizer uses ----
    async def states(self) -> list[dict]:
        return await self.call({"type": "get_states"})

    async def ha_config(self) -> dict:
        return await self.call({"type": "get_config"})

    async def service(self, domain: str, service: str, data: dict | None = None, target: dict | None = None,
                      response: bool = False) -> Any:
        msg = {"type": "call_service", "domain": domain, "service": service, "service_data": data or {}}
        if target:
            msg["target"] = target
        if response:
            msg["return_response"] = True
        res = await self.call(msg)
        return (res or {}).get("response") if response else res

    async def history(self, start_iso: str, end_iso: str, ids: list[str], attributes: bool) -> dict:
        """{entity_id: [{"s": state, "a": attributes, "lu": seconds, ...}]}, the compressed form."""
        if not ids:
            return {}
        return await self.call({
            "type": "history/history_during_period", "start_time": start_iso, "end_time": end_iso,
            "entity_ids": ids, "include_start_time_state": True, "significant_changes_only": False,
            "minimal_response": not attributes, "no_attributes": not attributes,
        }, timeout=300)

    async def statistics(self, start_iso: str, end_iso: str, ids: list[str], period: str = "5minute",
                         types: tuple[str, ...] = ("mean",)) -> dict:
        if not ids:
            return {}
        return await self.call({
            "type": "recorder/statistics_during_period", "start_time": start_iso, "end_time": end_iso,
            "statistic_ids": ids, "period": period, "types": list(types),
        }, timeout=300)


async def _wait(ev: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(ev.wait(), seconds)
    except asyncio.TimeoutError:
        pass
