"""The service end to end against the simulated apartment."""

import json
from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

from climate_optimizer.app import create_app
from climate_optimizer.demo import demo_service
from climate_optimizer.model.layout import Params, default_layout, default_params


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    svc = demo_service(tmp_path_factory.mktemp("data"))
    # skip the first training: the defaults are good enough to plan with
    p = default_params(default_layout())
    svc.store.put("params", p.to_dict())
    svc.planner.params = p
    with TestClient(create_app(service=svc)) as c:
        c.svc = svc
        yield c


def test_status_and_plan(client):
    st = client.get("/api/status").json()
    assert st["ha"]["connected"]
    assert set(st["units"]) == {"office", "bedroom", "living", "bath"}
    r = client.post("/api/plan").json()
    plan = r["plan"]
    assert plan["mode"] in ("heat", "cool")
    assert len(plan["t"]) == len(plan["price"]) + 1
    assert set(plan["temp"]) == {"office", "bedroom", "living", "bath", "hall"}
    assert r["info"]["units"]["office"]["proxy"] == "climate.office_proxy"


def test_command_answers_the_proxy(client):
    client.post("/api/plan")
    body = client.svc.ha.world.request("living")
    a = client.post("/api/v1/command", json=body).json()
    assert a.get("hvac_mode") in ("heat", "cool", "off")
    assert "valid_until" in a and a["reason"]
    if a["hvac_mode"] != "off":
        assert 0 <= a["fan_level"] <= 3
    unknown = client.post("/api/v1/command", json={"entity_id": "climate.nobody"}).json()
    assert unknown["follow"] is False


def test_shadow_mode_defers(client):
    client.put("/api/config", json={"shadow": True})
    client.post("/api/plan")
    a = client.post("/api/v1/command", json=client.svc.ha.world.request("office")).json()
    assert a["follow"] is False and a["reason"].startswith("shadow")
    client.put("/api/config", json={"shadow": False})


def test_config_validation_and_secrets(client):
    r = client.put("/api/config", json={"rules": [{"id": "x", "name": "x", "action": "comfort", "function": "nope"}]})
    assert r.status_code == 422
    client.put("/api/config", json={"ha": {"token": "secret"}})
    assert client.get("/api/config").json()["ha"]["token"] == "********"
    client.put("/api/config", json={"ha": {"token": "********"}})  # masked value keeps the stored one
    assert client.svc.cfgs.config.ha.token == "secret"


def test_api_token(client):
    client.put("/api/config", json={"api_token": "tok"})
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", headers={"Authorization": "Bearer tok"}).status_code == 200
    client.put("/api/config", json={"api_token": ""}, headers={"Authorization": "Bearer tok"})
    assert client.get("/api/status").status_code == 200


def test_schedule_preview_and_history(client):
    pv = client.get("/api/schedule/preview?days=7").json()
    assert len(pv["function"]["living"]) == 7 * 48
    assert "relaxed" in pv["function"]["living"]
    h = client.get("/api/history?hours=24").json()
    assert "error" not in h, h.get("error")
    assert len(h["t"]) > 200 and set(h["temp"]) >= {"office", "living"}
    assert h["prices"]


def test_ui_is_served(client):
    assert "Climate optimizer" in client.get("/").text
    r = client.get("/static/app.js")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    assert client.get("/").headers["cache-control"] == "no-cache"
    assert "cache-control" not in client.get("/healthz").headers


def test_fan_modes_come_from_the_proxy(client):
    svc = client.svc
    old = svc.cfgs.config.mapping
    m = {**old, "units": {u: {k: v for k, v in d.items() if k != "fan_modes"} for u, d in old["units"].items()}}
    client.put("/api/config", json={"mapping": m})
    body = {**svc.ha.world.request("office"), "source_fan_modes": ["silent", "low", "high", "turbo"]}
    client.post("/api/v1/command", json=body)
    assert svc.planner.mapping()["units"]["office"]["fan_modes"] == ["silent", "low", "high", "turbo"]
    client.put("/api/config", json={"mapping": old})


def test_diagnostics_roundtrip(client, tmp_path):
    import gzip
    import json

    from climate_optimizer.model.dataset import build_grid, parse_rows

    client.put("/api/config", json={"ha": {"token": "secret"}, "api_token": ""})
    r = client.get("/api/diagnostics?days=3")
    assert r.status_code == 200 and r.headers["content-type"] == "application/gzip"
    d = json.loads(gzip.decompress(r.content))
    assert d["kind"] == "climate-optimizer diagnostics"
    assert "secret" not in json.dumps(d)
    assert d["config"]["ha"]["token"] == "" and d["params"]
    # it is an export: it builds the training grid as it is, and loads back in
    svc = client.svc
    g = build_grid(parse_rows(d), svc.cfgs.config.layout_obj(), d["config"]["mapping"], min_hours=24)
    assert g.K > 500 and g.valid.mean() > 0.9
    n = client.post("/api/import", content=r.content).json()["rows"]
    assert n > 1000


def test_diagnostics_cli(client, tmp_path, monkeypatch):
    import gzip
    import json
    import sys

    from climate_optimizer import diagnostics

    out = tmp_path / "d.json.gz"
    monkeypatch.setattr(sys, "argv", ["x", "--data", str(client.svc.store_path.parent), "--days", "2", "--out", str(out)])
    monkeypatch.setattr(diagnostics.os, "geteuid", lambda: 1000)
    diagnostics.main()
    d = json.loads(gzip.decompress(out.read_bytes()))
    assert d["stats_5min"] and d["history"]


def test_training_progress(client, tmp_path):
    from climate_optimizer.trainer import write_progress

    t = client.svc.trainer
    write_progress(t.progress_path, {"stage": "fit", "evaluations": 50, "budget": 200})
    assert t.progress() is None  # nothing running
    t.running, t.started = True, 0.0
    try:
        p = client.get("/api/model").json()["progress"]
        assert p["stage"] == "fit" and p["evaluations"] == 50 and p["budget"] == 200
    finally:
        t.running = False


def test_a_room_whose_units_are_off_costs_no_comfort(client):
    """Pricing it would only have the plan heat the neighbours to warm it."""
    svc = client.svc
    world = svc.ha.world
    world.units["bath"]["proxy_mode"] = "off"
    try:
        for s in world.states():  # let the collector see it
            svc.collector.states[s["entity_id"]] = s
        r = client.post("/api/plan").json()
        assert "bath" not in r["plan"]["comfort"]
        assert r["info"]["no_comfort"] == {"bath": "proxy is off"}
        assert set(r["plan"]["comfort_cost"]) <= {"office", "bedroom", "living"}
    finally:
        world.units["bath"]["proxy_mode"] = "auto"
        for s in world.states():
            svc.collector.states[s["entity_id"]] = s


def test_environment_wins_over_saved_settings(tmp_path, monkeypatch):
    from climate_optimizer.config import ConfigStore

    cs = ConfigStore(tmp_path / "config.json")
    cs.update({"ha": {"url": "http://saved:8123", "token": "saved"}})
    monkeypatch.setenv("HA_URL", "http://host.docker.internal:8123")
    cs = ConfigStore(tmp_path / "config.json")
    assert cs.config.ha.url == "http://host.docker.internal:8123" and cs.config.ha.token == "saved"
    cs.update({"ha": {"url": "http://ui:8123"}})  # the UI cannot override it
    assert cs.config.ha.url == "http://host.docker.internal:8123"
    assert cs.config.public()["env"] == {"ha_url": True, "ha_token": False}


def test_shadow_mode_plans_units_and_comfort_as_set(client):
    """Watching beside another automation: plan a unit whose proxy is off, with our own comfort temperature."""
    svc = client.svc
    world = svc.ha.world
    world.units["bath"]["proxy_mode"] = "off"
    for s in world.states():
        svc.collector.states[s["entity_id"]] = s
    try:
        client.put("/api/config", json={"shadow": True, "shadow_units": ["bath"], "shadow_comfort": {"bath": 21.5, "living": 19.0}})
        r = client.post("/api/plan").json()
        assert r["plan"]["comfort"]["bath"][0] == 21.5
        assert r["plan"]["comfort"]["living"][-1] == 19.0  # the living room's proxy says 21.5
        assert r["info"]["units"]["bath"]["anyway"] and r["info"]["units"]["bath"]["note"] is None
        assert "no_comfort" not in r["info"]
        # outside shadow mode the proxies decide again
        client.put("/api/config", json={"shadow": False})
        r = client.post("/api/plan").json()
        assert "bath" not in r["plan"]["comfort"] and r["plan"]["comfort"]["living"][-1] == 21.5
    finally:
        client.put("/api/config", json={"shadow": False, "shadow_units": [], "shadow_comfort": {}})
        world.units["bath"]["proxy_mode"] = "auto"
        for s in world.states():
            svc.collector.states[s["entity_id"]] = s


def test_preview_shows_the_calendar(client):
    svc = client.svc
    client.put("/api/config", json={"calendar": "calendar.klima"})
    pv = client.get("/api/schedule/preview?days=7&refresh=true").json()
    cal = pv["calendar"]
    assert cal["entity"] == "calendar.klima" and cal["error"] is None and cal["fetched"]
    by = {e["summary"]: e for e in cal["events"]}
    assert by["Wohnzimmer: Strict"]["understood"] and by["Wohnzimmer: Strict"]["effect"] == "Strict in Living room & kitchen"
    assert by["Schlafzimmer: aus"]["effect"] == "off: Bedroom"
    assert not by["Zahnarzt"]["understood"]
    # tomorrow 18:00-23:00 the living room is strict, from the calendar
    k = (24 + 19) * 2
    assert pv["function"]["living"][k] == "strict" and pv["event_zone"]["living"][k] and pv["source"]["living"][k] == "Wohnzimmer: Strict"
    k = (3 * 24 + 14) * 2
    assert pv["locked"]["bedroom"][k] and pv["event_unit"]["bedroom"][k]
    # without refresh the cached calendar is used
    t = cal["fetched"]
    assert client.get("/api/schedule/preview?days=7").json()["calendar"]["fetched"] == t
    client.put("/api/config", json={"calendar": ""})
    assert client.get("/api/schedule/preview?days=7").json()["calendar"]["events"] == []


def test_airing_is_planned_and_published(client):
    svc = client.svc
    world = svc.ha.world
    client.put("/api/config", json={"airing": {"enabled": True, "duration_min": 10, "per_day": 1}})
    try:
        r = client.post("/api/plan").json()
        plan, info = r["plan"], r["info"]
        assert info["airing"]["per_day"] == 1 and "done_today" in info["airing"]
        assert plan["airing"], info["airing"]
        a0, a1 = plan["airing"][0]
        assert a1 - a0 == 10 * 60_000 and a0 % (30 * 60_000) == 0
        # never at night, by the default rule, and the units in the window rooms stay off then
        tz = svc.planner.tz()
        for s, _ in plan["airing"]:
            assert 7 <= datetime.fromtimestamp(s / 1000, tz).hour < 22
        state, attrs = world.extra["sensor.climate_optimizer_airing"]
        assert datetime.fromisoformat(state).timestamp() * 1000 == a0
        assert attrs["device_class"] == "timestamp" and attrs["duration_min"] == 10
        assert set(attrs["rooms"]) == {"Office", "Bedroom", "Living room & kitchen"}
        assert info["airing"]["suggested"] == [a0, a1]
        # 'Lüften: aus' in the calendar is understood
        client.put("/api/config", json={"calendar": "calendar.klima"})
        pv = client.get("/api/schedule/preview?days=3&refresh=true").json()
        ev = next(e for e in pv["calendar"]["events"] if e["summary"] == "Lüften: aus")
        assert ev["effect"] == "no airing suggested"
        assert pv["no_airing"][(48 + 12) * 2] and pv["no_airing"][23 * 2] and not pv["no_airing"][(24 + 12) * 2]
    finally:
        client.put("/api/config", json={"airing": {"enabled": False}, "calendar": ""})
    client.post("/api/plan")
    assert world.extra["sensor.climate_optimizer_airing"][0] == "unknown"


def test_airing_done_today_counts_the_windows(client):
    from climate_optimizer.config import AiringConfig

    svc = client.svc
    p = svc.planner
    tz = p.tz()
    now = datetime.now(tz)
    if now.hour < 2:
        pytest.skip("needs a couple of hours of today")
    t0 = int(now.replace(hour=0, minute=30, second=0, microsecond=0).timestamp() * 1000)
    M = 60_000
    cfg = p.cfg
    old = {z: cfg.mapping["zones"][z].get("windows") for z in ("office", "bedroom", "living")}
    for z in old:
        cfg.mapping["zones"][z]["windows"] = [f"binary_sensor.test_window_{z}"]
    rows = {
        # 0:30-0:42 office and bedroom open: two of three rooms, an airing
        "office": [[t0, "on"], [t0 + 12 * M, "off"],
                   # 1:00-1:30 the office alone: a tilted window, not one
                   [t0 + 30 * M, "on"], [t0 + 60 * M, "off"]],
        "bedroom": [[t0 + 2 * M, "on"], [t0 + 10 * M, "off"]],
        "living": [[t0 - 3600 * 1000, "off"]],
    }
    for z, r in rows.items():
        svc.store.add_rows(f"binary_sensor.test_window_{z}", r)
    old_air = cfg.airing
    cfg.airing = AiringConfig(enabled=True, duration_min=10)
    try:
        done, last = p.airing_done_today(now.timestamp() * 1000, tz)
        assert done == 1 and last == t0 + 12 * M
        p.airing_now = (int(now.timestamp() * 1000) - M, int(now.timestamp() * 1000) + 9 * M)
        done, last = p.airing_done_today(now.timestamp() * 1000, tz)
        assert done == 2 and last == p.airing_now[1]  # one under way counts
    finally:
        for z, w in old.items():
            cfg.mapping["zones"][z]["windows"] = w
        cfg.airing = old_air
        p.airing_now = None


def _see(svc):
    for s in svc.ha.world.states():
        svc.collector.states[s["entity_id"]] = s


def test_room_override_until_reached(client):
    svc = client.svc
    world = svc.ha.world
    zi = svc.planner.cfg.layout_obj().zone_index("living")
    world.x[zi] = 20.0
    _see(svc)
    o = client.post("/api/overrides", json={"zone": "Wohnzimmer", "target": 23}).json()
    assert o["zone"] == "living" and o["direction"] == "up" and o["ends_at"] is None
    assert [x["id"] for x in client.get("/api/overrides").json()] == [o["id"]]
    r = client.post("/api/plan").json()
    assert r["plan"]["comfort"]["living"][0] == 23.0
    assert r["info"]["schedule"]["function"]["living"][0] == "override-up"
    t = r["plan"]["t"]
    soon = [v for x, v in zip(t, r["plan"]["share"]["living"]) if x < t[0] + 15 * 60_000]
    assert max(soon) > 0.5  # heats at once (after the compressor's pause), whatever the price
    assert world.extra["sensor.climate_optimizer_override_living"][0] == "23"
    # reached: back to the plan as it was
    world.x[zi] = 23.0
    _see(svc)
    r = client.post("/api/plan").json()
    assert client.get("/api/overrides").json() == []
    assert r["plan"]["comfort"]["living"][0] != 23.0 or r["info"]["schedule"]["function"]["living"][0] != "override-up"
    assert world.extra["sensor.climate_optimizer_override_living"][0] == "off"


def test_room_override_by_a_time_and_held(client):
    import time as _t

    svc = client.svc
    now = _t.time() * 1000
    at = now + 3 * 3.6e6
    o = client.post("/api/overrides", json={"zone": "office", "target": 22.5, "at": int(at), "hold_min": 60}).json()
    assert o["ready_at"] == int(at)
    r = client.post("/api/plan").json()
    t = r["plan"]["t"]
    k = next(i for i, x in enumerate(t) if x >= at + 15 * 60_000)
    assert r["plan"]["comfort"]["office"][0] != 22.5 and r["plan"]["comfort"]["office"][k] == 22.5
    # reached and held: it stays an hour, then goes
    ov = svc.planner.overrides.items[0]
    ov.ready_at, ov.reached = int(now - 2 * 3.6e6), int(now - 30 * 60_000)
    assert svc.planner.overrides.tick(now, {}) == []
    ov.reached = int(now - 61 * 60_000)
    assert [why for _, why in svc.planner.overrides.tick(now, {})] == ["held as long as asked"]
    client.delete("/api/overrides?zone=office")


def test_run_override_starts_the_heater_at_once(client):
    svc = client.svc
    world = svc.ha.world
    world.units["bath"]["on"] = False
    world.pokes.clear()
    o = client.post("/api/overrides", json={"unit": "climate.bath_proxy", "run_min": 15, "toggle": True}).json()
    assert o["unit"] == "bath" and o["end"] - o["created"] == 15 * 60_000
    assert ["climate.bath_proxy"] in world.pokes  # told to ask now ...
    assert world.units["bath"]["on"] and world.units["bath"]["answer"]["force"]  # ... and running
    a = client.post("/api/v1/command", json=world.request("bath")).json()
    assert a["hvac_mode"] == "heat" and a["force"] and a["reason"].startswith("asked to run until")
    r = client.post("/api/plan").json()
    assert r["plan"]["share"]["bath"][0] > 0.99
    assert world.extra["sensor.climate_optimizer_run_bath"][0] == "on"
    # pressed again: over
    assert client.post("/api/overrides", json={"unit": "bath", "toggle": True}).json() == {"cancelled": True, "unit": "bath"}
    assert svc.planner.overrides.run_for("bath", o["created"] + 60_000) is None
    # a run that is over goes by itself
    o = client.post("/api/overrides", json={"unit": "bath", "run_min": 15}).json()
    svc.planner.overrides.items[-1].end = o["created"] - 1
    client.post("/api/plan")
    assert client.get("/api/overrides").json() == []
    assert world.extra["sensor.climate_optimizer_run_bath"][0] == "off"


def test_override_validation(client):
    assert client.post("/api/overrides", json={"zone": "Keller", "target": 22}).status_code == 422
    assert client.post("/api/overrides", json={"zone": "living", "target": 40}).status_code == 422
    assert client.post("/api/overrides", json={"zone": "living"}).status_code == 422
    assert client.post("/api/overrides", json={"unit": "nothing", "run_min": 15}).status_code == 422
    assert client.post("/api/overrides", json={"unit": "bath", "run_min": 999}).status_code == 422


def test_calendar_temperature_event(client):
    client.put("/api/config", json={"calendar": "calendar.klima"})
    try:
        pv = client.get("/api/schedule/preview?days=7&refresh=true").json()
        ev = next(e for e in pv["calendar"]["events"] if e["summary"] == "Büro: 22,5 °C")
        assert ev["effect"] == "22.5 °C in Office"
        k = (4 * 24 + 8) * 2
        assert pv["target"]["office"][k] == 22.5 and pv["event_zone"]["office"][k]
        assert pv["target"]["office"][k - 4] is None
    finally:
        client.put("/api/config", json={"calendar": ""})


def test_energy_per_day_planned_and_measured(client):
    svc = client.svc
    svc.store.put("forecasts", {})
    client.post("/api/plan")
    fc = svc.store.get("forecasts")
    assert len(fc) == 1
    (day, f), = fc.items()
    assert f["hours"] and all(v >= 0 for v in f["hours"].values())
    client.post("/api/plan")  # the day's first plan stays the one kept
    assert svc.store.get("forecasts")[day]["made"] == f["made"]
    days = client.get("/api/energy/days?days=3").json()
    assert [d["date"] for d in days][-1] == day and len(days) == 3
    today = days[-1]
    assert today["forecast_made"] == f["made"] and today["planned_day_kwh"] > 0
    assert today["compared_hours"] == 0 and today["planned_kwh"] is None  # the plan's hours have not come yet
    assert 23 <= len(today["hours"]) <= 25
    assert days[0]["planned_kwh"] is None and days[0]["measured_day_kwh"] > 0  # the demo's history: measured only
    # an hour that has come and gone, planned and measured, is compared
    t = days[0]["hours"][5][0]
    svc.store.put("forecasts", {**svc.store.get("forecasts"), days[0]["date"]: {"made": t - 1, "hours": {str(t): 0.5}, "eur": 0.1}})
    d0 = client.get("/api/energy/days?days=3").json()[0]
    assert d0["compared_hours"] == 1 and d0["planned_kwh"] == 0.5 and d0["measured_kwh"] is not None


def test_retrain_due():
    from zoneinfo import ZoneInfo

    from climate_optimizer.app import retrain_due

    tz = ZoneInfo("Europe/Berlin")
    at3 = datetime(2026, 10, 31, 3, 5, tzinfo=tz)
    evening = datetime(2026, 10, 30, 21, 28, tzinfo=tz).timestamp()
    assert retrain_due(evening, at3, 3)  # a training by hand in the evening does not skip the night's
    assert not retrain_due(datetime(2026, 10, 31, 3, 1, tzinfo=tz).timestamp(), at3, 3)  # done this hour already
    assert not retrain_due(evening, at3.replace(hour=4), 3)


def test_a_run_stopped_by_hand_goes_off_at_once(client):
    """Stopped: plan without it, then tell the proxy, which stops past its own minimum cycle."""
    import time as _t

    svc = client.svc
    world = svc.ha.world
    zi = svc.planner.cfg.layout_obj().zone_index("bath")
    world.x[zi] = 25.0  # warm: once the run is over, the plan has the heater off
    _see(svc)
    client.post("/api/overrides", json={"unit": "bath", "run_min": 15})
    assert world.units["bath"]["on"]
    world.pokes.clear()
    assert client.post("/api/overrides", json={"unit": "bath", "toggle": True}).json()["cancelled"]
    for _ in range(100):  # the plan and the poke come in the background
        if world.pokes:
            break
        _t.sleep(0.1)
    assert ["climate.bath_proxy"] in world.pokes
    ans = world.units["bath"]["answer"]
    assert ans["hvac_mode"] == "off" and ans["force"] and not world.units["bath"]["on"]


def test_quiet_rule_caps_the_answer(client):
    svc = client.svc
    world = svc.ha.world
    rules = client.get("/api/config").json()["rules"]
    quiet = {"id": "quiet", "name": "quiet", "action": "max_level", "units": ["living"], "level": 0, "start": "00:00", "end": "24:00"}
    client.put("/api/config", json={"rules": rules + [quiet]})
    zi = svc.planner.cfg.layout_obj().zone_index("living")
    world.x[zi] = 17.0  # cold: it would run high
    _see(svc)
    try:
        r = client.post("/api/plan").json()
        assert all(c == 0 for c in r["info"]["schedule"]["max_level"]["living"])
        a = client.post("/api/v1/command", json=world.request("living")).json()
        if a.get("hvac_mode") not in (None, "off"):
            assert a["fan_level"] == 0
        pv = client.get("/api/schedule/preview?days=1").json()
        assert pv["max_level"]["living"][0] == 0 and pv["max_level"]["office"][0] is None
    finally:
        client.put("/api/config", json={"rules": rules})


def test_airing_can_be_skipped(client):
    import time as _t

    svc = client.svc
    world = svc.ha.world
    client.put("/api/config", json={"airing": {"enabled": True}})
    svc.planner.airing_done_today = lambda *a: (0, None)
    try:
        r = client.post("/api/plan").json()
        first = r["plan"]["airing"][0] if r["plan"]["airing"] else None
        out = client.post("/api/airing/skip", json={"skip": "2h"}).json()
        now = _t.time() * 1000
        assert abs(out["skipped_until"] - (now + 2 * 3.6e6)) < 60_000
        r = client.post("/api/plan").json()
        assert all(a >= out["skipped_until"] for a, _ in r["plan"]["airing"])
        assert r["info"]["airing"]["skipped_until"] == out["skipped_until"]
        if first and first[0] < out["skipped_until"]:
            assert world.extra["sensor.climate_optimizer_airing"][0] != datetime.fromtimestamp(first[0] / 1000).astimezone().isoformat()
        today = client.post("/api/airing/skip", json={"skip": "today"}).json()["skipped_until"]
        d = datetime.fromtimestamp(today / 1000, svc.planner.tz())
        assert (d.hour, d.minute) == (0, 0) and d.date() > datetime.now(svc.planner.tz()).date()
        r = client.post("/api/plan").json()
        assert all(datetime.fromtimestamp(a / 1000, svc.planner.tz()).date() >= d.date() for a, _ in r["plan"]["airing"])
        assert client.post("/api/airing/skip", json={"skip": "soon"}).status_code == 422
        assert client.post("/api/airing/skip", json={"skip": ""}).json() == {"skipped_until": None}
        assert "skipped_until" not in client.post("/api/plan").json()["info"]["airing"]
    finally:
        del svc.planner.airing_done_today
        client.put("/api/config", json={"airing": {"enabled": False}})
        svc.store.put("airing_skip", {})


def test_close_windows_sensor(client):
    import time as _t

    svc = client.svc
    world = svc.ha.world
    eid = "binary_sensor.office_balkontuer"
    world.windows["office"] = True
    _see(svc)
    opened = svc.collector.states[eid]["last_changed"]
    try:
        client.post("/api/plan")
        state, attrs = world.extra["sensor.climate_optimizer_close_windows"]
        t_open = datetime.fromisoformat(opened.replace("Z", "+00:00")).timestamp() * 1000
        assert abs(datetime.fromisoformat(state).timestamp() * 1000 - (t_open + 10 * 60_000)) < 1000  # heating: after the airing minutes
        assert attrs["rooms"] == ["Office"] and "heat" in attrs["reason"]
        later = client.post("/api/windows/later", json={"minutes": 20}).json()["close_not_before"]
        client.post("/api/plan")
        state, _ = world.extra["sensor.climate_optimizer_close_windows"]
        assert abs(datetime.fromisoformat(state).timestamp() * 1000 - later) < 1000
        assert client.post("/api/windows/later", json={"minutes": 0}).status_code == 422
    finally:
        world.windows["office"] = False
        _see(svc)
    client.post("/api/plan")
    assert world.extra["sensor.climate_optimizer_close_windows"][0] == "unknown"
    assert not svc.store.get("close_later")


def test_venting_sensor_and_rules(client):
    from climate_optimizer.comfort import Rule

    Rule.from_dict({"id": "v", "name": "v", "action": "no_venting"})
    client.put("/api/config", json={"airing": {"venting": True}})
    try:
        r = client.post("/api/plan").json()
        assert r["plan"]["vent"] is not None and "venting" in r["info"]
        state, attrs = client.svc.ha.world.extra["sensor.climate_optimizer_venting"]
        assert state == "unknown" or "until" in attrs  # the demo heats: rarely worth opening
    finally:
        client.put("/api/config", json={"airing": {"venting": False}})


def test_a_silent_thermometer_plans_from_its_last_reading(client):
    import time as _t

    svc = client.svc
    p = svc.planner
    eid = p.cfg.mapping["zones"]["bedroom"]["thermometer"]
    now = int(_t.time() * 1000)
    svc.store.add_rows(eid, [[now - 20 * 60_000, 15.2]])
    _see(svc)
    svc.collector.states[eid] = {**svc.collector.states[eid], "state": "unavailable"}
    try:
        r = client.post("/api/plan").json()
        zi = p.cfg.layout_obj().zone_index("bedroom")
        last = p.last_reading(eid, _t.time() * 1000)
        others = [x for i, x in enumerate(r["info"]["x0"][:5]) if i != zi]
        assert r["info"]["x0"][zi] == pytest.approx(last[1]) and last[0] >= now - 20 * 60_000
        assert any("bedroom" in n.lower() and "last reading" in n for n in r["info"]["notes"])
        st = client.get("/api/status").json()["zones"]["bedroom"]
        assert st["last_reading"][1] == pytest.approx(last[1])
    finally:
        _see(svc)


def test_the_plan_knows_what_it_told_the_proxies(client):
    """A unit just told to stop is off in the next plan, its pause counted from then,
    even while the proxy still reports it heating."""
    import time as _t

    svc = client.svc
    p = svc.planner
    now = _t.time() * 1000
    p._answered["living"] = [False, now - 2 * 60_000, now - 30_000]
    try:
        r = client.post("/api/plan").json()  # fresh, no hold; the pause still holds
        t = r["plan"]["t"]
        early = [v for x, v in zip(t, r["plan"]["share"]["living"]) if x < now + 5 * 60_000]
        assert max(early) < 0.01
    finally:
        p._answered.pop("living", None)


def test_a_heater_is_off_a_moment_after_its_run(client):
    import time as _t

    svc = client.svc
    world = svc.ha.world
    zi = svc.planner.cfg.layout_obj().zone_index("bath")
    world.x[zi] = 17.0  # cold: the plan would have it on
    _see(svc)
    client.post("/api/overrides", json={"unit": "bath", "run_min": 15})
    world.pokes.clear()
    client.delete("/api/overrides?unit=bath")
    assert ["climate.bath_proxy"] in world.pokes  # at once, not after the plan
    a = client.post("/api/v1/command", json=world.request("bath")).json()
    assert a["hvac_mode"] == "off" and a["force"] and a["reason"] == "run over"
    svc.planner._run_over["bath"] = _t.time() * 1000 - 3 * 60_000  # the moment is over: the plan again
    client.post("/api/plan")
    a = client.post("/api/v1/command", json=world.request("bath")).json()
    assert a["reason"] != "run over"
    svc.planner._run_over.pop("bath", None)


def test_setpoint_power_control_is_noted(client):
    p = client.svc.planner
    body = client.svc.ha.world.request("living")
    p.command(body, now_ms=1_000_000)
    assert "climate.living_proxy" not in (client.svc.store.get("power_control") or {})  # fan control: nothing to note
    sp = {**body, "power_control": "setpoint", "setpoint_offsets": [1, 2, 3], "source_fan_modes": ["+1 K", "+2 K", "+3 K"]}
    p.command(sp, now_ms=2_000_000)
    p.command(sp, now_ms=3_000_000)  # the same again: since stays
    pc = client.svc.store.get("power_control")["climate.living_proxy"]
    assert pc["mode"] == "setpoint" and pc["since"] == 2_000_000 and pc["fan_modes_before"] == body["source_fan_modes"]
    u = p.mapping()["units"]["living"]
    assert u["fan_modes"] == ["+1 K", "+2 K", "+3 K"] and u["setpoint_offsets"] == [1.0, 2.0, 3.0]
    assert u["power_since"] == 2_000_000 and p.power_switch() == 2_000_000
    from climate_optimizer.trainer import power_window
    day = 864e5
    assert power_window(0, 2 * day, 3 * day)[0] == 0  # one day of it: all days still
    assert power_window(0, 2 * day, 6 * day)[0] == 2 * day  # three: those alone
    assert power_window(5 * day, 2 * day, 6 * day) == (5 * day, None)  # out of the window anyway
    p.command({**sp, "setpoint_offsets": [1, 2, 4]}, now_ms=3_500_000)  # other offsets: what is learned holds along them
    assert client.svc.store.get("power_control")["climate.living_proxy"]["since"] == 2_000_000
    p.command({**sp, "setpoint_fan_modes": ["quiet", "low", "high"]}, now_ms=3_600_000)  # other fan modes: anew
    assert client.svc.store.get("power_control")["climate.living_proxy"]["since"] == 3_600_000
    p.command(body, now_ms=4_000_000)  # back to fan control
    client.svc.store.put("power_control", {})
    client.svc.store.put("fan_modes", {})


def test_earlier_models_can_be_used_again(client):
    p, store = client.svc.planner, client.svc.store
    first = p.params.to_dict()
    rep = {"finished": 1.0, "rmse_val": [0.5, 0.7], "days": {"train": [1, 2], "val": [3]}}
    a = p.remember_model(first, first, rep)
    second = json.loads(json.dumps(first))
    second["units"]["living"]["heat"] = [9.0 for _ in second["units"]["living"]["heat"]]
    b = p.remember_model(second, second, {**rep, "finished": 2.0})
    p.params, p.report = Params.from_dict(second), {**rep, "finished": 2.0}
    ms = client.get("/api/model").json()["models"]
    assert [m["id"] for m in ms][:2] == [b, a] and ms[0]["current"] and ms[1]["rmse_val"] == 0.6
    assert client.post(f"/api/model/use/{a}").json()["ok"]
    assert p.params.units["living"]["heat"] == first["units"]["living"]["heat"]
    m = client.get("/api/model").json()
    assert m["kept"] and m["models"][1]["current"]
    assert client.post("/api/model/use/nope").status_code == 404
    store.put("model_kept", None)


def test_trying_every_level(client):
    import time as _t

    import numpy as np

    p = client.svc.planner
    assert client.post("/api/model/explore", json={"days": 99}).status_code == 422
    ex = client.post("/api/model/explore", json={"days": 1}).json()["explore"]
    assert client.get("/api/model").json()["explore"]["until"] == ex["until"]
    now = _t.time() * 1000
    t = np.arange(now, now + 2 * 864e5, 15 * 60_000)
    a, b = p.explore_caps(0, 4, t, p.tz()), p.explore_caps(1, 4, t, p.tz())
    inside = t[:-1] < ex["until"]
    assert set(a[inside]) == {0.0, 1.0, 2.0, 3.0} and np.isnan(a[~inside]).all()  # every level, then any
    assert (a[inside] != b[inside]).all()  # the units take different turns
    r = client.post("/api/plan").json()
    assert any("Trying every level" in n for n in r["info"]["notes"])
    client.post("/api/model/explore", json={"days": 0})
    assert client.get("/api/model").json()["explore"] is None


def test_a_window_counts_once_open_a_moment(client):
    svc = client.svc
    world = svc.ha.world
    world.windows["office"] = True
    _see(svc)
    try:
        r = client.post("/api/plan").json()
        assert any("window just opened" in n for n in r["info"]["notes"])
        eid = "binary_sensor.office_balkontuer"
        st = svc.collector.states[eid]
        st["last_changed"] = datetime.fromtimestamp(datetime.fromisoformat(st["last_changed"].replace("Z", "+00:00")).timestamp() - 180).isoformat()
        r = client.post("/api/plan").json()
        assert not any("window just opened" in n for n in r["info"]["notes"])
    finally:
        world.windows["office"] = False
        _see(svc)


def test_planning_with_a_fixed_tariff(client):
    old = client.get("/api/config").json()["prices"]
    r = client.put("/api/config", json={"prices": {**old, "source": "fixed", "fixed_price": 0.31,
                                                   "fixed_periods": [{"start": "22:00", "end": "06:00", "price": 0.22}]}})
    assert r.status_code == 200, r.text
    try:
        plan = client.post("/api/plan").json()["plan"]
        assert set(round(p, 3) for p in plan["price"]) == {0.31, 0.22} and plan["t"][-1] - plan["t"][0] >= 24 * 3.6e6
        assert client.put("/api/config", json={"prices": {**old, "source": "fixed", "fixed_periods": [
            {"start": "22:00", "end": "6", "price": 0.2}]}}).status_code == 422
    finally:
        client.put("/api/config", json={"prices": old})


def test_a_training_keeps_the_model_it_replaces(client, monkeypatch):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from climate_optimizer import trainer as trainer_mod

    p, store = client.svc.planner, client.svc.store
    store.put("models", [])
    old = p.params.to_dict()
    new = json.loads(json.dumps(old))
    new["units"]["living"]["heat"] = [7.0 for _ in new["units"]["living"]["heat"]]
    p.report = p.report or {"finished": 1.0}
    monkeypatch.setattr(trainer_mod, "train_job", lambda *a, **k: {"params": new, "checked": new, "report": {"rmse_val": [0.5]}})
    t = trainer_mod.Trainer(p)
    t._pool = ThreadPoolExecutor(1)
    asyncio.run(t.run())
    assert t.error is None, t.error
    hist = store.get("models")
    assert hist[0]["params"]["units"]["living"]["heat"] == new["units"]["living"]["heat"]
    assert hist[1]["params"]["units"]["living"]["heat"] == old["units"]["living"]["heat"]  # the one before, kept
    p.params = Params.from_dict(old)
    store.put("params", old)
    store.put("models", [])


def test_a_new_home_assistant_address_needs_its_token_again(client):
    cfgs = client.svc.cfgs
    old = cfgs.config.ha.model_dump()
    try:
        cfgs.config.ha.token = "secret"
        url = cfgs.config.ha.url
        client.put("/api/config", json={"ha": {"url": url, "token": "********"}})  # the same address: kept
        assert cfgs.config.ha.token == "secret"
        client.put("/api/config", json={"ha": {"url": "http://elsewhere.example:8123", "token": "********"}})
        assert cfgs.config.ha.token == ""  # not sent to another address
    finally:
        client.put("/api/config", json={"ha": old})


def test_an_empty_setting_keeps_its_default(client):
    from climate_optimizer import mpc

    assert mpc.Settings.from_dict({"min_run_min": None}).min_run_min == mpc.Settings().min_run_min


def test_tariffs_compared(client):
    old = client.get("/api/config").json()
    r = client.put("/api/config", json={"prices": {**old["prices"], "monthly_fee": 10.0}, "tariffs": [
        {"name": "Fixed", "price": 0.30, "monthly_fee": 12.0},
        {"name": "Day and night", "price": 0.34, "periods": [{"start": "22:00", "end": "06:00", "price": 0.24}]}]})
    assert r.status_code == 200, r.text
    try:
        res = client.get("/api/tariffs/compare?days=7").json()
        assert res["days"] > 1 and res["kwh"] > 0 and res["phases"] >= 1
        now, fixed, dn = res["tariffs"]
        assert now["current"] and fixed["name"] == "Fixed" and now["diff_month_eur"] == 0
        assert abs(fixed["energy_eur"] - 0.30 * res["kwh"]) < 0.05 * res["kwh"] * 0.30 + 0.01  # one price: the kWh at it
        assert abs(fixed["fee_eur"] - 12.0 * res["days"] / (365.25 / 12)) < 0.05
        assert 24 <= dn["ct_per_kwh"] <= 34
        assert abs(fixed["month_eur"] - fixed["total_eur"] / res["days"] * 365.25 / 12) < 0.01 * fixed["month_eur"]
    finally:
        client.put("/api/config", json={"prices": old["prices"], "tariffs": []})


def test_plans_are_corrected_by_their_misses(client):
    """The planner's side: the misses over the last hour of its recent grid move the stored
    correction (once per five minutes), and the plan carries it."""
    import sys as _sys

    _sys.path.insert(0, "tests")
    from test_model import _grid_and_hidden

    from climate_optimizer.model.correction import with_correction

    p, store = client.svc.planner, client.svc.store
    layout, params = p.model()
    g, (Tm, Tw) = _grid_and_hidden(with_correction(params, {"living": -0.4}), layout)
    store.put("correction", {})
    k = 40 * 12
    while g.open[k - 13:k + 1].any():  # an hour without a window open
        k += 1
    cut = {f: (getattr(g, f)[:k + 1] if isinstance(getattr(g, f), np.ndarray) and len(getattr(g, f)) == g.K else getattr(g, f))
           for f in g.__dataclass_fields__}
    cut["ac_frac"] = {u: a[:k + 1] for u, a in g.ac_frac.items()}
    cut["heater_kw"] = {u: a[:k + 1] for u, a in g.heater_kw.items()}
    cut["others"] = [a[:k + 1] for a in g.others]
    g2 = type(g)(**cut)
    recent = {"grid": g2, "Tm": Tm[:k + 1], "Tw": Tw[:k + 1], "how": ""}
    now = float(g2.t[-1]) + 6 * 60_000
    zones = p.update_correction(now, layout, params, recent)
    assert zones["living"]["kw"] < -0.005 and zones["living"]["miss"] > 0  # the model ends it too warm: less heat
    assert p.update_correction(now + 60_000, layout, params, recent) == zones  # not again within five minutes
    stale = p.update_correction(now + 60 * 60_000, layout, params, recent)  # readings an hour old: only fades
    assert zones["living"]["kw"] <= stale["living"]["kw"] < 0
    store.put("correction", {})


def test_when_plans_start(client):
    from climate_optimizer import mpc

    p = client.svc.planner
    took = list(p._plan_took)
    try:
        p._plan_took.clear()
        assert p.plan_lead(mpc.Settings()) == 150  # a few plans first
        p._plan_took.extend([1.7, 2.0, 1.5])
        assert p.plan_lead(mpc.Settings()) == 60  # quick plans: a minute ahead, on fresher readings
        p._plan_took.extend([100.0] * 5)
        assert p.plan_lead(mpc.Settings()) == 180  # slow ones: started earlier, done in time
        p._plan_took.extend([400.0] * 5)
        assert p.plan_lead(mpc.Settings()) == 270
        assert p.plan_lead(mpc.Settings(plan_lead_s=200)) == 200  # as set
        info = client.post("/api/plan").json()["info"]
        assert info["plan_seconds"] >= info["solve_seconds"] >= 0
        assert client.get("/api/status").json()["plan_lead_s"] > 0
    finally:
        p._plan_took.clear()
        p._plan_took.extend(took)


def test_a_plan_under_way_shows(client):
    import threading
    import time as _t

    from climate_optimizer import mpc

    p = client.svc.planner
    st = client.get("/api/status").json()["plan"]
    assert st["running"] is None and st["time_limit_s"] > 0
    real = mpc.solve
    gate = threading.Event()

    def slow(pr):
        gate.wait(10)
        return real(pr)

    mpc.solve = slow
    try:
        done = threading.Thread(target=lambda: client.post("/api/plan"))
        done.start()
        for _ in range(100):
            run = client.get("/api/status").json()["plan"]["running"]
            if run and run["stage"] == "solving":
                break
            _t.sleep(0.05)
        assert run and run["stage"] == "solving" and run["seconds"] >= 0 and run["since"] > 0
    finally:
        gate.set()
        done.join(30)
        mpc.solve = real
    st = client.get("/api/status").json()["plan"]
    assert st["running"] is None and st["took_s"] is not None and p.waiting == 0
