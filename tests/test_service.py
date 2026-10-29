"""The service end to end against the simulated apartment."""

import pytest
from fastapi.testclient import TestClient

from climate_optimizer.app import create_app
from climate_optimizer.demo import demo_service
from climate_optimizer.model.layout import default_layout, default_params


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
    assert client.get("/static/app.js").status_code == 200


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
