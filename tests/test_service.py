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
