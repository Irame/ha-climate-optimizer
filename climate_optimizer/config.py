"""Everything the user sets, kept as one JSON file in the data directory."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .comfort import ComfortFunction, Rule, default_functions, default_rules
from .model.layout import Layout, default_layout
from .mpc import Settings

DATA = Path(os.environ.get("DATA_DIR", "/data"))


class HAConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    url: str = ""  # e.g. http://homeassistant.local:8123; empty inside a Home Assistant add-on
    token: str = ""  # long-lived access token
    verify_ssl: bool = True


class PriceConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # tibber_prices: the action tibber_prices.get_price; attributes: a sensor that lists the
    # coming prices in an attribute (Nord Pool, EPEX Spot, Tibber and similar)
    source: Literal["tibber_prices", "attributes"] = "tibber_prices"
    entry_id: str = ""  # tibber_prices config entry, if there is more than one
    sensor: str = ""  # for source attributes; also the price history shown in the UI


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")
    ha: HAConfig = Field(default_factory=HAConfig)
    layout: dict = Field(default_factory=lambda: default_layout().to_dict())
    # which Home Assistant entities stand for what, see model/dataset.py build_grid
    mapping: dict = Field(default_factory=dict)
    prices: PriceConfig = Field(default_factory=PriceConfig)
    location: dict | None = None  # {"lat", "lon"}; default: Home Assistant's
    functions: list[dict] = Field(default_factory=lambda: [f.to_dict() for f in default_functions()])
    defaults: dict[str, str] = Field(default_factory=dict)  # zone -> comfort function when no rule applies
    comfort: dict[str, float] = Field(default_factory=dict)  # zone -> comfort temperature when no proxy sets one
    rules: list[dict] = Field(default_factory=lambda: [r.to_dict() for r in default_rules()])
    calendar: str = ""  # calendar entity whose events override the rules
    settings: dict = Field(default_factory=lambda: Settings().__dict__.copy())
    shadow: bool = True  # plan and show, but let the proxies' own thermostats decide
    api_token: str = ""  # required from the proxies and the UI when set
    retrain_hour: int = 3
    history_days: int = 30

    @field_validator("layout")
    @classmethod
    def _layout(cls, v):
        Layout.from_dict(v)
        return v

    @field_validator("functions")
    @classmethod
    def _functions(cls, v):
        ids = [ComfortFunction.from_dict(f).id for f in v]
        if len(set(ids)) != len(ids):
            raise ValueError("comfort function ids must be unique")
        if not ids:
            raise ValueError("at least one comfort function")
        return v

    @field_validator("rules")
    @classmethod
    def _rules(cls, v):
        for r in v:
            Rule.from_dict(r)
        return v

    @field_validator("settings")
    @classmethod
    def _settings(cls, v):
        return Settings.from_dict(v).__dict__.copy()

    @model_validator(mode="after")
    def _references(self):
        fids = {f["id"] for f in self.functions}
        for z, f in self.defaults.items():
            if f not in fids:
                raise ValueError(f"default for {z}: no comfort function {f}")
        for r in self.rules:
            if r.get("action") == "comfort" and r.get("function") not in fids:
                raise ValueError(f"rule {r.get('name')}: no comfort function {r.get('function')}")
        return self

    # typed views
    def layout_obj(self) -> Layout:
        return Layout.from_dict(self.layout)

    def function_objs(self) -> dict[str, ComfortFunction]:
        return {f["id"]: ComfortFunction.from_dict(f) for f in self.functions}

    def rule_objs(self) -> list[Rule]:
        return [Rule.from_dict(r) for r in self.rules]

    def settings_obj(self) -> Settings:
        return Settings.from_dict(self.settings)

    def public(self) -> dict:
        d = self.model_dump()
        d["ha"]["token"] = "********" if self.ha.token else ""
        d["api_token"] = "********" if self.api_token else ""
        d["env"] = {"ha_url": bool(os.environ.get("HA_URL")), "ha_token": bool(os.environ.get("HA_TOKEN"))}
        return d


class ConfigStore:
    def __init__(self, path: Path | None = None):
        self.path = path or DATA / "config.json"
        self._lock = threading.Lock()
        self.config = self._load()

    def _load(self) -> Config:
        c = Config.model_validate(json.loads(self.path.read_text())) if self.path.exists() else Config()
        return self._env(c)

    @staticmethod
    def _env(c: Config) -> Config:
        """HA_URL and HA_TOKEN, when the container sets them, win over what was saved,
        or a changed compose file would seem to do nothing."""
        if os.environ.get("HA_URL"):
            c.ha.url = os.environ["HA_URL"]
        if os.environ.get("HA_TOKEN"):
            c.ha.token = os.environ["HA_TOKEN"]
        return c

    def update(self, patch: dict) -> Config:
        """Merge a partial config; masked secrets keep their stored value."""
        with self._lock:
            cur = self.config.model_dump()
            patch = json.loads(json.dumps(patch))
            if patch.get("ha", {}).get("token") == "********":
                patch["ha"].pop("token")
            if patch.get("api_token") == "********":
                patch.pop("api_token")
            for k, v in patch.items():
                if isinstance(v, dict) and isinstance(cur.get(k), dict) and k in ("ha", "prices"):
                    cur[k].update(v)
                else:
                    cur[k] = v
            new = self._env(Config.model_validate(cur))
            self.save(new)
            self.config = new
            return new

    def save(self, c: Config) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(c.model_dump(), indent=2))
        tmp.replace(self.path)
