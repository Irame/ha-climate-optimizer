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


WEEKDAYS = ("mo", "tu", "we", "th", "fr", "sa", "su")


class PricePeriod(BaseModel):
    """A time of a fixed tariff with a price of its own (night rate, weekend)."""
    model_config = ConfigDict(extra="ignore")
    start: str = "22:00"
    end: str = "06:00"  # before the start: runs past midnight (on the days it starts)
    price: float = Field(0.25, ge=-1, le=5)  # €/kWh
    days: list[str] = []  # mo .. su; empty: every day

    @field_validator("start", "end")
    @classmethod
    def _hm(cls, v):
        h, _, m = str(v).partition(":")
        if not (h.isdigit() and m.isdigit() and 0 <= int(h) <= 24 and 0 <= int(m) < 60 and int(h) * 60 + int(m) <= 1440):
            raise ValueError(f"{v}: a time as HH:MM")
        return f"{int(h):02d}:{int(m):02d}"

    @field_validator("days")
    @classmethod
    def _days(cls, v):
        v = [str(d).lower()[:2] for d in v]
        if set(v) - set(WEEKDAYS):
            raise ValueError(f"days: some of {', '.join(WEEKDAYS)}")
        return v


class PriceConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # tibber_prices: the action tibber_prices.get_price; attributes: a sensor that lists the
    # coming prices in an attribute (Nord Pool, EPEX Spot, Tibber and similar); fixed: a tariff
    # with one price, or prices by time (day and night rate)
    source: Literal["tibber_prices", "attributes", "fixed"] = "tibber_prices"
    entry_id: str = ""  # tibber_prices config entry, if there is more than one
    sensor: str = ""  # for source attributes; also the price history shown in the UI
    fixed_price: float = Field(0.30, ge=-1, le=5)  # €/kWh, whenever no period below holds
    fixed_periods: list[PricePeriod] = []  # later ones win where they overlap
    monthly_fee: float = Field(0.0, ge=0, le=1000)  # €, for comparing tariffs


class Tariff(BaseModel):
    """A tariff to compare with the one paid now: a price, times with their own, a monthly fee."""
    model_config = ConfigDict(extra="ignore")
    name: str = "Fixed price"
    price: float = Field(0.30, ge=-1, le=5)  # €/kWh
    periods: list[PricePeriod] = []
    monthly_fee: float = Field(0.0, ge=0, le=1000)  # €


class AiringConfig(BaseModel):
    """One airing of the whole apartment at a time: every room with a window sensor at once."""
    model_config = ConfigDict(extra="ignore")
    enabled: bool = False
    duration_min: float = Field(10.0, ge=2, le=60)
    per_day: int = Field(1, ge=1, le=4)
    min_gap_h: float = Field(4.0, ge=0.5, le=12)
    sensor: str = "sensor.climate_optimizer_airing"  # where Home Assistant gets the next suggestion
    # to cool: windows open as long as the outside air does better than the units (summer nights)
    venting: bool = False
    vent_hassle_eur: float = Field(0.10, ge=0, le=5)  # what opening them is worth avoiding, each time
    vent_sensor: str = "sensor.climate_optimizer_venting"
    close_sensor: str = "sensor.climate_optimizer_close_windows"  # when open windows are best closed


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
    airing: AiringConfig = Field(default_factory=AiringConfig)
    settings: dict = Field(default_factory=lambda: Settings().__dict__.copy())
    shadow: bool = True  # plan and show, but let the proxies' own thermostats decide
    # in shadow mode, to watch the optimizer beside another automation:
    shadow_units: list[str] = Field(default_factory=list)  # plan these even when their proxy is off or not in auto
    shadow_comfort: dict[str, float] = Field(default_factory=dict)  # zone -> comfort temperature instead of the proxy's
    api_token: str = ""  # required from the proxies and the UI when set
    retrain_hour: int = 3
    history_days: int = 30
    tariffs: list[Tariff] = Field(default_factory=list)  # to compare with what is paid now

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
        d["settings"] = self.settings_obj().__dict__  # every setting, the ones added since saved included
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
            new_url = str(patch.get("ha", {}).get("url") or "").rstrip("/")
            if new_url and new_url != str(cur["ha"].get("url") or "").rstrip("/") and not os.environ.get("HA_URL"):
                # the stored token goes to no other address than the one it was given for
                if os.environ.get("HA_TOKEN"):
                    raise ValueError("ha.url: the token comes from the container (HA_TOKEN); set HA_URL there too")
                if not patch["ha"].get("token"):
                    patch["ha"]["token"] = ""
            if patch.get("api_token") == "********":
                patch.pop("api_token")
            for k, v in patch.items():
                if isinstance(v, dict) and isinstance(cur.get(k), dict) and k in ("ha", "prices", "airing"):
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
