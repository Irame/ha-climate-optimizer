"""Which zones the apartment has, how they connect, and what heats them.

The layout is configuration; the numbers that describe how each zone holds
and loses heat are the model's parameters (see thermal.py) and are learned.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Zone:
    id: str
    name: str
    windows: bool = False  # has door/window sensors that raise its losses when open
    solar: bool = True  # gets sun through its own windows
    facades: tuple[str, ...] = ()  # where its windows face, of N/E/S/W; empty: not known
    aliases: tuple[str, ...] = ()  # more names for it, as calendar events use them

    def sun_from(self, facade: str) -> bool:
        """Whether direct sun on this facade (E, S or W) can reach the zone."""
        return self.solar and (not self.facades or facade in self.facades)


@dataclass(frozen=True)
class Unit:
    """Something the optimizer switches: an AC indoor unit or an electric heater."""

    id: str
    name: str
    zone: str
    kind: str  # "ac" | "heater"
    levels: int = 4  # fan levels of an AC unit, quietest first


@dataclass(frozen=True)
class Layout:
    zones: tuple[Zone, ...]
    links: tuple[tuple[str, str], ...]  # zones whose air exchanges heat directly
    units: tuple[Unit, ...]
    hub: str | None = None  # zone everything else is linked to, if any

    @property
    def zone_ids(self) -> list[str]:
        return [z.id for z in self.zones]

    def zone_index(self, zid: str) -> int:
        return self.zone_ids.index(zid)

    @property
    def ac_units(self) -> list[Unit]:
        return [u for u in self.units if u.kind == "ac"]

    @property
    def heaters(self) -> list[Unit]:
        return [u for u in self.units if u.kind == "heater"]

    @property
    def window_zones(self) -> list[str]:
        return [z.id for z in self.zones if z.windows]

    def to_dict(self) -> dict:
        return {
            "zones": [{**z.__dict__, "facades": list(z.facades), "aliases": list(z.aliases)} for z in self.zones],
            "links": [list(lk) for lk in self.links],
            "units": [u.__dict__ for u in self.units],
            "hub": self.hub,
        }

    @staticmethod
    def from_dict(d: dict) -> Layout:
        zones = tuple(Zone(**{**z, "facades": tuple(z.get("facades") or ()), "aliases": tuple(z.get("aliases") or ())})
                      for z in d["zones"])
        for z in zones:
            if set(z.facades) - set("NESW"):
                raise ValueError(f"zone {z.id}: facades are N, E, S or W")
        units = tuple(Unit(**u) for u in d["units"])
        ids = {z.id for z in zones}
        links = tuple((a, b) for a, b in d.get("links", []))
        for a, b in links:
            if a not in ids or b not in ids:
                raise ValueError(f"link {a}-{b} names an unknown zone")
        for u in units:
            if u.zone not in ids:
                raise ValueError(f"unit {u.id} sits in unknown zone {u.zone}")
            if u.kind not in ("ac", "heater"):
                raise ValueError(f"unit {u.id}: kind must be ac or heater")
        return Layout(zones=zones, links=links, units=units, hub=d.get("hub"))


def default_layout() -> Layout:
    """Office, bedroom and living room with kitchen each have an AC unit on one
    outdoor unit; the bath has an electric heater; the hallway connects them."""
    zones = (
        Zone("office", "Office", windows=True),
        Zone("bedroom", "Bedroom", windows=True),
        Zone("living", "Living room & kitchen", windows=True),
        Zone("bath", "Bathroom"),
        Zone("hall", "Hallway", solar=False),
    )
    links = tuple((z, "hall") for z in ("office", "bedroom", "living", "bath"))
    units = (
        Unit("office", "Office", "office", "ac"),
        Unit("bedroom", "Bedroom", "bedroom", "ac"),
        Unit("living", "Living room", "living", "ac"),
        Unit("bath", "Bath heater", "bath", "heater", levels=1),
    )
    return Layout(zones=zones, links=links, units=units, hub="hall")


@dataclass
class Params:
    """Learned numbers, JSON-friendly. Units: C kWh/K, H kW/K, q kW, s m²."""

    zones: dict[str, dict[str, float]]
    links: dict[str, float]  # "a|b" -> kW/K
    units: dict[str, dict]  # ac: {"heat": [kW per level], "cool_ratio": f}; heater: {"kw": P}
    plant: dict[str, float]  # Qmax, eta, standby, cap_slope
    building: dict[str, float] = field(default_factory=lambda: {"Tnb": 20.0})
    version: int = 1

    def to_dict(self) -> dict:
        return {"version": self.version, "zones": self.zones, "links": self.links, "units": self.units,
                "plant": self.plant, "building": self.building}

    @staticmethod
    def from_dict(d: dict) -> Params:
        return Params(zones=d["zones"], links=d["links"], units=d["units"], plant=d["plant"],
                      building=d.get("building", {"Tnb": 20.0}), version=d.get("version", 1))


def link_key(a: str, b: str) -> str:
    return f"{a}|{b}"


def default_params(layout: Layout) -> Params:
    """A plausible apartment: the example, and the prior for training."""
    big = {"living"}
    zones = {}
    for z in layout.zones:
        if z.id in big:
            p = dict(Ca=0.6, Cm=2.5, Cw=7.0, Ham=0.8, Hmw=0.35, Hwo=0.032, Hwn=0.016, H=0.010, Hw=0.20, q=0.12,
                     sD=1.2, sE=0.0, sS=1.0, sW=0.8)
        elif z.id == layout.hub:
            p = dict(Ca=0.2, Cm=0.6, Cw=2.0, Ham=0.3, Hmw=0.12, Hwo=0.002, Hwn=0.008, H=0.002, Hw=0.0, q=0.02,
                     sD=0.0, sE=0.0, sS=0.0, sW=0.0)
        elif any(u.zone == z.id and u.kind == "heater" for u in layout.units):
            p = dict(Ca=0.12, Cm=0.4, Cw=1.0, Ham=0.2, Hmw=0.08, Hwo=0.005, Hwn=0.004, H=0.002, Hw=0.0, q=0.01,
                     sD=0.1, sE=0.0, sS=0.0, sW=0.0)
        else:
            p = dict(Ca=0.25, Cm=1.0, Cw=3.0, Ham=0.35, Hmw=0.15, Hwo=0.014, Hwn=0.008, H=0.005, Hw=0.12, q=0.04,
                     sD=0.6, sE=0.0, sS=0.8, sW=0.0)
        if not z.windows:
            p["Hw"] = 0.0
        if not z.solar:
            p.update(sD=0.0, sE=0.0, sS=0.0, sW=0.0)
        for f in "ESW":
            if not z.sun_from(f):
                p["s" + f] = 0.0
            elif z.facades:
                p["s" + f] = max(p["s" + f], 0.6)  # a known sunny facade: start from a typical window
        p.update(qc=0.0, qs=0.0)
        zones[z.id] = p
    links = {}
    for a, b in layout.links:
        links[link_key(a, b)] = 0.09 if "living" in (a, b) else 0.012
    units = {}
    for u in layout.units:
        if u.kind == "ac":
            top = 4.0 if u.zone in big else 3.0
            n = max(1, u.levels)
            units[u.id] = {"heat": [round(top * (0.35 + 0.65 * i / max(1, n - 1)), 3) for i in range(n)], "cool_ratio": 0.9}
        else:
            units[u.id] = {"kw": 2.0}
    plant = {"Qmax": 7.0, "eta": 0.42, "standby": 0.03, "cap_slope": 0.02}
    return Params(zones=zones, links=links, units=units, plant=plant)
