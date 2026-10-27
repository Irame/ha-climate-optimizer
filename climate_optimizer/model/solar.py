"""Sun position and a rough split of solar irradiance onto vertical facades.

Good enough to give each zone a solar input the fit can scale; the fitted
coefficients absorb window sizes, shading and the crudeness here.
"""

from __future__ import annotations

import math

RAD = math.pi / 180
FACADE_AZ = {"E": 90 * RAD, "S": 180 * RAD, "W": 270 * RAD}


def sun_position(t_ms: float, lat: float, lon: float) -> tuple[float, float]:
    """Elevation and azimuth (from north, clockwise) in radians, about 0.01° accurate."""
    d = t_ms / 864e5 + 2440587.5 - 2451545.0
    g = (357.529 + 0.98560028 * d) * RAD
    q = 280.459 + 0.98564736 * d
    L = (q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) * RAD
    e = (23.439 - 0.00000036 * d) * RAD
    ra = math.atan2(math.cos(e) * math.sin(L), math.cos(L))
    dec = math.asin(math.sin(e) * math.sin(L))
    gmst = (18.697374558 + 24.06570982441908 * d) % 24
    ha = (gmst * 15 + lon) * RAD - ra
    phi = lat * RAD
    elev = math.asin(math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(ha))
    az = math.atan2(-math.cos(dec) * math.sin(ha), math.sin(dec) * math.cos(phi) - math.cos(dec) * math.sin(phi) * math.cos(ha))
    if az < 0:
        az += 2 * math.pi
    return elev, az


def irradiance(t_ms: float, lat: float, lon: float, cloud: float | None) -> dict:
    """kW/m²: D = diffuse and reflected light any window sees; E/S/W = direct
    sun on a vertical east, south or west facade. cloud: 0..1, None = 0.5."""
    elev, az = sun_position(t_ms, lat, lon)
    s = math.sin(elev)
    if s <= 0.01:
        return {"D": 0.0, "E": 0.0, "S": 0.0, "W": 0.0, "ghi": 0.0}
    c = 0.5 if cloud is None or not math.isfinite(cloud) else min(1.0, max(0.0, cloud))
    clear = 1.098 * s * math.exp(-0.057 / s)  # Haurwitz clear-sky GHI
    ghi = clear * (1 - 0.75 * c ** 3.4)  # Kasten-Czeplak
    kd = min(1.0, 0.15 + 0.85 * c * c)
    dhi = kd * ghi
    dni = min(1.0, (ghi - dhi) / max(s, 0.05))
    out = {"D": 0.5 * dhi + 0.1 * ghi, "ghi": ghi}
    for f, a in FACADE_AZ.items():
        out[f] = dni * max(0.0, math.cos(elev) * math.cos(az - a))
    return out
