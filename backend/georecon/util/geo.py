"""WGS84 <-> local ENU (metres) and UTM helpers. Vectorised over numpy arrays.

ENU is built from ECEF (EPSG:4979 geographic-3D -> EPSG:4978 geocentric) plus the
standard east/north/up rotation about the origin's geodetic lat/lon.
"""

from __future__ import annotations

import numpy as np
from pyproj import Transformer

_LLA_TO_ECEF = Transformer.from_crs(4979, 4978, always_xy=True)
_ECEF_TO_LLA = Transformer.from_crs(4978, 4979, always_xy=True)
_utm_cache: dict[int, Transformer] = {}


def _origin(origin) -> tuple[float, float, float]:
    if isinstance(origin, dict):
        return float(origin["lat"]), float(origin["lon"]), float(origin.get("alt", 0.0))
    o = list(origin)
    return float(o[0]), float(o[1]), float(o[2] if len(o) > 2 else 0.0)


def _enu_rotation(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    lat0, lon0 = np.radians(lat0_deg), np.radians(lon0_deg)
    sla, cla, slo, clo = np.sin(lat0), np.cos(lat0), np.sin(lon0), np.cos(lon0)
    return np.array([
        [-slo,        clo,       0.0],
        [-sla * clo, -sla * slo, cla],
        [ cla * clo,  cla * slo, sla],
    ])


def wgs84_to_enu(lat, lon, alt, origin):
    """(lat, lon, alt) degrees/metres -> (E, N, U) metres about ``origin``."""
    lat0, lon0, alt0 = _origin(origin)
    x, y, z = (np.asarray(v, dtype=float) for v in
               _LLA_TO_ECEF.transform(np.asarray(lon, float),
                                      np.asarray(lat, float),
                                      np.asarray(alt, float)))
    x0, y0, z0 = _LLA_TO_ECEF.transform(lon0, lat0, alt0)
    d = np.stack([x - x0, y - y0, z - z0], axis=-1)
    enu = d @ _enu_rotation(lat0, lon0).T
    return enu[..., 0], enu[..., 1], enu[..., 2]


def enu_to_wgs84(e, n, u, origin):
    """(E, N, U) metres about ``origin`` -> (lat, lon, alt)."""
    lat0, lon0, alt0 = _origin(origin)
    x0, y0, z0 = _LLA_TO_ECEF.transform(lon0, lat0, alt0)
    enu = np.stack([np.asarray(e, float), np.asarray(n, float),
                    np.asarray(u, float)], axis=-1)
    d = enu @ _enu_rotation(lat0, lon0)          # inverse of (d @ R.T)
    lon, lat, alt = (np.asarray(v, dtype=float) for v in
                     _ECEF_TO_LLA.transform(d[..., 0] + x0, d[..., 1] + y0,
                                            d[..., 2] + z0))
    return lat, lon, alt


def utm_epsg(lat: float, lon: float) -> int:
    zone = int((float(lon) + 180.0) // 6) % 60 + 1
    return (32600 if float(lat) >= 0 else 32700) + zone


def to_utm(lat, lon):
    """Project to the UTM zone of the mean coordinate. Returns (easting, northing, epsg)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    epsg = utm_epsg(float(np.mean(lat)), float(np.mean(lon)))
    tr = _utm_cache.get(epsg)
    if tr is None:
        tr = _utm_cache[epsg] = Transformer.from_crs(4326, epsg, always_xy=True)
    e, n = tr.transform(lon, lat)
    return np.asarray(e, float), np.asarray(n, float), epsg
