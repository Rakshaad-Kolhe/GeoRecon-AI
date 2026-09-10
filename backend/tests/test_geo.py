import numpy as np

from georecon.util.geo import enu_to_wgs84, to_utm, utm_epsg, wgs84_to_enu

ORIGIN = {"lat": 18.52, "lon": 73.85, "alt": 560.0}


def test_enu_wgs84_roundtrip_sub_millimetre():
    rng = np.random.default_rng(0)
    enu = rng.uniform(-800.0, 800.0, size=(300, 3))
    lat, lon, alt = enu_to_wgs84(enu[:, 0], enu[:, 1], enu[:, 2], ORIGIN)
    e, n, u = wgs84_to_enu(lat, lon, alt, ORIGIN)
    back = np.stack([e, n, u], axis=-1)
    assert np.abs(back - enu).max() < 1e-3


def test_origin_maps_to_zero():
    e, n, u = wgs84_to_enu(ORIGIN["lat"], ORIGIN["lon"], ORIGIN["alt"], ORIGIN)
    assert abs(float(e)) < 1e-6 and abs(float(n)) < 1e-6 and abs(float(u)) < 1e-6


def test_utm_epsg_zones():
    assert utm_epsg(18.52, 73.85) == 32643        # Pune -> 43N
    assert utm_epsg(-33.9, 151.2) == 32756        # Sydney -> 56S
    _, _, epsg = to_utm([18.52, 18.53], [73.85, 73.86])
    assert epsg == 32643
