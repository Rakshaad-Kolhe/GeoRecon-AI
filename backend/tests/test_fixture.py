import pytest

import make_aukerman_fixture as mk


def test_gps_ref_negative_normalises_variants():
    assert mk._ref_is_negative("W", "W")
    assert mk._ref_is_negative(b"W", "W")
    assert mk._ref_is_negative("W\x00", "W")
    assert mk._ref_is_negative(" s ", "S")
    assert not mk._ref_is_negative("N", "S")
    assert not mk._ref_is_negative("E", "W")
    assert not mk._ref_is_negative(None, "W")


def test_parse_gps_applies_hemisphere_refs():
    lat, lon, alt = mk._parse_gps({
        "GPSLatitude": (41.0, 18.0, 13.75), "GPSLatitudeRef": "N",
        "GPSLongitude": (81.0, 45.0, 1.7), "GPSLongitudeRef": b"W",
        "GPSAltitude": 300.0, "GPSAltitudeRef": 0,
    })
    assert lat == pytest.approx(41 + 18 / 60 + 13.75 / 3600)
    assert lon == pytest.approx(-(81 + 45 / 60 + 1.7 / 3600))     # W -> negative
    assert alt == pytest.approx(300.0)

    lat, lon, alt = mk._parse_gps({
        "GPSLatitude": (33.0, 0, 0), "GPSLatitudeRef": "S\x00",
        "GPSLongitude": (151.0, 0, 0), "GPSLongitudeRef": "E",
        "GPSAltitude": 5.0, "GPSAltitudeRef": 1,
    })
    assert lat == pytest.approx(-33.0)                            # S -> negative
    assert lon == pytest.approx(151.0)
    assert alt == pytest.approx(-5.0)                             # below sea level
