import numpy as np
import pytest

from georecon.stages.export import _conf_from_views, _write_geotiff
from georecon.stages.validate import _coverage_pct, project_10min


def test_conf_from_views_range_and_scaling():
    views = np.array([0, 2, 4, 8, 16, 40], np.uint8)
    conf = _conf_from_views(views)
    assert conf.dtype == np.uint8
    assert conf.min() == 0 and conf.max() <= 255
    # p95 ~ 40 -> denom 40; views=8 -> ~51; views>=40 -> 255
    assert conf[-1] == 255
    assert conf[0] == 0
    # a cloud where every point has few views -> denom floored at 8
    low = _conf_from_views(np.full(100, 2, np.uint8))
    assert low.max() == int(2 / 8 * 255)


def test_project_10min_math():
    per = {"ingest": 1.0, "keyframes": 4.0, "masking": 20.0, "sfm": 25.0,
           "georef": 5.0, "dense": 100.0, "mesh": 10.0, "export": 5.0}
    total = sum(per.values())                              # 170
    kf10, proj = project_10min(per, total, keyframes=50, duration_s=30.0,
                               max_keyframes=150)
    # kf10 = min(150, round(50/30*600)) = min(150, 1000) = 150
    assert kf10 == 150
    frame_dep = 1 + 4 + 20 + 25 + 100 + 10                 # 160
    fixed = total - frame_dep                              # 10 (georef+export)
    assert proj == pytest.approx(round(fixed + frame_dep / 50 * 150, 1))
    assert project_10min(per, total, 0, 30.0, 150) == (None, None)


def test_coverage_pct_grid_with_hole(tmp_path):
    pytest.importorskip("rasterio")
    cell = 1.0
    e0, n1 = 500_000.0, 4_570_020.0
    dsm = np.full((20, 20), 5.0, np.float32)
    dsm[10, 10] = -9999.0                                  # one hole
    _write_geotiff(tmp_path / "d.tif", dsm, e0, n1, cell, 32617, nodata=-9999.0)

    # dense points spanning the whole raster footprint (hull covers all cells)
    gx = e0 + np.linspace(0.5, 19.5, 12)
    gy = (n1 - 20) + np.linspace(0.5, 19.5, 12)
    xx, yy = np.meshgrid(gx, gy)
    dense_en = np.column_stack([xx.ravel(), yy.ravel()])

    cov = _coverage_pct(tmp_path / "d.tif", dense_en, cell)
    assert 98.0 <= cov <= 100.0                            # 399/400 occupied
    assert cov < 100.0
