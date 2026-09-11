import json

import numpy as np
import pytest

from georecon import pipeline
from georecon.config import JobConfig
from georecon.pipeline import JobPaths
from georecon.stages import validate
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


# --------------------------------------------------------------------------- #
def _write_validate_job(tmp_path, *, georeferenced):
    paths = JobPaths.for_job(tmp_path / "job").ensure()
    georef_metrics = {
        "georeferenced": georeferenced, "branch": "sim3" if georeferenced else "none",
        "rmse_h": 1.1, "rmse_v": 0.8, "holdout_rmse_h": 1.5, "holdout_rmse_v": 1.0,
        "inliers": 18, "pairs": 20, "scale_drift_pct": 0.3, "seconds": 1.0,
    }
    stages = {
        "ingest": {"video": {"duration_s": 20.0}, "seconds": 1.0},
        "keyframes": {"keyframes": 30, "seconds": 2.0},
        "masking": {"seconds": 1.0},
        "sfm": {"registered_pct": 90.0, "mean_reproj_px": 0.8, "seconds": 3.0},
        "georef": georef_metrics,
        "dense": {"dense": True, "points": 1000, "points_per_m2": 50.0,
                 "excluded_images": [], "seconds": 4.0},
        "mesh": {"surface_area_m2": 200.0, "seconds": 1.0},
        "export": {"dsm_cell_m": 0.5, "seconds": 1.0},
    }
    paths.metrics_json.write_text(json.dumps({"stages": stages, "total_seconds": 14.0}),
                                  encoding="utf-8")
    return paths


def test_validate_accuracy_na_without_gps(tmp_path, monkeypatch):
    paths = _write_validate_job(tmp_path, georeferenced=False)
    monkeypatch.setattr(pipeline, "STAGES", [("validate", validate.run)])
    pipeline.run_job(tmp_path / "job", JobConfig(video_path="x"))

    summary = json.loads(paths.metrics_json.read_text())["stages"]["validate"]["summary"]
    assert summary["accuracy"] == "n/a — no GPS telemetry"
    assert summary["completeness"]["registered_pct"] == 90.0
    assert summary["completeness"]["dense_points"] == 1000
    assert summary["speed"]["total_seconds"] > 0
    md = paths.summary_md.read_text(encoding="utf-8")
    assert "n/a — no GPS telemetry" in md


def test_validate_accuracy_present_with_gps(tmp_path, monkeypatch):
    paths = _write_validate_job(tmp_path, georeferenced=True)
    monkeypatch.setattr(pipeline, "STAGES", [("validate", validate.run)])
    pipeline.run_job(tmp_path / "job", JobConfig(video_path="x"))

    summary = json.loads(paths.metrics_json.read_text())["stages"]["validate"]["summary"]
    assert isinstance(summary["accuracy"], dict)
    assert summary["accuracy"]["holdout_rmse_h"] == 1.5
    assert summary["completeness"]["registered_pct"] == 90.0
