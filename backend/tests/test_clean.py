import json

import numpy as np
import pandas as pd
import pytest

from georecon import pipeline
from georecon.config import CleanCfg, JobConfig
from georecon.pipeline import JobPaths
from georecon.stages import clean
from georecon.stages.clean import (
    _classify_and_mask,
    far_grazing_mask,
    radius_outlier_mask,
    statistical_outlier_mask,
)
from georecon.util.ply import write_ply


def test_roi_classifies_orbit_and_keeps_points_near_hull():
    rng = np.random.default_rng(0)
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    C = np.column_stack([50 * np.cos(theta), 50 * np.sin(theta), np.full(40, 80.0)])
    cfg = CleanCfg()

    near = rng.uniform(-40, 40, size=(300, 2))
    far = rng.uniform(500, 600, size=(50, 2))
    xy = np.vstack([near, far])
    xyz = np.column_stack([xy, np.zeros(len(xy))])

    mask, kind, polygon, buffer = _classify_and_mask(xyz, C, cfg, z_lo=-10.0, z_hi=90.0)
    assert kind == "orbit"
    assert mask[:300].mean() > 0.95            # near-centre points kept
    assert not mask[300:].any()                # far-away points dropped
    assert buffer > 0
    assert len(polygon) >= 3


def test_roi_classifies_strip_and_buffers_the_track():
    C = np.column_stack([np.linspace(0, 200, 30), np.zeros(30), np.full(30, 50.0)])
    cfg = CleanCfg(roi_strip_buffer_mult=1.0)

    on_track = np.column_stack([np.linspace(0, 200, 100), np.zeros(100), np.zeros(100)])
    off_track = np.column_stack([np.linspace(0, 200, 100), np.full(100, 500.0), np.zeros(100)])
    xyz = np.vstack([on_track, off_track])

    mask, kind, polygon, buffer = _classify_and_mask(xyz, C, cfg, z_lo=-10.0, z_hi=60.0)
    assert kind == "strip"
    assert mask[:100].all()                    # on the flight line: kept
    assert not mask[100:].any()                # 500 m off to the side: dropped
    assert buffer == pytest.approx(60.0, rel=0.01)    # 1.0 * median camera height (50 - z_lo=-10)


def test_far_grazing_mask_drops_points_far_from_every_camera():
    C = np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 10.0]])
    near = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    far = np.array([[1000.0, 1000.0, 1000.0]])
    xyz = np.vstack([near, far])

    mask, med = far_grazing_mask(xyz, C, mult=2.5)
    assert mask[:3].all()
    assert not mask[3]
    assert med > 0


def test_statistical_and_radius_outlier_masks_drop_isolated_points():
    rng = np.random.default_rng(2)
    cluster = rng.normal(scale=0.1, size=(200, 3))
    isolated = np.array([[100.0, 100.0, 100.0]])
    xyz = np.vstack([cluster, isolated])

    sor = statistical_outlier_mask(xyz, k=16, std_mult=2.0)
    assert sor[:200].mean() > 0.9
    assert not sor[200]

    rad, med_sp = radius_outlier_mask(xyz, min_neighbors=4, radius_mult=3.0)
    assert rad[:200].mean() > 0.9
    assert not rad[200]
    assert med_sp > 0


def _write_clean_job(tmp_path, *, n_cams=30, n_pts=500, seed=0):
    paths = JobPaths.for_job(tmp_path / "job").ensure()
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n_cams, endpoint=False)
    C = np.column_stack([50 * np.cos(theta), 50 * np.sin(theta), np.full(n_cams, 80.0)])
    names = [f"f_{i:06d}.jpg" for i in range(n_cams)]
    pd.DataFrame({
        "name": names, "registered": [1] * n_cams,
        "E": C[:, 0], "N": C[:, 1], "U": C[:, 2],
        "qw": [1.0] * n_cams, "qx": [0.0] * n_cams, "qy": [0.0] * n_cams, "qz": [0.0] * n_cams,
    }).to_csv(paths.georef_cameras, index=False)

    xy = rng.uniform(-40, 40, size=(n_pts, 2))
    z = rng.normal(scale=0.2, size=n_pts)
    xyz = np.column_stack([xy, z]).astype(np.float32)
    normals = np.tile([0.0, 0.0, 1.0], (n_pts, 1)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(n_pts, 3)).astype(np.uint8)
    views = rng.integers(3, 10, n_pts).astype(np.uint8)
    write_ply(paths.dense_fused, xyz, {
        "nx": normals[:, 0], "ny": normals[:, 1], "nz": normals[:, 2],
        "red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2], "views": views,
    })
    return paths


def test_clean_stage_writes_clean_ply_and_roi_json(tmp_path, monkeypatch):
    paths = _write_clean_job(tmp_path)
    monkeypatch.setattr(pipeline, "STAGES", [("clean", clean.run)])
    pipeline.run_job(tmp_path / "job", JobConfig(video_path="x"))

    m = json.loads(paths.metrics_json.read_text())["stages"]["clean"]
    assert m["roi_kind"] == "orbit"
    assert m["counts"]["raw"] == 500
    assert 0.0 <= m["kept_pct"] <= 100.0
    assert paths.dense_clean.exists()
    roi = json.loads(paths.dense_roi.read_text())
    assert roi["kind"] == "orbit"
    assert len(roi["polygon"]) >= 3
