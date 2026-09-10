import json

import numpy as np
import pandas as pd
import pytest

from georecon import pipeline
from georecon.config import GeorefCfg, JobConfig
from georecon.pipeline import JobPaths
from georecon.stages import georef
from georecon.stages.georef import (
    collinearity_ratio,
    contiguous_folds,
    solve_collinear,
    solve_sim3,
)
from georecon.util.ply import write_ply
from georecon.util.sim3 import apply, rmse


def _rand_rot(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# --------------------------------------------------------------------------- #
def test_contiguous_folds_are_contiguous():
    folds = contiguous_folds(10, 3)
    assert [f.tolist() for f in folds] == [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
    assert len(contiguous_folds(2, 5)) == 2                      # k clamped to n
    for f in contiguous_folds(23, 4):
        assert list(f) == list(range(int(f[0]), int(f[-1]) + 1))


def test_collinearity_ratio():
    line = np.column_stack([np.arange(50.0), np.zeros(50), np.zeros(50)])
    assert collinearity_ratio(line) < 1e-6
    blob = np.random.default_rng(0).normal(size=(50, 3))
    assert collinearity_ratio(blob) > 0.3


def test_solve_sim3_recovers_scale_under_2m_gps_noise():
    rng = np.random.default_rng(7)
    C = rng.normal(size=(90, 3)) * np.array([40.0, 40.0, 6.0])
    s, R, t = 3.5, _rand_rot(rng), np.array([10.0, 20.0, -5.0])
    U = apply(s, R, t, C) + rng.normal(scale=2.0, size=(90, 3))
    cfg = GeorefCfg(ransac_thr_m=8.0, ransac_iters=800)
    ss, RR, tt, inl, _ = solve_sim3(C, U, cfg, seed=0)
    assert abs(ss - s) / s < 0.02
    assert inl.mean() > 0.8


def test_rmse_h_reported_over_inliers_not_all_pairs():
    rng = np.random.default_rng(3)
    n = 40
    C = rng.normal(size=(n, 3)) * np.array([30.0, 30.0, 5.0])
    s, R, t = 2.0, _rand_rot(rng), np.array([1.0, 2.0, 3.0])
    U = apply(s, R, t, C) + rng.normal(scale=0.5, size=(n, 3))
    U[7] += np.array([120.0, 0.0, 0.0])                       # one gross GPS outlier
    cfg = GeorefCfg(ransac_thr_m=5.0, ransac_iters=500)

    ss, RR, tt, inl, _ = solve_sim3(C, U, cfg, seed=0)
    h = np.linalg.norm((apply(ss, RR, tt, C) - U)[:, :2], axis=1)
    assert not inl[7]
    assert rmse(h[inl]) < 2.0            # headline rmse_h: outlier excluded
    assert rmse(h) > 15.0               # rmse_h_all: outlier visible
    assert h.max() > 100.0


def test_solve_collinear_levels_tilted_ground_and_recovers_scale():
    rng = np.random.default_rng(11)
    # ---- ground truth in ENU ----
    n_true = 40
    U_cam = np.column_stack([np.linspace(0, 120, n_true), np.zeros(n_true),
                             np.full(n_true, 55.0)])
    U_cam += rng.normal(scale=0.05, size=U_cam.shape)             # tiny track jitter
    g_xy = rng.uniform(-40, 40, size=(400, 2))
    G_enu = np.column_stack([g_xy, np.zeros(len(g_xy))])          # flat ground at U=0
    G_enu += rng.normal(scale=0.1, size=G_enu.shape)
    view_enu = np.tile([0.0, 0.0, -1.0], (n_true, 1))             # looking straight down

    # ---- put everything in an arbitrary SfM frame ----
    s_t, R_t, t_t = 0.5, _rz(np.radians(40)) @ _rx(np.radians(25)), np.array([3.0, -2.0, 1.0])
    inv = R_t.T
    C_sfm = (U_cam - t_t) @ inv.T / s_t
    P_sfm = (G_enu - t_t) @ inv.T / s_t
    views_sfm = view_enu @ inv.T

    cfg = GeorefCfg(ransac_thr_m=6.0, ransac_iters=400, plane_thr_frac=0.05)
    s, R, t, inl, up = solve_collinear(C_sfm, U_cam, P_sfm, views_sfm, cfg, seed=0)

    assert abs(s - s_t) / s_t < 0.03
    assert up == pytest.approx(1.0)                              # all cameras look down
    ground_n_sfm = inv @ np.array([0.0, 0.0, 1.0])
    lifted = R @ ground_n_sfm
    ang = np.degrees(np.arccos(np.clip(abs(lifted[2]) / np.linalg.norm(lifted), -1, 1)))
    assert ang < 2.0
    assert np.linalg.norm((apply(s, R, t, C_sfm) - U_cam)[:, :2], axis=1).mean() < 2.0


# --------------------------------------------------------------------------- #
def _write_job(tmp_path, *, with_gps):
    paths = JobPaths.for_job(tmp_path / "job").ensure()
    names = [f"f_{i:06d}.jpg" for i in range(5)]
    pd.DataFrame({
        "name": names, "registered": [1] * 5,
        "cx": np.arange(5.0), "cy": np.zeros(5), "cz": np.zeros(5),
        "qw": [1.0] * 5, "qx": [0.0] * 5, "qy": [0.0] * 5, "qz": [0.0] * 5,
    }).to_csv(paths.sfm / "cameras.csv", index=False)
    lat = np.linspace(18.52, 18.5203, 5) if with_gps else [np.nan] * 5
    lon = np.linspace(73.85, 73.8504, 5) if with_gps else [np.nan] * 5
    pd.DataFrame({
        "name": names, "frame_idx": range(5), "t": np.arange(5.0),
        "lat": lat, "lon": lon, "alt": [560.0] * 5 if with_gps else [np.nan] * 5,
        "sharpness": [1.0] * 5, "disp_px": [1.0] * 5,
    }).to_csv(paths.frames / "frames.csv", index=False)
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(80, 3))
    write_ply(paths.sfm / "sparse.ply", xyz, {
        "red": np.zeros(80, np.uint8), "green": np.zeros(80, np.uint8),
        "blue": np.zeros(80, np.uint8), "error": np.zeros(80, np.float32),
        "track_len": np.ones(80, np.float32),
    })
    return paths


def test_no_telemetry_writes_identity_transform(tmp_path, monkeypatch):
    paths = _write_job(tmp_path, with_gps=False)
    monkeypatch.setattr(pipeline, "STAGES", [("georef", georef.run)])
    pipeline.run_job(tmp_path / "job", JobConfig(video_path="x"))

    m = json.loads((tmp_path / "job" / "report" / "metrics.json").read_text())["stages"]["georef"]
    assert m["georeferenced"] is False and m["branch"] == "none" and m["scale"] == 1.0
    tr = json.loads(paths.georef_transform.read_text())
    assert tr["R"] == np.eye(3).tolist() and tr["t"] == [0.0, 0.0, 0.0]
    assert (paths.georef / "cameras_enu.csv").exists()
    assert (paths.georef / "sparse_enu.ply").exists()


def test_georef_stage_with_gps_produces_transform(tmp_path, monkeypatch):
    paths = _write_job(tmp_path, with_gps=True)
    monkeypatch.setattr(pipeline, "STAGES", [("georef", georef.run)])
    pipeline.run_job(tmp_path / "job", JobConfig(video_path="x"))

    m = json.loads((tmp_path / "job" / "report" / "metrics.json").read_text())["stages"]["georef"]
    assert m["georeferenced"] is True
    assert m["pairs"] == 5
    origin = json.loads(paths.georef_origin.read_text())
    assert origin["utm_epsg"] == 32643
    res = pd.read_csv(paths.georef / "residuals.csv")
    assert list(res["name"]) == [f"f_{i:06d}.jpg" for i in range(5)]
