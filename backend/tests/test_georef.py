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
    holdout_rmse,
    solve_collinear,
    solve_none,
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


def test_holdout_rmse_robust_to_single_outlier():
    rng = np.random.default_rng(21)
    n = 40
    C = np.column_stack([np.linspace(0, 200, n), rng.normal(scale=8, size=n),
                         rng.normal(scale=3, size=n)])
    s, R, t = 1.5, _rand_rot(rng), np.array([4.0, -1.0, 2.0])
    U = apply(s, R, t, C) + rng.normal(scale=0.4, size=(n, 3))
    cfg = GeorefCfg(ransac_thr_m=3.0, ransac_iters=400, holdout_folds=5)

    _, _, _, inl_clean, _ = solve_sim3(C, U, cfg, 0)
    clean = holdout_rmse("sim3", C, U, np.zeros((0, 3)), None, inl_clean, cfg)["h"]

    U_out = U.copy()
    U_out[18] += np.array([100.0, 0.0, 0.0])
    _, _, _, inl_out, _ = solve_sim3(C, U_out, cfg, 0)
    assert not inl_out[18]
    ho = holdout_rmse("sim3", C, U_out, np.zeros((0, 3)), None, inl_out, cfg)
    assert ho["h"] <= 1.5 * clean
    assert ho["h_all"] > 5.0 * ho["h"]                  # outlier-inclusive still sees it
    assert len(ho["folds_h"]) == len(ho["folds_v"]) == len(ho["folds_h_all"]) == 5
    # fold holding frame 18: inlier-only fold RMSE stays small, *_all spikes.
    k = next(i for i, fold in enumerate(georef.contiguous_folds(len(C), 5))
             if 18 in fold)
    assert ho["folds_h"][k] <= 1.5 * clean
    assert ho["folds_h_all"][k] > 5.0 * ho["folds_h"][k]


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


def _tilted_plane_scene(rng, tilt_deg, altitude, n_cams=20, n_pts=300):
    """Cameras flying at ``altitude`` above a plane tilted ``tilt_deg`` off
    horizontal (no GPS involved: the "SfM frame" is ground truth here)."""
    tilt = np.radians(tilt_deg)
    nrm = np.array([-np.sin(tilt), 0.0, np.cos(tilt)])          # plane normal
    e1 = np.array([np.cos(tilt), 0.0, np.sin(tilt)])            # in-plane basis
    e2 = np.array([0.0, 1.0, 0.0])
    u = rng.uniform(-40, 40, n_pts)
    v = rng.uniform(-40, 40, n_pts)
    P = u[:, None] * e1 + v[:, None] * e2 + rng.normal(scale=0.05, size=(n_pts, 3))
    cx = np.linspace(-20, 20, n_cams)
    C = np.column_stack([cx, np.zeros(n_cams), np.zeros(n_cams)]) + altitude * nrm
    return C, P, nrm


def test_solve_none_levels_tilted_plane_and_normalises_scale():
    rng = np.random.default_rng(5)
    C, P, nrm = _tilted_plane_scene(rng, tilt_deg=15.0, altitude=50.0)
    cfg = GeorefCfg(plane_thr_frac=0.05, ransac_iters=400)

    s, R, t, level_angle_deg, scale_source, units = solve_none(P, C, cfg, seed=0)

    assert abs(level_angle_deg - 15.0) < 2.0                    # recovers the true tilt
    lifted = R @ nrm                                            # fitted normal -> should land on +Z
    ang = np.degrees(np.arccos(np.clip(lifted[2], -1.0, 1.0)))
    assert ang < 2.0
    assert scale_source == "normalised"
    assert units == "rel. units"
    C_out = apply(s, R, t, C)
    assert abs(np.median(C_out[:, 2]) - 100.0) < 1.0            # normalised height
    P_out = apply(s, R, t, P)
    assert abs(np.median(P_out[:, 2])) < 1.0                    # ground centred at 0


def test_solve_none_scales_from_assumed_altitude():
    rng = np.random.default_rng(6)
    C, P, nrm = _tilted_plane_scene(rng, tilt_deg=10.0, altitude=50.0)
    cfg = GeorefCfg(plane_thr_frac=0.05, ransac_iters=400, assumed_altitude_m=80.0)

    s, R, t, level_angle_deg, scale_source, units = solve_none(P, C, cfg, seed=0)

    assert scale_source == "assumed_altitude"
    assert units == "≈ m (from altitude)"
    C_out = apply(s, R, t, C)
    assert abs(np.median(C_out[:, 2]) - 80.0) < 1.0


# --------------------------------------------------------------------------- #
def _write_none_job(tmp_path, *, tilt_deg=15.0, altitude=50.0, n_cams=20):
    paths = JobPaths.for_job(tmp_path / "job").ensure()
    rng = np.random.default_rng(1)
    C, P, _ = _tilted_plane_scene(rng, tilt_deg, altitude, n_cams=n_cams)
    names = [f"f_{i:06d}.jpg" for i in range(n_cams)]

    pd.DataFrame({
        "name": names, "registered": [1] * n_cams,
        "cx": C[:, 0], "cy": C[:, 1], "cz": C[:, 2],
        "qw": [1.0] * n_cams, "qx": [0.0] * n_cams, "qy": [0.0] * n_cams, "qz": [0.0] * n_cams,
    }).to_csv(paths.sfm / "cameras.csv", index=False)
    pd.DataFrame({
        "name": names, "frame_idx": range(n_cams), "t": np.arange(float(n_cams)),
        "lat": [np.nan] * n_cams, "lon": [np.nan] * n_cams, "alt": [np.nan] * n_cams,
        "sharpness": [1.0] * n_cams, "disp_px": [1.0] * n_cams,
    }).to_csv(paths.frames / "frames.csv", index=False)
    write_ply(paths.sfm / "sparse.ply", P, {
        "red": np.zeros(len(P), np.uint8), "green": np.zeros(len(P), np.uint8),
        "blue": np.zeros(len(P), np.uint8), "error": np.zeros(len(P), np.float32),
        "track_len": np.ones(len(P), np.float32),
    })
    return paths


def test_georef_none_branch_levels_and_normalises_end_to_end(tmp_path, monkeypatch):
    paths = _write_none_job(tmp_path, tilt_deg=15.0, altitude=50.0)
    monkeypatch.setattr(pipeline, "STAGES", [("georef", georef.run)])
    pipeline.run_job(tmp_path / "job", JobConfig(video_path="x"))

    m = json.loads((tmp_path / "job" / "report" / "metrics.json").read_text())["stages"]["georef"]
    assert m["georeferenced"] is False
    assert m["branch"] == "none"
    assert m["scale_source"] == "normalised"
    assert m["units"] == "rel. units"
    assert abs(m["level_angle_deg"] - 15.0) < 2.0               # recovers the true tilt
    assert not paths.georef_origin.exists()

    tr = json.loads(paths.georef_transform.read_text())
    assert tr["scale_source"] == "normalised"
    s, R, t = tr["s"], np.array(tr["R"]), np.array(tr["t"])
    cams = pd.read_csv(paths.sfm / "cameras.csv")
    C_out = apply(s, R, t, cams[["cx", "cy", "cz"]].to_numpy())
    assert abs(np.median(C_out[:, 2]) - 100.0) < 1.0
    assert (paths.georef / "cameras_enu.csv").exists()
    assert (paths.georef / "sparse_enu.ply").exists()

    from georecon.util.ply import read_ply

    ground = read_ply(paths.georef / "sparse_enu.ply")
    ground_z = np.asarray(ground["z"], float)
    assert abs(np.median(ground_z)) < 1.0                       # ground levelled to z=0
    assert np.std(ground_z) < 5.0                                # flat, not still tilted


def test_georef_none_branch_uses_assumed_altitude_end_to_end(tmp_path, monkeypatch):
    paths = _write_none_job(tmp_path, tilt_deg=10.0, altitude=50.0)
    monkeypatch.setattr(pipeline, "STAGES", [("georef", georef.run)])
    cfg = JobConfig(video_path="x", georef=GeorefCfg(assumed_altitude_m=80.0))
    pipeline.run_job(tmp_path / "job", cfg)

    m = json.loads((tmp_path / "job" / "report" / "metrics.json").read_text())["stages"]["georef"]
    assert m["scale_source"] == "assumed_altitude"
    assert m["units"] == "≈ m (from altitude)"
    tr = json.loads(paths.georef_transform.read_text())
    s, R, t = tr["s"], np.array(tr["R"]), np.array(tr["t"])
    cams = pd.read_csv(paths.sfm / "cameras.csv")
    C_out = apply(s, R, t, cams[["cx", "cy", "cz"]].to_numpy())
    assert abs(np.median(C_out[:, 2]) - 80.0) < 1.0


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


def test_no_telemetry_is_not_georeferenced_and_writes_no_origin(tmp_path, monkeypatch):
    paths = _write_job(tmp_path, with_gps=False)
    monkeypatch.setattr(pipeline, "STAGES", [("georef", georef.run)])
    pipeline.run_job(tmp_path / "job", JobConfig(video_path="x"))

    m = json.loads((tmp_path / "job" / "report" / "metrics.json").read_text())["stages"]["georef"]
    assert m["georeferenced"] is False and m["branch"] == "none"
    assert not paths.georef_origin.exists()
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
