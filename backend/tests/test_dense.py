import json

import numpy as np

from georecon import pipeline
from georecon.config import JobConfig, settings
from georecon.pipeline import JobPaths
from georecon.stages import dense
from georecon.stages.dense import _points_per_m2, _transform_cloud
from georecon.util import colmap_cli
from georecon.util.ply import read_ply, write_ply


def _rand_rot(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_transform_cloud_keeps_unit_normals():
    rng = np.random.default_rng(1)
    n = 200
    xyz = rng.normal(size=(n, 3)) * 5.0
    nrm = rng.normal(size=(n, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    d = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
         "nx": nrm[:, 0], "ny": nrm[:, 1], "nz": nrm[:, 2],
         "red": np.zeros(n, np.uint8), "green": np.zeros(n, np.uint8),
         "blue": np.zeros(n, np.uint8)}
    R = _rand_rot(rng)
    out = _transform_cloud(d, 7.3, R, np.array([10.0, -3.0, 2.0]))

    on = np.stack([out["nx"], out["ny"], out["nz"]], -1)
    assert np.allclose(np.linalg.norm(on, axis=1), 1.0, atol=1e-6)
    assert np.allclose(on, nrm @ R.T, atol=1e-6)          # rotation only
    oxyz = np.stack([out["x"], out["y"], out["z"]], -1)
    assert np.allclose(oxyz, 7.3 * (xyz @ R.T) + np.array([10.0, -3.0, 2.0]))


def test_points_per_m2_square_and_degenerate():
    sq = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], float)
    assert abs(_points_per_m2(sq) - 0.04) < 1e-9        # 4 pts / 100 m^2
    collinear = np.array([[i, i, 0] for i in range(5)], float)
    assert _points_per_m2(collinear) == 0.0             # QhullError -> 0, not a crash


def test_dense_skips_and_copies_sparse_when_no_colmap(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "colmap_bin", "")
    colmap_cli.reset_probe_cache()

    paths = JobPaths.for_job(tmp_path / "job").ensure()
    paths.georef_transform.write_text(json.dumps(
        {"s": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]}))
    rng = np.random.default_rng(0)
    write_ply(paths.georef / "sparse_enu.ply", rng.normal(size=(60, 3)), {
        "red": np.zeros(60, np.uint8), "green": np.zeros(60, np.uint8),
        "blue": np.zeros(60, np.uint8), "error": np.zeros(60, np.float32),
        "track_len": np.ones(60, np.float32),
    })

    monkeypatch.setattr(pipeline, "STAGES", [("dense", dense.run)])
    pipeline.run_job(tmp_path / "job", JobConfig(video_path="x"))

    m = json.loads((tmp_path / "job" / "report" / "metrics.json").read_text())["stages"]["dense"]
    assert m["dense"] is False
    assert paths.dense_fused.exists()
    a, b = read_ply(paths.georef / "sparse_enu.ply"), read_ply(paths.dense_fused)
    assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["track_len"], b["track_len"])
