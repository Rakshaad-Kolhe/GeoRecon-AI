import json

import numpy as np

from georecon import pipeline
from georecon.config import DenseCfg, JobConfig, settings
from georecon.pipeline import JobPaths
from georecon.stages import dense
from georecon.stages.dense import (
    _georef_outliers,
    _points_per_m2,
    _rewrite_patchmatch_cfg,
    _transform_cloud,
    _write_clean_model,
)
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


def test_rewrite_patchmatch_cfg_pins_every_source_spec(tmp_path):
    cfg = tmp_path / "patch-match.cfg"
    cfg.write_text("a.jpg\n__auto__, 20\nb.jpg\n__auto__, 20\nc.jpg\n__auto__, 20\n")
    n_ref = _rewrite_patchmatch_cfg(cfg, 8)
    lines = cfg.read_text().splitlines()
    assert n_ref == 3
    assert lines[0::2] == ["a.jpg", "b.jpg", "c.jpg"]
    assert set(lines[1::2]) == {"__auto__, 8"}


def test_rewrite_patchmatch_cfg_ref_stride_explicit_sources_from_kept(tmp_path):
    cfg = tmp_path / "patch-match.cfg"
    names = [f"f_{i}.jpg" for i in range(6)]
    cfg.write_text("".join(f"{n}\n__auto__, 20\n" for n in names))
    n_ref = _rewrite_patchmatch_cfg(cfg, 8, ref_stride=2)
    lines = cfg.read_text().splitlines()
    kept = {"f_0.jpg", "f_2.jpg", "f_4.jpg"}
    assert n_ref == 3
    assert lines[0::2] == ["f_0.jpg", "f_2.jpg", "f_4.jpg"]
    for ref, src in zip(lines[0::2], lines[1::2]):
        srcs = [s.strip() for s in src.split(",")]
        assert "__auto__" not in src                     # explicit, not auto
        assert set(srcs) <= kept and ref not in srcs      # sources are kept refs


def test_georef_outliers_reads_inlier_zero(tmp_path):
    csvp = tmp_path / "residuals.csv"
    csvp.write_text("name,dE,dN,dU,inlier\n"
                    "f_0.jpg,0.1,0.1,0.1,1\n"
                    "f_1.jpg,50,2,3,0\n"
                    "f_2.jpg,0.2,0.1,0.0,1\n")
    assert _georef_outliers(csvp) == ["f_1.jpg"]
    assert _georef_outliers(tmp_path / "missing.csv") == []


class _FakeImage:
    def __init__(self, name, frame_id):
        self.name, self.frame_id, self.has_pose = name, frame_id, True


class _FakeRecon:
    def __init__(self, _path):
        self.images = {i: _FakeImage(f"f_{i}.jpg", 10 + i) for i in range(4)}
        self.deregistered, self.written_to = [], None

    def deregister_frame(self, fid):
        self.deregistered.append(fid)

    def write(self, path):
        self.written_to = path


def test_write_clean_model_deregisters_named_outliers(tmp_path, monkeypatch):
    import pycolmap

    monkeypatch.setattr(pycolmap, "Reconstruction", _FakeRecon)
    captured = {}
    real_init = _FakeRecon.__init__

    def _spy_init(self, path):
        real_init(self, path)
        captured["recon"] = self

    monkeypatch.setattr(_FakeRecon, "__init__", _spy_init)

    out = tmp_path / "sparse_clean"
    excluded = _write_clean_model(tmp_path / "model", ["f_1.jpg", "f_9.jpg"], out)
    assert excluded == ["f_1.jpg"]                       # f_9 not in the model
    assert captured["recon"].deregistered == [11]        # frame_id of f_1
    assert captured["recon"].written_to == str(out)


def test_resolved_dense_preset_override_and_explicit():
    assert JobConfig(preset="fast").resolved_dense == DenseCfg(
        num_src_images=8, window_radius=4, num_iterations=4, ref_stride=2)
    assert JobConfig(preset="accurate").resolved_dense == DenseCfg(
        num_src_images=12, window_radius=5, num_iterations=5, ref_stride=1)
    # explicit --set override beats the preset
    over = DenseCfg(num_src_images=6, window_radius=4, num_iterations=4)
    assert JobConfig(preset="accurate", dense=over).resolved_dense == over


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
