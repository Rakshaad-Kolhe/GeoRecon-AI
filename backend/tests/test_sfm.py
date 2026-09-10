import json
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from georecon import pipeline
from georecon.config import JobConfig, SfmCfg
from georecon.stages.sfm import (
    _camera_rows,
    build_reader_options,
    largest_unregistered_gap,
    mask_dir_for,
    want_gpu,
)

REAL = Path(__file__).resolve().parents[2] / "data" / "samples" / "real"


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flags,expected", [
    ([True] * 5, 0),
    ([False] * 4, 4),
    ([True, False, False, True, False, False, False, True], 3),
    ([], 0),
    ([True, False, True], 1),
    ([False, False, True, False], 2),
])
def test_largest_unregistered_gap(flags, expected):
    assert largest_unregistered_gap(flags) == expected


def test_want_gpu():
    assert want_gpu(True) is True
    assert want_gpu(False) is False
    assert want_gpu("false") is False
    assert want_gpu("auto") == bool(pycolmap.has_cuda)


def test_mask_dir_for(tmp_path):
    md = tmp_path / "masks"
    md.mkdir()
    assert mask_dir_for(True, md) is None            # enabled but nothing written
    (md / "f_000000.jpg.png").write_bytes(b"x")
    assert mask_dir_for(True, md) == md
    assert mask_dir_for(False, md) is None           # disabled -> no mask path


def test_build_reader_options_mask_path(tmp_path):
    cfg = SfmCfg()
    off = build_reader_options(cfg, None, None)
    assert str(off.mask_path) in (".", "")           # default, i.e. unset
    on = build_reader_options(cfg, tmp_path, "1200.0,960.0,540.0,0.0")
    assert str(on.mask_path) == str(tmp_path)
    assert on.camera_model == "SIMPLE_RADIAL"
    assert "1200" in on.camera_params


# --------------------------------------------------------------------------- #
# cameras.csv from a duck-typed reconstruction
# --------------------------------------------------------------------------- #
class _Rot:
    def __init__(self, quat):
        self.quat = np.asarray(quat, dtype=float)      # [x, y, z, w]


class _Rigid:
    def __init__(self, quat):
        self.rotation = _Rot(quat)


class _Img:
    def __init__(self, image_id, name, center, quat_xyzw):
        self.image_id = image_id
        self.name = name
        self._c = np.asarray(center, dtype=float)
        self._q = quat_xyzw

    def projection_center(self):
        return self._c

    def cam_from_world(self):
        return _Rigid(self._q)


class _Recon:
    def __init__(self, imgs, reg_ids):
        self._imgs = imgs
        self._reg = list(reg_ids)

    @property
    def images(self):
        imgs = self._imgs

        class _Map:
            def values(self):
                return list(imgs)

        return _Map()

    def reg_image_ids(self):
        return list(self._reg)


def test_camera_rows_registered_blank_and_quat_order():
    imgs = [
        _Img(1, "f_000000.jpg", [1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]),
        _Img(2, "f_000005.jpg", [4.0, 5.0, 6.0], [0.1, 0.2, 0.3, 0.9]),
    ]
    recon = _Recon(imgs, reg_ids=[1])                 # img 2 exists but is unregistered
    names = ["f_000000.jpg", "f_000005.jpg", "f_000010.jpg"]

    rows = _camera_rows(recon, names)
    assert [r["registered"] for r in rows] == [1, 0, 0]
    assert (rows[0]["cx"], rows[0]["cy"], rows[0]["cz"]) == (1.0, 2.0, 3.0)
    assert (rows[0]["qw"], rows[0]["qx"], rows[0]["qy"], rows[0]["qz"]) == (1.0, 0.0, 0.0, 0.0)
    assert rows[1]["cx"] == "" and rows[2]["qw"] == ""


# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.skipif(not (REAL / "video.mp4").exists(), reason="no data/samples/real fixture")
def test_sfm_full_stage_on_real(tmp_path):
    job_dir = tmp_path / "real"
    cfg = JobConfig(video_path=str(REAL / "video.mp4"),
                    telemetry_path=str(REAL / "telemetry.csv"),
                    preset="fast", mask_dynamic=False)
    state = pipeline.run_job(job_dir, cfg)
    assert state["state"] == "done", state
    m = json.loads((job_dir / "report" / "metrics.json").read_text("utf-8"))["stages"]["sfm"]
    assert m["registered"] >= 3 and m["points3d"] > 0
    assert (job_dir / "sfm" / "sparse" / "0" / "cameras.bin").exists()
    assert (job_dir / "sfm" / "sparse.ply").exists()
    assert (job_dir / "sfm" / "cameras.csv").exists()
