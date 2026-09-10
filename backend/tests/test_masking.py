import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from georecon import pipeline
from georecon.config import JobConfig, MaskCfg
from georecon.pipeline import JobPaths
from georecon.stages import masking
from georecon.stages.masking import Det, YoloSegmenter

SQUARE = np.array([[10, 8], [30, 8], [30, 28], [10, 28]], dtype=float)  # 20x20 box


@pytest.fixture(autouse=True)
def _masking_only(monkeypatch):
    monkeypatch.setattr(pipeline, "STAGES", [("masking", masking.run)])


@pytest.fixture
def job_with_frames(tmp_path):
    def _make(n=4, w=64, h=48):
        paths = JobPaths.for_job(tmp_path / "job").ensure()
        rows = []
        for i in range(n):
            nm = f"f_{i:06d}.jpg"
            cv2.imwrite(str(paths.frames / nm), np.full((h, w, 3), 120, np.uint8))
            rows.append({"name": nm, "frame_idx": i, "t": round(i * 0.1, 2),
                         "lat": 18.5, "lon": 73.85, "alt": 10.0,
                         "sharpness": 100.0, "disp_px": 1.0})
        pd.DataFrame(rows).to_csv(paths.frames / "frames.csv", index=False)
        return tmp_path / "job", paths
    return _make


class FakeSeg:
    device = "cpu"

    def __init__(self, det_frames=None, poly=SQUARE):
        self.det_frames = det_frames          # None => detection on every frame
        self.poly = poly
        self.calls: list[int] = []

    def predict(self, frames):
        self.calls.append(len(frames))
        out = []
        for i in range(len(frames)):
            if self.det_frames is None or i in self.det_frames:
                out.append([Det(0, 0.9, self.poly.copy())])
            else:
                out.append([])
        return out


def _run(job_dir, **cfg_kw):
    return pipeline.run_job(job_dir, JobConfig(**cfg_kw))


# --------------------------------------------------------------------------- #
def test_masks_written_for_every_frame(job_with_frames, monkeypatch):
    job_dir, paths = job_with_frames(n=4)
    fake = FakeSeg(det_frames={0, 1})
    monkeypatch.setattr(masking, "get_segmenter", lambda *a, **k: fake)

    state = _run(job_dir)
    assert state["state"] == "done"

    pngs = sorted(p.name for p in paths.masks.glob("*.png"))
    assert pngs == [f"f_{i:06d}.jpg.png" for i in range(4)]
    for i, name in enumerate(pngs):
        m = cv2.imread(str(paths.masks / name), cv2.IMREAD_UNCHANGED)
        assert m.shape == (48, 64) and m.dtype == np.uint8
        if i in (0, 1):
            assert m[18, 20] == 0           # inside the polygon
            assert (m == 255).any()         # ... but not the whole frame
        else:
            assert (m == 255).all()         # no detections -> fully usable


def test_polygon_region_masked_and_grows_with_dilation(job_with_frames, monkeypatch):
    job_dir, paths = job_with_frames(n=1)
    monkeypatch.setattr(masking, "get_segmenter", lambda *a, **k: FakeSeg(det_frames={0}))
    _run(job_dir)

    m = cv2.imread(str(paths.masks / "f_000000.jpg.png"), cv2.IMREAD_UNCHANGED)
    raw = np.full((48, 64), 255, np.uint8)
    cv2.fillPoly(raw, [SQUARE.round().astype(np.int32)], 0)
    assert (m == 0).sum() > (raw == 0).sum()      # dilation enlarged the ignore region


def test_disabled_writes_nothing(job_with_frames, monkeypatch):
    job_dir, paths = job_with_frames()
    hit = []
    monkeypatch.setattr(masking, "get_segmenter", lambda *a, **k: hit.append(1))

    _run(job_dir, mask_dynamic=False)
    m = json.loads((job_dir / "report" / "metrics.json").read_text("utf-8"))["stages"]["masking"]
    assert m["enabled"] is False
    assert not list(paths.masks.glob("*.png"))
    assert not hit


def test_class_filter_passed_through_to_model_predict():
    seg = YoloSegmenter.__new__(YoloSegmenter)   # skip lazy torch/ultralytics import
    rec = {}

    class _Model:
        def predict(self, chunk, **kw):
            rec.update(kw)
            rec["n"] = len(chunk)
            return []

    seg.model = _Model()
    seg.imgsz, seg.conf, seg.classes, seg.batch = 1280, 0.25, [0, 2, 7], 8
    seg.device, seg.half = "cpu", False

    seg.predict([np.zeros((4, 4, 3), np.uint8)])
    assert rec["classes"] == [0, 2, 7]
    assert rec["imgsz"] == 1280 and rec["conf"] == 0.25


def test_overlay_sheet_created(job_with_frames, monkeypatch):
    job_dir, paths = job_with_frames(n=3)
    monkeypatch.setattr(masking, "get_segmenter", lambda *a, **k: FakeSeg())
    _run(job_dir)
    assert (paths.report / "masks_overlay.jpg").exists()


def test_metrics_report_device_and_classes(job_with_frames, monkeypatch):
    job_dir, paths = job_with_frames(n=3)
    monkeypatch.setattr(masking, "get_segmenter", lambda *a, **k: FakeSeg(det_frames={0, 2}))
    _run(job_dir)
    m = json.loads((job_dir / "report" / "metrics.json").read_text("utf-8"))["stages"]["masking"]
    assert m["enabled"] is True
    assert m["device"] == "cpu"
    assert m["frames"] == 3 and m["frames_with_dets"] == 2
    assert m["dets_by_class"] == {"person": 2}


# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.skipif(not Path("models/yolo11n-seg.pt").exists(),
                    reason="real YOLO weights not present")
def test_real_yolo_smoke(job_with_frames):
    job_dir, paths = job_with_frames(n=2, w=320, h=240)
    state = _run(job_dir)
    assert state["state"] == "done"
    assert len(list(paths.masks.glob("*.png"))) == 2
