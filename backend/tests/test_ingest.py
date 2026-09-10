import json

import cv2
import numpy as np
import pytest

from georecon import pipeline
from georecon.config import JobConfig

FPS = 10
N_FRAMES = 20  # -> 2.0 s
W, H = 160, 120

SRT = """1
00:00:00,000 --> 00:00:01,000
[latitude: 18.5204] [longitude: 73.8567] [rel_alt: 50.0 abs_alt: 610.0]

2
00:00:01,000 --> 00:00:02,000
[latitude: 18.5210] [longitude: 73.8572] [rel_alt: 51.0 abs_alt: 611.0]

3
00:00:02,000 --> 00:00:03,000
[latitude: 18.5216] [longitude: 73.8577] [rel_alt: 52.0 abs_alt: 612.0]
"""


def _make_video(path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, FPS, (W, H))
    if not vw.isOpened():
        pytest.skip("no mp4v encoder available in this OpenCV build")
    rng = np.random.default_rng(0)
    for _ in range(N_FRAMES):
        vw.write(rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
    vw.release()
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("mp4v encoder produced no output")


def test_ingest_probe_and_coverage(tmp_path):
    video = tmp_path / "clip.mp4"
    _make_video(video)
    srt = tmp_path / "clip.srt"
    srt.write_text(SRT, encoding="utf-8")

    job_dir = tmp_path / "job"
    cfg = JobConfig(video_path=str(video), telemetry_path=str(srt))
    state = pipeline.run_job(job_dir, cfg)
    assert state["state"] == "done"

    metrics = json.loads((job_dir / "report" / "metrics.json").read_text(encoding="utf-8"))
    ing = metrics["stages"]["ingest"]
    assert ing["video"]["width"] == W
    assert ing["video"]["height"] == H
    assert ing["video"]["fps"] == pytest.approx(FPS, abs=0.5)
    assert ing["video"]["frame_count"] == N_FRAMES
    assert ing["video"]["duration_s"] == pytest.approx(2.0, abs=0.25)
    assert ing["telemetry"]["coverage"] == pytest.approx(1.0, abs=0.05)
    assert ing["telemetry"]["rows"] == 3

    # media copied into the workspace
    assert (job_dir / "input" / "video.mp4").exists()
    assert (job_dir / "input" / "telemetry.srt").exists()
    assert (job_dir / "input" / "telemetry_norm.csv").exists()
    assert not state["warnings"]  # full coverage -> no warning


def test_ingest_without_telemetry_warns(tmp_path):
    video = tmp_path / "clip.mp4"
    _make_video(video)
    job_dir = tmp_path / "job"
    state = pipeline.run_job(job_dir, JobConfig(video_path=str(video)))
    assert state["state"] == "done"
    assert any("unscaled" in w for w in state["warnings"])
