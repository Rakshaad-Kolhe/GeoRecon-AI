import json

import numpy as np
import pandas as pd
import pytest

from georecon import pipeline
from georecon.config import JobConfig, KeyframeCfg
from georecon.stages.keyframes import select

CFG = KeyframeCfg()


def _moving(n):
    lat = 18.5 + np.arange(n) * 1e-4
    lon = 73.85 + np.arange(n) * 1e-4
    return lat, lon


# --------------------------------------------------------------------------- #
# select() - pure
# --------------------------------------------------------------------------- #
def test_select_hover_yields_single_keyframe():
    n = 200
    sharp = np.full(n, 100.0)
    disp = np.full(n, 0.01)                 # essentially stationary
    lat = np.full(n, 18.5)
    lon = np.full(n, 73.85)
    pos, reasons, thr = select(sharp, disp, lat, lon, CFG, 20.0, max_keyframes=300)
    assert pos == [0]


def test_select_never_picks_blurred_frames():
    n = 240
    sharp = np.full(n, 100.0)
    sharp[95:105] = 8.0                     # narrow blur dip
    disp = np.full(n, 2.0)
    lat, lon = _moving(n)
    pos, reasons, thr = select(sharp, disp, lat, lon, CFG, 20.0, max_keyframes=300)
    assert len(pos) > 3
    assert all(not (95 <= p <= 104) for p in pos)
    assert reasons["rejected_blur"] >= 10


def test_select_respects_cap_and_raises_threshold():
    n = 400
    sharp = np.full(n, 100.0)
    disp = np.full(n, 2.0)
    lat, lon = _moving(n)
    pos, reasons, thr = select(sharp, disp, lat, lon, CFG, 20.0, max_keyframes=8)
    assert len(pos) <= 8
    assert thr > 20.0


def test_select_uniform_motion_spacing_near_threshold():
    n = 300
    sharp = np.full(n, 100.0)
    disp = np.full(n, 2.0)                  # thr / disp = 10
    lat, lon = _moving(n)
    pos, reasons, thr = select(sharp, disp, lat, lon, CFG, 20.0, max_keyframes=300)
    gaps = np.diff(pos)
    assert abs(float(np.median(gaps)) - 10.0) <= 1.0


def test_select_forces_boundaries_across_lost_tracking_with_moving_gps():
    n = 120
    sharp = np.full(n, 100.0)
    disp = np.full(n, 2.0)
    disp[40:75] = np.nan                    # long lost-tracking stretch
    lat = 18.5 + np.arange(n) * 2.0e-5      # ~2.2 m per analysed step -> not hover
    lon = np.full(n, 73.85)
    pos, reasons, thr = select(sharp, disp, lat, lon, CFG, 20.0, max_keyframes=500)
    assert reasons["forced_lost"] > 0
    assert any(40 <= p <= 74 for p in pos), "no keyframe across the lost stretch"

    # with the same lost stretch but a *stationary* GPS track it must not force
    lat_hover = np.full(n, 18.5)
    lon_hover = np.full(n, 73.85)
    p2, r2, _ = select(sharp, disp, lat_hover, lon_hover, CFG, 20.0, max_keyframes=500)
    assert r2["forced_lost"] == 0


# --------------------------------------------------------------------------- #
# full stage on a synthetic clip
# --------------------------------------------------------------------------- #
def test_keyframes_integration(tmp_path, synth_clip, monkeypatch):
    monkeypatch.setattr(pipeline, "STAGES", pipeline.STAGES[:2])  # ingest + keyframes only
    video, srt = synth_clip(seconds=6.0, fps=20, w=640, h=360,
                            speed_px=5.0, blur_seg=(2.0, 2.5))
    job_dir = tmp_path / "job"
    cfg = JobConfig(video_path=str(video), telemetry_path=str(srt), preset="fast")
    state = pipeline.run_job(job_dir, cfg)
    assert state["state"] == "done", state

    m = json.loads((job_dir / "report" / "metrics.json").read_text("utf-8"))
    kf = m["stages"]["keyframes"]
    assert kf["keyframes"] > 5

    fdf = pd.read_csv(job_dir / "frames" / "frames.csv")
    jpgs = sorted((job_dir / "frames").glob("f_*.jpg"))
    assert len(fdf) == len(jpgs) == kf["keyframes"]
    assert fdf["lat"].notna().all() and fdf["lon"].notna().all()
    assert not ((fdf["t"] >= 2.0) & (fdf["t"] < 2.5)).any()
    assert (job_dir / "report" / "keyframes_contact.jpg").exists()
    assert (job_dir / "report" / "keyframes_timeline.csv").exists()
