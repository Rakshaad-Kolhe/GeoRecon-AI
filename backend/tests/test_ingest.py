import json

import pytest

from georecon import pipeline
from georecon.config import JobConfig
from georecon.stages import ingest as ingest_stage


@pytest.fixture(autouse=True)
def _ingest_only(monkeypatch):
    """Keep these tests scoped to the ingest stage."""
    monkeypatch.setattr(pipeline, "STAGES", [("ingest", ingest_stage.run)])


def test_ingest_probe_and_coverage(tmp_path, synth_clip):
    video, srt = synth_clip(seconds=6.0, fps=20, w=640, h=360)
    job_dir = tmp_path / "job"
    state = pipeline.run_job(job_dir, JobConfig(video_path=str(video),
                                               telemetry_path=str(srt)))
    assert state["state"] == "done"

    ing = json.loads((job_dir / "report" / "metrics.json").read_text("utf-8"))["stages"]["ingest"]
    assert ing["video"]["width"] == 640
    assert ing["video"]["height"] == 360
    assert ing["video"]["fps"] == pytest.approx(20, abs=1)
    assert ing["video"]["frame_count"] == pytest.approx(120, abs=3)
    assert ing["video"]["duration_s"] == pytest.approx(6.0, abs=0.3)
    assert ing["telemetry"]["coverage"] == pytest.approx(1.0, abs=0.05)
    assert ing["telemetry"]["t0_rule"] == "srt_timestamp"

    assert (job_dir / "input" / "video.mp4").exists()
    assert (job_dir / "input" / "telemetry.srt").exists()
    assert (job_dir / "input" / "telemetry_norm.csv").exists()
    assert not state["warnings"]


def test_ingest_without_telemetry_warns(tmp_path, synth_clip):
    video, _ = synth_clip(with_srt=False)
    job_dir = tmp_path / "job"
    state = pipeline.run_job(job_dir, JobConfig(video_path=str(video)))
    assert state["state"] == "done"
    assert any("unscaled" in w for w in state["warnings"])
