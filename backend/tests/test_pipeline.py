import json

import pytest

from georecon import pipeline
from georecon.config import JobConfig


def _mk_stage(name, counter):
    def _run(ctx):
        counter[name] = counter.get(name, 0) + 1
        return {"runs": counter[name]}
    return _run


def _read_status(job_dir):
    return json.loads((job_dir / "status.json").read_text(encoding="utf-8"))


def test_second_run_skips_cached_stages(tmp_path, monkeypatch):
    counter = {}
    monkeypatch.setattr(pipeline, "STAGES", [
        ("a", _mk_stage("a", counter)),
        ("b", _mk_stage("b", counter)),
    ])
    job_dir = tmp_path / "job"
    cfg = JobConfig()

    pipeline.run_job(job_dir, cfg)
    assert counter == {"a": 1, "b": 1}

    pipeline.run_job(job_dir, cfg)  # everything cached
    assert counter == {"a": 1, "b": 1}

    state = _read_status(job_dir)
    assert state["state"] == "done"
    assert state["progress"] == 1.0


def test_force_from_reruns_that_stage_and_later_only(tmp_path, monkeypatch):
    counter = {}
    monkeypatch.setattr(pipeline, "STAGES", [
        ("a", _mk_stage("a", counter)),
        ("b", _mk_stage("b", counter)),
        ("c", _mk_stage("c", counter)),
    ])
    job_dir = tmp_path / "job"

    pipeline.run_job(job_dir, JobConfig())
    assert counter == {"a": 1, "b": 1, "c": 1}

    pipeline.run_job(job_dir, JobConfig(force_from="b"))
    assert counter == {"a": 1, "b": 2, "c": 2}


def test_failing_stage_sets_failed_status_and_message(tmp_path, monkeypatch):
    def boom(ctx):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(pipeline, "STAGES", [
        ("ok", _mk_stage("ok", {})),
        ("boom", boom),
    ])
    job_dir = tmp_path / "job"

    with pytest.raises(RuntimeError):
        pipeline.run_job(job_dir, JobConfig())

    state = _read_status(job_dir)
    assert state["state"] == "failed"
    assert state["message"] == "boom: RuntimeError: kaboom"


def test_status_json_always_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "STAGES", [("a", _mk_stage("a", {}))])
    job_dir = tmp_path / "job"
    pipeline.run_job(job_dir, JobConfig())

    state = _read_status(job_dir)  # raises if not valid JSON
    for key in ("job_id", "state", "stage", "progress", "message", "warnings", "updated_at"):
        assert key in state
    metrics = json.loads((job_dir / "report" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stages"]["a"]["runs"] == 1
    assert "seconds" in metrics["stages"]["a"]
    assert "total_seconds" in metrics
