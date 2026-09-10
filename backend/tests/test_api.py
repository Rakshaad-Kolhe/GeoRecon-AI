import json
import queue

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api import jobs
from api.main import _serve, app
from georecon.config import settings


class _FakeProc:
    """Stand-in for the CLI subprocess: on wait() it writes a finished job."""

    def __init__(self, job_dir):
        self.job_dir = job_dir
        self.pid = -1

    def wait(self, timeout=None):
        (self.job_dir / "report").mkdir(parents=True, exist_ok=True)
        (self.job_dir / "report" / "metrics.json").write_text(json.dumps(
            {"stages": {"ingest": {"seconds": 0.1}, "sfm": {"seconds": 1.0}},
             "total_seconds": 1.1}))
        jobs.write_status(self.job_dir,
                          {"state": "done", "message": "ok", "progress": 1.0})
        return 0


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "jobs_dir", str(tmp_path / "jobs"))
    monkeypatch.setattr(jobs, "start_worker", lambda: None)      # no background thread
    jobs._q = queue.Queue()
    jobs._cancelled = set()
    jobs._running = {"job_id": None, "proc": None}
    monkeypatch.setattr(jobs, "_spawn",
                        lambda job_id: _FakeProc(jobs.jobs_root() / job_id))
    with TestClient(app) as client:
        yield client


def _post_job(client, *, video=("clip.mp4", b"video-bytes"), tel=None, **form):
    files = {"video": (video[0], video[1], "application/octet-stream")}
    if tel is not None:
        files["telemetry"] = (tel[0], tel[1], "text/plain")
    return client.post("/api/jobs", files=files, data={"preset": "fast", **form})


def test_health(api):
    h = api.get("/api/health").json()
    assert set(h) >= {"torch_cuda", "gpu_name", "colmap", "pycolmap_version",
                      "queue_len", "running_job"}
    assert h["colmap"].keys() >= {"available", "version", "cuda"}


def test_create_then_queued_then_done(api):
    r = _post_job(api, tel=("t.csv", b"t,lat,lon\n0,1,2\n"))
    assert r.status_code == 201
    jid = r.json()["job_id"]

    st = api.get(f"/api/jobs/{jid}").json()
    assert st["state"] == "queued" and st["preset"] == "fast"
    assert st["created_at"] and st["metrics"] is None

    jobs._run_one(jid)                                          # drain synchronously

    st = api.get(f"/api/jobs/{jid}").json()
    assert st["state"] == "done"
    assert st["metrics"]["total_seconds"] == 1.1
    listing = api.get("/api/jobs").json()
    row = next(j for j in listing if j["job_id"] == jid)
    assert row["total_seconds"] == 1.1 and row["preset"] == "fast"


def test_upload_too_large_413(api, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    r = _post_job(api, video=("big.mp4", b"x" * (2 << 20)))
    assert r.status_code == 413
    assert not (jobs.jobs_root()).exists() or not any(jobs.jobs_root().iterdir())


def test_bad_extension_400(api):
    assert _post_job(api, video=("notes.txt", b"nope")).status_code == 400
    assert _post_job(api, tel=("t.json", b"{}")).status_code == 400


def test_path_traversal_403(api, tmp_path):
    base = tmp_path / "outputs"
    (base).mkdir()
    (tmp_path / "secret.txt").write_text("shhh")
    with pytest.raises(HTTPException) as exc:
        _serve(base, "../secret.txt")
    assert exc.value.status_code == 403


def test_legacy_job_listed_with_mtime_created_at(api, tmp_path):
    d = jobs.jobs_root() / "legacy-run"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"preset": "balanced", "video_path": "x"}))
    (d / "status.json").write_text(json.dumps(
        {"job_id": "legacy-run", "state": "done", "stage": "validate", "progress": 1.0}))

    rows = api.get("/api/jobs").json()
    row = next(j for j in rows if j["job_id"] == "legacy-run")
    from api.main import _mtime_iso

    assert row["created_at"] == _mtime_iso(d / "config.json")
    assert row["created_at"] is not None


def test_cancel_queued_job(api):
    jid = _post_job(api).json()["job_id"]
    assert api.get(f"/api/jobs/{jid}").json()["state"] == "queued"

    r = api.post(f"/api/jobs/{jid}/cancel")
    assert r.status_code == 200 and r.json()["cancelled"] is True
    st = api.get(f"/api/jobs/{jid}").json()
    assert st["state"] == "failed" and st["message"] == "cancelled"


def test_startup_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "jobs_dir", str(tmp_path / "jobs"))
    jobs._q = queue.Queue()
    root = jobs.jobs_root()
    for name, state, created in [("r1", "running", "2026-01-01T00:00:00+00:00"),
                                 ("q1", "queued", "2026-01-02T00:00:00+00:00"),
                                 ("q2", "queued", "2026-01-01T12:00:00+00:00")]:
        d = root / name
        d.mkdir(parents=True)
        (d / "status.json").write_text(json.dumps(
            {"job_id": name, "state": state, "created_at": created}))

    jobs.recover_on_startup()

    assert json.loads((root / "r1" / "status.json").read_text())["state"] == "failed"
    assert "interrupted" in json.loads((root / "r1" / "status.json").read_text())["message"]
    assert [jobs._q.get_nowait() for _ in range(2)] == ["q2", "q1"]   # by created_at
