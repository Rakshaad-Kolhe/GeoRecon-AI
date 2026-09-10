"""Background job queue for the demo API.

One worker thread drains a FIFO queue; each job runs as an isolated subprocess
``python -m georecon.cli run --job <id>`` (cwd = ``backend/``) so a crash or a
GPU-memory leak can't touch the API and COLMAP's memory is reclaimed per job.
Job state lives entirely on the filesystem (``jobs/<id>/status.json``).
"""

from __future__ import annotations

import json
import os
import queue
import secrets
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import psutil

from georecon.config import settings

BACKEND_DIR = Path(__file__).resolve().parents[1]          # .../backend
CHUNK = 1 << 20                                            # 1 MiB upload chunks
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi"}
TELEMETRY_EXT = {".srt", ".csv"}


class UploadTooLarge(Exception):
    pass


class BadUpload(Exception):
    pass


def jobs_root() -> Path:
    root = Path(settings.jobs_dir)
    if not root.is_absolute():
        root = BACKEND_DIR / root
    return root


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)


def _read_status(job_dir: Path) -> dict:
    p = job_dir / "status.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def write_status(job_dir: Path, patch: dict) -> dict:
    """Atomic merge-write of ``status.json`` (temp file + os.replace)."""
    cur = _read_status(job_dir)
    cur.update(patch)
    cur["updated_at"] = _now_iso()
    tmp = job_dir / "status.json.tmp"
    tmp.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    os.replace(tmp, job_dir / "status.json")
    return cur


def init_status(job_dir: Path, job_id: str, preset: str) -> dict:
    return write_status(job_dir, {
        "job_id": job_id, "state": "queued", "stage": None, "progress": 0.0,
        "message": "", "warnings": [], "created_at": _now_iso(), "preset": preset,
    })


# --------------------------------------------------------------------------- #
# queue + single worker
# --------------------------------------------------------------------------- #
_q: "queue.Queue[str]" = queue.Queue()
_lock = threading.Lock()
_running: dict = {"job_id": None, "proc": None}
_cancelled: set[str] = set()
_worker: threading.Thread | None = None


def queue_len() -> int:
    return _q.qsize()


def running_job() -> str | None:
    with _lock:
        return _running["job_id"]


def enqueue(job_id: str) -> None:
    _q.put(job_id)


def _spawn(job_id: str) -> subprocess.Popen:
    """Launch the CLI runner as an isolated subprocess. Patched out in tests."""
    log = open(jobs_root() / job_id / "input" / "runner.log", "wb")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    return subprocess.Popen(
        [sys.executable, "-m", "georecon.cli", "run", "--job", job_id],
        cwd=str(BACKEND_DIR), stdout=log, stderr=subprocess.STDOUT,
        creationflags=flags,
    )


def _run_one(job_id: str) -> None:
    job_dir = jobs_root() / job_id
    with _lock:
        skip = job_id in _cancelled
    if skip:
        with _lock:
            _cancelled.discard(job_id)
        write_status(job_dir, {"state": "failed", "message": "cancelled"})
        return

    write_status(job_dir, {"state": "running", "message": ""})
    proc = _spawn(job_id)
    with _lock:
        _running.update(job_id=job_id, proc=proc)
    try:
        proc.wait()
    finally:
        with _lock:
            _running.update(job_id=None, proc=None)
            was_cancelled = job_id in _cancelled
            _cancelled.discard(job_id)

    st = _read_status(job_dir)
    if was_cancelled:
        write_status(job_dir, {"state": "failed", "message": "cancelled"})
    elif st.get("state") not in ("done", "failed"):
        write_status(job_dir, {"state": "failed",
                               "message": "runner exited without finishing"})


def _worker_loop() -> None:
    while True:
        job_id = _q.get()
        try:
            _run_one(job_id)
        except Exception as exc:                           # noqa: BLE001
            try:
                write_status(jobs_root() / job_id,
                             {"state": "failed", "message": f"runner: {exc}"})
            except OSError:
                pass
        finally:
            _q.task_done()


def start_worker() -> None:
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_worker_loop, daemon=True,
                                   name="georecon-worker")
        _worker.start()


# --------------------------------------------------------------------------- #
# cancel + startup recovery
# --------------------------------------------------------------------------- #
def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        parent = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return
    for child in parent.children(recursive=True):          # COLMAP etc.
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        parent.kill()
    except psutil.NoSuchProcess:
        pass


def cancel(job_id: str) -> bool:
    job_dir = jobs_root() / job_id
    if not job_dir.exists():
        return False
    with _lock:
        if _running["job_id"] == job_id and _running["proc"] is not None:
            proc = _running["proc"]
            _cancelled.add(job_id)
        else:
            proc = None
            state = _read_status(job_dir).get("state")
            if state in ("queued", None, "running"):
                _cancelled.add(job_id)                     # skipped on dequeue
                write_status(job_dir, {"state": "failed", "message": "cancelled"})
                return True
            return False
    _kill_tree(proc)
    write_status(job_dir, {"state": "failed", "message": "cancelled"})
    return True


def recover_on_startup() -> None:
    root = jobs_root()
    if not root.exists():
        return
    resume: list[tuple[str, str]] = []
    for d in sorted(root.iterdir()):
        if not (d / "status.json").exists():
            continue
        st = _read_status(d)
        if st.get("state") == "running":
            write_status(d, {"state": "failed",
                             "message": "interrupted (server restart)"})
        elif st.get("state") == "queued":
            resume.append((st.get("created_at") or "", d.name))
    for _, job_id in sorted(resume):
        enqueue(job_id)
