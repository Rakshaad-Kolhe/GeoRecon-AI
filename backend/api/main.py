"""FastAPI app for the GeoRecon AI demo — job submission, status, artifacts."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from georecon.config import PRESETS, GeorefCfg, JobConfig, settings
from georecon.util.version import code_version

from . import jobs

_MEDIA = {
    ".ply": "application/octet-stream", ".las": "application/octet-stream",
    ".glb": "model/gltf-binary", ".obj": "text/plain", ".tif": "image/tiff",
    ".geojson": "application/geo+json", ".png": "image/png",
    ".md": "text/markdown", ".json": "application/json",
}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    jobs.start_worker()
    jobs.recover_on_startup()
    yield


app = FastAPI(title="GeoRecon AI", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:5173"],
    allow_methods=["*"], allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _job_dir(job_id: str) -> Path:
    d = jobs.jobs_root() / job_id
    if "/" in job_id or "\\" in job_id or not d.exists():
        raise HTTPException(404, f"unknown job {job_id!r}")
    return d


def _mtime_iso(p: Path) -> str | None:
    if not p.exists():
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()


def _check_ext(name: str, allowed: set[str], kind: str) -> str:
    ext = Path(name or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"{kind} must be one of {sorted(allowed)}, got {ext or '?'}")
    return ext


async def _stream_upload(upload: UploadFile, dst: Path, limit_bytes: int) -> Path:
    written = 0
    with open(dst, "wb") as fh:
        while True:
            chunk = await upload.read(jobs.CHUNK)
            if not chunk:
                break
            written += len(chunk)
            if written > limit_bytes:
                raise jobs.UploadTooLarge()
            fh.write(chunk)
    return dst


def _serve(base: Path, rel: str) -> FileResponse:
    base = base.resolve()
    target = (base / rel).resolve()
    if target != base and base not in target.parents:
        raise HTTPException(403, "path escapes the job directory")
    if not target.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(target, media_type=_MEDIA.get(target.suffix.lower()))


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict:
    import pycolmap

    from georecon.util import colmap_cli

    try:
        import torch

        torch_cuda = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if torch_cuda else None
    except Exception:                                       # noqa: BLE001
        torch_cuda, gpu_name = False, None
    return {
        "torch_cuda": torch_cuda,
        "gpu_name": gpu_name,
        "colmap": colmap_cli.probe(),
        "pycolmap_version": pycolmap.__version__,
        "queue_len": jobs.queue_len(),
        "running_job": jobs.running_job(),
        "code_version": code_version(),
    }


@app.post("/api/jobs", status_code=201)
async def create_job(
    video: UploadFile = File(...),
    telemetry: UploadFile | None = File(None),
    preset: str = Form("fast"),
    mask_dynamic: bool = Form(True),
    telemetry_offset_s: float = Form(0.0),
    assumed_altitude_m: float | None = Form(None),
) -> dict:
    if preset not in PRESETS:
        raise HTTPException(400, f"unknown preset {preset!r}; choose {sorted(PRESETS)}")
    v_ext = _check_ext(video.filename, jobs.VIDEO_EXT, "video")
    have_tel = telemetry is not None and (telemetry.filename or "")
    t_ext = _check_ext(telemetry.filename, jobs.TELEMETRY_EXT, "telemetry") if have_tel else None

    job_id = jobs.new_id()
    job_dir = jobs.jobs_root() / job_id
    (job_dir / "input").mkdir(parents=True, exist_ok=True)
    limit = settings.max_upload_mb * (1 << 20)
    try:
        v_path = await _stream_upload(video, job_dir / "input" / f"video{v_ext}", limit)
        used = v_path.stat().st_size
        t_path = None
        if have_tel:
            t_path = await _stream_upload(
                telemetry, job_dir / "input" / f"telemetry{t_ext}", limit - used)
    except jobs.UploadTooLarge:
        import shutil

        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(413, f"upload exceeds {settings.max_upload_mb} MB")

    cv = code_version()
    cfg = JobConfig(
        video_path=str(v_path),
        telemetry_path=str(t_path) if t_path else None,
        preset=preset, mask_dynamic=mask_dynamic,
        telemetry_offset_s=telemetry_offset_s,
        georef=GeorefCfg(assumed_altitude_m=assumed_altitude_m),
        code_commit=cv["commit"], code_dirty=cv["dirty"],
    )
    cfg.to_json(job_dir)
    jobs.init_status(job_dir, job_id, preset)
    jobs.enqueue(job_id)
    return {"job_id": job_id}


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    root = jobs.jobs_root()
    out: list[dict] = []
    if not root.exists():
        return out
    for d in root.iterdir():
        if not (d / "status.json").exists():
            continue
        try:
            st = json.loads((d / "status.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        total = None
        mj = d / "report" / "metrics.json"
        if mj.exists():
            try:
                total = json.loads(mj.read_text(encoding="utf-8")).get("total_seconds")
            except (OSError, json.JSONDecodeError):
                pass
        out.append({
            **st,
            "job_id": d.name,
            "created_at": st.get("created_at") or _mtime_iso(d / "config.json"),
            "preset": st.get("preset"),
            "total_seconds": total,
        })
    out.sort(key=lambda j: j.get("created_at") or "", reverse=True)
    return out


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str) -> dict:
    d = _job_dir(job_id)
    st = {"job_id": job_id}
    if (d / "status.json").exists():
        st = json.loads((d / "status.json").read_text(encoding="utf-8"))
    mj = d / "report" / "metrics.json"
    st["metrics"] = json.loads(mj.read_text(encoding="utf-8")) if mj.exists() else None
    return st


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def job_log(job_id: str, tail: int = Query(200, ge=1, le=20000)) -> str:
    log = _job_dir(job_id) / "log.txt"
    if not log.exists():
        return ""
    with open(log, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 256 * 1024))                 # tail of the file only
        text = fh.read().decode("utf-8", "replace")
    return "\n".join(text.splitlines()[-tail:])


@app.get("/api/jobs/{job_id}/files")
def job_files(job_id: str) -> list[dict]:
    out_dir = _job_dir(job_id) / "outputs"
    if not out_dir.exists():
        return []
    return [
        {"path": p.relative_to(out_dir).as_posix(), "bytes": p.stat().st_size}
        for p in sorted(out_dir.rglob("*")) if p.is_file()
    ]


@app.get("/api/jobs/{job_id}/files/{path:path}")
def job_file(job_id: str, path: str) -> FileResponse:
    return _serve(_job_dir(job_id) / "outputs", path)


@app.get("/api/jobs/{job_id}/report/{path:path}")
def job_report_file(job_id: str, path: str) -> FileResponse:
    return _serve(_job_dir(job_id) / "report", path)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    _job_dir(job_id)
    return {"cancelled": jobs.cancel(job_id)}


# --------------------------------------------------------------------------- #
# optional: serve a built frontend from the same process (demo convenience)
# --------------------------------------------------------------------------- #
_DIST = jobs.BACKEND_DIR.parent / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="spa")
