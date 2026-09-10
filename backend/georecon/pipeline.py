"""Stage runner: workspace paths, per-job logging, caching and status updates."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from georecon.config import JobConfig, Preset
from georecon.stages import ingest as _ingest_stage

# Ordered stage registry. Later PRs append their stages here.
STAGES: list[tuple[str, Callable[["StageContext"], dict]]] = [
    ("ingest", _ingest_stage.run),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload) -> None:
    """Write JSON via a temp file + os.replace, retrying the replace on Windows
    where a virus scanner or indexer can briefly lock the destination."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05)


@dataclass
class JobPaths:
    """One attribute per directory / file in the job workspace contract."""

    root: Path
    input: Path
    frames: Path
    frames_csv: Path
    masks: Path
    sfm: Path
    sfm_db: Path
    sfm_sparse: Path
    georef: Path
    georef_transform: Path
    georef_origin: Path
    dense: Path
    dense_fused: Path
    outputs: Path
    outputs_web: Path
    report: Path
    metrics_json: Path
    status_json: Path
    log_txt: Path
    stages_dir: Path

    @classmethod
    def for_job(cls, job_dir) -> "JobPaths":
        r = Path(job_dir)
        return cls(
            root=r,
            input=r / "input",
            frames=r / "frames",
            frames_csv=r / "frames" / "frames.csv",
            masks=r / "masks",
            sfm=r / "sfm",
            sfm_db=r / "sfm" / "database.db",
            sfm_sparse=r / "sfm" / "sparse",
            georef=r / "georef",
            georef_transform=r / "georef" / "transform.json",
            georef_origin=r / "georef" / "origin.json",
            dense=r / "dense",
            dense_fused=r / "dense" / "fused.ply",
            outputs=r / "outputs",
            outputs_web=r / "outputs" / "web",
            report=r / "report",
            metrics_json=r / "report" / "metrics.json",
            status_json=r / "status.json",
            log_txt=r / "log.txt",
            stages_dir=r / ".stages",
        )

    def ensure(self) -> "JobPaths":
        for d in (
            self.input, self.frames, self.masks, self.sfm, self.sfm_sparse,
            self.georef, self.dense, self.outputs, self.outputs_web,
            self.report, self.stages_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self


class _StageFilter(logging.Filter):
    """Injects the current stage name into every log record."""

    def __init__(self) -> None:
        super().__init__()
        self.stage = "-"

    def filter(self, record: logging.LogRecord) -> bool:
        record.stage = self.stage
        return True


class StageContext:
    """Everything a stage needs: paths, config, resolved preset, logger, warn()."""

    def __init__(self, job_id, paths, cfg, preset, log, warnings, persist_status):
        self.job_id: str = job_id
        self.paths: JobPaths = paths
        self.cfg: JobConfig = cfg
        self.preset: Preset = preset
        self.log: logging.Logger = log
        self.warnings: list[str] = warnings
        self._persist_status = persist_status

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self.log.warning(msg)
        self._persist_status()


def _setup_logging(job_id: str, log_path: Path):
    logger = logging.getLogger(f"georecon.job.{job_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)

    stage_filter = _StageFilter()
    fmt = logging.Formatter("%(asctime)s [%(stage)s] %(levelname)s %(message)s")
    handlers = [logging.FileHandler(log_path, encoding="utf-8"),
                logging.StreamHandler(sys.stdout)]
    for h in handlers:
        h.setFormatter(fmt)
        h.addFilter(stage_filter)
        logger.addHandler(h)
    return logger, stage_filter, handlers


def _marker_path(paths: JobPaths, name: str) -> Path:
    return paths.stages_dir / f"{name}.json"


def _rebuild_metrics(paths: JobPaths, names: list[str]) -> None:
    stages: dict[str, dict] = {}
    total = 0.0
    for n in names:
        m = _marker_path(paths, n)
        if not m.exists():
            continue
        data = json.loads(m.read_text(encoding="utf-8"))
        seconds = float(data.get("seconds", 0.0) or 0.0)
        entry = dict(data.get("metrics") or {})
        entry["seconds"] = seconds
        stages[n] = entry
        total += seconds
    _atomic_write_json(paths.metrics_json, {"stages": stages, "total_seconds": round(total, 3)})


def run_job(job_dir, cfg: JobConfig) -> dict:
    """Run every registered stage in order, with caching and status updates."""

    paths = JobPaths.for_job(job_dir).ensure()
    job_id = paths.root.name
    logger, stage_filter, handlers = _setup_logging(job_id, paths.log_txt)

    warnings: list[str] = []
    state = {
        "job_id": job_id,
        "state": "queued",
        "stage": None,
        "progress": 0.0,
        "message": "",
        "warnings": warnings,
        "updated_at": None,
    }

    def write_status() -> None:
        state["updated_at"] = _now_iso()
        _atomic_write_json(paths.status_json, state)

    stages = list(STAGES)
    names = [n for n, _ in stages]
    total = len(stages)

    if cfg.force_from:
        if cfg.force_from not in names:
            raise ValueError(
                f"force_from={cfg.force_from!r} is not a known stage: {names}"
            )
        for n in names[names.index(cfg.force_from):]:
            _marker_path(paths, n).unlink(missing_ok=True)

    state["state"] = "running"
    write_status()

    done = 0
    try:
        for name, fn in stages:
            stage_filter.stage = name
            state["stage"] = name
            marker = _marker_path(paths, name)
            if marker.exists():
                logger.info("skip (cached)")
                done += 1
                state["progress"] = done / total
                write_status()
                _rebuild_metrics(paths, names)
                continue

            write_status()
            ctx = StageContext(job_id, paths, cfg, cfg.resolved_preset,
                               logger, warnings, write_status)
            logger.info("start")
            t0 = time.time()
            metrics = fn(ctx) or {}
            seconds = round(time.time() - t0, 3)
            marker.write_text(
                json.dumps({"metrics": metrics, "seconds": seconds,
                            "finished_at": _now_iso()}, indent=2),
                encoding="utf-8",
            )
            done += 1
            state["progress"] = done / total
            logger.info("done (%.3fs)", seconds)
            write_status()
            _rebuild_metrics(paths, names)

        state["state"] = "done"
        state["progress"] = 1.0
        state["message"] = "ok"
        write_status()
    except Exception as exc:
        stage = state["stage"]
        logger.error("failed\n%s", traceback.format_exc())
        state["state"] = "failed"
        state["message"] = f"{stage}: {type(exc).__name__}: {exc}"
        write_status()
        raise
    finally:
        _rebuild_metrics(paths, names)
        for h in handlers:
            h.close()
            logger.removeHandler(h)

    return state
