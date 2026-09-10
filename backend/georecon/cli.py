"""Command-line entry point: ``python -m georecon.cli``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from georecon.config import JobConfig, PRESETS, settings
from georecon.pipeline import run_job


def _job_dir(name: str) -> Path:
    return Path(settings.jobs_dir) / name


def _compact(path: Path) -> str:
    return json.dumps(json.loads(path.read_text(encoding="utf-8")), separators=(",", ":"))


def cmd_run(a: argparse.Namespace) -> int:
    job_dir = _job_dir(a.job)
    job_dir.mkdir(parents=True, exist_ok=True)

    cfg = JobConfig(
        preset=a.preset,
        mask_dynamic=a.mask_dynamic,
        hfov_deg=a.hfov,
        force_from=a.force_from,
        video_path=str(Path(a.video).resolve()),
        telemetry_path=str(Path(a.telemetry).resolve()) if a.telemetry else None,
    )
    cfg.to_json(job_dir)

    try:
        state = run_job(job_dir, cfg)
    except Exception as exc:  # noqa: BLE001 - surfaced to the operator
        print(f"[FAILED] {type(exc).__name__}: {exc}")
        return 1

    print(f"[{state['state'].upper()}] job={state['job_id']} "
          f"stage={state['stage']} progress={state['progress']:.0%}")
    for w in state["warnings"]:
        print(f"  warn: {w}")
    metrics = job_dir / "report" / "metrics.json"
    if metrics.exists():
        print(f"  metrics: {_compact(metrics)}")
    return 0 if state["state"] == "done" else 1


def cmd_status(a: argparse.Namespace) -> int:
    job_dir = _job_dir(a.job)
    status = job_dir / "status.json"
    metrics = job_dir / "report" / "metrics.json"
    if not status.exists():
        print(f"no status.json for job {a.job!r} ({job_dir})")
        return 1
    print(f"status : {_compact(status)}")
    if metrics.exists():
        print(f"metrics: {_compact(metrics)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="georecon", description="GeoRecon AI pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline for a job")
    r.add_argument("--video", required=True)
    r.add_argument("--telemetry", default=None)
    r.add_argument("--job", required=True, help="job name (workspace dir under jobs/)")
    r.add_argument("--preset", choices=sorted(PRESETS), default="fast")
    r.add_argument("--no-mask", dest="mask_dynamic", action="store_false",
                   help="disable dynamic-object masking")
    r.add_argument("--hfov", type=float, default=None, help="horizontal FOV in degrees")
    r.add_argument("--force-from", dest="force_from", default=None,
                   help="rerun this stage and every later one")
    r.set_defaults(func=cmd_run, mask_dynamic=True)

    s = sub.add_parser("status", help="print status.json + metrics.json for a job")
    s.add_argument("--job", required=True)
    s.set_defaults(func=cmd_status)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
