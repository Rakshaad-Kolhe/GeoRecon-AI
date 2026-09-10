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


def parse_set_overrides(items) -> dict:
    """``["sfm.seq_overlap=20", "preset=accurate"]`` -> nested dict.

    Each value is JSON-parsed, falling back to the raw string.
    """
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = out
        parts = [p for p in key.split(".") if p]
        if not parts:
            raise ValueError(f"--set has an empty key: {item!r}")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"--set path conflict at {part!r} in {item!r}")
        node[parts[-1]] = value
    return out


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def cmd_run(a: argparse.Namespace) -> int:
    job_dir = _job_dir(a.job)
    job_dir.mkdir(parents=True, exist_ok=True)

    base = dict(
        preset=a.preset,
        mask_dynamic=a.mask_dynamic,
        hfov_deg=a.hfov,
        force_from=a.force_from,
        video_path=str(Path(a.video).resolve()),
        telemetry_path=str(Path(a.telemetry).resolve()) if a.telemetry else None,
        telemetry_offset_s=a.telemetry_offset,
    )
    try:
        merged = _deep_merge(base, parse_set_overrides(a.set))
        cfg = JobConfig(**merged)
    except (ValueError, TypeError) as exc:
        print(f"[FAILED] bad --set override: {exc}")
        return 1
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
    r.add_argument("--telemetry-offset", dest="telemetry_offset", type=float, default=0.0,
                   help="seconds to add to video time when sampling telemetry")
    r.add_argument("--force-from", dest="force_from", default=None,
                   help="rerun this stage and every later one")
    r.add_argument("--set", dest="set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any JobConfig field, e.g. --set sfm.seq_overlap=20 "
                        "(repeatable; value is JSON-parsed)")
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
