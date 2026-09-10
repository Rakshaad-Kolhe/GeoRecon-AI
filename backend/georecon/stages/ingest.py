"""Ingest stage: copy source media into the workspace, probe video, normalise telemetry."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from georecon.ingest.telemetry import load_telemetry, telemetry_stats

if TYPE_CHECKING:
    from georecon.pipeline import StageContext


def _probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            frame_count = 0
            while cap.read()[0]:
                frame_count += 1
    finally:
        cap.release()

    if width <= 0 or height <= 0 or frame_count <= 0:
        raise ValueError(f"unreadable video (dims/frame count) : {path}")
    duration_s = frame_count / fps if fps > 0 else 0.0
    return {
        "fps": round(fps, 4),
        "frame_count": frame_count,
        "duration_s": round(duration_s, 4),
        "width": width,
        "height": height,
    }


def _ingest_media(src: Path, input_dir: Path, stem: str) -> Path:
    """Return the workspace copy of ``src`` (``input/<stem>.<ext>``), copying it
    in only when it does not already live in ``input/`` (API uploads stream
    straight there)."""
    src = src.resolve()
    dst = (input_dir / f"{stem}{src.suffix.lower()}").resolve()
    if src.parent == input_dir.resolve() or src == dst:
        return src
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def run(ctx: "StageContext") -> dict:
    cfg = ctx.cfg
    paths = ctx.paths

    if not cfg.video_path:
        raise ValueError("job config has no video_path")
    src_video = Path(cfg.video_path)
    if not src_video.exists():
        raise ValueError(f"video not found: {src_video}")

    dst_video = _ingest_media(src_video, paths.input, "video")

    probe = _probe_video(dst_video)
    ctx.log.info(
        "video %dx%d @ %.3f fps, %d frames, %.2fs",
        probe["width"], probe["height"], probe["fps"],
        probe["frame_count"], probe["duration_s"],
    )
    metrics: dict = {"video": probe}

    if cfg.telemetry_path:
        src_tel = Path(cfg.telemetry_path)
        if not src_tel.exists():
            raise ValueError(f"telemetry not found: {src_tel}")
        dst_tel = _ingest_media(src_tel, paths.input, "telemetry")

        df = load_telemetry(dst_tel)
        df.to_csv(paths.input / "telemetry_norm.csv", index=False)
        stats = telemetry_stats(df)
        t0_rule = df.attrs.get("t0_rule", "first_sample")

        vid_span = probe["duration_s"]
        overlap = max(0.0, min(stats["t_end"], vid_span) - max(stats["t_start"], 0.0))
        coverage = overlap / vid_span if vid_span > 0 else 0.0
        metrics["telemetry"] = {**stats, "coverage": round(coverage, 4), "t0_rule": t0_rule}
        ctx.log.info("telemetry %d rows @ %.3f Hz, coverage %.1f%%, t0=%s",
                     stats["rows"], stats["hz"], coverage * 100, t0_rule)
        if coverage < 0.9:
            ctx.warn(f"telemetry covers only {coverage:.0%} of the video span")
    else:
        ctx.warn("no telemetry: model will be unscaled")
        metrics["telemetry"] = None

    return metrics
