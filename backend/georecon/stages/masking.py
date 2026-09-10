"""Dynamic-object masking stage.

Runs YOLO segmentation over every selected keyframe and writes a COLMAP-style
mask per frame (``masks/<image_name>.png``: 0 = ignore, 255 = use). Masks are
written for *all* frames, including those with no detections.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple, Protocol, TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd

from georecon.config import MaskCfg, settings

if TYPE_CHECKING:
    from georecon.pipeline import StageContext

# Enough COCO names for the default class set; unknown ids fall back to str(id).
_COCO = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe",
}


class Det(NamedTuple):
    cls: int
    conf: float
    poly: np.ndarray          # (N, 2) float, original-image pixel coords


class Segmenter(Protocol):
    def predict(self, frames: list[np.ndarray]) -> list[list[Det]]:
        ...


class YoloSegmenter:
    """ultralytics YOLO segmentation wrapper. Imports torch/ultralytics lazily."""

    def __init__(self, weights, cfg: MaskCfg, log):
        import torch
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        cuda = torch.cuda.is_available()
        self.device = 0 if cuda else "cpu"
        self.half = bool(cuda)
        self.imgsz = cfg.imgsz
        self.conf = cfg.conf
        self.classes = list(cfg.classes)
        self.batch = max(1, cfg.batch)
        self._log = log

    def predict(self, frames: list[np.ndarray]) -> list[list[Det]]:
        out: list[list[Det]] = []
        for start in range(0, len(frames), self.batch):
            chunk = frames[start:start + self.batch]
            results = self.model.predict(
                chunk, imgsz=self.imgsz, conf=self.conf, classes=self.classes,
                device=self.device, half=self.half, verbose=False,
            )
            for res in results:
                dets: list[Det] = []
                if res.masks is None:
                    out.append(dets)
                    continue
                cls = res.boxes.cls.cpu().numpy().astype(int)
                conf = res.boxes.conf.cpu().numpy()
                for poly, c, q in zip(res.masks.xy, cls, conf):
                    if poly is None or len(poly) < 3:
                        continue
                    dets.append(Det(int(c), float(q), np.asarray(poly, dtype=np.float32)))
                out.append(dets)
        return out


def get_segmenter(cfg: MaskCfg, models_dir: Path, log) -> Segmenter:
    """Build the segmenter. Tests monkeypatch this."""
    models_dir.mkdir(parents=True, exist_ok=True)
    weights = models_dir / cfg.model
    if not weights.exists():
        log.info("segmentation weights %s absent — ultralytics will download them once",
                 weights)
    return YoloSegmenter(weights, cfg, log)


def _grid(thumbs: list[np.ndarray], out_path: Path) -> None:
    if not thumbs:
        return
    cols = min(4, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    ch = max(t.shape[0] for t in thumbs)
    cw = max(t.shape[1] for t in thumbs)
    sheet = np.zeros((rows * ch, cols * cw, 3), np.uint8)
    for i, th in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * ch:r * ch + th.shape[0], c * cw:c * cw + th.shape[1]] = th
    cv2.imwrite(str(out_path), sheet)


def _overlay_sheet(pool, frames_dir: Path, masks_dir: Path, report_dir: Path) -> None:
    pool = sorted(pool, key=lambda pn: -pn[0])[:12]
    thumbs = []
    for pct, name in pool:
        im = cv2.imread(str(frames_dir / name))
        m = cv2.imread(str(masks_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if im is None or m is None:
            continue
        red = im.copy()
        sel = m == 0
        red[sel] = (0.5 * red[sel] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        s = 240.0 / max(red.shape[:2])
        th = cv2.resize(red, (max(1, round(red.shape[1] * s)), max(1, round(red.shape[0] * s))))
        label = f"{name} {pct:.0%}"
        base = th.shape[0] - 6
        cv2.putText(th, label, (4, base), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
        cv2.putText(th, label, (4, base), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        thumbs.append(th)
    _grid(thumbs, report_dir / "masks_overlay.jpg")


def run(ctx: "StageContext") -> dict:
    cfg = ctx.cfg
    mcfg: MaskCfg = cfg.masking
    paths = ctx.paths

    frames_csv = paths.frames / "frames.csv"
    if not frames_csv.exists():
        raise ValueError("masking: frames.csv not found (run keyframes first)")
    names = pd.read_csv(frames_csv)["name"].astype(str).tolist()

    if not cfg.mask_dynamic:
        ctx.log.info("masking disabled (mask_dynamic=false); no masks written")
        return {"enabled": False}

    t0 = time.time()
    if not names:
        ctx.warn("masking: no keyframes to mask")
        return {"enabled": True, "frames": 0, "seconds": round(time.time() - t0, 3)}

    seg = get_segmenter(mcfg, Path(settings.models_dir), ctx.log)
    device = str(getattr(seg, "device", "cpu"))
    if device == "cpu":
        ctx.warn("masking is running on CPU — expect this stage to be slow")

    imgs = []
    for nm in names:
        im = cv2.imread(str(paths.frames / nm))
        if im is None:
            raise ValueError(f"masking: cannot read keyframe {nm}")
        imgs.append(im)

    t_inf = time.time()
    dets_per_frame = seg.predict(imgs)
    infer_s = time.time() - t_inf
    if len(dets_per_frame) != len(names):
        raise ValueError("masking: segmenter returned wrong number of results")

    dets_by_class: dict[str, int] = {}
    masked_pcts: list[float] = []
    overlay_pool: list[tuple[float, str]] = []
    frames_with_dets = 0
    over_warn = 0
    over_total = 0

    for nm, im, dets in zip(names, imgs, dets_per_frame):
        h, w = im.shape[:2]
        dpx = max(5, round(mcfg.dilate_frac * max(h, w)))
        mask = np.full((h, w), 255, np.uint8)
        for d in dets:
            pts = np.round(d.poly).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 0)
            key = _COCO.get(d.cls, str(d.cls))
            dets_by_class[key] = dets_by_class.get(key, 0) + 1
        if dets:
            frames_with_dets += 1
        if (mask == 0).any():
            ker = np.ones((dpx * 2 + 1, dpx * 2 + 1), np.uint8)
            mask = cv2.erode(mask, ker)          # grow the 0 (ignore) region
        cv2.imwrite(str(paths.masks / f"{nm}.png"), mask)

        pct = float(np.count_nonzero(mask == 0)) / (h * w)
        masked_pcts.append(pct)
        overlay_pool.append((pct, nm))
        if pct > mcfg.max_masked_warn:
            over_total += 1
            if over_warn < 5:
                ctx.warn(f"{nm}: {pct:.0%} of the frame masked (> {mcfg.max_masked_warn:.0%})")
                over_warn += 1
    if over_total > over_warn:
        ctx.warn(f"{over_total} frames exceeded {mcfg.max_masked_warn:.0%} masked "
                 f"(first {over_warn} listed above)")

    _overlay_sheet(overlay_pool, paths.frames, paths.masks, paths.report)

    seconds = time.time() - t0
    pcts = np.asarray(masked_pcts, dtype=float)
    ctx.log.info("masked %d frames on %s (%d with detections), mean %.1f%% / max %.1f%%",
                 len(names), device, frames_with_dets, pcts.mean() * 100, pcts.max() * 100)
    return {
        "enabled": True,
        "device": device,
        "model": mcfg.model,
        "frames": len(names),
        "frames_with_dets": frames_with_dets,
        "dets_by_class": dict(sorted(dets_by_class.items(), key=lambda kv: -kv[1])),
        "masked_pct_mean": round(float(pcts.mean()), 4),
        "masked_pct_max": round(float(pcts.max()), 4),
        "infer_fps": round(len(names) / infer_s, 2) if infer_s > 0 else 0.0,
        "seconds": round(seconds, 3),
    }
