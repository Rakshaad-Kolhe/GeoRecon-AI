"""Structure-from-Motion stage (pycolmap).

Verified against pycolmap 4.2.0 (``python -c "import pycolmap"`` on this machine,
``pycolmap.__version__`` / ``pycolmap.has_cuda``). Only the names below are used:

  pycolmap.has_cuda                      -> bool attribute (NOT callable); False on CPU wheel
  pycolmap.Device.{auto,cpu,cuda}
  pycolmap.CameraMode.{AUTO,PER_FOLDER,PER_IMAGE,SINGLE}
  pycolmap.extract_features(database_path, image_path, image_names=[],
        camera_mode=CameraMode.AUTO, reader_options=ImageReaderOptions(),
        extraction_options=FeatureExtractionOptions(), device=Device.auto) -> None
  ImageReaderOptions: .camera_model (str) .camera_params (str "f,cx,cy,k")
        .mask_path (dir; COLMAP looks for <mask_path>/<image_name>.png) .existing_camera_id
  FeatureExtractionOptions: .use_gpu (bool) .max_image_size .gpu_index
        .sift.max_num_features
  pycolmap.match_sequential(database_path, matching_options=FeatureMatchingOptions(),
        pairing_options=SequentialPairingOptions(),
        verification_options=TwoViewGeometryOptions(), device=Device.auto) -> None
  SequentialPairingOptions: .overlap (int) .quadratic_overlap (bool) .loop_detection (bool)
  FeatureMatchingOptions: .use_gpu (bool) .max_num_matches
  pycolmap.global_mapping(database_path, image_path, output_path,
        options=GlobalPipelineOptions()) -> dict[int, Reconstruction]     # GLOMAP
  pycolmap.incremental_mapping(database_path, image_path, output_path,
        options=IncrementalPipelineOptions(), input_path='') -> dict[int, Reconstruction]
  GlobalPipelineOptions / IncrementalPipelineOptions: .min_model_size .min_num_matches
  Reconstruction: .num_reg_images() .reg_image_ids() -> list[int] .images (ImageMap)
        .points3D (Point3DMap) .cameras (CameraMap) .num_points3D()
        .compute_mean_track_length() .compute_mean_reprojection_error()
        .write(dir)  # COLMAP binary: cameras.bin/images.bin/points3D.bin
  Image: .name .image_id .camera_id .has_pose (bool)
        .projection_center() -> np(3)   # camera centre C in the SfM frame
        .cam_from_world() -> Rigid3d    # world -> cam;  .rotation.quat is [x,y,z,w]
  Camera: .model_name .width .height .params (np)  .focal_length
  Point3D: .xyz (np3) .color (np3 uint8) .error (float) .track.length()
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd
import pycolmap

from georecon.config import SfmCfg
from georecon.util import colmap_cli
from georecon.util.ply import write_ply

if TYPE_CHECKING:
    from georecon.pipeline import StageContext


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def largest_unregistered_gap(registered_flags) -> int:
    """Longest run of consecutive False (unregistered) values, in order."""
    best = run = 0
    for ok in registered_flags:
        run = 0 if ok else run + 1
        best = max(best, run)
    return best


def want_gpu(use_gpu) -> bool:
    if isinstance(use_gpu, bool):
        return use_gpu
    s = str(use_gpu).strip().lower()
    if s == "auto":
        return bool(pycolmap.has_cuda)
    return s in ("true", "1", "yes", "gpu", "cuda")


def gpu_not_disabled(use_gpu) -> bool:
    """True unless GPU was turned off explicitly. Gates the CUDA-CLI hybrid,
    which is worth trying even when the pycolmap wheel is CPU-only
    (``pycolmap.has_cuda`` False) — that is the whole reason to shell out."""
    if isinstance(use_gpu, bool):
        return use_gpu
    return str(use_gpu).strip().lower() not in ("false", "0", "no", "cpu", "off")


def build_reader_options(cfg: SfmCfg, mask_dir: Path | None,
                         cam_params: str | None) -> pycolmap.ImageReaderOptions:
    ro = pycolmap.ImageReaderOptions()
    ro.camera_model = cfg.camera_model
    if mask_dir is not None:
        ro.mask_path = str(mask_dir)
    if cam_params:
        ro.camera_params = cam_params
    return ro


def mask_dir_for(mask_dynamic: bool, masks_dir: Path) -> Path | None:
    """The masks directory to hand COLMAP, or None (disabled / no masks written)."""
    if not mask_dynamic:
        return None
    return masks_dir if any(masks_dir.glob("*.png")) else None


def _best(recons: dict) -> "pycolmap.Reconstruction | None":
    if not recons:
        return None
    return max(recons.values(), key=lambda r: r.num_reg_images())


def _camera_rows(recon, names: list[str]) -> list[dict]:
    """One row per frame name: pose (C + cam_from_world quat) or blanks."""
    reg = set(recon.reg_image_ids())
    by_name = {im.name: im for im in recon.images.values()}
    rows = []
    for nm in names:
        im = by_name.get(nm)
        if im is not None and im.image_id in reg:
            c = np.asarray(im.projection_center(), dtype=float).ravel()
            q = np.asarray(im.cam_from_world().rotation.quat, dtype=float).ravel()  # xyzw
            rows.append({"name": nm, "registered": 1,
                         "cx": c[0], "cy": c[1], "cz": c[2],
                         "qw": q[3], "qx": q[0], "qy": q[1], "qz": q[2]})
        else:
            rows.append({"name": nm, "registered": 0, "cx": "", "cy": "", "cz": "",
                         "qw": "", "qx": "", "qy": "", "qz": ""})
    return rows


def _write_sparse_ply(recon, path: Path) -> None:
    pts = list(recon.points3D.values())
    if pts:
        xyz = np.array([p.xyz for p in pts], dtype=np.float64)
        rgb = np.array([p.color for p in pts], dtype=np.uint8)
        err = np.array([p.error for p in pts], dtype=np.float32)
        tlen = np.array([p.track.length() for p in pts], dtype=np.float32)
    else:
        xyz = np.zeros((0, 3)); rgb = np.zeros((0, 3), np.uint8)
        err = np.zeros((0,), np.float32); tlen = np.zeros((0,), np.float32)
    write_ply(path, xyz, {"red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2],
                          "error": err, "track_len": tlen})


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #
def _extract(db: Path, frames_dir: Path, names, ro, fe, device) -> None:
    pycolmap.extract_features(
        str(db), str(frames_dir), image_names=list(names),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=ro, extraction_options=fe, device=device,
    )


def _extract_match_cli(ctx, cfg: SfmCfg, paths, frames_dir: Path, names,
                       mask_dir, cam_params):
    """GPU feature extraction + sequential matching via the native COLMAP CLI.
    Mapping still runs through pycolmap. Returns (extract_s, match_s)."""
    img_list = paths.sfm / "image_list.txt"
    img_list.write_text("\n".join(names) + "\n", encoding="utf-8")

    fe_opts = {
        "database_path": str(paths.sfm_db),
        "image_path": str(frames_dir),
        "image_list_path": str(img_list),
        "ImageReader.single_camera": True,
        "ImageReader.camera_model": cfg.camera_model,
        "ImageReader.mask_path": str(mask_dir) if mask_dir is not None else None,
        "ImageReader.camera_params": cam_params,
        "SiftExtraction.max_num_features": cfg.max_features,
        "FeatureExtraction.use_gpu": True,
    }
    t0 = time.time()
    colmap_cli.run("feature_extractor", fe_opts, ctx.log)
    extract_s = time.time() - t0
    if mask_dir is not None:
        ctx.log.info("CLI feature extraction used masks in %s", mask_dir)

    mt_opts = {
        "database_path": str(paths.sfm_db),
        "SequentialMatching.overlap": int(cfg.seq_overlap),
        "SequentialMatching.quadratic_overlap": bool(cfg.quadratic_overlap),
        "SequentialMatching.loop_detection": False,
        "FeatureMatching.use_gpu": True,
    }
    t0 = time.time()
    colmap_cli.run("sequential_matcher", mt_opts, ctx.log)
    match_s = time.time() - t0
    return extract_s, match_s


def run(ctx: "StageContext") -> dict:
    cfg: SfmCfg = ctx.cfg.sfm
    paths = ctx.paths
    frames_dir = paths.frames
    frames_csv = frames_dir / "frames.csv"
    if not frames_csv.exists():
        raise ValueError("sfm: frames.csv not found (run keyframes first)")
    names = pd.read_csv(frames_csv)["name"].astype(str).tolist()
    if len(names) < 3:
        raise ValueError(f"sfm: need >= 3 keyframes, have {len(names)}")

    # fresh database + sparse dir every run
    if paths.sfm_db.exists():
        paths.sfm_db.unlink()
    if paths.sfm_sparse.exists():
        shutil.rmtree(paths.sfm_sparse)
    paths.sfm_sparse.mkdir(parents=True, exist_ok=True)

    # camera focal prior from hfov, if given
    im0 = None
    for nm in names:
        im0 = paths.frames / nm
        if im0.exists():
            break
    probe = cv2.imread(str(im0))
    if probe is None:
        raise ValueError(f"sfm: cannot read keyframe {im0}")
    h, w = probe.shape[:2]
    cam_params = None
    if ctx.cfg.hfov_deg:
        f = (w / 2.0) / math.tan(math.radians(ctx.cfg.hfov_deg) / 2.0)
        cam_params = f"{f:.4f},{w / 2:.1f},{h / 2:.1f},0.0"
        ctx.log.info("hfov %.1f deg -> focal prior %.1f px", ctx.cfg.hfov_deg, f)

    mask_dir = mask_dir_for(ctx.cfg.mask_dynamic, paths.masks)
    gpu = want_gpu(cfg.use_gpu)
    cli = colmap_cli.probe()
    # SfM feature extraction + matching share the database file, so the CLI and
    # pycolmap must agree on major.minor. Dense (bin model reader) is stable
    # across 4.x, so a mismatch only disables the SfM hybrid, not dense.
    db_compat = None
    if cli["available"] and cli["version"]:
        db_compat = (".".join(cli["version"].split(".")[:2])
                     == ".".join(pycolmap.__version__.split(".")[:2]))
    gpu_cli = gpu_not_disabled(cfg.use_gpu) and cli["available"] and cli["cuda"]
    hybrid = gpu_cli and db_compat is not False
    if gpu_cli and db_compat is False:
        ctx.warn(f"COLMAP CLI {cli['version']} vs pycolmap {pycolmap.__version__}: "
                 f"major.minor differ — SfM hybrid disabled (shared DB format); "
                 f"CLI still used for dense")

    device = pycolmap.Device.cuda if (gpu and not hybrid) else pycolmap.Device.cpu
    device_str = "cuda-cli" if hybrid else ("cuda" if gpu and pycolmap.has_cuda else "cpu")
    ctx.log.info("pycolmap %s | has_cuda=%s | colmap-cli=%s | device=%s | %d frames",
                 pycolmap.__version__, pycolmap.has_cuda,
                 cli["version"] if cli["available"] else "no", device_str, len(names))

    if hybrid:
        extract_s, match_s = _extract_match_cli(ctx, cfg, paths, frames_dir, names,
                                                mask_dir, cam_params)
    else:
        # ---- feature extraction (GPU -> CPU retry) -----------------------------
        fe = pycolmap.FeatureExtractionOptions()
        fe.sift.max_num_features = cfg.max_features
        fe.use_gpu = (device == pycolmap.Device.cuda)
        ro = build_reader_options(cfg, mask_dir, cam_params)

        t0 = time.time()
        try:
            _extract(paths.sfm_db, frames_dir, names, ro, fe, device)
        except Exception as exc:                   # noqa: BLE001
            if device != pycolmap.Device.cuda:
                raise
            ctx.warn(f"GPU feature extraction failed ({type(exc).__name__}: {exc}); "
                     f"retrying on CPU")
            device, device_str = pycolmap.Device.cpu, "cpu"
            fe.use_gpu = False
            paths.sfm_db.unlink(missing_ok=True)
            _extract(paths.sfm_db, frames_dir, names, ro, fe, device)
        extract_s = time.time() - t0
        if mask_dir is not None:
            ctx.log.info("feature extraction used masks in %s", mask_dir)

        # ---- sequential matching --------------------------------------------
        mm = pycolmap.FeatureMatchingOptions()
        mm.use_gpu = (device == pycolmap.Device.cuda)
        sp = pycolmap.SequentialPairingOptions()
        sp.overlap = int(cfg.seq_overlap)
        sp.quadratic_overlap = bool(cfg.quadratic_overlap)
        sp.loop_detection = False
        t0 = time.time()
        pycolmap.match_sequential(str(paths.sfm_db), matching_options=mm,
                                  pairing_options=sp, device=device)
        match_s = time.time() - t0

    # ---- mapping: global (GLOMAP) with incremental fallback ----------------
    scratch = paths.sfm_sparse / "_work"
    scratch.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    recons: dict = {}
    mapper_used = None
    if cfg.mapper == "global" and hasattr(pycolmap, "global_mapping"):
        try:
            gopts = pycolmap.GlobalPipelineOptions()
            gopts.min_model_size = cfg.min_model_size
            recons = pycolmap.global_mapping(str(paths.sfm_db), str(frames_dir),
                                             str(scratch), gopts)
            mapper_used = "global"
        except Exception as exc:                   # noqa: BLE001
            ctx.warn(f"global mapper failed ({type(exc).__name__}: {exc}); "
                     f"falling back to incremental")
            recons = {}

    best = _best(recons)
    if best is None or best.num_reg_images() < cfg.min_model_size:
        if mapper_used == "global":
            ctx.warn("global model smaller than min_model_size; retrying incremental")
        iopts = pycolmap.IncrementalPipelineOptions()
        iopts.min_model_size = cfg.min_model_size
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True, exist_ok=True)
        recons = pycolmap.incremental_mapping(str(paths.sfm_db), str(frames_dir),
                                              str(scratch), iopts)
        mapper_used = "incremental"
        best = _best(recons)
    map_s = time.time() - t0
    num_models = len(recons)
    shutil.rmtree(scratch, ignore_errors=True)

    if best is None:
        raise ValueError("sfm: mapping produced no reconstruction")

    # ---- write outputs ---------------------------------------------------
    out0 = paths.sfm_sparse / "0"
    out0.mkdir(parents=True, exist_ok=True)
    best.write(str(out0))
    _write_sparse_ply(best, paths.sfm / "sparse.ply")

    rows = _camera_rows(best, names)
    pd.DataFrame(rows, columns=["name", "registered", "cx", "cy", "cz",
                                "qw", "qx", "qy", "qz"]).to_csv(
        paths.sfm / "cameras.csv", index=False)

    cam = next(iter(best.cameras.values()))
    (paths.sfm / "intrinsics.json").write_text(json.dumps({
        "model": cam.model_name, "w": int(cam.width), "h": int(cam.height),
        "params": [float(v) for v in cam.params],
    }, indent=2), encoding="utf-8")

    # ---- metrics + warnings --------------------------------------------
    registered = best.num_reg_images()
    reg_pct = 100.0 * registered / len(names)
    gap = largest_unregistered_gap([r["registered"] == 1 for r in rows])
    reproj = float(best.compute_mean_reprojection_error())
    metrics = {
        "pycolmap_version": pycolmap.__version__,
        "device": device_str,
        "colmap_db_compatible": db_compat,
        "mapper_used": mapper_used,
        "frames": len(names),
        "registered": registered,
        "registered_pct": round(reg_pct, 2),
        "largest_gap": gap,
        "num_models": num_models,
        "points3d": int(best.num_points3D()),
        "mean_track_len": round(float(best.compute_mean_track_length()), 3),
        "mean_reproj_px": round(reproj, 4),
        "focal_px": round(float(cam.focal_length), 2),
        "extract_s": round(extract_s, 3),
        "match_s": round(match_s, 3),
        "map_s": round(map_s, 3),
    }
    ctx.log.info("sfm: %d/%d registered (%.0f%%) via %s, %d pts, reproj %.2f px",
                 registered, len(names), reg_pct, mapper_used,
                 metrics["points3d"], reproj)
    if reg_pct < 60:
        ctx.warn(f"only {reg_pct:.0f}% of frames registered (<60%)")
    if gap > 5:
        ctx.warn(f"{gap} consecutive frames unregistered (track break)")
    if reproj > 2:
        ctx.warn(f"mean reprojection error {reproj:.2f}px (>2px)")
    return metrics
