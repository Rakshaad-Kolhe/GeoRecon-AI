"""Dense MVS stage (native CUDA COLMAP) -> georeferenced dense point cloud.

Needs a CUDA COLMAP CLI (``GEORECON_COLMAP_BIN``). Without one the stage is a
no-op: it copies ``georef/sparse_enu.ply`` to ``dense/fused.ply`` and reports
``dense=false`` so downstream meshing still has an input.

MVS runs in the SfM frame, undistorting from a copy of ``sfm/sparse/0`` with
georef outliers deregistered (``dense/sparse_clean``); the fused cloud is then
mapped to ENU with ``georef/transform.json`` (xyz: sRX+t, normals: Rn).
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from georecon.util import colmap_cli
from georecon.util.colmap_vis import read_vis
from georecon.util.ply import read_ply, write_ply

if TYPE_CHECKING:
    from georecon.pipeline import StageContext


def _georef_outliers(residuals_csv: Path) -> list[str]:
    """Image names flagged ``inlier=0`` by georef (residuals.csv)."""
    if not residuals_csv.exists():
        return []
    with open(residuals_csv, newline="") as fh:
        return [r["name"] for r in csv.DictReader(fh) if r.get("inlier") == "0"]


def _write_clean_model(sfm_model: Path, outliers: list[str], out_dir: Path) -> list[str]:
    """Copy the SfM model with ``outliers`` deregistered (pycolmap frame API).
    Returns the names actually deregistered."""
    import pycolmap

    recon = pycolmap.Reconstruction(str(sfm_model))
    name_to_frame = {im.name: im.frame_id for im in recon.images.values()
                     if im.has_pose}
    excluded = []
    for name in outliers:
        fid = name_to_frame.get(name)
        if fid is not None:
            recon.deregister_frame(fid)
            excluded.append(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    recon.write(str(out_dir))
    return excluded


def _rewrite_patchmatch_cfg(cfg_path: Path, num_src: int) -> int:
    """COLMAP writes patch-match.cfg as alternating <ref-image> / <source-spec>
    lines. Pin every source spec to ``__auto__, <num_src>``. Returns the ref
    count."""
    lines = [ln for ln in cfg_path.read_text().splitlines() if ln.strip()]
    for i in range(1, len(lines), 2):
        lines[i] = f"__auto__, {num_src}"
    cfg_path.write_text("\n".join(lines) + "\n")
    return (len(lines) + 1) // 2


def _transform_cloud(d: dict, s: float, R: np.ndarray, t: np.ndarray) -> dict:
    xyz = np.stack([d["x"], d["y"], d["z"]], axis=-1).astype(float)
    if len(xyz):
        xyz = s * (xyz @ R.T) + t
    out = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    if {"nx", "ny", "nz"} <= d.keys():
        n = np.stack([d["nx"], d["ny"], d["nz"]], axis=-1).astype(float)
        if len(n):
            n = n @ R.T
            norm = np.linalg.norm(n, axis=1, keepdims=True)
            n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 1e-12)
        out["nx"], out["ny"], out["nz"] = n[:, 0], n[:, 1], n[:, 2]
    for k in ("red", "green", "blue"):
        if k in d:
            out[k] = d[k].astype(np.uint8)
    return out


def _points_per_m2(xyz: np.ndarray) -> float:
    if len(xyz) < 3:
        return 0.0
    from scipy.spatial import ConvexHull  # required dep — let ImportError surface
    from scipy.spatial import QhullError

    try:
        area = float(ConvexHull(xyz[:, :2]).volume)   # 2-D hull "volume" == area
    except QhullError:                                # collinear / degenerate XY
        return 0.0
    return float(len(xyz) / area) if area > 0 else 0.0


def run(ctx: "StageContext") -> dict:
    paths = ctx.paths
    t_start = perf_counter()

    tr = json.loads(paths.georef_transform.read_text(encoding="utf-8"))
    s = float(tr["s"])
    R = np.asarray(tr["R"], float)
    t = np.asarray(tr["t"], float)

    cli = colmap_cli.probe()
    if not (cli["available"] and cli["cuda"]):
        ctx.warn("no CUDA COLMAP: dense skipped")
        src = paths.georef / "sparse_enu.ply"
        if src.exists():
            paths.dense.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, paths.dense_fused)
        return {"dense": False, "seconds": round(perf_counter() - t_start, 3)}

    colmap_cli.check_version_compat(ctx.log)
    dcfg = ctx.cfg.resolved_dense
    ws = paths.dense / "ws"
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True, exist_ok=True)
    max_size = int(ctx.preset.mvs_max_image_size)

    # MVS on a copy of the SfM model with georef outliers deregistered, so a
    # mis-registered frame cannot poison depth maps / fusion.
    clean_model = paths.dense / "sparse_clean"
    excluded = _write_clean_model(
        paths.sfm_sparse / "0", _georef_outliers(paths.georef / "residuals.csv"),
        clean_model,
    )
    if excluded:
        ctx.log.info("deregistered %d georef outlier(s) for MVS: %s",
                     len(excluded), ", ".join(excluded))

    t0 = perf_counter()
    colmap_cli.run("image_undistorter", {
        "image_path": str(paths.frames),
        "input_path": str(clean_model),
        "output_path": str(ws),
        "output_type": "COLMAP",
        "max_image_size": max_size,
        "num_patch_match_src_images": dcfg.num_src_images,
    }, ctx.log)
    undistort_s = perf_counter() - t0

    pm_cfg = ws / "stereo" / "patch-match.cfg"
    n_ref = _rewrite_patchmatch_cfg(pm_cfg, dcfg.num_src_images) if pm_cfg.exists() else 0

    def _stereo(size: int) -> None:
        colmap_cli.run("patch_match_stereo", {
            "workspace_path": str(ws), "workspace_format": "COLMAP",
            "PatchMatchStereo.geom_consistency": True,
            "PatchMatchStereo.max_image_size": size,
            "PatchMatchStereo.window_radius": dcfg.window_radius,
            "PatchMatchStereo.num_iterations": dcfg.num_iterations,
        }, ctx.log)

    retried = False
    t0 = perf_counter()
    try:
        _stereo(max_size)
    except RuntimeError as exc:
        max_size = int(max_size * 0.75)
        retried = True
        ctx.warn(f"patch_match_stereo failed ({exc.args[0].splitlines()[0]}); "
                 f"retrying once at max_image_size={max_size}")
        _stereo(max_size)
    stereo_s = perf_counter() - t0

    fused_sfm = paths.dense / "fused_sfm.ply"
    t0 = perf_counter()
    colmap_cli.run("stereo_fusion", {
        "workspace_path": str(ws), "workspace_format": "COLMAP",
        "input_type": "geometric", "output_path": str(fused_sfm),
    }, ctx.log)
    fusion_s = perf_counter() - t0

    d = read_ply(fused_sfm)
    n_pts = len(d["x"])
    counts, status = read_vis(str(fused_sfm) + ".vis")
    if counts is None or len(counts) != n_pts:
        ctx.warn(f"fused.ply.vis unusable ({status}); writing views=0")
        views = np.zeros(n_pts, np.uint8)
        mean_views = 0.0
    else:
        views = np.clip(counts, 0, 255).astype(np.uint8)
        mean_views = float(counts.mean()) if n_pts else 0.0

    out = _transform_cloud(d, s, R, t)
    out["views"] = views
    write_ply(paths.dense_fused, np.stack([out.pop("x"), out.pop("y"), out.pop("z")], -1), out)

    xyz_enu = read_ply(paths.dense_fused)
    xyz_enu = np.stack([xyz_enu["x"], xyz_enu["y"], xyz_enu["z"]], -1)

    return {
        "dense": True,
        "points": n_pts,
        "excluded_images": excluded,
        "num_src_images": dcfg.num_src_images,
        "window_radius": dcfg.window_radius,
        "num_iterations": dcfg.num_iterations,
        "max_image_size_used": max_size,
        "retried": retried,
        "mean_views": round(mean_views, 3),
        "points_per_m2": round(_points_per_m2(xyz_enu), 3),
        # wall-clock per perf_counter — a machine sleep mid-run inflates these.
        "undistort_s": round(undistort_s, 3),
        "stereo_s": round(stereo_s, 3),
        "stereo_s_per_image": round(stereo_s / n_ref, 3) if n_ref else None,
        "fusion_s": round(fusion_s, 3),
        "seconds": round(perf_counter() - t_start, 3),
    }
