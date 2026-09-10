"""Dense MVS stage (native CUDA COLMAP) -> georeferenced dense point cloud.

Needs a CUDA COLMAP CLI (``GEORECON_COLMAP_BIN``). Without one the stage is a
no-op: it copies ``georef/sparse_enu.ply`` to ``dense/fused.ply`` and reports
``dense=false`` so downstream meshing still has an input.

MVS runs in the SfM frame (undistort from ``sfm/sparse/0``); the fused cloud is
then mapped to ENU with ``georef/transform.json`` (xyz: sRX+t, normals: Rn).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from georecon.util import colmap_cli
from georecon.util.colmap_vis import read_vis
from georecon.util.ply import read_ply, write_ply

if TYPE_CHECKING:
    from georecon.pipeline import StageContext


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
    try:
        from scipy.spatial import ConvexHull

        area = float(ConvexHull(xyz[:, :2]).volume)   # 2-D hull "volume" == area
    except Exception:                                 # noqa: BLE001
        return 0.0
    return float(len(xyz) / area) if area > 0 else 0.0


def run(ctx: "StageContext") -> dict:
    paths = ctx.paths
    t_start = time.time()

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
        return {"dense": False, "seconds": round(time.time() - t_start, 3)}

    colmap_cli.check_version_compat(ctx.log)
    ws = paths.dense / "ws"
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True, exist_ok=True)
    max_size = int(ctx.preset.mvs_max_image_size)

    t0 = time.time()
    colmap_cli.run("image_undistorter", {
        "image_path": str(paths.frames),
        "input_path": str(paths.sfm_sparse / "0"),
        "output_path": str(ws),
        "output_type": "COLMAP",
        "max_image_size": max_size,
    }, ctx.log)
    undistort_s = time.time() - t0

    retried = False
    t0 = time.time()
    try:
        colmap_cli.run("patch_match_stereo", {
            "workspace_path": str(ws), "workspace_format": "COLMAP",
            "PatchMatchStereo.geom_consistency": True,
            "PatchMatchStereo.max_image_size": max_size,
        }, ctx.log)
    except RuntimeError as exc:
        max_size = int(max_size * 0.75)
        retried = True
        ctx.warn(f"patch_match_stereo failed ({exc.args[0].splitlines()[0]}); "
                 f"retrying once at max_image_size={max_size}")
        colmap_cli.run("patch_match_stereo", {
            "workspace_path": str(ws), "workspace_format": "COLMAP",
            "PatchMatchStereo.geom_consistency": True,
            "PatchMatchStereo.max_image_size": max_size,
        }, ctx.log)
    stereo_s = time.time() - t0

    fused_sfm = paths.dense / "fused_sfm.ply"
    t0 = time.time()
    colmap_cli.run("stereo_fusion", {
        "workspace_path": str(ws), "workspace_format": "COLMAP",
        "input_type": "geometric", "output_path": str(fused_sfm),
    }, ctx.log)
    fusion_s = time.time() - t0

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
        "max_image_size_used": max_size,
        "retried": retried,
        "mean_views": round(mean_views, 3),
        "points_per_m2": round(_points_per_m2(xyz_enu), 3),
        "undistort_s": round(undistort_s, 3),
        "stereo_s": round(stereo_s, 3),
        "fusion_s": round(fusion_s, 3),
        "seconds": round(time.time() - t_start, 3),
    }
