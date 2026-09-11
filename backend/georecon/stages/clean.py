"""Clean stage: dense point cloud -> ``dense/clean.ply`` (ENU, normals, rgb,
views). Runs between ``dense`` and ``mesh``; the raw ``dense/fused.ply`` is
kept untouched so re-cleaning never needs to re-run MVS.

Pipeline: ROI crop (camera-track shaped) -> far/grazing filter -> views>=3
(when the ``.vis`` file was usable) -> statistical outlier removal -> radius
filter. Each step's surviving count is reported; a warning fires if fewer
than ``clean.warn_kept_frac`` of the raw points survive.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from georecon.config import CleanCfg
from georecon.util.ply import read_ply, write_ply

if TYPE_CHECKING:
    from georecon.pipeline import StageContext


# --------------------------------------------------------------------------- #
# ROI
# --------------------------------------------------------------------------- #
def _offset_convex_polygon(verts: np.ndarray, buffer: float) -> np.ndarray:
    """Outward mitre-offset of a CCW convex polygon by ``buffer`` (sharp
    corners, not rounded — a slight underestimate of a true disc buffer, fine
    for a display/metadata polygon)."""
    n = len(verts)
    if n < 3 or buffer <= 0:
        return verts
    edges = np.roll(verts, -1, axis=0) - verts
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    centroid = verts.mean(0)
    mid0 = (verts[0] + verts[1]) / 2.0
    if np.dot(normals[0], centroid - mid0) > 0:      # normals must point outward
        normals = -normals

    out = np.zeros_like(verts)
    for i in range(n):
        n0, n1 = normals[i - 1], normals[i]
        b0 = n0 @ verts[i] + buffer
        b1 = n1 @ verts[i] + buffer
        A = np.array([n0, n1])
        try:
            out[i] = np.linalg.solve(A, [b0, b1])
        except np.linalg.LinAlgError:
            out[i] = verts[i] + buffer * n1
    return out


def _roi_orbit(C_xy: np.ndarray, cfg: CleanCfg):
    from scipy.spatial import ConvexHull

    hull = ConvexHull(C_xy)
    hull_area = float(hull.volume)
    equiv_r = math.sqrt(hull_area / math.pi) if hull_area > 0 else 0.0
    buffer = cfg.roi_orbit_buffer_frac * equiv_r
    A = hull.equations[:, :2]
    b = hull.equations[:, 2]

    def mask_fn(xy: np.ndarray) -> np.ndarray:
        return np.all(xy @ A.T + b <= buffer, axis=1)

    polygon = _offset_convex_polygon(C_xy[hull.vertices], buffer)
    return mask_fn, polygon, buffer


def _roi_strip(C_xy: np.ndarray, buffer: float):
    from scipy.spatial import cKDTree

    # densify the track so nearest-neighbour distance approximates true
    # point-to-polyline distance
    if len(C_xy) >= 2:
        seg = np.linalg.norm(np.diff(C_xy, axis=0), axis=1)
        n_sub = np.maximum(1, np.ceil(seg / max(buffer / 4, 1e-6))).astype(int)
        pts = []
        for i, ns in enumerate(n_sub):
            t = np.linspace(0.0, 1.0, int(ns), endpoint=False)[:, None]
            pts.append(C_xy[i] + t * (C_xy[i + 1] - C_xy[i]))
        pts.append(C_xy[-1:])
        dense_track = np.concatenate(pts, axis=0)
    else:
        dense_track = C_xy

    tree = cKDTree(dense_track)

    def mask_fn(xy: np.ndarray) -> np.ndarray:
        d, _ = tree.query(xy, k=1)
        return d <= buffer

    # display polygon: a squared-off "sausage" around the track (left/right
    # offset lines, no rounded caps)
    if len(C_xy) >= 2:
        edges = np.diff(C_xy, axis=0)
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 1e-12)
        vnorm = np.vstack([normals, normals[-1:]])
        left = C_xy + buffer * vnorm
        right = C_xy - buffer * vnorm
        polygon = np.vstack([left, right[::-1]])
    else:
        polygon = C_xy
    return mask_fn, polygon, buffer


def _classify_and_mask(xyz: np.ndarray, C: np.ndarray, cfg: CleanCfg, z_lo: float, z_hi: float):
    """Returns (roi_mask, roi_kind, polygon_xy, buffer_m)."""
    C_xy = C[:, :2]
    bbox_area = float(np.ptp(C_xy[:, 0]) * np.ptp(C_xy[:, 1]))
    if len(C_xy) >= 3 and bbox_area > 0:
        from scipy.spatial import ConvexHull

        hull_area = float(ConvexHull(C_xy).volume)
    else:
        hull_area = 0.0
    is_orbit = bbox_area > 0 and hull_area > cfg.roi_orbit_hull_area_frac * bbox_area

    if is_orbit and len(C_xy) >= 3:
        mask_fn, polygon, buffer = _roi_orbit(C_xy, cfg)
        kind = "orbit"
    else:
        cam_h_med = float(np.median(C[:, 2])) - z_lo if len(C) else 0.0
        buffer = cfg.roi_strip_buffer_mult * max(cam_h_med, 0.0)
        mask_fn, polygon, buffer = _roi_strip(C_xy, buffer)
        kind = "strip"

    xy_mask = mask_fn(xyz[:, :2])
    z_mask = (xyz[:, 2] >= z_lo) & (xyz[:, 2] <= z_hi)
    return xy_mask & z_mask, kind, polygon, buffer


# --------------------------------------------------------------------------- #
# filters
# --------------------------------------------------------------------------- #
def far_grazing_mask(xyz: np.ndarray, C: np.ndarray, mult: float):
    from scipy.spatial import cKDTree

    if len(xyz) == 0 or len(C) == 0:
        return np.ones(len(xyz), bool), 0.0
    dist, _ = cKDTree(C).query(xyz, k=1)
    med = float(np.median(dist))
    thr = mult * med
    return dist <= thr, med


def statistical_outlier_mask(xyz: np.ndarray, k: int, std_mult: float):
    from scipy.spatial import cKDTree

    n = len(xyz)
    if n < k + 1:
        return np.ones(n, bool)
    d, _ = cKDTree(xyz).query(xyz, k=k + 1)
    mean_dist = d[:, 1:].mean(axis=1)
    mu, sigma = float(mean_dist.mean()), float(mean_dist.std())
    return mean_dist <= mu + std_mult * sigma


def radius_outlier_mask(xyz: np.ndarray, min_neighbors: int, radius_mult: float):
    from scipy.spatial import cKDTree

    n = len(xyz)
    if n < 3:
        return np.ones(n, bool), 0.0
    tree = cKDTree(xyz)
    d2, _ = tree.query(xyz, k=2)
    med_spacing = float(np.median(d2[:, 1]))
    radius = radius_mult * med_spacing
    counts = tree.query_ball_point(xyz, r=radius, return_length=True) - 1     # exclude self
    return counts >= min_neighbors, med_spacing


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #
def _load_cameras(paths) -> np.ndarray:
    df = pd.read_csv(paths.georef_cameras)
    reg = df[df["registered"] == 1]
    if not len(reg):
        return np.zeros((0, 3))
    return reg[["E", "N", "U"]].to_numpy(float)


def run(ctx: "StageContext") -> dict:
    paths = ctx.paths
    cfg: CleanCfg = ctx.cfg.clean
    t0 = perf_counter()

    if not paths.dense_fused.exists():
        ctx.warn("no dense/fused.ply: clean skipped")
        return {"roi_kind": None, "counts": {}, "kept_pct": 0.0,
                "seconds": round(perf_counter() - t0, 3)}

    d = read_ply(paths.dense_fused)
    xyz = np.column_stack([d["x"], d["y"], d["z"]]).astype(float)
    n_raw = len(xyz)
    has_normals = {"nx", "ny", "nz"} <= d.keys()
    normals = (np.column_stack([d["nx"], d["ny"], d["nz"]]).astype(float)
              if has_normals else np.zeros((n_raw, 3)))
    rgb = np.column_stack([d["red"], d["green"], d["blue"]]).astype(np.uint8)
    views = np.asarray(d.get("views", np.zeros(n_raw, np.uint8)))

    C = _load_cameras(paths)
    counts = {"raw": n_raw}

    if n_raw == 0 or len(C) < 2:
        write_ply(paths.dense_clean, xyz, {
            "nx": normals[:, 0], "ny": normals[:, 1], "nz": normals[:, 2],
            "red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2], "views": views,
        })
        paths.dense_roi.write_text(json.dumps({"kind": None, "polygon": [], "buffer_m": 0.0,
                                              "z_lo": None, "z_hi": None}),
                                   encoding="utf-8")
        return {"roi_kind": None, "counts": counts, "kept_pct": 100.0 if n_raw else 0.0,
                "seconds": round(perf_counter() - t0, 3)}

    # ---- ground reference + Z bounds -----------------------------------
    z_p1 = float(np.percentile(xyz[:, 2], 1))
    z_extent = float(xyz[:, 2].max() - xyz[:, 2].min())
    z_lo = z_p1 - (cfg.roi_z_below_pct / 100.0) * z_extent
    z_hi = float(C[:, 2].max())

    # ---- ROI ------------------------------------------------------------
    roi_mask, roi_kind, polygon, buffer_m = _classify_and_mask(xyz, C, cfg, z_lo, z_hi)
    idx = roi_mask
    xyz, normals, rgb, views = xyz[idx], normals[idx], rgb[idx], views[idx]
    counts["roi"] = len(xyz)

    # ---- far / grazing ----------------------------------------------------
    far_mask, med_cam_dist = far_grazing_mask(xyz, C, cfg.far_filter_mult)
    xyz, normals, rgb, views = xyz[far_mask], normals[far_mask], rgb[far_mask], views[far_mask]
    counts["far"] = len(xyz)

    # ---- views >= min_views (only when .vis produced real counts) --------
    if views.size and views.max() > 0:
        v_mask = views >= cfg.min_views
        xyz, normals, rgb, views = xyz[v_mask], normals[v_mask], rgb[v_mask], views[v_mask]
    counts["views"] = len(xyz)

    # ---- statistical outlier removal --------------------------------------
    sor_mask = statistical_outlier_mask(xyz, cfg.sor_k, cfg.sor_std)
    xyz, normals, rgb, views = xyz[sor_mask], normals[sor_mask], rgb[sor_mask], views[sor_mask]
    counts["sor"] = len(xyz)

    # ---- radius filter -----------------------------------------------------
    radius_mask, med_spacing = radius_outlier_mask(xyz, cfg.radius_min_neighbors, cfg.radius_mult)
    xyz, normals, rgb, views = (xyz[radius_mask], normals[radius_mask],
                                rgb[radius_mask], views[radius_mask])
    counts["radius"] = len(xyz)

    kept_pct = round(100.0 * len(xyz) / n_raw, 2) if n_raw else 0.0
    if kept_pct < cfg.warn_kept_frac * 100.0:
        ctx.warn(f"clean: kept {kept_pct:.0f}% of raw dense points "
                 f"(<{cfg.warn_kept_frac:.0%})")

    write_ply(paths.dense_clean, xyz, {
        "nx": normals[:, 0], "ny": normals[:, 1], "nz": normals[:, 2],
        "red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2], "views": views,
    })
    paths.dense_roi.write_text(json.dumps({
        "kind": roi_kind,
        "polygon": np.round(polygon, 3).tolist(),
        "buffer_m": round(float(buffer_m), 3),
        "z_lo": round(z_lo, 3), "z_hi": round(z_hi, 3),
    }, indent=2), encoding="utf-8")

    ctx.log.info("clean: roi=%s raw=%d roi=%d far=%d views=%d sor=%d radius=%d (%.1f%% kept)",
                 roi_kind, counts["raw"], counts["roi"], counts["far"], counts["views"],
                 counts["sor"], counts["radius"], kept_pct)
    return {
        "roi_kind": roi_kind,
        "roi_buffer_m": round(float(buffer_m), 3),
        "counts": counts,
        "kept_pct": kept_pct,
        "median_cam_dist_m": round(med_cam_dist, 3),
        "median_spacing_m": round(med_spacing, 4),
        "seconds": round(perf_counter() - t0, 3),
    }
