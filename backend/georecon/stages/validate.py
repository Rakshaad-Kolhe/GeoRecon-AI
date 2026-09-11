"""Validation stage: roll the per-stage metrics up into an accuracy /
completeness / speed ``summary`` (added to ``report/metrics.json``) and a
slide-ready ``report/summary.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from georecon.util import geo
from georecon.util.ply import read_ply

if TYPE_CHECKING:
    from georecon.pipeline import StageContext

# stages whose wall time scales with the number of frames / keyframes
_FRAME_DEPENDENT = ("ingest", "keyframes", "masking", "sfm", "dense", "mesh")


def project_10min(per_stage_s: dict, total_s: float, keyframes: int,
                  duration_s: float, max_keyframes: int):
    """(kf10, projected_seconds) for a hypothetical 10-minute clip, or
    (None, None) when the inputs are missing."""
    if not keyframes or not duration_s:
        return None, None
    kf10 = min(int(max_keyframes), int(round(keyframes / duration_s * 600)))
    frame_dep = sum(per_stage_s.get(k, 0.0) for k in _FRAME_DEPENDENT)
    fixed = total_s - frame_dep
    return kf10, round(fixed + frame_dep / keyframes * kf10, 1)


def _coverage_pct(dsm_tif: Path, dense_en: np.ndarray, cell: float) -> float:
    """Occupied DSM cells / DSM cells whose centre lies inside the XY convex
    hull of the dense cloud."""
    import rasterio
    from scipy.spatial import ConvexHull, Delaunay

    with rasterio.open(dsm_tif) as src:
        arr = src.read(1)
        nodata = src.nodata
        tr = src.transform
    h, w = arr.shape
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    cx = tr.c + (cols + 0.5) * tr.a
    cy = tr.f + (rows + 0.5) * tr.e
    centres = np.column_stack([cx.ravel(), cy.ravel()])

    if len(dense_en) < 3:
        return 0.0
    hull = ConvexHull(dense_en)
    inside = Delaunay(dense_en[hull.vertices]).find_simplex(centres) >= 0
    denom = int(inside.sum())
    if denom == 0:
        return 0.0
    occupied = ((arr.ravel() != nodata) & inside).sum()
    return round(100.0 * occupied / denom, 2)


def run(ctx: "StageContext") -> dict:
    paths = ctx.paths
    t0 = perf_counter()
    m = json.loads(paths.metrics_json.read_text(encoding="utf-8"))
    st = m["stages"]
    g, s, dn, mh, ex = (st.get(k, {}) for k in
                        ("georef", "sfm", "dense", "mesh", "export"))
    ing = st.get("ingest", {})
    kf = st.get("keyframes", {})

    # ---- accuracy ----------------------------------------------------
    if g.get("georeferenced"):
        accuracy = {
            "branch": g.get("branch"),
            "rmse_h": g.get("rmse_h"), "rmse_v": g.get("rmse_v"),
            "holdout_rmse_h": g.get("holdout_rmse_h"),
            "holdout_rmse_v": g.get("holdout_rmse_v"),
            "inliers": g.get("inliers"), "pairs": g.get("pairs"),
            "excluded_images": len(dn.get("excluded_images", []) or []),
            "scale_drift_pct": g.get("scale_drift_pct"),
            "mean_reproj_px": s.get("mean_reproj_px"),
        }
    else:
        accuracy = "n/a — no GPS telemetry"

    # ---- completeness ---------------------------------------------
    duration_s = float(ing.get("video", {}).get("duration_s") or 0.0)
    keyframes = int(kf.get("keyframes") or 0)
    coverage = None
    if dn.get("dense") and (paths.outputs / "dsm.tif").exists():
        d = read_ply(paths.dense_fused)
        origin = json.loads(paths.georef_origin.read_text(encoding="utf-8"))
        lat, lon, _ = geo.enu_to_wgs84(d["x"], d["y"], d["z"], origin)
        from pyproj import Transformer

        e, n = Transformer.from_crs(4326, int(origin["utm_epsg"]),
                                    always_xy=True).transform(lon, lat)
        coverage = _coverage_pct(paths.outputs / "dsm.tif",
                                 np.column_stack([e, n]),
                                 float(ex.get("dsm_cell_m") or 1.0))
    completeness = {
        "registered_pct": s.get("registered_pct"),
        "dense_points": dn.get("points"),
        "points_per_m2": dn.get("points_per_m2"),
        "coverage_pct": coverage,
        "mesh_surface_area_m2": mh.get("surface_area_m2"),
    }

    # ---- speed -----------------------------------------------------
    per_stage = {k: round(float(v.get("seconds") or 0.0), 3) for k, v in st.items()}
    total = float(m.get("total_seconds") or sum(per_stage.values()))
    kf10, projected = project_10min(per_stage, total, keyframes, duration_s,
                                    ctx.preset.max_keyframes)
    speed = {
        "per_stage_s": per_stage,
        "total_seconds": round(total, 3),
        "video_duration_s": duration_s,
        "keyframes": keyframes,
        "kf10": kf10,
        "projected_10min_s": projected,
    }

    summary = {"accuracy": accuracy, "completeness": completeness, "speed": speed}
    if coverage is not None and coverage < 80.0:
        ctx.warn(f"DSM coverage {coverage:.0f}% (<80%) — thin / holey dense cloud")

    _write_summary_md(paths.summary_md, summary, kf10)
    holdout_h = accuracy["holdout_rmse_h"] if isinstance(accuracy, dict) else None
    ctx.log.info("validate: holdout_h=%.2fm registered=%.0f%% coverage=%s%%",
                 holdout_h or 0.0,
                 completeness["registered_pct"] or 0.0, coverage)
    return {"summary": summary, "coverage_pct": coverage,
            "seconds": round(perf_counter() - t0, 3)}


def _fmt(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _write_summary_md(path: Path, summary: dict, kf10) -> None:
    a, c, sp = summary["accuracy"], summary["completeness"], summary["speed"]
    lines = ["# GeoRecon AI — run summary", ""]

    if isinstance(a, dict):
        lines += ["## Accuracy", "",
                  "| metric | value |", "|---|---|",
                  f"| Hold-out RMSE horizontal | {_fmt(a['holdout_rmse_h'])} m |",
                  f"| Hold-out RMSE vertical | {_fmt(a['holdout_rmse_v'])} m |",
                  f"| Fit RMSE h / v | {_fmt(a['rmse_h'])} / {_fmt(a['rmse_v'])} m |",
                  f"| GPS inliers | {_fmt(a['inliers'])} / {_fmt(a['pairs'])} |",
                  f"| Outlier frames excluded | {_fmt(a['excluded_images'])} |",
                  f"| Scale drift | {_fmt(a['scale_drift_pct'])} % |",
                  f"| Mean reprojection error | {_fmt(a['mean_reproj_px'])} px |",
                  f"| Georef branch | {_fmt(a['branch'])} |", ""]
    else:
        lines += ["## Accuracy", "", a, ""]

    lines += ["## Completeness", "",
              "| metric | value |", "|---|---|",
              f"| Registered images | {_fmt(c['registered_pct'])} % |",
              f"| Dense points | {_fmt(c['dense_points'])} |",
              f"| Points / m² | {_fmt(c['points_per_m2'])} |",
              f"| DSM coverage | {_fmt(c['coverage_pct'])} % |",
              f"| Mesh surface area | {_fmt(c['mesh_surface_area_m2'])} m² |", ""]

    ps = sp["per_stage_s"]
    lines += ["## Speed", "",
              "| stage | seconds |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v, 1)} |" for k, v in ps.items()]
    lines += [f"| **total** | **{_fmt(sp['total_seconds'], 1)}** |", "",
              f"Clip: {_fmt(sp['video_duration_s'], 1)} s, "
              f"{_fmt(sp['keyframes'])} keyframes.", ""]
    if sp["projected_10min_s"] is not None:
        lines += [f"**Projected for a 10-min clip: ~{_fmt(sp['projected_10min_s'], 0)} s "
                  f"(estimate).**", "",
                  "Assumptions: frame-dependent stages "
                  f"({', '.join(_FRAME_DEPENDENT)}) scale linearly with keyframe "
                  f"count to kf10={kf10} (= min(preset.max_keyframes, "
                  "keyframes/duration × 600)); georef/export/validate held fixed; "
                  "same preset, hardware and scene texture.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
