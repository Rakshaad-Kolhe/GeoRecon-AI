"""Georeferencing stage: align the SfM reconstruction to a local ENU frame.

See CLAUDE.md "Georeferencing" / "Coordinate conventions". Processing frame is
local ENU in metres with origin = GPS of the first registered keyframe.

Branches:
  none       - no telemetry / < 3 GPS pairs -> identity, unscaled.
  sim3       - GPS track well spread: robust Umeyama sim(3) with RANSAC.
  collinear  - near-straight pass (sigma2/sigma1 < collinear_ratio): roll is
               unobservable from positions, so fit the ground plane to the
               sparse points, level it to +Z, then solve a 2D similarity in XY.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from georecon.config import GeorefCfg
from georecon.util.geo import utm_epsg, wgs84_to_enu
from georecon.util.ply import read_ply, write_ply
from georecon.util.sim3 import (
    R_to_quat_wxyz,
    align_vector_to_z,
    apply,
    fit_plane_ransac,
    quat_wxyz_to_R,
    ransac_sim3,
    rmse,
    umeyama,
)

if TYPE_CHECKING:
    from georecon.pipeline import StageContext

_IDENTITY = (1.0, np.eye(3), np.zeros(3))


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def collinearity_ratio(enu: np.ndarray) -> float:
    """sigma2 / sigma1 of the centred track (0 = perfectly collinear)."""
    x = np.asarray(enu, float)
    if len(x) < 2:
        return 0.0
    sv = np.linalg.svd(x - x.mean(0), compute_uv=False)
    return float(sv[1] / sv[0]) if sv[0] > 0 else 0.0


def contiguous_folds(n: int, k: int) -> list[np.ndarray]:
    """k contiguous index blocks (sequential data -> no random folds)."""
    k = max(1, min(int(k), n))
    edges = np.linspace(0, n, k + 1).astype(int)
    return [np.arange(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def solve_sim3(C, U, cfg: GeorefCfg, seed):
    s, R, t, inl = ransac_sim3(C, U, cfg.ransac_thr_m, cfg.ransac_iters, seed)
    return s, R, t, inl, None


def solve_collinear(C, U, pts, view_dirs, cfg: GeorefCfg, seed):
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0))) if len(pts) else 1.0
    extent = extent or 1.0
    nrm, d, _ = fit_plane_ransac(pts, cfg.plane_thr_frac * extent, cfg.ransac_iters, seed)
    if float(np.mean(C @ nrm + d)) < 0:          # normal points toward the cameras
        nrm, d = -nrm, -d
    up_agreement = (float(np.mean((view_dirs @ nrm) < 0))
                    if view_dirs is not None and len(view_dirs) else None)

    R_up = align_vector_to_z(nrm)
    P = C @ R_up.T                                # cameras with the ground levelled
    s, R2, t2 = umeyama(P[:, :2], U[:, :2], dim=2)
    yaw = math.atan2(R2[1, 0], R2[0, 0])
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                   [math.sin(yaw),  math.cos(yaw), 0.0],
                   [0.0, 0.0, 1.0]])
    R = Rz @ R_up
    tz = float(np.median(U[:, 2] - s * P[:, 2]))
    t = np.array([t2[0], t2[1], tz])
    inl = np.linalg.norm((apply(s, R, t, C) - U)[:, :2], axis=1) <= cfg.ransac_thr_m
    return s, R, t, inl, up_agreement


def _fit(branch, C, U, pts, view_dirs, cfg, seed):
    if branch == "collinear":
        return solve_collinear(C, U, pts, view_dirs, cfg, seed)[:3]
    return solve_sim3(C, U, cfg, seed)[:3]


def holdout_rmse(branch, C, U, pts, view_dirs, full_inliers, cfg):
    """Contiguous-block hold-out. Each fold is fitted with the same robust
    branch on the *training inliers of the full fit* (never plain umeyama, and
    never on a full-fit outlier). Headline h/v pool held-out frames that are
    full-fit inliers; ``*_all`` keep every held-out frame.
    """
    fin = np.asarray(full_inliers, bool)
    hi, vi, ha, va, folds_h = [], [], [], [], []
    for fold in contiguous_folds(len(C), cfg.holdout_folds):
        keep = np.ones(len(C), bool)
        keep[fold] = False
        train = keep & fin
        if train.sum() < 3:
            continue
        vd = view_dirs[train] if view_dirs is not None else None
        s, R, t = _fit(branch, C[train], U[train], pts, vd, cfg, cfg.seed)
        res = apply(s, R, t, C[fold]) - U[fold]
        h = np.linalg.norm(res[:, :2], axis=1)
        v = np.abs(res[:, 2])
        folds_h.append(round(float(rmse(h)), 4))
        ha.append(h)
        va.append(v)
        m = fin[fold]
        if m.any():
            hi.append(h[m])
            vi.append(v[m])
    if not ha:
        nan = float("nan")
        return {"h": nan, "v": nan, "h_all": nan, "v_all": nan, "folds_h": folds_h}
    ha, va = np.concatenate(ha), np.concatenate(va)
    hi_c = np.concatenate(hi) if hi else ha
    vi_c = np.concatenate(vi) if vi else va
    return {"h": rmse(hi_c), "v": rmse(vi_c),
            "h_all": rmse(ha), "v_all": rmse(va), "folds_h": folds_h}


def scale_drift_pct(branch, C, U, pts, view_dirs, full_inliers, cfg, s_full):
    fin = np.asarray(full_inliers, bool)
    m = len(C) // 2
    a, b = fin[:m], fin[m:]
    if a.sum() < 3 or b.sum() < 3 or s_full == 0:
        return 0.0
    vd1 = view_dirs[:m][a] if view_dirs is not None else None
    vd2 = view_dirs[m:][b] if view_dirs is not None else None
    s1 = _fit(branch, C[:m][a], U[:m][a], pts, vd1, cfg, cfg.seed)[0]
    s2 = _fit(branch, C[m:][b], U[m:][b], pts, vd2, cfg, cfg.seed)[0]
    return float(abs(s1 - s2) / abs(s_full) * 100.0)


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #
def _load_pairs(paths):
    cams = pd.read_csv(paths.sfm / "cameras.csv")
    frames = pd.read_csv(paths.frames / "frames.csv")
    frames = frames.set_index("name")
    order = frames.index.tolist()

    rows = []
    for _, r in cams.iterrows():
        if int(r["registered"]) != 1 or r["name"] not in frames.index:
            continue
        fr = frames.loc[r["name"]]
        lat, lon, alt = fr.get("lat"), fr.get("lon"), fr.get("alt")
        if not (np.isfinite(lat) and np.isfinite(lon)):
            continue
        R_cw = quat_wxyz_to_R(r["qw"], r["qx"], r["qy"], r["qz"])
        rows.append({
            "name": r["name"],
            "C": np.array([r["cx"], r["cy"], r["cz"]], float),
            "view": R_cw[2, :],                    # camera optical axis in the SfM frame
            "lat": float(lat), "lon": float(lon),
            "alt": float(alt) if np.isfinite(alt) else 0.0,
        })
    rows.sort(key=lambda d: order.index(d["name"]))
    return rows


def run(ctx: "StageContext") -> dict:
    cfg: GeorefCfg = ctx.cfg.georef
    paths = ctx.paths
    if not (paths.sfm / "cameras.csv").exists():
        raise ValueError("georef: sfm/cameras.csv not found (run sfm first)")
    t0 = time.time()

    pairs = _load_pairs(paths)
    C = np.array([p["C"] for p in pairs], float) if pairs else np.zeros((0, 3))
    views = np.array([p["view"] for p in pairs], float) if pairs else np.zeros((0, 3))

    # ---- origin + GPS -> ENU ------------------------------------------------
    branch = "none"
    georeferenced = len(pairs) >= 3
    if pairs:
        o = pairs[0]
        origin = {"lat": o["lat"], "lon": o["lon"], "alt": o["alt"],
                  "utm_epsg": utm_epsg(o["lat"], o["lon"])}
        paths.georef_origin.write_text(json.dumps(origin, indent=2), encoding="utf-8")
        e, n, u = wgs84_to_enu([p["lat"] for p in pairs], [p["lon"] for p in pairs],
                               [p["alt"] for p in pairs], origin)
        U = np.stack([np.atleast_1d(e), np.atleast_1d(n), np.atleast_1d(u)], axis=-1)
    else:
        U = np.zeros((0, 3))

    track_len = (float(np.sum(np.linalg.norm(np.diff(U, axis=0), axis=1)))
                 if len(U) > 1 else 0.0)
    collinearity = collinearity_ratio(U) if len(U) >= 2 else 0.0

    # ---- choose + solve branch -------------------------------------------
    up_agreement = None
    if not georeferenced:
        ctx.warn("no usable GPS pairs: model will be unscaled (identity transform)")
        s, R, t = _IDENTITY
        inliers = np.zeros(len(C), bool)
    else:
        if cfg.mode == "sim3":
            branch = "sim3"
        elif cfg.mode == "collinear":
            branch = "collinear"
        else:
            branch = "collinear" if collinearity < cfg.collinear_ratio else "sim3"

        P = np.zeros((0, 3))
        if branch == "collinear":
            d = read_ply(paths.sfm / "sparse.ply")
            P = np.stack([d["x"], d["y"], d["z"]], axis=-1).astype(float)
            if len(P) < 10:
                ctx.warn("collinear branch needs a sparse cloud for the ground "
                         "plane; falling back to sim3")
                branch = "sim3"

        if branch == "collinear":
            s, R, t, inliers, up_agreement = solve_collinear(C, U, P, views, cfg, cfg.seed)
            if up_agreement is not None and up_agreement < 0.7:
                ctx.warn(f"camera up-direction agreement {up_agreement:.0%} (<70%); "
                         f"ground orientation may be flipped")
        else:
            s, R, t, inliers, _ = solve_sim3(C, U, cfg, cfg.seed)

    # ---- residuals + metrics -------------------------------------------
    # rmse_h / rmse_v are over the RANSAC inliers (a rejected outlier must not
    # dominate the headline fit quality); rmse_h_all + the resid_h_* percentiles
    # keep the full picture visible.
    if len(C):
        res = apply(s, R, t, C) - U
        h_all = np.linalg.norm(res[:, :2], axis=1)
        v_all = np.abs(res[:, 2])
        sel = inliers.astype(bool)
        if not sel.any():
            sel = np.ones(len(res), bool)
        rmse_h, rmse_v = rmse(h_all[sel]), rmse(v_all[sel])
        rmse_h_all = rmse(h_all)
        resid_h_median = float(np.median(h_all))
        resid_h_p90 = float(np.percentile(h_all, 90))
        resid_h_max = float(h_all.max())
    else:
        res = np.zeros((0, 3))
        rmse_h = rmse_v = rmse_h_all = 0.0
        resid_h_median = resid_h_p90 = resid_h_max = 0.0

    ho = {"h": 0.0, "v": 0.0, "h_all": 0.0, "v_all": 0.0, "folds_h": []}
    drift = 0.0
    if georeferenced:
        pts_for = P if branch == "collinear" else np.zeros((0, 3))
        ho = holdout_rmse(branch, C, U, pts_for, views, inliers, cfg)
        drift = scale_drift_pct(branch, C, U, pts_for, views, inliers, cfg, s)
    hold_h = ho["h"]

    n_inl = int(inliers.sum()) if len(C) else 0
    _write_outputs(paths, pairs, s, R, t, branch, collinearity, res, inliers)

    metrics = {
        "georeferenced": bool(georeferenced),
        "branch": branch,
        "collinearity": round(collinearity, 5),
        "pairs": len(pairs),
        "inliers": n_inl,
        "scale": round(float(s), 6),
        "rmse_h": round(rmse_h, 4),
        "rmse_v": round(rmse_v, 4),
        "rmse_h_all": round(rmse_h_all, 4),
        "resid_h_median": round(resid_h_median, 4),
        "resid_h_p90": round(resid_h_p90, 4),
        "resid_h_max": round(resid_h_max, 4),
        "holdout_rmse_h": round(ho["h"], 4),
        "holdout_rmse_v": round(ho["v"], 4),
        "holdout_rmse_h_all": round(ho["h_all"], 4),
        "holdout_rmse_v_all": round(ho["v_all"], 4),
        "holdout_folds_h": ho["folds_h"],
        "scale_drift_pct": round(drift, 3),
        "up_agreement": (round(up_agreement, 4) if up_agreement is not None else None),
        "track_length_m": round(track_len, 3),
        "seconds": round(time.time() - t0, 3),
    }
    ctx.log.info("georef: branch=%s pairs=%d inliers=%d scale=%.4f "
                 "rmse_h=%.2fm (all %.2fm) holdout_h=%.2fm (all %.2fm) folds=%s drift=%.1f%%",
                 branch, len(pairs), n_inl, s, rmse_h, rmse_h_all,
                 ho["h"], ho["h_all"], ho["folds_h"], drift)
    if georeferenced and n_inl < 0.7 * len(pairs):
        ctx.warn(f"only {n_inl}/{len(pairs)} GPS pairs are inliers (<70%)")
    if georeferenced and np.isfinite(hold_h) and hold_h > 5.0:
        ctx.warn(f"hold-out horizontal RMSE {hold_h:.1f} m (>5 m)")
    if georeferenced and drift > 5.0:
        ctx.warn(f"scale drift {drift:.1f}% between first and second half (>5%)")
    return metrics


def _write_outputs(paths, pairs, s, R, t, branch, collinearity, res, inliers):
    paths.georef.mkdir(parents=True, exist_ok=True)
    paths.georef_transform.write_text(json.dumps({
        "s": float(s), "R": np.asarray(R, float).tolist(),
        "t": np.asarray(t, float).tolist(),
        "branch": branch, "collinearity": float(collinearity),
    }, indent=2), encoding="utf-8")

    rr = pd.DataFrame({
        "name": [p["name"] for p in pairs],
        "dE": res[:, 0] if len(res) else [],
        "dN": res[:, 1] if len(res) else [],
        "dU": res[:, 2] if len(res) else [],
        "inlier": inliers.astype(int) if len(pairs) else [],
    })
    rr.to_csv(paths.georef / "residuals.csv", index=False)

    # sparse cloud -> ENU
    sparse = paths.sfm / "sparse.ply"
    if sparse.exists():
        d = read_ply(sparse)
        xyz = np.stack([d["x"], d["y"], d["z"]], axis=-1).astype(float)
        xyz = apply(s, R, t, xyz) if len(xyz) else xyz
        props = {k: v for k, v in d.items() if k not in ("x", "y", "z")}
        write_ply(paths.georef / "sparse_enu.ply", xyz, props)

    # cameras -> ENU
    cams = pd.read_csv(paths.sfm / "cameras.csv")
    Rt = np.asarray(R, float).T
    out = []
    for _, r in cams.iterrows():
        if int(r["registered"]) == 1:
            C = np.array([r["cx"], r["cy"], r["cz"]], float)
            enu = apply(s, R, t, C)
            R_ce = quat_wxyz_to_R(r["qw"], r["qx"], r["qy"], r["qz"]) @ Rt
            q = R_to_quat_wxyz(R_ce)
            out.append({"name": r["name"], "registered": 1,
                        "E": enu[0], "N": enu[1], "U": enu[2],
                        "qw": q[0], "qx": q[1], "qy": q[2], "qz": q[3]})
        else:
            out.append({"name": r["name"], "registered": 0, "E": "", "N": "", "U": "",
                        "qw": "", "qx": "", "qy": "", "qz": ""})
    pd.DataFrame(out, columns=["name", "registered", "E", "N", "U",
                               "qw", "qx", "qy", "qz"]).to_csv(
        paths.georef / "cameras_enu.csv", index=False)
