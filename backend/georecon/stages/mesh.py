"""Mesh stage: dense point cloud -> triangle mesh (ENU metres) + preview PNG.

Poisson path: COLMAP ``poisson_mesher`` directly on ``dense/clean.ply`` (ENU,
already carries normals from the clean stage) -> trimesh cleanup -> ROI crop
-> ``mesh/mesh.ply``. Falls back to a NumPy height-field when Poisson is
unavailable, errors, runs long, or the cloud is too sparse (no-CUDA path).

``poisson_mesher`` options verified live against COLMAP 4.2.0 ``-h``:
  --PoissonMeshing.depth --PoissonMeshing.trim --PoissonMeshing.point_weight
  --PoissonMeshing.color --PoissonMeshing.num_threads
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from georecon.util import colmap_cli
from georecon.util.depth_map import read_array as read_depth_array
from georecon.util.ply import read_ply

if TYPE_CHECKING:
    from georecon.pipeline import StageContext


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _median_nn(xyz: np.ndarray, sample: int = 20_000, seed: int = 0) -> float:
    """Median nearest-neighbour spacing (subsampled for speed)."""
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, float)
    if len(xyz) > sample:
        idx = np.random.default_rng(seed).choice(len(xyz), sample, replace=False)
        xyz = xyz[idx]
    if len(xyz) < 2:
        return 0.0
    d, _ = cKDTree(xyz).query(xyz, k=2)
    return float(np.median(d[:, 1]))


def _nearest_colors(query_xyz: np.ndarray, src_xyz: np.ndarray,
                    src_rgb: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    _, idx = cKDTree(np.asarray(src_xyz, float)).query(np.asarray(query_xyz, float), k=1)
    return np.asarray(src_rgb)[idx]


def _crop_far_vertices(mesh, dense_xyz: np.ndarray, max_dist: float):
    """Drop vertices farther than ``max_dist`` from any dense point (kills the
    Poisson dome / skirt) plus every face that touches them."""
    from scipy.spatial import cKDTree

    d, _ = cKDTree(np.asarray(dense_xyz, float)).query(np.asarray(mesh.vertices), k=1)
    keep_v = d <= max_dist
    n_cropped = int((~keep_v).sum())
    if n_cropped:
        mesh.update_faces(keep_v[mesh.faces].all(axis=1))
        mesh.remove_unreferenced_vertices()
    return mesh, n_cropped


def _bilinear_sample(img: np.ndarray, u: np.ndarray, v: np.ndarray):
    """Bilinear-sample ``img`` (H,W) or (H,W,C) at pixel coords ``(u,v)``
    (x=col, y=row, both float). Returns (values, valid) — ``valid`` is False
    where the 2x2 support square falls outside the image."""
    h, w = img.shape[:2]
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1, v1 = u0 + 1, v0 + 1
    valid = (u0 >= 0) & (v0 >= 0) & (u1 < w) & (v1 < h)
    u0c, u1c = np.clip(u0, 0, w - 1), np.clip(u1, 0, w - 1)
    v0c, v1c = np.clip(v0, 0, h - 1), np.clip(v1, 0, h - 1)
    fu, fv = (u - u0), (v - v0)
    Ia, Ib = img[v0c, u0c], img[v0c, u1c]
    Ic, Id = img[v1c, u0c], img[v1c, u1c]
    wa, wb = (1 - fu) * (1 - fv), fu * (1 - fv)
    wc, wd = (1 - fu) * fv, fu * fv
    if img.ndim == 3:
        wa, wb, wc, wd = wa[:, None], wb[:, None], wc[:, None], wd[:, None]
    return wa * Ia + wb * Ib + wc * Ic + wd * Id, valid


# Number of best candidate views to blend per vertex (weighted median)
_TOP_K = 3


def view_based_vertex_colors(vertices: np.ndarray, normals: np.ndarray, ws_dir: Path,
                             cameras_enu_csv: Path, scale: float,
                             tol_frac: float = 0.02, facing_thr: float = -0.2):
    """Per-vertex colour using multi-view weighted-median blending.

    For each vertex up to ``_TOP_K`` unoccluded candidate views are collected,
    weighted by ``w = cos(theta) / dist``.  Before blending, each camera's
    sampled colours are gain-compensated: the per-channel median of (sampled /
    multi-view median) over shared vertices is clamped to [0.7, 1.4] and
    applied so that exposure differences between images are removed.  The
    final colour is the per-channel weighted median of the compensated samples.

    Camera extrinsics come from ``georef/cameras_enu.csv`` (E,N,U + quaternion
    already in the mesh's ENU/local frame — same values georef.py wrote);
    only intrinsics (+ image/depth files) come from ``dense/ws``.

    Returns ``(colors uint8 (N,3), covered bool (N,))``.
    """
    n = len(vertices)
    colors = np.zeros((n, 3), np.uint8)
    covered = np.zeros(n, bool)
    sparse, images_dir = ws_dir / "sparse", ws_dir / "images"
    depth_dir = ws_dir / "stereo" / "depth_maps"
    if not (sparse.exists() and images_dir.exists() and depth_dir.exists()
            and cameras_enu_csv.exists()):
        return colors, covered

    import cv2
    import pandas as pd
    import pycolmap

    from georecon.util.sim3 import quat_wxyz_to_R

    recon = pycolmap.Reconstruction(str(sparse))
    intrinsics = {im.name: recon.cameras[im.camera_id] for im in recon.images.values()}

    cams = pd.read_csv(cameras_enu_csv)
    cams = cams[cams["registered"] == 1]

    V = np.asarray(vertices, float)
    Nrm = np.asarray(normals, float)
    nlen = np.linalg.norm(Nrm, axis=1, keepdims=True)
    Nrm = np.divide(Nrm, nlen, out=np.zeros_like(Nrm), where=nlen > 1e-9)

    # Accumulate per-vertex: list of (score, rgb_float) from each qualified camera.
    # We store only the top-K per vertex to cap memory.
    top_scores: list[np.ndarray] = [np.full(n, -np.inf)] * _TOP_K   # shape (K, N)
    top_rgb: list[np.ndarray] = [np.zeros((n, 3), float)] * _TOP_K  # shape (K, N, 3)
    top_scores = [np.full(n, -np.inf) for _ in range(_TOP_K)]
    top_rgb = [np.zeros((n, 3), float) for _ in range(_TOP_K)]

    for row in cams.itertuples():
        name = row.name
        cam = intrinsics.get(name)
        img_path, depth_path = images_dir / name, depth_dir / f"{name}.geometric.bin"
        if cam is None or not img_path.exists() or not depth_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        try:
            depth = read_depth_array(depth_path).astype(float) * float(scale)
        except (ValueError, OSError):
            continue

        cam_center = np.array([row.E, row.N, row.U], float)
        M = quat_wxyz_to_R(row.qw, row.qx, row.qy, row.qz)   # ENU world -> camera
        view_dir = M[2, :]                                    # ENU-frame optical axis

        Xc = (V - cam_center) @ M.T
        z = Xc[:, 2]
        front = z > 1e-6
        if not front.any():
            continue
        uv = np.zeros((n, 2))
        uv[front] = cam.img_from_cam(Xc[front])
        u, v_px = uv[:, 0], uv[:, 1]
        h, w = img.shape[:2]
        facing = (Nrm @ view_dir) < facing_thr
        cand = front & facing & (u >= 0) & (u < w - 1) & (v_px >= 0) & (v_px < h - 1)
        if not cand.any():
            continue

        cand_idx = np.flatnonzero(cand)
        d_sampled, dvalid = _bilinear_sample(depth, u[cand_idx], v_px[cand_idx])
        good = dvalid & (d_sampled > 1e-6) & (np.abs(z[cand_idx] - d_sampled) <= tol_frac * d_sampled)
        cand_idx = cand_idx[good]
        if not len(cand_idx):
            continue

        dist = np.linalg.norm(V[cand_idx] - cam_center, axis=1)
        frontal = -(Nrm[cand_idx] @ view_dir)
        score = frontal / np.maximum(dist, 1e-6)

        col_bgr, _ = _bilinear_sample(img.astype(float), u[cand_idx], v_px[cand_idx])
        col_rgb = col_bgr[:, ::-1]   # BGR -> RGB

        # Insert into per-vertex top-K heap (slot 0 = best, slot K-1 = weakest kept)
        for k in range(_TOP_K):
            take = score > top_scores[k][cand_idx]
            if not take.any():
                break
            take_idx = cand_idx[take]
            # push current slot-k down to slot k+1 before overwriting
            if k + 1 < _TOP_K:
                for kk in range(_TOP_K - 1, k, -1):
                    top_scores[kk][take_idx] = top_scores[kk - 1][take_idx]
                    top_rgb[kk][take_idx] = top_rgb[kk - 1][take_idx]
            top_scores[k][take_idx] = score[take]
            top_rgb[k][take_idx] = col_rgb[take]
            # remaining candidates (score <= slot-k) might still belong in lower slots
            cand_idx = cand_idx[~take]
            score = score[~take]
            col_rgb = col_rgb[~take]
            if not len(cand_idx):
                break

    # ---- per-image gain compensation (shared across all vertices) ----------------
    # Stack collected candidates: shape (K, N, 3); only slots with score > -inf are valid.
    stacked = np.stack(top_rgb, axis=0)            # (K, N, 3)
    valid_k = np.stack([s > -np.inf for s in top_scores], axis=0)  # (K, N)

    any_valid = valid_k.any(axis=0)               # (N,)
    if any_valid.any():
        # Global multi-view median per vertex channel (over valid slots)
        mv_med = np.zeros((n, 3), float)
        for c in range(3):
            ch = stacked[:, :, c]                  # (K, N)
            for vi in np.flatnonzero(any_valid):
                vals = ch[valid_k[:, vi], vi]
                mv_med[vi, c] = float(np.median(vals))

        # Per-camera gain = median of (sampled / mv_med) over vertices where
        # both the slot and mv_med are valid. Apply per-slot.
        for k in range(_TOP_K):
            vk = valid_k[k]                        # (N,) — vertices filled in slot k
            if not vk.any():
                continue
            ref = mv_med[vk]                       # (Mv, 3)
            smp = stacked[k][vk]                  # (Mv, 3)
            denom = np.where(ref > 1e-3, ref, np.nan)
            ratio = np.where(ref > 1e-3, smp / denom, np.nan)
            # per-channel median gain across vertices covered by this slot
            gain = np.nanmedian(ratio, axis=0)     # (3,)
            gain = np.clip(gain, 0.7, 1.4)
            top_rgb[k][vk] = np.clip(smp * gain, 0, 255)

        # ---- weighted median per vertex ----------------------------------------
        for vi in np.flatnonzero(any_valid):
            w_all = np.array([top_scores[k][vi] for k in range(_TOP_K)])
            c_all = np.array([top_rgb[k][vi] for k in range(_TOP_K)])
            valid_k_vi = w_all > -np.inf
            w = w_all[valid_k_vi]
            c = c_all[valid_k_vi]
            w = w / w.sum()
            # weighted median: sort by value per channel, cumsum weights, pick >= 0.5
            out = np.zeros(3, float)
            for ch in range(3):
                order = np.argsort(c[:, ch])
                cum = np.cumsum(w[order])
                out[ch] = c[order][cum >= 0.5][0, ch]
            colors[vi] = np.clip(out, 0, 255).astype(np.uint8)
            covered[vi] = True

    return colors, covered


def compute_colour_error(
    mesh,
    ws_dir: Path,
    cameras_enu_csv: Path,
    scale: float,
    keyframes: int = 5,
    n_samples: int = 2000,
    seed: int = 42,
) -> float | None:
    """Mean absolute RGB error (0-255) between mesh vertex colour and source
    pixels over ``n_samples`` random unoccluded vertices across ``keyframes``
    fixed keyframes.  Returns None if the workspace is unavailable.

    Uses the same camera selection + occlusion logic as
    ``view_based_vertex_colors`` so the metric is directly comparable across
    methods.  ``keyframes`` fixed images are picked evenly from the registered
    camera list so the same frames are always used for all methods.
    """
    sparse = ws_dir / "sparse"
    images_dir = ws_dir / "images"
    depth_dir = ws_dir / "stereo" / "depth_maps"
    if not (sparse.exists() and images_dir.exists() and depth_dir.exists()
            and cameras_enu_csv.exists()):
        return None

    import cv2
    import pandas as pd
    import pycolmap

    from georecon.util.sim3 import quat_wxyz_to_R

    recon = pycolmap.Reconstruction(str(sparse))
    intrinsics = {im.name: recon.cameras[im.camera_id] for im in recon.images.values()}
    cams = pd.read_csv(cameras_enu_csv)
    cams = cams[cams["registered"] == 1].reset_index(drop=True)
    if len(cams) == 0:
        return None

    # Pick keyframe indices evenly spaced
    kf_idx = np.linspace(0, len(cams) - 1, min(keyframes, len(cams)), dtype=int)
    kf_rows = [cams.iloc[i] for i in kf_idx]

    V = np.asarray(mesh.vertices, float)
    Nrm = np.asarray(mesh.vertex_normals, float)
    nlen = np.linalg.norm(Nrm, axis=1, keepdims=True)
    Nrm = np.divide(Nrm, nlen, out=np.zeros_like(Nrm), where=nlen > 1e-9)
    vcol = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(float)

    rng = np.random.default_rng(seed)
    errors: list[float] = []
    tol_frac = 0.02
    facing_thr = -0.2

    for row in kf_rows:
        name = row["name"]
        cam = intrinsics.get(name)
        img_path = images_dir / name
        depth_path = depth_dir / f"{name}.geometric.bin"
        if cam is None or not img_path.exists() or not depth_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        try:
            depth = read_depth_array(depth_path).astype(float) * float(scale)
        except (ValueError, OSError):
            continue

        cam_center = np.array([row["E"], row["N"], row["U"]], float)
        M = quat_wxyz_to_R(row["qw"], row["qx"], row["qy"], row["qz"])
        view_dir = M[2, :]

        Xc = (V - cam_center) @ M.T
        z = Xc[:, 2]
        front = z > 1e-6
        if not front.any():
            continue
        n = len(V)
        uv = np.zeros((n, 2))
        uv[front] = cam.img_from_cam(Xc[front])
        u, v_px = uv[:, 0], uv[:, 1]
        h, w = img.shape[:2]
        facing = (Nrm @ view_dir) < facing_thr
        cand = front & facing & (u >= 0) & (u < w - 1) & (v_px >= 0) & (v_px < h - 1)
        if not cand.any():
            continue
        cand_idx = np.flatnonzero(cand)
        d_sampled, dvalid = _bilinear_sample(depth, u[cand_idx], v_px[cand_idx])
        good = dvalid & (d_sampled > 1e-6) & (np.abs(z[cand_idx] - d_sampled) <= tol_frac * d_sampled)
        cand_idx = cand_idx[good]
        if not len(cand_idx):
            continue

        # Sample up to n_samples from unoccluded vertices
        sample_size = min(n_samples, len(cand_idx))
        sel = rng.choice(cand_idx, sample_size, replace=False)
        img_col, _ = _bilinear_sample(img.astype(float), u[sel], v_px[sel])
        img_rgb = img_col[:, ::-1]   # BGR -> RGB
        mesh_rgb = vcol[sel]
        errors.append(float(np.mean(np.abs(img_rgb - mesh_rgb))))

    return float(np.mean(errors)) if errors else None


def _crop_to_roi(mesh, roi: dict):
    """Drop vertices outside the clean stage's ROI polygon / Z bounds
    (``dense/roi.json``) plus every face that touches them."""
    poly = roi.get("polygon") or []
    if len(poly) < 3:
        return mesh, 0
    from matplotlib.path import Path as MplPath

    v = np.asarray(mesh.vertices, float)
    keep_v = MplPath(np.asarray(poly, float)).contains_points(v[:, :2])
    z_lo, z_hi = roi.get("z_lo"), roi.get("z_hi")
    if z_lo is not None:
        keep_v &= v[:, 2] >= float(z_lo)
    if z_hi is not None:
        keep_v &= v[:, 2] <= float(z_hi)
    n_cropped = int((~keep_v).sum())
    if n_cropped:
        mesh.update_faces(keep_v[mesh.faces].all(axis=1))
        mesh.remove_unreferenced_vertices()
    return mesh, n_cropped


def _drop_small_components(mesh, min_frac: float = 0.01):
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh, 0
    thresh = min_frac * len(mesh.faces)
    keep = [p for p in parts if len(p.faces) >= thresh]
    if not keep:
        return mesh, 0
    import trimesh

    return trimesh.util.concatenate(keep), len(parts) - len(keep)


def _clean_topology(mesh):
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    return mesh


def _decimate(mesh, target: int, dense_xyz: np.ndarray, dense_rgb: np.ndarray):
    """Quadric decimation via fast_simplification; re-sample vertex colours from
    the dense cloud (fast_simplification drops them)."""
    if len(mesh.faces) <= target:
        return mesh
    import fast_simplification as fs
    import trimesh

    v, f = fs.simplify(np.asarray(mesh.vertices, float),
                       np.asarray(mesh.faces, np.int64), target_count=int(target))
    colors = _nearest_colors(v, dense_xyz, dense_rgb)
    return trimesh.Trimesh(vertices=v, faces=f, vertex_colors=colors, process=False)


# --------------------------------------------------------------------------- #
# height-field fallback (NumPy only)
# --------------------------------------------------------------------------- #
def heightfield_mesh(xyz: np.ndarray, rgb: np.ndarray, cell: float):
    """Grid the cloud: per-cell max-z + mean rgb, fill holes <= 2 cells, two
    triangles per fully-valid 2x2 quad. Returns a trimesh.Trimesh."""
    import trimesh

    xyz = np.asarray(xyz, float)
    rgb = np.asarray(rgb, float)
    org = xyz[:, :2].min(axis=0)
    ij = np.floor((xyz[:, :2] - org) / cell).astype(int)
    nx, ny = int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1

    z = np.full((nx, ny), -np.inf)
    np.maximum.at(z, (ij[:, 0], ij[:, 1]), xyz[:, 2])
    csum = np.zeros((nx, ny, 3))
    cnt = np.zeros((nx, ny))
    np.add.at(csum, (ij[:, 0], ij[:, 1]), rgb)
    np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1.0)
    valid = cnt > 0
    col = np.zeros((nx, ny, 3))
    col[valid] = csum[valid] / cnt[valid, None]

    # fill holes up to 2 cells wide: 2 passes of 4-neighbour mean over empties.
    for _ in range(2):
        filled = valid.copy()
        for ax in (0, 1):
            for sh in (-1, 1):
                nb = np.roll(valid, sh, axis=ax)
                nz = np.roll(np.where(valid, z, 0.0), sh, axis=ax)
                nc = np.roll(np.where(valid[..., None], col, 0.0), sh, axis=ax)
                take = (~filled) & nb
                z = np.where(take, nz, z)
                col = np.where(take[..., None], nc, col)
                filled |= take
        valid = filled

    # vertex per cell; face only where all four quad corners are valid.
    xs = org[0] + np.arange(nx) * cell
    ys = org[1] + np.arange(ny) * cell
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    verts = np.stack([gx, gy, np.where(valid, z, 0.0)], axis=-1).reshape(-1, 3)
    vcol = (np.clip(col, 0, 255)).astype(np.uint8).reshape(-1, 3)

    q = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    ii, jj = np.nonzero(q)
    v00 = ii * ny + jj
    v10 = (ii + 1) * ny + jj
    v01 = ii * ny + (jj + 1)
    v11 = (ii + 1) * ny + (jj + 1)
    faces = np.vstack([np.stack([v00, v10, v11], axis=1),
                       np.stack([v00, v11, v01], axis=1)])

    m = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vcol, process=False)
    m.remove_unreferenced_vertices()
    return m


# --------------------------------------------------------------------------- #
# preview raster (NumPy + cv2, no GUI)
# --------------------------------------------------------------------------- #
def render_preview(vertices: np.ndarray, faces: np.ndarray, vcol: np.ndarray,
                   out_png: Path, width: int = 1024) -> None:
    """Top-down orthographic raster: scatter each face's centroid colour into an
    image (higher z wins), close gaps by dilation, modulate by a hillshade of
    the z raster. cv2 + NumPy only, no renderer."""
    import cv2

    v = np.asarray(vertices, float)
    fv = v[faces]
    cen = fv.mean(axis=1)                             # (F,3)
    fcol = np.asarray(vcol, float)[:, :3][faces].mean(axis=1)   # (F,3)

    lo, hi = v[:, :2].min(axis=0), v[:, :2].max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    scale = (width - 2) / span[0]
    h = int(np.clip(span[1] * scale + 2, 4, 8192))
    px = ((cen[:, :2] - lo) * scale + 1).astype(np.int64)
    px[:, 1] = h - 1 - px[:, 1]                       # +Y up
    np.clip(px[:, 0], 0, width - 1, out=px[:, 0])
    np.clip(px[:, 1], 0, h - 1, out=px[:, 1])

    order = np.argsort(cen[:, 2])                     # low z first -> high z wins
    lin = px[order, 1] * width + px[order, 0]
    color = np.zeros((h * width, 3), np.float32)
    zbuf = np.full(h * width, -np.inf, np.float32)
    color[lin] = fcol[order]
    zbuf[lin] = cen[order, 2]
    color = color.reshape(h, width, 3)
    zbuf = zbuf.reshape(h, width)

    k = np.ones((3, 3), np.uint8)
    for _ in range(8):
        hole = ~np.isfinite(zbuf) | (zbuf == -np.inf)
        if not hole.any():
            break
        dz = cv2.dilate(np.where(hole, -1e30, zbuf).astype(np.float32), k)
        dc = cv2.dilate(color, k)
        take = hole & (dz > -1e29)
        zbuf = np.where(take, dz, zbuf)
        color = np.where(take[..., None], dc, color)

    bg = ~np.isfinite(zbuf) | (zbuf == -np.inf)
    zfill = np.where(bg, np.median(zbuf[~bg]) if (~bg).any() else 0.0, zbuf)
    gy, gx = np.gradient(zfill)
    n = np.dstack([-gx, -gy, np.ones_like(zfill)])
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-9
    light = np.array([-0.5, 0.5, 0.7])
    light /= np.linalg.norm(light)
    shade = np.clip(n @ light, 0.15, 1.0)

    img = np.clip(color * shade[..., None], 0, 255).astype(np.uint8)
    img[bg] = 32
    cv2.imwrite(str(out_png), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #
def _load_ply_xyz_rgb(path: Path):
    d = read_ply(path)
    xyz = np.stack([d["x"], d["y"], d["z"]], axis=-1).astype(float)
    if {"red", "green", "blue"} <= d.keys():
        rgb = np.stack([d["red"], d["green"], d["blue"]], axis=-1).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 200, np.uint8)
    return xyz, rgb


def poisson_args(src_ply: Path, out_ply: Path, mcfg) -> dict:
    """poisson_mesher options — keys verified against COLMAP 4.2.0 ``-h``."""
    return {
        "input_path": str(src_ply),
        "output_path": str(out_ply),
        "PoissonMeshing.depth": mcfg.poisson_depth,
        "PoissonMeshing.trim": mcfg.poisson_trim,
        "PoissonMeshing.point_weight": 1,
        "PoissonMeshing.color": True,
        "PoissonMeshing.num_threads": -1,
    }


def choose_poisson(method: str, fused_sfm_exists: bool, cli_available: bool,
                   n_points: int, min_points: int) -> bool:
    if method == "poisson":
        return True
    if method == "heightfield":
        return False
    return fused_sfm_exists and cli_available and n_points >= min_points


def _poisson(ctx, mcfg, src_ply: Path, out_ply: Path) -> None:
    colmap_cli.run("poisson_mesher", poisson_args(src_ply, out_ply, mcfg),
                   ctx.log, timeout=mcfg.max_seconds + 120)


def _empty_metrics(method: str, secs: float, **extra) -> dict:
    m = {"method": method, "poisson_depth": None, "trim": None,
         "triangles_raw": 0, "triangles": 0, "vertices": 0,
         "components_removed": 0, "cropped_vertices": 0,
         "surface_area_m2": 0.0, "seconds": round(secs, 3)}
    m.update(extra)
    return m


def run(ctx: "StageContext") -> dict:
    import trimesh

    paths = ctx.paths
    t0 = perf_counter()
    mcfg = ctx.cfg.resolved_mesh
    paths.mesh.mkdir(parents=True, exist_ok=True)

    src_cloud = paths.dense_clean
    if not src_cloud.exists():
        src_cloud = paths.dense_fused
        if src_cloud.exists():
            ctx.warn("dense/clean.ply missing; meshing from uncleaned dense/fused.ply")
    if not src_cloud.exists():
        ctx.warn("no dense/clean.ply or dense/fused.ply: mesh skipped")
        return _empty_metrics("none", perf_counter() - t0)

    enu_xyz, enu_rgb = _load_ply_xyz_rgb(src_cloud)
    cli = colmap_cli.probe()
    want_poisson = choose_poisson(mcfg.method, src_cloud.exists(), cli["available"],
                                  len(enu_xyz), mcfg.min_points_for_poisson)

    roi: dict = {}
    if paths.dense_roi.exists():
        roi = json.loads(paths.dense_roi.read_text(encoding="utf-8"))

    method = "heightfield"
    depth = trim = None
    comps = cropped = 0
    tris_raw = 0
    mesh = None

    if want_poisson:
        try:
            # src_cloud is already ENU with normals (from the clean stage) —
            # poisson_mesher's output needs no coordinate transform.
            p_ply = paths.mesh / "poisson.ply"
            t_p = perf_counter()
            _poisson(ctx, mcfg, src_cloud, p_ply)
            if perf_counter() - t_p > mcfg.max_seconds:
                raise RuntimeError(f"poisson_mesher exceeded {mcfg.max_seconds:.0f}s")
            m = trimesh.load(p_ply, process=False, force="mesh")
            tris_raw = len(m.faces)
            m, cropped = _crop_to_roi(m, roi)
            m, far_cropped = _crop_far_vertices(m, enu_xyz, 3.0 * _median_nn(enu_xyz))
            cropped += far_cropped
            m = _clean_topology(m)
            m, comps = _drop_small_components(m, 0.01)
            m = _clean_topology(m)
            m = _decimate(m, mcfg.max_tris, enu_xyz, enu_rgb)
            mesh, method, depth, trim = m, "poisson", mcfg.poisson_depth, mcfg.poisson_trim
        except Exception as exc:                       # noqa: BLE001
            ctx.warn(f"poisson meshing failed ({type(exc).__name__}: {exc}); "
                     f"falling back to height-field")

    if mesh is None:
        cell = max(0.1, 2.0 * _median_nn(enu_xyz))
        mesh = heightfield_mesh(enu_xyz, enu_rgb, cell)
        tris_raw = len(mesh.faces)
        if len(mesh.faces) > mcfg.max_tris:
            mesh = _decimate(mesh, mcfg.max_tris, enu_xyz, enu_rgb)

    # ---- view-based vertex colours (nearest-point colours are the fallback) --
    vcol = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.uint8)
    colour_view_coverage = 0.0
    ws_dir = paths.dense / "ws"
    scale = 1.0
    if paths.georef_transform.exists():
        try:
            scale = float(json.loads(
                paths.georef_transform.read_text(encoding="utf-8"))["s"])
        except Exception:  # noqa: BLE001
            pass
    if ws_dir.exists():
        try:
            view_colors, covered = view_based_vertex_colors(
                mesh.vertices, mesh.vertex_normals, ws_dir, paths.georef_cameras, scale)
            vcol[covered] = view_colors[covered]
            colour_view_coverage = round(float(covered.mean()), 4) if len(covered) else 0.0
        except Exception as exc:                       # noqa: BLE001
            ctx.warn(f"view-based vertex colouring failed ({type(exc).__name__}: {exc}); "
                     f"keeping nearest-point colours")
    mesh.visual.vertex_colors = vcol
    # binary PLY *with faces* (util.ply is vertex-only) — exports read this back.
    paths.mesh_ply.write_bytes(mesh.export(file_type="ply", encoding="binary"))
    try:
        render_preview(mesh.vertices, mesh.faces, vcol, paths.mesh_preview)
    except Exception as exc:                           # noqa: BLE001
        ctx.warn(f"mesh preview failed ({type(exc).__name__}: {exc})")

    # ---- shared colour_error metric -----------------------------------------
    colour_error: float | None = None
    if ws_dir.exists():
        try:
            colour_error = compute_colour_error(
                mesh, ws_dir, paths.georef_cameras, scale)
        except Exception as exc:                       # noqa: BLE001
            ctx.warn(f"colour_error metric failed ({type(exc).__name__}: {exc})")

    ctx.log.info("mesh: method=%s tris %d->%d verts=%d comps_removed=%d cropped=%d "
                 "colour_view_coverage=%.1f%% colour_error=%s",
                 method, tris_raw, len(mesh.faces), len(mesh.vertices), comps, cropped,
                 colour_view_coverage * 100.0,
                 f"{colour_error:.2f}" if colour_error is not None else "n/a")
    return {
        "method": method,
        "roi_kind": roi.get("kind"),
        "colour_view_coverage": colour_view_coverage,
        "colour_error": colour_error,
        "poisson_depth": depth,
        "trim": trim,
        "triangles_raw": int(tris_raw),
        "triangles": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "components_removed": int(comps),
        "cropped_vertices": int(cropped),
        "surface_area_m2": round(float(mesh.area), 3),
        "seconds": round(perf_counter() - t0, 3),
    }
