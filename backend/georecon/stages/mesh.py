"""Mesh stage: dense point cloud -> triangle mesh (ENU metres) + preview PNG.

Poisson path: COLMAP ``poisson_mesher`` on ``dense/fused_sfm.ply`` (SfM frame,
carries normals) -> trimesh cleanup -> georef transform -> ``mesh/mesh.ply``.
Falls back to a NumPy height-field when Poisson is unavailable, errors, runs
long, or the cloud is too sparse (no-CUDA path).

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
from georecon.util.ply import read_ply, write_ply

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


def _transform_vertices(v: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return s * (np.asarray(v, float) @ np.asarray(R, float).T) + np.asarray(t, float)


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

    if not paths.dense_fused.exists():
        ctx.warn("no dense/fused.ply: mesh skipped")
        return _empty_metrics("none", perf_counter() - t0)

    enu_xyz, enu_rgb = _load_ply_xyz_rgb(paths.dense_fused)
    fused_sfm = paths.dense / "fused_sfm.ply"
    cli = colmap_cli.probe()
    want_poisson = choose_poisson(mcfg.method, fused_sfm.exists(), cli["available"],
                                  len(enu_xyz), mcfg.min_points_for_poisson)

    tr_path = paths.georef_transform
    s = R = t = None
    if tr_path.exists():
        tr = json.loads(tr_path.read_text(encoding="utf-8"))
        s, R, t = float(tr["s"]), np.asarray(tr["R"], float), np.asarray(tr["t"], float)

    method = "heightfield"
    depth = trim = None
    comps = cropped = 0
    tris_raw = 0
    mesh = None

    if want_poisson and s is not None:
        try:
            p_ply = paths.mesh / "poisson_sfm.ply"
            t_p = perf_counter()
            _poisson(ctx, mcfg, fused_sfm, p_ply)
            if perf_counter() - t_p > mcfg.max_seconds:
                raise RuntimeError(f"poisson_mesher exceeded {mcfg.max_seconds:.0f}s")
            m = trimesh.load(p_ply, process=False, force="mesh")
            tris_raw = len(m.faces)
            m.vertices = _transform_vertices(m.vertices, s, R, t)
            m, cropped = _crop_far_vertices(m, enu_xyz, 3.0 * _median_nn(enu_xyz))
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

    vcol = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.uint8)
    write_ply(paths.mesh_ply, np.asarray(mesh.vertices, float), {
        "red": vcol[:, 0], "green": vcol[:, 1], "blue": vcol[:, 2],
    })
    try:
        render_preview(mesh.vertices, mesh.faces, vcol, paths.mesh_preview)
    except Exception as exc:                           # noqa: BLE001
        ctx.warn(f"mesh preview failed ({type(exc).__name__}: {exc})")

    ctx.log.info("mesh: method=%s tris %d->%d verts=%d comps_removed=%d cropped=%d",
                 method, tris_raw, len(mesh.faces), len(mesh.vertices), comps, cropped)
    return {
        "method": method,
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
