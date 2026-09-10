"""Export stage: turn the ENU dense cloud + mesh into shareable deliverables.

Processing frame is local ENU metres (origin = first registered keyframe GPS,
``georef/origin.json``). LAS / GeoTIFF are re-projected to the origin's UTM zone
with **ellipsoidal** height (origin alt + U); the browser assets stay in ENU.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from georecon.util import geo
from georecon.util.ply import read_ply, write_ply

if TYPE_CHECKING:
    from georecon.pipeline import StageContext

WEB_MAX_POINTS = 1_500_000


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _median_nn(xyz: np.ndarray, sample: int = 20_000, seed: int = 0) -> float:
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, float)
    if len(xyz) > sample:
        xyz = xyz[np.random.default_rng(seed).choice(len(xyz), sample, replace=False)]
    if len(xyz) < 2:
        return 0.0
    d, _ = cKDTree(xyz).query(xyz, k=2)
    return float(np.median(d[:, 1]))


def _voxel_downsample(xyz: np.ndarray, cols: dict, voxel: float):
    key = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx.sort()
    return xyz[idx], {k: v[idx] for k, v in cols.items()}


def _enu_to_utm(e, n, u, origin: dict):
    """ENU metres -> (easting, northing, ellipsoidal_z) in the origin's UTM."""
    lat, lon, alt = geo.enu_to_wgs84(e, n, u, origin)
    from pyproj import Transformer

    tr = Transformer.from_crs(4326, int(origin["utm_epsg"]), always_xy=True)
    east, north = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(east, float), np.asarray(north, float), alt


def _registered_cameras(cameras_csv: Path, frames_csv: Path, residuals_csv: Path):
    """[(name, E, N, U, t, inlier)] for registered cameras, in file order."""
    t_by = {}
    if frames_csv.exists():
        for r in csv.DictReader(open(frames_csv, newline="")):
            t_by[r["name"]] = float(r["t"])
    inl_by = {}
    if residuals_csv.exists():
        for r in csv.DictReader(open(residuals_csv, newline="")):
            inl_by[r["name"]] = r.get("inlier") == "1"
    out = []
    for r in csv.DictReader(open(cameras_csv, newline="")):
        if r.get("registered") != "1":
            continue
        out.append((r["name"], float(r["E"]), float(r["N"]), float(r["U"]),
                    t_by.get(r["name"]), inl_by.get(r["name"], False)))
    return out


def _write_las(path: Path, e, n, z, rgb, epsg: int) -> None:
    import laspy
    import pyproj

    xyz = np.column_stack([e, n, z]).astype(float)
    hdr = laspy.LasHeader(point_format=3, version="1.4")
    hdr.scales = [0.001, 0.001, 0.001]
    hdr.offsets = np.floor(xyz.min(axis=0))
    hdr.add_crs(pyproj.CRS.from_epsg(int(epsg)))
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r16 = np.asarray(rgb, np.uint16) * 257
    las.red, las.green, las.blue = r16[:, 0], r16[:, 1], r16[:, 2]
    las.write(str(path))


def _rasterise(e, n, z, rgb, cell: float):
    """UTM grid: per-cell max-Z (float32, nodata -9999) and mean-RGB (uint8)."""
    e0, n1 = float(np.min(e)), float(np.max(n))               # NW corner
    col = np.floor((e - e0) / cell).astype(int)
    row = np.floor((n1 - n) / cell).astype(int)
    w = int(col.max()) + 1
    h = int(row.max()) + 1

    dsm = np.full((h, w), -9999.0, np.float32)
    np.maximum.at(dsm, (row, col), z.astype(np.float32))
    csum = np.zeros((h, w, 3), np.float64)
    cnt = np.zeros((h, w), np.float64)
    np.add.at(csum, (row, col), np.asarray(rgb, float))
    np.add.at(cnt, (row, col), 1.0)
    color = np.zeros((h, w, 3), np.uint8)
    ok = cnt > 0
    color[ok] = np.clip(csum[ok] / cnt[ok, None], 0, 255).astype(np.uint8)
    return dsm, color, e0, n1


def _write_geotiff(path: Path, arr: np.ndarray, e0: float, n1: float,
                   cell: float, epsg: int, nodata=None) -> None:
    import rasterio
    from rasterio.transform import from_origin

    bands = 1 if arr.ndim == 2 else arr.shape[2]
    data = arr[None] if arr.ndim == 2 else np.moveaxis(arr, 2, 0)
    h, w = arr.shape[:2]
    blk = max(16, min(256, (min(h, w) // 16) * 16 or 16))
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=bands,
        dtype=data.dtype, crs=f"EPSG:{int(epsg)}",
        transform=from_origin(e0, n1, cell, cell), nodata=nodata,
        compress="deflate", tiled=True, blockxsize=blk, blockysize=blk,
    ) as dst:
        dst.write(data)


def _glb_axes(v_enu: np.ndarray) -> np.ndarray:
    """ENU (x=E,y=N,z=U) -> glTF right-handed +Y up: x=E, y=U, z=-N."""
    return np.column_stack([v_enu[:, 0], v_enu[:, 2], -v_enu[:, 1]])


def _conf_from_views(views: np.ndarray) -> np.ndarray:
    """Per-point confidence 0..255 = clip(views / max(8, p95(views)) * 255)."""
    v = np.asarray(views, float)
    denom = max(8.0, float(np.percentile(v, 95)) if len(v) else 8.0)
    return np.clip(v / denom * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #
def run(ctx: "StageContext") -> dict:
    import trimesh

    paths = ctx.paths
    t0 = perf_counter()
    out = paths.outputs
    web = paths.outputs_web
    out.mkdir(parents=True, exist_ok=True)
    web.mkdir(parents=True, exist_ok=True)

    origin = json.loads(paths.georef_origin.read_text(encoding="utf-8"))
    epsg = int(origin["utm_epsg"])

    d = read_ply(paths.dense_fused)
    xyz = np.column_stack([d["x"], d["y"], d["z"]]).astype(np.float32)
    rgb = np.column_stack([d["red"], d["green"], d["blue"]]).astype(np.uint8)
    views = np.asarray(d.get("views", np.zeros(len(xyz), np.uint8)))
    med_sp = _median_nn(xyz)

    files: dict[str, int] = {}

    def _record(p: Path):
        files[str(p.relative_to(out))] = p.stat().st_size

    # ---- point cloud (ENU, all props) -------------------------------------
    shutil.copy2(paths.dense_fused, out / "pointcloud.ply")
    _record(out / "pointcloud.ply")

    # ---- LAS (UTM, ellipsoidal Z) ---------------------------------------
    e, n, ell = _enu_to_utm(xyz[:, 0], xyz[:, 1], xyz[:, 2], origin)
    _write_las(out / "pointcloud.las", e, n, ell, rgb, epsg)
    _record(out / "pointcloud.las")

    # ---- mesh: OBJ (ENU) + GLB (+Y up) --------------------------------
    mesh = trimesh.load(paths.mesh_ply, process=False, force="mesh")
    vcol = np.asarray(mesh.visual.vertex_colors)[:, :4].astype(np.uint8)
    (out / "mesh.obj").write_bytes(mesh.export(file_type="obj").encode())
    _record(out / "mesh.obj")

    glb = trimesh.Trimesh(vertices=_glb_axes(np.asarray(mesh.vertices, float)),
                          faces=np.asarray(mesh.faces), vertex_colors=vcol,
                          process=False)
    (out / "model.glb").write_bytes(glb.export(file_type="glb"))
    _record(out / "model.glb")

    (out / "georef.json").write_text(json.dumps({
        "origin": {k: origin[k] for k in ("lat", "lon", "alt")},
        "utm_epsg": epsg,
        "frame": "local ENU, +Z up, metres",
        "height_ref": "ellipsoidal (GPS)",
    }, indent=2), encoding="utf-8")
    _record(out / "georef.json")

    # ---- DSM + colour raster (UTM) --------------------------------------
    from rasterio.fill import fillnodata

    cell = max(0.1, 2.0 * med_sp)
    dsm, color, e0, n1 = _rasterise(e, n, ell, rgb, cell)
    dsm_filled = fillnodata(dsm.copy(), mask=(dsm != -9999.0).astype(np.uint8),
                            max_search_distance=3)
    _write_geotiff(out / "dsm.tif", dsm_filled, e0, n1, cell, epsg, nodata=-9999.0)
    _write_geotiff(out / "color.tif", color, e0, n1, cell, epsg)
    _record(out / "dsm.tif")
    _record(out / "color.tif")
    dsm_h, dsm_w = dsm.shape

    # ---- trajectory GeoJSON --------------------------------------------
    cams = _registered_cameras(paths.georef_cameras, paths.frames_csv,
                               paths.georef_residuals)
    traj_enu = [[round(c[1], 3), round(c[2], 3), round(c[3], 3)] for c in cams]
    feats = []
    if cams:
        clat, clon, calt = geo.enu_to_wgs84(
            np.array([c[1] for c in cams]), np.array([c[2] for c in cams]),
            np.array([c[3] for c in cams]), origin)
        line = [[round(float(lo), 8), round(float(la), 8), round(float(al), 3)]
                for la, lo, al in zip(clat, clon, calt)]
        feats.append({"type": "Feature", "properties": {"kind": "trajectory"},
                      "geometry": {"type": "LineString", "coordinates": line}})
        for (name, _e, _n, _u, t, inl), pt in zip(cams, line):
            feats.append({"type": "Feature",
                          "properties": {"name": name, "t": t, "inlier": bool(inl)},
                          "geometry": {"type": "Point", "coordinates": pt}})
    (out / "trajectory.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    _record(out / "trajectory.geojson")

    # ---- web assets ---------------------------------------------------
    w_xyz, w_cols = xyz, {"red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2],
                          "views": views}
    if len(w_xyz) > WEB_MAX_POINTS:
        vx = max(cell / 4, med_sp)
        while len(w_xyz) > WEB_MAX_POINTS:
            w_xyz, w_cols = _voxel_downsample(xyz, w_cols, vx)
            vx *= 1.3
    conf = _conf_from_views(w_cols["views"])
    write_ply(web / "pointcloud.ply", w_xyz.astype(np.float32), {
        "red": w_cols["red"].astype(np.uint8),
        "green": w_cols["green"].astype(np.uint8),
        "blue": w_cols["blue"].astype(np.uint8),
        "conf": conf,
    })
    shutil.copy2(paths.mesh_ply, web / "mesh.ply")

    lo = xyz.min(axis=0).round(3).tolist()
    hi = xyz.max(axis=0).round(3).tolist()
    (web / "meta.json").write_text(json.dumps({
        "origin": {k: origin[k] for k in ("lat", "lon", "alt")},
        "utm_epsg": epsg, "units": "m", "up": "+Z",
        "bbox_enu": {"min": lo, "max": hi},
        "points": int(len(xyz)),
        "triangles": int(len(mesh.faces)),
        "median_spacing_m": round(med_sp, 4),
        "trajectory_enu": traj_enu,
        "height_ref": "ellipsoidal (GPS)",
    }, indent=2), encoding="utf-8")

    for p in (web / "pointcloud.ply", web / "mesh.ply", web / "meta.json"):
        _record(p)

    ctx.log.info("export: %d files, DSM %dx%d @ %.2fm", len(files), dsm_w, dsm_h, cell)
    return {
        "files": files,
        "dsm_cell_m": round(cell, 4),
        "dsm_size_px": [dsm_w, dsm_h],
        "formats": ["ply", "las", "obj", "glb", "tif", "geojson", "json"],
        "seconds": round(perf_counter() - t0, 3),
    }
