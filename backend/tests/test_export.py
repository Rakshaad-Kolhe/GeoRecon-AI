import json

import numpy as np
import pandas as pd
import pytest
import trimesh

from georecon import pipeline
from georecon.config import JobConfig
from georecon.pipeline import JobPaths
from georecon.stages import export
from georecon.stages.export import (
    _glb_axes,
    _rasterise,
    _voxel_downsample,
    _write_geotiff,
    _write_las,
)
from georecon.util.ply import write_ply


def test_las_roundtrip_crs_and_coords(tmp_path):
    laspy = pytest.importorskip("laspy")
    rng = np.random.default_rng(0)
    e = 500_000 + rng.uniform(0, 100, 400)
    n = 4_570_000 + rng.uniform(0, 100, 400)
    z = 300 + rng.uniform(0, 20, 400)
    rgb = rng.integers(0, 256, size=(400, 3)).astype(np.uint8)

    p = tmp_path / "pc.las"
    _write_las(p, e, n, z, rgb, epsg=32617)
    r = laspy.read(str(p))
    assert r.header.point_format.id == 3
    assert r.header.parse_crs().to_epsg() == 32617
    assert np.max(np.abs(np.asarray(r.x) - e)) < 1e-2      # 0.001 scale
    assert np.max(np.abs(np.asarray(r.z) - z)) < 1e-2
    assert int(r.red[0]) == int(rgb[0, 0]) * 257           # 8-bit -> 16-bit


def test_geotiff_crs_bounds_nodata(tmp_path):
    pytest.importorskip("rasterio")
    import rasterio

    arr = np.full((40, 60), -9999.0, np.float32)
    arr[5:35, 5:55] = 12.5
    _write_geotiff(tmp_path / "d.tif", arr, e0=500_000.0, n1=4_570_100.0,
                   cell=0.5, epsg=32617, nodata=-9999.0)
    with rasterio.open(tmp_path / "d.tif") as s:
        assert s.crs.to_epsg() == 32617
        assert s.nodata == -9999.0
        assert s.res == pytest.approx((0.5, 0.5))
        assert s.bounds.left == pytest.approx(500_000.0)
        assert s.bounds.top == pytest.approx(4_570_100.0)
        assert s.bounds.right == pytest.approx(500_030.0)     # 60 * 0.5


def test_glb_axes_enu_to_gltf():
    v = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])       # E, N, U
    g = _glb_axes(v)
    assert np.allclose(g, [[1.0, 3.0, -2.0], [-4.0, -6.0, -5.0]])  # x=E, y=U, z=-N


def test_voxel_downsample_reduces_and_keeps_columns():
    xyz = np.repeat(np.array([[0.0, 0, 0], [10, 10, 10]]), 50, axis=0)
    cols = {"r": np.arange(100, dtype=np.uint8)}
    dx, dc = _voxel_downsample(xyz, cols, voxel=1.0)
    assert len(dx) == 2 and len(dc["r"]) == 2


def test_rasterise_max_z_and_mean_rgb():
    # two cells, cell=1 m; cell (0,0) has z {1,3}, cell (1,0) has z {2}
    e = np.array([0.2, 0.7, 1.5])
    n = np.array([0.5, 0.5, 0.5])
    z = np.array([1.0, 3.0, 2.0])
    rgb = np.array([[0, 0, 0], [100, 100, 100], [40, 40, 40]], np.uint8)
    dsm, color, e0, n1 = _rasterise(e, n, z, rgb, cell=1.0)
    assert dsm.shape == (1, 2)
    assert dsm[0, 0] == 3.0 and dsm[0, 1] == 2.0
    assert tuple(color[0, 0]) == (50, 50, 50)                # mean of 0 and 100


# --------------------------------------------------------------------------- #
def _write_export_job(tmp_path, *, georeferenced, n_pts=200, n_cams=5):
    paths = JobPaths.for_job(tmp_path / "job").ensure()
    rng = np.random.default_rng(3)
    xyz = rng.uniform(-10, 10, size=(n_pts, 3)).astype(np.float32)
    xyz[:, 2] = rng.uniform(0, 5, n_pts)
    rgb = rng.integers(0, 256, size=(n_pts, 3)).astype(np.uint8)
    views = rng.integers(1, 20, n_pts).astype(np.uint8)
    write_ply(paths.dense_fused, xyz, {
        "red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2], "views": views,
    })

    verts = np.array([[-5.0, -5.0, 0.0], [5.0, -5.0, 0.0],
                      [5.0, 5.0, 0.0], [-5.0, 5.0, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual.vertex_colors = np.tile([200, 100, 50, 255], (4, 1)).astype(np.uint8)
    paths.mesh_ply.write_bytes(mesh.export(file_type="ply", encoding="binary"))

    transform = {
        "s": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0],
        "branch": "sim3" if georeferenced else "none", "collinearity": 0.0,
        "scale_source": None if georeferenced else "normalised",
        "units": None if georeferenced else "rel. units",
    }
    paths.georef_transform.write_text(json.dumps(transform), encoding="utf-8")
    if georeferenced:
        origin = {"lat": 18.52, "lon": 73.85, "alt": 560.0, "utm_epsg": 32643}
        paths.georef_origin.write_text(json.dumps(origin), encoding="utf-8")

    names = [f"f_{i:06d}.jpg" for i in range(n_cams)]
    pd.DataFrame({
        "name": names, "registered": [1] * n_cams,
        "E": np.linspace(-5, 5, n_cams), "N": np.zeros(n_cams), "U": np.full(n_cams, 50.0),
        "qw": [1.0] * n_cams, "qx": [0.0] * n_cams, "qy": [0.0] * n_cams, "qz": [0.0] * n_cams,
    }).to_csv(paths.georef_cameras, index=False)
    pd.DataFrame({
        "name": names, "frame_idx": range(n_cams), "t": np.arange(float(n_cams)),
        "lat": [np.nan] * n_cams, "lon": [np.nan] * n_cams, "alt": [np.nan] * n_cams,
        "sharpness": [1.0] * n_cams, "disp_px": [1.0] * n_cams,
    }).to_csv(paths.frames_csv, index=False)
    pd.DataFrame({
        "name": names, "dE": [0.0] * n_cams, "dN": [0.0] * n_cams, "dU": [0.0] * n_cams,
        "inlier": [1] * n_cams,
    }).to_csv(paths.georef_residuals, index=False)
    return paths


def test_export_georeferenced_writes_geotiffs_and_crs_las(tmp_path, monkeypatch):
    pytest.importorskip("rasterio")
    laspy = pytest.importorskip("laspy")
    paths = _write_export_job(tmp_path, georeferenced=True)
    monkeypatch.setattr(pipeline, "STAGES", [("export", export.run)])
    pipeline.run_job(tmp_path / "job", JobConfig(video_path="x"))

    out = paths.outputs
    for name in ("pointcloud.ply", "pointcloud.las", "mesh.obj", "model.glb",
                "dsm.tif", "color.tif", "trajectory.geojson", "georef.json"):
        assert (out / name).exists(), name

    meta = json.loads((paths.outputs_web / "meta.json").read_text())
    assert meta["georeferenced"] is True
    assert meta["origin"] is not None
    assert meta["units"] == "m"

    las = laspy.read(str(out / "pointcloud.las"))
    assert las.header.parse_crs() is not None


def test_export_none_branch_skips_crs_outputs(tmp_path, monkeypatch):
    laspy = pytest.importorskip("laspy")
    paths = _write_export_job(tmp_path, georeferenced=False)
    monkeypatch.setattr(pipeline, "STAGES", [("export", export.run)])
    pipeline.run_job(tmp_path / "job", JobConfig(video_path="x"))

    out = paths.outputs
    for name in ("pointcloud.ply", "pointcloud.las", "mesh.obj", "model.glb", "georef.json"):
        assert (out / name).exists(), name
    for name in ("dsm.tif", "color.tif", "trajectory.geojson"):
        assert not (out / name).exists(), name
    assert (paths.outputs_web / "pointcloud.ply").exists()
    assert (paths.outputs_web / "mesh.ply").exists()

    meta = json.loads((paths.outputs_web / "meta.json").read_text())
    assert meta["georeferenced"] is False
    assert meta["origin"] is None
    assert meta["scale_source"] == "normalised"
    assert meta["units"] == "rel. units"
    assert len(meta["trajectory_enu"]) == 5

    las = laspy.read(str(out / "pointcloud.las"))
    assert las.header.parse_crs() is None
