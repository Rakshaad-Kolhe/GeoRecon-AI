import numpy as np
import pytest

from georecon.stages.export import (
    _glb_axes,
    _rasterise,
    _voxel_downsample,
    _write_geotiff,
    _write_las,
)


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
