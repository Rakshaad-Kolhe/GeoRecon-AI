import numpy as np
import pandas as pd
import pytest
import trimesh

from georecon.config import JobConfig, MeshCfg
from georecon.stages.mesh import (
    _crop_far_vertices,
    _crop_to_roi,
    _drop_small_components,
    choose_poisson,
    heightfield_mesh,
    poisson_args,
    view_based_vertex_colors,
)
from georecon.util.colmap_cli import build_args
from georecon.util.depth_map import write_array
from georecon.util.sim3 import R_to_quat_wxyz


def test_choose_poisson_auto_thresholds():
    # auto: needs fused_sfm + CLI + enough points
    assert choose_poisson("auto", True, True, 60_000, 50_000) is True
    assert choose_poisson("auto", True, True, 40_000, 50_000) is False
    assert choose_poisson("auto", False, True, 99_000, 50_000) is False
    assert choose_poisson("auto", True, False, 99_000, 50_000) is False
    # explicit overrides ignore everything else
    assert choose_poisson("poisson", False, False, 0, 50_000) is True
    assert choose_poisson("heightfield", True, True, 10**7, 0) is False


def test_poisson_args_only_verified_options():
    args = poisson_args("in.ply", "out.ply", MeshCfg(poisson_depth=10, poisson_trim=7))
    assert set(args) == {
        "input_path", "output_path", "PoissonMeshing.depth", "PoissonMeshing.trim",
        "PoissonMeshing.point_weight", "PoissonMeshing.color", "PoissonMeshing.num_threads",
    }
    flat = build_args(args)
    assert "--PoissonMeshing.depth" in flat and "10" in flat
    assert flat[flat.index("--PoissonMeshing.trim") + 1] == "7"
    assert flat[flat.index("--PoissonMeshing.color") + 1] == "1"


def test_heightfield_on_synthetic_plane():
    g = np.arange(0, 20, 1.0)
    gx, gy = np.meshgrid(g, g)
    xyz = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 5.0)])
    rgb = np.full((len(xyz), 3), 128, np.uint8)

    m = heightfield_mesh(xyz, rgb, cell=2.0)             # nx = ny = 10
    assert m.faces.shape[0] == 2 * 9 * 9                 # two tris per valid quad
    assert np.allclose(m.vertices[:, 2], 5.0)
    assert np.allclose(m.bounds[:, 2], [5.0, 5.0])


def test_crop_far_vertices_removes_dome():
    # one triangle near the cloud, one lifted 100 m away
    verts = np.array([[0, 0, 0.0], [1, 0, 0], [0, 1, 0],       # near
                      [0, 0, 100.0], [1, 0, 100], [0, 1, 100]])  # far
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    cloud = np.random.default_rng(0).uniform(-1, 1, size=(500, 3))
    cloud[:, 2] *= 0.05

    m2, n_cropped = _crop_far_vertices(m, cloud, max_dist=1.0)
    assert n_cropped == 3
    assert len(m2.faces) == 1
    assert m2.vertices[:, 2].max() < 1.0


def test_drop_small_components():
    big = trimesh.creation.icosphere(subdivisions=3)          # 1280 faces
    speck = trimesh.Trimesh(
        vertices=np.array([[100, 100, 0.0], [101, 100, 0], [100, 101, 0]]),
        faces=np.array([[0, 1, 2]]), process=False)
    combo = trimesh.util.concatenate([big, speck])

    out, removed = _drop_small_components(combo, min_frac=0.01)   # thresh ~12.8
    assert removed == 1
    assert len(out.faces) == len(big.faces)


def test_resolved_mesh_preset_values():
    assert JobConfig(preset="fast").resolved_mesh.poisson_depth == 9
    assert JobConfig(preset="fast").resolved_mesh.max_tris == 200_000
    assert JobConfig(preset="accurate").resolved_mesh.poisson_depth == 11
    assert JobConfig(preset="accurate").resolved_mesh.max_tris == 800_000
    over = MeshCfg(method="heightfield")
    assert JobConfig(preset="accurate", mesh=over).resolved_mesh == over


def test_crop_to_roi_drops_outside_polygon_and_z_bounds():
    verts = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],    # inside polygon+Z
        [100.0, 100.0, 0.0], [101.0, 100.0, 0.0], [100.0, 101.0, 0.0],  # outside polygon
        [0.2, 0.2, 500.0], [1.2, 0.2, 500.0], [0.2, 1.2, 500.0],        # inside XY, above z_hi
    ])
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    roi = {"polygon": [[-10, -10], [10, -10], [10, 10], [-10, 10]], "z_lo": -1.0, "z_hi": 5.0}

    out, n_cropped = _crop_to_roi(mesh, roi)
    assert len(out.faces) == 1
    assert n_cropped == 6


def _write_synthetic_camera_scene(tmp_path, *, depth_a: float, depth_b: float):
    """Two PINHOLE cameras directly above the origin looking straight down
    (world -Z): camera A's depth map matches the true distance (unoccluded),
    camera B's is deliberately wrong (occluded / mismatched)."""
    import cv2
    import pycolmap

    ws = tmp_path / "ws"
    images_dir, depth_dir, sparse_dir = ws / "images", ws / "stereo" / "depth_maps", ws / "sparse"
    for d in (images_dir, depth_dir, sparse_dir):
        d.mkdir(parents=True)

    recon = pycolmap.Reconstruction()
    cam_a = pycolmap.Camera.create_from_model_name(1, "PINHOLE", 20.0, 20, 20)
    cam_b = pycolmap.Camera.create_from_model_name(2, "PINHOLE", 20.0, 20, 20)
    recon.add_camera_with_trivial_rig(cam_a)
    recon.add_camera_with_trivial_rig(cam_b)
    pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.zeros(3))
    recon.add_image_with_trivial_frame(
        pycolmap.Image(name="camA.jpg", camera_id=1, image_id=1), pose)
    recon.add_image_with_trivial_frame(
        pycolmap.Image(name="camB.jpg", camera_id=2, image_id=2), pose)
    recon.write(str(sparse_dir))

    cv2.imwrite(str(images_dir / "camA.jpg"), np.full((20, 20, 3), (50, 50, 200), np.uint8))  # BGR->RGB red
    cv2.imwrite(str(images_dir / "camB.jpg"), np.full((20, 20, 3), (200, 50, 50), np.uint8))  # BGR->RGB blue
    write_array(depth_dir / "camA.jpg.geometric.bin", np.full((20, 20), depth_a, np.float32))
    write_array(depth_dir / "camB.jpg.geometric.bin", np.full((20, 20), depth_b, np.float32))

    # cameras looking straight down (world +Z maps to camera "behind"): M = diag(1,-1,-1)
    M = np.diag([1.0, -1.0, -1.0])
    q = R_to_quat_wxyz(M)
    cams_csv = tmp_path / "cameras_enu.csv"
    pd.DataFrame({
        "name": ["camA.jpg", "camB.jpg"], "registered": [1, 1],
        "E": [0.0, 0.0], "N": [0.0, 0.0], "U": [10.0, 10.0],
        "qw": [q[0]] * 2, "qx": [q[1]] * 2, "qy": [q[2]] * 2, "qz": [q[3]] * 2,
    }).to_csv(cams_csv, index=False)
    return ws, cams_csv


def test_view_based_colors_picks_unoccluded_frontal_camera(tmp_path):
    pytest.importorskip("pycolmap")
    # true distance to the vertex is 10 m; camera A's depth map agrees,
    # camera B's reports 5 m (something occluding it) -> B must be rejected
    ws, cams_csv = _write_synthetic_camera_scene(tmp_path, depth_a=10.0, depth_b=5.0)

    V = np.array([[0.0, 0.0, 0.0]])
    N = np.array([[0.0, 0.0, 1.0]])       # normal points straight up, toward both cameras
    colors, covered = view_based_vertex_colors(V, N, ws, cams_csv, scale=1.0)

    assert covered[0]
    assert colors[0, 0] > 150 and colors[0, 2] < 100          # camera A's red, not B's blue


def test_view_based_colors_covered_false_without_workspace(tmp_path):
    V = np.array([[0.0, 0.0, 0.0]])
    N = np.array([[0.0, 0.0, 1.0]])
    colors, covered = view_based_vertex_colors(
        V, N, tmp_path / "no_ws", tmp_path / "no_cams.csv", scale=1.0)
    assert not covered[0]
    assert tuple(colors[0]) == (0, 0, 0)
