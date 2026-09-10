import numpy as np
import trimesh

from georecon.config import JobConfig, MeshCfg
from georecon.stages.mesh import (
    _crop_far_vertices,
    _drop_small_components,
    choose_poisson,
    heightfield_mesh,
    poisson_args,
)
from georecon.util.colmap_cli import build_args


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
