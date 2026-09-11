"""Unit tests for Phase-A colour correctness additions.

Tests:
 1. BGR -> RGB channel ordering in view_based_vertex_colors
 2. Weighted-median blending picks the best-angle colour among candidates
 3. Per-image gain compensation clamps outlier exposures
 4. compute_colour_error returns a finite, positive mean-abs error
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from georecon.stages.mesh import (
    compute_colour_error,
    view_based_vertex_colors,
)
from georecon.util.depth_map import write_array
from georecon.util.sim3 import R_to_quat_wxyz


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _build_camera_scene(
    tmp_path,
    *,
    cam_configs: list[dict],
):
    """Minimal pycolmap reconstruction + cameras_enu CSV.

    Each entry in ``cam_configs`` is a dict with:
      color_bgr  – (B,G,R) fill colour for the synthetic 20x20 image
      depth      – float, depth-map value (SfM units, scale=1 so = ENU metres)
      cam_U      – camera height above origin in ENU metres
      name       – image filename (e.g. "cam0.jpg")
    """
    cv2 = pytest.importorskip("cv2")
    import pycolmap

    ws = tmp_path / "ws"
    images_dir = ws / "images"
    depth_dir = ws / "stereo" / "depth_maps"
    sparse_dir = ws / "sparse"
    for d in (images_dir, depth_dir, sparse_dir):
        d.mkdir(parents=True)

    recon = pycolmap.Reconstruction()
    cam_rows = []
    M = np.diag([1.0, -1.0, -1.0])   # camera looking straight down
    q = R_to_quat_wxyz(M)

    for i, cfg in enumerate(cam_configs):
        cam = pycolmap.Camera.create_from_model_name(i + 1, "PINHOLE", 20.0, 20, 20)
        recon.add_camera_with_trivial_rig(cam)
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.zeros(3))
        img_obj = pycolmap.Image(
            name=cfg["name"], camera_id=i + 1, image_id=i + 1)
        recon.add_image_with_trivial_frame(img_obj, pose)
        cv2.imwrite(
            str(images_dir / cfg["name"]),
            np.full((20, 20, 3), cfg["color_bgr"], np.uint8))
        write_array(
            depth_dir / f"{cfg['name']}.geometric.bin",
            np.full((20, 20), cfg["depth"], np.float32))
        cam_rows.append({
            "name": cfg["name"], "registered": 1,
            "E": 0.0, "N": 0.0, "U": cfg["cam_U"],
            "qw": q[0], "qx": q[1], "qy": q[2], "qz": q[3],
        })

    recon.write(str(sparse_dir))
    cams_csv = tmp_path / "cameras_enu.csv"
    pd.DataFrame(cam_rows).to_csv(cams_csv, index=False)
    return ws, cams_csv


# --------------------------------------------------------------------------- #
# 1. BGR -> RGB ordering
# --------------------------------------------------------------------------- #

def test_bgr_to_rgb_channel_ordering(tmp_path):
    """A blue OpenCV image (B=200,G=50,R=50) must produce a blue-channel-high
    vertex colour in the RGB output, not a red-channel-high one."""
    pytest.importorskip("pycolmap")

    ws, cams_csv = _build_camera_scene(tmp_path, cam_configs=[{
        "name": "blue.jpg",
        "color_bgr": (200, 50, 50),   # OpenCV stores BGR → blue channel
        "depth": 10.0,
        "cam_U": 10.0,
    }])
    V = np.array([[0.0, 0.0, 0.0]])
    N = np.array([[0.0, 0.0, 1.0]])
    colors, covered = view_based_vertex_colors(V, N, ws, cams_csv, scale=1.0)

    assert covered[0], "vertex should be covered"
    # After BGR -> RGB: R=50, G=50, B=200
    assert colors[0, 2] > 150, f"B channel should be high, got {colors[0]}"
    assert colors[0, 0] < 100, f"R channel should be low, got {colors[0]}"


# --------------------------------------------------------------------------- #
# 2. Weighted-median prefers best-angle candidate
# --------------------------------------------------------------------------- #

def test_weighted_median_picks_best_angle(tmp_path):
    """Two cameras at the same height; one at a more frontal angle (U=10),
    one oblique (U=100, but vertex at origin, so it won't be visible at a
    straight-down view — we'll use same position but different colours so we
    can verify blending direction).

    Here we test the simpler case: two equally-positioned cameras, one red
    (frontal = highest score because closer) and one blue (farther).  The
    blended result should be dominated by the closer camera's colour.
    """
    pytest.importorskip("pycolmap")

    # cam0: closer (U=5), red in RGB => BGR=(50,50,200)
    # cam1: farther (U=15), blue in RGB => BGR=(200,50,50)
    ws, cams_csv = _build_camera_scene(tmp_path, cam_configs=[
        {"name": "red_close.jpg", "color_bgr": (50, 50, 200), "depth": 5.0, "cam_U": 5.0},
        {"name": "blue_far.jpg",  "color_bgr": (200, 50, 50), "depth": 15.0, "cam_U": 15.0},
    ])
    V = np.array([[0.0, 0.0, 0.0]])
    N = np.array([[0.0, 0.0, 1.0]])
    colors, covered = view_based_vertex_colors(V, N, ws, cams_csv, scale=1.0)

    assert covered[0], "vertex should be covered by both cameras"
    # The closer camera has a higher score (cos(theta)/dist = 1/5 > 1/15).
    # After gain compensation and weighted median the red channel should dominate.
    assert colors[0, 0] > colors[0, 2], (
        f"Red channel should dominate over blue; got {colors[0]}")


# --------------------------------------------------------------------------- #
# 3. Gain compensation clamps extreme gains
# --------------------------------------------------------------------------- #

def test_gain_compensation_clamped_to_range(tmp_path):
    """A 5× overexposed image (all 255) and a correctly exposed 128-grey image
    at the same position: after gain compensation the blended vertex colour
    should not differ wildly from 128 grey (i.e. the gain is clamped at 1.4).
    """
    pytest.importorskip("pycolmap")

    # cam0: normal exposure (128,128,128 in RGB) => BGR=(128,128,128)
    # cam1: overexposed (255,255,255) — raw ratio would be ~2×, clamped to 1.4
    ws, cams_csv = _build_camera_scene(tmp_path, cam_configs=[
        {"name": "normal.jpg",     "color_bgr": (128, 128, 128), "depth": 10.0, "cam_U": 10.0},
        {"name": "overexposed.jpg","color_bgr": (255, 255, 255), "depth": 10.0, "cam_U": 10.0},
    ])
    V = np.array([[0.0, 0.0, 0.0]])
    N = np.array([[0.0, 0.0, 1.0]])
    colors, covered = view_based_vertex_colors(V, N, ws, cams_csv, scale=1.0)

    assert covered[0]
    # If gain were unclamped the merged result could be >> 200.
    # With clamp [0.7, 1.4] the corrected overexposed value is ≤ 128 × 1.4 ≈ 179.
    assert all(c <= 200 for c in colors[0]), (
        f"Gain compensation should prevent run-away brightness; got {colors[0]}")


# --------------------------------------------------------------------------- #
# 4. compute_colour_error returns finite positive value
# --------------------------------------------------------------------------- #

def test_compute_colour_error_returns_finite_positive(tmp_path):
    """With a grey camera image and a grey mesh the colour_error should be ~0;
    with mismatched colours it should be > 0 and finite."""
    pytest.importorskip("pycolmap")
    import trimesh

    # Camera colour: red (BGR=50,50,200 → RGB=200,50,50)
    ws, cams_csv = _build_camera_scene(tmp_path, cam_configs=[
        {"name": "red.jpg", "color_bgr": (50, 50, 200), "depth": 10.0, "cam_U": 10.0},
    ])

    # Mesh: a flat quad at z=0, coloured grey (128,128,128)
    verts = np.array([
        [-0.1, -0.1, 0.0], [0.1, -0.1, 0.0],
        [0.1,  0.1, 0.0], [-0.1,  0.1, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    vcol = np.full((4, 4), 128, np.uint8)
    vcol[:, 3] = 255  # alpha
    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                           vertex_colors=vcol, process=False)

    err = compute_colour_error(mesh, ws, cams_csv, scale=1.0,
                               keyframes=1, n_samples=4, seed=0)
    # The mesh is grey and the camera is red → error should be finite and positive
    assert err is not None, "Should return a float, not None"
    assert np.isfinite(err), f"colour_error should be finite, got {err}"
    assert err > 0, "Grey mesh vs red camera should have non-zero error"


def test_compute_colour_error_returns_none_without_workspace(tmp_path):
    """If the workspace does not exist, compute_colour_error must return None
    gracefully (no exception)."""
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=1)
    err = compute_colour_error(
        mesh, tmp_path / "no_ws", tmp_path / "no_cams.csv", scale=1.0)
    assert err is None
