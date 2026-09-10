import numpy as np
import pytest

from georecon.util.sim3 import (
    R_to_quat_wxyz,
    apply,
    fit_plane_ransac,
    quat_wxyz_to_R,
    ransac_sim3,
    rmse,
    umeyama,
)


def test_rmse_known_vector():
    assert rmse([3.0, 4.0]) == pytest.approx(12.5 ** 0.5)
    assert rmse(np.array([[1.0, -1.0], [2.0, -2.0]])) == pytest.approx(2.5 ** 0.5)
    assert rmse([2.0, 2.0, 2.0]) == pytest.approx(2.0)
    assert rmse([]) == 0.0


def _rand_rot(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_umeyama_recovers_sim3_3d():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(50, 3)) * 10.0
    s, R, t = 2.3, _rand_rot(rng), np.array([5.0, -3.0, 7.0])
    dst = apply(s, R, t, src)
    s2, R2, t2 = umeyama(src, dst)
    assert abs(s2 - s) / s < 1e-8
    assert np.allclose(R2, R, atol=1e-7)
    assert np.allclose(t2, t, atol=1e-4)
    assert np.isclose(np.linalg.det(R2), 1.0)


def test_umeyama_2d():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(40, 2)) * 5.0
    ang, s = 0.7, 1.7
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = s * (src @ R.T) + np.array([2.0, -1.0])
    s2, R2, t2 = umeyama(src, dst, dim=2)
    assert abs(s2 - s) / s < 1e-8
    assert np.allclose(R2, R, atol=1e-7)


def test_ransac_sim3_rejects_20pct_outliers():
    rng = np.random.default_rng(3)
    n = 120
    src = rng.normal(size=(n, 3)) * 10.0
    s, R, t = 2.0, _rand_rot(rng), np.array([1.0, 2.0, 3.0])
    dst = apply(s, R, t, src)
    bad = rng.choice(n, n // 5, replace=False)
    dst[bad] += rng.normal(size=(n // 5, 3)) * 60.0

    s2, R2, t2, inl = ransac_sim3(src, dst, thr_m=3.0, iters=600, seed=0)
    assert abs(s2 - s) / s < 0.02
    assert np.allclose(R2, R, atol=1e-2)
    assert inl.sum() >= int(0.75 * n)
    assert not inl[bad].any()


def test_fit_plane_ransac_tilted():
    rng = np.random.default_rng(4)
    nrm_true = np.array([0.1, -0.2, 1.0])
    nrm_true /= np.linalg.norm(nrm_true)
    xy = rng.uniform(-20, 20, size=(200, 2))
    z = -(nrm_true[0] * xy[:, 0] + nrm_true[1] * xy[:, 1]) / nrm_true[2]
    pts = np.column_stack([xy, z]) + rng.normal(scale=0.05, size=(200, 3))
    nrm, d, inl = fit_plane_ransac(pts, thr=0.3, iters=300, seed=0)
    assert inl.mean() > 0.9
    cos = abs(float(nrm @ nrm_true))
    assert cos > np.cos(np.radians(2))


def test_quat_matrix_roundtrip():
    rng = np.random.default_rng(5)
    R = _rand_rot(rng)
    q = R_to_quat_wxyz(R)
    assert np.allclose(quat_wxyz_to_R(*q), R, atol=1e-8)
