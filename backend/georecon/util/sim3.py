"""Pure-numpy similarity transforms, RANSAC and plane fitting.

sim(3): ``X' = s R X + t`` with ``det(R) = +1``. Used to align SfM camera
centres to a GPS-derived ENU track.
"""

from __future__ import annotations

import math

import numpy as np


def apply(s, R, t, X):
    """s R X + t for X of shape (N, dim) or (dim,)."""
    return float(s) * (np.asarray(X, float) @ np.asarray(R, float).T) + np.asarray(t, float)


def umeyama(src, dst, w=None, dim=3):
    """Least-squares similarity (Umeyama 1991). Returns (s, R, t), det(R) = +1."""
    src = np.asarray(src, float)[:, :dim]
    dst = np.asarray(dst, float)[:, :dim]
    n = src.shape[0]
    w = np.ones(n) if w is None else np.asarray(w, float)
    wsum = w.sum()
    mu_s = (w[:, None] * src).sum(0) / wsum
    mu_d = (w[:, None] * dst).sum(0) / wsum
    sc, dc = src - mu_s, dst - mu_d
    cov = (w[:, None] * dc).T @ sc / wsum
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    var_s = (w * (sc ** 2).sum(1)).sum() / wsum
    s = float((D * np.diag(S)).sum() / var_s) if var_s > 0 else 1.0
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def _triangle_area(p):
    return 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))


def ransac_sim3(src, dst, thr_m, iters=1000, seed=0):
    """Robust sim(3). Returns (s, R, t, inlier_mask), refined on the inliers."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    if n < 3:
        s, R, t = umeyama(src, dst)
        res = np.linalg.norm(apply(s, R, t, src) - dst, axis=1)
        return s, R, t, res <= thr_m

    rng = np.random.default_rng(seed)
    best_mask, best_cnt = None, -1
    for _ in range(int(iters)):
        idx = rng.choice(n, 3, replace=False)
        if _triangle_area(src[idx]) < 1e-9 or _triangle_area(dst[idx]) < 1e-9:
            continue
        try:
            s, R, t = umeyama(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        res = np.linalg.norm(apply(s, R, t, src) - dst, axis=1)
        mask = res <= thr_m
        if int(mask.sum()) > best_cnt:
            best_mask, best_cnt = mask, int(mask.sum())

    if best_mask is None or best_mask.sum() < 3:
        best_mask = np.ones(n, bool)
    s, R, t = umeyama(src[best_mask], dst[best_mask])
    res = np.linalg.norm(apply(s, R, t, src) - dst, axis=1)
    mask = res <= thr_m
    if mask.sum() >= 3:
        s, R, t = umeyama(src[mask], dst[mask])
        res = np.linalg.norm(apply(s, R, t, src) - dst, axis=1)
        mask = res <= thr_m
    return s, R, t, mask


def fit_plane_ransac(pts, thr, iters=1000, seed=0):
    """Returns (unit_normal, d, inlier_mask) for the plane ``n·x + d = 0``."""
    pts = np.asarray(pts, float)
    n = len(pts)
    rng = np.random.default_rng(seed)
    best_mask, best_cnt = None, -1
    for _ in range(int(iters)):
        p = pts[rng.choice(n, 3, replace=False)]
        nrm = np.cross(p[1] - p[0], p[2] - p[0])
        ln = np.linalg.norm(nrm)
        if ln < 1e-12:
            continue
        nrm = nrm / ln
        d = -nrm @ p[0]
        mask = np.abs(pts @ nrm + d) <= thr
        if int(mask.sum()) > best_cnt:
            best_mask, best_cnt = mask, int(mask.sum())

    q = pts[best_mask] if best_mask is not None and best_mask.sum() >= 3 else pts
    c = q.mean(0)
    *_, Vt = np.linalg.svd(q - c)
    nrm = Vt[-1] / np.linalg.norm(Vt[-1])
    d = float(-nrm @ c)
    return nrm, d, np.abs(pts @ nrm + d) <= thr


def align_vector_to_z(nrm) -> np.ndarray:
    """Rotation R with R @ nrm ≈ +Z."""
    nrm = np.asarray(nrm, float)
    nrm = nrm / np.linalg.norm(nrm)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(nrm, z)
    c = float(nrm @ z)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def quat_wxyz_to_R(qw, qx, qy, qz) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def R_to_quat_wxyz(R) -> np.ndarray:
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)
