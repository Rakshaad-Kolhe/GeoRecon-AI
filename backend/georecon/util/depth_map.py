"""Parse a COLMAP dense depth/normal map (``stereo/depth_maps/<img>.geometric.bin``
or ``.photometric.bin``).

Layout: ASCII header ``width&height&channels&`` (three ``&``-terminated
decimal ints) followed by ``width*height*channels`` little-endian float32
values in column-major (Fortran) order over ``(width, height, channels)`` —
COLMAP's own ``read_array`` reference implementation reshapes then
transposes to ``(height, width, channels)``; we do the same and squeeze a
single-channel map down to plain ``(height, width)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_array(path) -> np.ndarray:
    """(height, width) for a single-channel map, else (height, width, channels)."""
    data = Path(path).read_bytes()
    pos = 0
    dims: list[int] = []
    for _ in range(3):
        end = data.index(b"&", pos)
        dims.append(int(data[pos:end]))
        pos = end + 1
    width, height, channels = dims
    n = width * height * channels
    arr = np.frombuffer(data, dtype=np.float32, offset=pos, count=n)
    if arr.size < n:
        raise ValueError(f"depth map truncated: got {arr.size}, expected {n}")
    arr = arr.reshape((width, height, channels), order="F")
    arr = np.transpose(arr, (1, 0, 2))
    return arr[:, :, 0] if channels == 1 else arr


def write_array(path, arr: np.ndarray) -> None:
    """Inverse of ``read_array`` — used by tests. ``arr`` is (H,W) or (H,W,C)."""
    a = np.asarray(arr, np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    h, w, c = a.shape
    header = f"{w}&{h}&{c}&".encode("ascii")
    wh_c = np.transpose(a, (1, 0, 2))                      # (h,w,c) -> (w,h,c)
    flat = np.asfortranarray(wh_c).reshape(-1, order="F")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + flat.astype(np.float32).tobytes())
