"""Parse a COLMAP dense ``fused.ply.vis`` sidecar.

Layout (little-endian): uint64 num_points, then per point uint32 k followed by
k * uint32 image ids. A clean parse must consume the file exactly.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def read_vis(path) -> tuple[np.ndarray | None, str]:
    """Return ``(per_point_view_count (N,), status)``. On any inconsistency
    (short read, trailing bytes) return ``(None, reason)`` so the caller can
    warn and skip the ``views`` column."""
    data = Path(path).read_bytes()
    if len(data) < 8:
        return None, f"file too small ({len(data)} bytes)"
    try:
        (n,) = struct.unpack_from("<Q", data, 0)
        off = 8
        counts = np.empty(int(n), dtype=np.int64)
        end = len(data)
        for i in range(int(n)):
            if off + 4 > end:
                return None, f"truncated at point {i}/{n}"
            (k,) = struct.unpack_from("<I", data, off)
            off += 4 + 4 * k
            if off > end:
                return None, f"truncated in ids of point {i}/{n}"
            counts[i] = k
        if off != end:
            return None, f"trailing bytes (consumed {off}, file {end})"
        return counts, "ok"
    except struct.error as exc:
        return None, f"struct error: {exc}"


def write_vis(path, view_id_lists) -> None:
    """Inverse of read_vis — used by tests."""
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(view_id_lists)))
        for ids in view_id_lists:
            fh.write(struct.pack("<I", len(ids)))
            for v in ids:
                fh.write(struct.pack("<I", int(v)))
