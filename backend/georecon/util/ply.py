"""Minimal binary little-endian PLY reader/writer.

Supports a single ``vertex`` element with arbitrary scalar properties whose
dtype is either float (``float`` = f4, ``double`` = f8) or ``uchar`` (u1).
Reused across stages (SfM sparse cloud, dense, exports).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_NP_TO_PLY = {"f4": "float", "f8": "double", "u1": "uchar", "i4": "int", "u4": "uint"}
_PLY_TO_NP = {v: k for k, v in _NP_TO_PLY.items()}


def _ply_type(arr: np.ndarray) -> str:
    kind = arr.dtype.str[1:]  # e.g. 'f4'
    if kind in _NP_TO_PLY:
        return _NP_TO_PLY[kind]
    if arr.dtype.kind == "f":
        return "float"
    if arr.dtype.kind in "bu" and arr.dtype.itemsize == 1:
        return "uchar"
    return "float"


def write_ply(path, xyz, props: dict[str, np.ndarray] | None = None) -> None:
    """Write ``xyz`` (N,3) plus optional per-vertex ``props`` (name -> (N,) array)."""
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be (N, 3)")
    n = xyz.shape[0]
    props = props or {}

    cols: list[tuple[str, np.ndarray, str]] = [
        ("x", xyz[:, 0].astype(np.float32), "float"),
        ("y", xyz[:, 1].astype(np.float32), "float"),
        ("z", xyz[:, 2].astype(np.float32), "float"),
    ]
    for name, arr in props.items():
        arr = np.asarray(arr)
        if arr.shape[0] != n:
            raise ValueError(f"property {name!r} has length {arr.shape[0]}, expected {n}")
        pt = _ply_type(arr)
        cols.append((name, arr.astype(_PLY_TO_NP[pt]), pt))

    dtype = np.dtype([(name, arr.dtype) for name, arr, _ in cols])
    rec = np.empty(n, dtype=dtype)
    for name, arr, _ in cols:
        rec[name] = arr

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property {pt} {name}" for name, _, pt in cols]
    header.append("end_header")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(rec.tobytes())


def read_ply(path) -> dict[str, np.ndarray]:
    """Return ``{property_name: (N,) array}`` (includes ``x``, ``y``, ``z``)."""
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError("not a PLY file")
        fmt = fh.readline().strip()
        if fmt != b"format binary_little_endian 1.0":
            raise ValueError(f"unsupported PLY format: {fmt!r}")
        n = 0
        fields: list[tuple[str, str]] = []
        while True:
            line = fh.readline().strip()
            if line == b"end_header":
                break
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            elif line.startswith(b"property"):
                _, pt, name = line.split()
                fields.append((name.decode(), pt.decode()))
        dtype = np.dtype([(name, _PLY_TO_NP.get(pt, "f4")) for name, pt in fields])
        rec = np.frombuffer(fh.read(n * dtype.itemsize), dtype=dtype, count=n)
    return {name: np.array(rec[name]) for name, _ in fields}
