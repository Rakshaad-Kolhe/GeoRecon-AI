"""Quick sanity report for a dense (or sparse-fallback) ENU point cloud.

    python scripts/inspect_cloud.py [jobs/real/dense/fused.ply]

numpy only. Prints point count, ENU bounding box, z percentiles, the angle
between a least-squares plane normal and +Z (a flat site should be < ~10 deg),
and height_above_ground = mean registered-camera U - cloud z p50.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_PLY_TO_NP = {"float": "f4", "double": "f8", "uchar": "u1", "uint8": "u1",
              "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4"}


def read_ply_xyz(path: Path) -> np.ndarray:
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError("not a PLY file")
        fmt = fh.readline().strip()
        if fmt != b"format binary_little_endian 1.0":
            raise ValueError(f"unsupported PLY format: {fmt!r}")
        n, fields = 0, []
        while True:
            line = fh.readline().strip()
            if line == b"end_header":
                break
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            elif line.startswith(b"property list"):
                raise ValueError("list properties not supported")
            elif line.startswith(b"property"):
                _, pt, name = line.split()
                fields.append((name.decode(), _PLY_TO_NP.get(pt.decode(), "f4")))
        dtype = np.dtype([(nm, tp) for nm, tp in fields])
        rec = np.frombuffer(fh.read(n * dtype.itemsize), dtype=dtype, count=n)
    return np.stack([rec["x"], rec["y"], rec["z"]], axis=-1).astype(float)


def plane_normal_angle_to_z(xyz: np.ndarray) -> float:
    c = xyz - xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    n = vt[-1]
    cos = abs(float(n @ np.array([0.0, 0.0, 1.0])))
    return float(np.degrees(np.arccos(min(1.0, cos))))


def mean_registered_camera_u(cameras_csv: Path) -> float | None:
    if not cameras_csv.exists():
        return None
    us = []
    with open(cameras_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("registered") not in ("1", "1.0"):
                continue
            try:
                us.append(float(row["U"]))
            except (KeyError, ValueError):
                pass
    return float(np.mean(us)) if us else None


def main() -> None:
    ply = Path(sys.argv[1] if len(sys.argv) > 1 else "jobs/real/dense/fused.ply")
    xyz = read_ply_xyz(ply)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    z1, z50, z99 = (float(v) for v in np.percentile(xyz[:, 2], [1, 50, 99]))
    ang = plane_normal_angle_to_z(xyz)
    cam_u = mean_registered_camera_u(ply.parent.parent / "georef" / "cameras_enu.csv")

    print(f"file                 {ply}")
    print(f"points               {len(xyz):,}")
    print(f"ENU bbox min (m)     E {lo[0]:9.2f}  N {lo[1]:9.2f}  U {lo[2]:9.2f}")
    print(f"ENU bbox max (m)     E {hi[0]:9.2f}  N {hi[1]:9.2f}  U {hi[2]:9.2f}")
    print(f"ENU bbox size (m)    E {hi[0]-lo[0]:9.2f}  N {hi[1]-lo[1]:9.2f}  U {hi[2]-lo[2]:9.2f}")
    print(f"z  p1 / p50 / p99    {z1:.2f} / {z50:.2f} / {z99:.2f}")
    print(f"plane normal vs +Z   {ang:.1f} deg   (flat site -> expect < 10)")
    if cam_u is None:
        print("height_above_ground  n/a (cameras_enu.csv missing)")
    else:
        print(f"mean camera U (m)    {cam_u:.2f}")
        print(f"height_above_ground  {cam_u - z50:.2f} m   (mean cam U - z p50)")


if __name__ == "__main__":
    main()
