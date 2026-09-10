"""Build a small real-world fixture from OpenDroneMap's `odm_data_aukerman`.

Downloads the 77 source JPEGs, reads EXIF GPS + capture time, keeps the longest
straight flight line (heading change > 45 deg splits lines), and writes a
2 fps clip + telemetry CSV to data/samples/real/.

    python scripts/make_aukerman_fixture.py [--all] [--fps 2] [--long-side 1920]

Network is required on first run; downloaded JPEGs are cached under data/raw/
(gitignored) and re-used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import GPSTAGS

REPO = "OpenDroneMap/odm_data_aukerman"
API = f"https://api.github.com/repos/{REPO}/contents/images"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "aukerman"
OUT = ROOT / "data" / "samples" / "real"
_EARTH_R = 6_371_000.0


def _download_images() -> list[Path]:
    RAW.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(API, headers={"User-Agent": "georecon-fixture"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        entries = json.load(resp)
    jpgs = [e for e in entries if e["name"].lower().endswith((".jpg", ".jpeg"))]
    paths = []
    for i, e in enumerate(sorted(jpgs, key=lambda e: e["name"])):
        dst = RAW / e["name"]
        if not dst.exists() or dst.stat().st_size != e.get("size", -1):
            print(f"  [{i + 1}/{len(jpgs)}] {e['name']}")
            with urllib.request.urlopen(
                urllib.request.Request(e["download_url"],
                                       headers={"User-Agent": "georecon-fixture"}),
                timeout=120,
            ) as r:
                dst.write_bytes(r.read())
        paths.append(dst)
    return paths


def _ratio(v) -> float:
    try:
        return float(v)
    except TypeError:
        return v[0] / v[1]


def _dms(vals) -> float:
    d, m, s = (_ratio(x) for x in vals)
    return d + m / 60.0 + s / 3600.0


def _ref_is_negative(ref, neg_letter: str) -> bool:
    """True when a GPS*Ref value denotes the negative hemisphere.

    Pillow returns these as 'S'/'W', b'W', 'W\\x00', ' w ', etc. — normalise.
    """
    if ref is None:
        return False
    if isinstance(ref, (bytes, bytearray)):
        ref = ref.decode("ascii", "ignore")
    return str(ref).strip().strip("\x00").upper()[:1] == neg_letter


def _parse_gps(g: dict):
    """(lat, lon, alt) from a {GPSTag: value} dict, applying hemisphere refs."""
    if "GPSLatitude" not in g or "GPSLongitude" not in g:
        return None
    lat = _dms(g["GPSLatitude"])
    lon = _dms(g["GPSLongitude"])
    if _ref_is_negative(g.get("GPSLatitudeRef"), "S"):
        lat = -lat
    if _ref_is_negative(g.get("GPSLongitudeRef"), "W"):
        lon = -lon
    alt = _ratio(g.get("GPSAltitude", 0) or 0)
    alt_ref = g.get("GPSAltitudeRef")
    if alt_ref in (1, b"\x01") or (isinstance(alt_ref, (bytes, bytearray))
                                   and alt_ref[:1] == b"\x01"):
        alt = -alt
    return lat, lon, alt


def _read_meta(path: Path):
    ex = Image.open(path).getexif()
    dto = ex.get_ifd(0x8769).get(0x9003) or ex.get(0x0132)
    gps_raw = ex.get_ifd(0x8825)
    if not dto or not gps_raw:
        return None
    gps = _parse_gps({GPSTAGS.get(k, k): v for k, v in gps_raw.items()})
    if gps is None:
        return None
    lat, lon, alt = gps
    t = datetime.strptime(str(dto).strip(), "%Y:%m:%d %H:%M:%S")
    return {"name": path.name, "path": path, "t": t, "lat": lat, "lon": lon, "alt": alt}


def _bearing(a, b) -> float:
    lat1, lat2 = math.radians(a["lat"]), math.radians(b["lat"])
    dlon = math.radians(b["lon"] - a["lon"])
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360.0


def _ang_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _dist_m(a, b) -> float:
    lat0 = math.radians((a["lat"] + b["lat"]) / 2)
    dx = math.radians(b["lon"] - a["lon"]) * math.cos(lat0) * _EARTH_R
    dy = math.radians(b["lat"] - a["lat"]) * _EARTH_R
    return math.hypot(dx, dy)


def _split_lines(metas: list[dict]) -> list[list[dict]]:
    if len(metas) < 3:
        return [metas]
    lines = [[metas[0], metas[1]]]
    prev_b = _bearing(metas[0], metas[1])
    for a, b in zip(metas[1:], metas[2:]):
        cur_b = _bearing(a, b)
        if _ang_diff(cur_b, prev_b) > 45.0:
            lines.append([b])
        else:
            lines[-1].append(b)
        prev_b = cur_b
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="use every image, skip line split")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--long-side", type=int, default=1920)
    args = ap.parse_args()

    if (OUT / "video.mp4").exists() and (OUT / "telemetry.csv").exists():
        print(f"fixture already present at {OUT} — nothing to do")
        return 0

    print(f"downloading {REPO}/images -> {RAW}")
    files = _download_images()
    metas = [m for m in (_read_meta(p) for p in files) if m]
    metas.sort(key=lambda m: m["t"])
    print(f"{len(metas)}/{len(files)} images with usable EXIF GPS + time")
    if len(metas) < 3:
        print("not enough geotagged images", file=sys.stderr)
        return 1

    if args.all:
        line = metas
        heading = float(np.mean([_bearing(a, b) for a, b in zip(line, line[1:])]))
    else:
        lines = _split_lines(metas)
        line = max(lines, key=len)
        heading = float(np.mean([_bearing(a, b) for a, b in zip(line, line[1:])]))
        print(f"{len(lines)} flight lines; longest has {len(line)} images")
        if len(line) < 15:
            print("longest line < 15 images — using all images instead")
            line = metas
            heading = float(np.mean([_bearing(a, b) for a, b in zip(line, line[1:])]))

    length_m = sum(_dist_m(a, b) for a, b in zip(line, line[1:]))
    print(f"selected {len(line)} images | line length {length_m:.1f} m | heading {heading:.1f} deg")

    OUT.mkdir(parents=True, exist_ok=True)
    h0, w0 = cv2.imread(str(line[0]["path"])).shape[:2]
    scale = min(1.0, args.long_side / max(h0, w0))
    w, h = int(round(w0 * scale)), int(round(h0 * scale))
    vw = cv2.VideoWriter(str(OUT / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (w, h))
    rows = ["t,lat,lon,alt"]
    for i, m in enumerate(line):
        img = cv2.imread(str(m["path"]))
        if scale < 1.0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        vw.write(img)
        rows.append(f"{i / args.fps:.3f},{m['lat']:.8f},{m['lon']:.8f},{m['alt']:.3f}")
    vw.release()
    (OUT / "telemetry.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'video.mp4'} ({w}x{h} @ {args.fps} fps, {len(line)} frames) "
          f"and telemetry.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
