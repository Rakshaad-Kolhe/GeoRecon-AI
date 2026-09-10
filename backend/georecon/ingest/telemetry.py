"""Parse DJI ``.SRT`` / CSV flight telemetry into a clean, time-sorted table.

All parsers return ``DataFrame[t, lat, lon, alt(, rel_alt)]`` where ``t`` is
seconds from the start of the video.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_LAT_RANGE = (-90.0, 90.0)
_LON_RANGE = (-180.0, 180.0)

_TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->")
_TAG_RE = re.compile(r"<[^>]+>")
_FLOAT = r"(-?\d+(?:\.\d+)?)"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_telemetry(path) -> pd.DataFrame:
    """Load telemetry from a ``.srt`` or ``.csv`` file, dispatching on extension."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".srt":
        df = _parse_srt(p)
    elif ext == ".csv":
        df = _parse_csv(p)
    else:
        raise ValueError(
            f"unsupported telemetry extension {ext!r} for {p.name} (expected .srt or .csv)"
        )
    t0_rule = df.attrs.get("t0_rule", "first_sample")
    df = _clean(df)
    if df.empty:
        raise ValueError(f"no usable telemetry rows parsed from {p.name}")
    df = df.sort_values("t").reset_index(drop=True)
    df.attrs["t0_rule"] = t0_rule
    return df


def sample_at_video_time(df: pd.DataFrame, video_t, offset_s: float = 0.0):
    """Interpolate telemetry at video timestamps ``video_t``.

    The only place the telemetry offset is applied: ``telemetry_t = video_t + offset_s``.
    Returns ``(lat, lon, alt, out_of_range_mask)`` like :func:`interpolate`.
    """
    tel_t = np.asarray(video_t, dtype=float) + float(offset_s)
    return interpolate(df, tel_t)


def interpolate(df: pd.DataFrame, t: np.ndarray):
    """Linearly interpolate (lat, lon, alt) at times ``t``, clamping at the edges.

    Returns ``(lat, lon, alt, out_of_range_mask)``.
    """
    t = np.asarray(t, dtype=float)
    src_t = df["t"].to_numpy(dtype=float)
    lo, hi = src_t[0], src_t[-1]
    oob = (t < lo) | (t > hi)

    lat = np.interp(t, src_t, df["lat"].to_numpy(dtype=float))
    lon = np.interp(t, src_t, df["lon"].to_numpy(dtype=float))

    alt_src = df["alt"].to_numpy(dtype=float) if "alt" in df else np.full(len(src_t), np.nan)
    good = ~np.isnan(alt_src)
    if good.any():
        alt = np.interp(t, src_t[good], alt_src[good])
    else:
        alt = np.full(t.shape, np.nan)
    return lat, lon, alt, oob


def telemetry_stats(df: pd.DataFrame) -> dict:
    t = df["t"].to_numpy(dtype=float)
    span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    hz = float((len(t) - 1) / span) if span > 0 else 0.0
    return {
        "rows": int(len(df)),
        "hz": round(hz, 4),
        "t_start": float(t[0]) if len(t) else 0.0,
        "t_end": float(t[-1]) if len(t) else 0.0,
    }


# --------------------------------------------------------------------------- #
# SRT
# --------------------------------------------------------------------------- #
def _search(pattern: str, text: str):
    m = re.search(pattern, text, re.I)
    return float(m.group(1)) if m else None


def _parse_srt(p: Path) -> pd.DataFrame:
    raw = p.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\r?\n\r?\n+", raw.strip())
    rows: list[dict] = []
    for blk in blocks:
        mt = _TS_RE.search(blk)
        if not mt:
            continue
        h, m, s, ms = map(int, mt.groups())
        t = h * 3600 + m * 60 + s + ms / 1000.0

        text = _TAG_RE.sub(" ", blk)
        lat = _search(rf"latitude\s*:\s*{_FLOAT}", text)
        lon = _search(rf"longitude\s*:\s*{_FLOAT}", text)
        rel_alt = None
        alt = None

        if lat is not None and lon is not None:
            # Format (a): bracketed key/value pairs.
            abs_alt = _search(rf"abs_alt\s*:\s*{_FLOAT}", text)
            rel_alt = _search(rf"rel_alt\s*:\s*{_FLOAT}", text)
            plain_alt = _search(rf"\baltitude\s*:\s*{_FLOAT}", text)
            if abs_alt is not None:
                alt = abs_alt
            elif rel_alt is not None:
                alt = rel_alt
            elif plain_alt is not None:
                alt = plain_alt
        else:
            # Format (b): legacy "GPS(lon,lat,alt)" plus optional "BAROMETER:x".
            mg = re.search(
                rf"GPS\s*\(\s*{_FLOAT}\s*,\s*{_FLOAT}\s*,\s*{_FLOAT}\s*\)", text, re.I
            )
            if mg:
                lon = float(mg.group(1))
                lat = float(mg.group(2))
                alt = float(mg.group(3))
                rel_alt = _search(rf"BAROMETER\s*:\s*{_FLOAT}", text)

        if lat is None or lon is None:
            continue
        rows.append({
            "t": t, "lat": lat, "lon": lon,
            "alt": np.nan if alt is None else alt,
            "rel_alt": np.nan if rel_alt is None else rel_alt,
        })
    df = pd.DataFrame(rows, columns=["t", "lat", "lon", "alt", "rel_alt"])
    df.attrs["t0_rule"] = "srt_timestamp"
    return df


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
_TIME_KEYS = ["t", "time", "time_s", "seconds", "elapsed", "timestamp", "time(millisecond)"]
_LAT_KEYS = ["lat", "latitude"]
_LON_KEYS = ["lon", "lng", "long", "longitude"]
_ALT_KEYS = ["alt", "altitude", "abs_alt", "altitude_above_sealevel(feet)"]
_VIDEO_KEYS = {"isvideo", "is_video", "recording"}


def _truthy(v) -> bool:
    if v is None or (isinstance(v, float) and v != v):
        return False
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "t"}
    try:
        return float(v) != 0.0
    except (TypeError, ValueError):
        return bool(v)


def _find_col(lower_map: dict, keys: list[str]):
    for k in keys:
        if k in lower_map:
            return lower_map[k]
    return None


def _csv_time(s: pd.Series) -> pd.Series:
    name = str(s.name).lower()
    if "millisecond" in name:
        return pd.to_numeric(s, errors="coerce") / 1000.0
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        return num
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return (dt - dt.iloc[0]).dt.total_seconds()


def _parse_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"empty CSV: {p.name}")

    lower = {str(c).strip().lower(): c for c in df.columns}
    tcol = _find_col(lower, _TIME_KEYS)
    latcol = _find_col(lower, _LAT_KEYS)
    loncol = _find_col(lower, _LON_KEYS)
    altcol = _find_col(lower, _ALT_KEYS)
    if latcol is None or loncol is None:
        raise ValueError(
            f"CSV {p.name} missing latitude/longitude columns; got {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["t"] = _csv_time(df[tcol]) if tcol is not None else np.arange(len(df), dtype=float)
    out["lat"] = pd.to_numeric(df[latcol], errors="coerce")
    out["lon"] = pd.to_numeric(df[loncol], errors="coerce")
    if altcol is not None:
        alt = pd.to_numeric(df[altcol], errors="coerce")
        if "(feet)" in str(altcol).lower():
            alt = alt * 0.3048
        out["alt"] = alt
    else:
        out["alt"] = np.nan

    # If a recording flag is present, put t=0 at the first "recording on" row
    # and drop everything before it.
    video_col = next((orig for low, orig in lower.items() if low in _VIDEO_KEYS), None)
    t0_rule = "first_sample"
    if video_col is not None:
        rec = df[video_col].map(_truthy).to_numpy()
        if rec.any():
            first = int(rec.argmax())
            out = out.iloc[first:].reset_index(drop=True)
            out["t"] = out["t"] - out["t"].iloc[0]
            t0_rule = "isvideo_column"
    out.attrs["t0_rule"] = t0_rule
    return out


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["t", "lat", "lon"])
    df = df[(df["lat"] != 0) & (df["lon"] != 0)]
    df = df[df["lat"].between(*_LAT_RANGE) & df["lon"].between(*_LON_RANGE)]
    df = df.drop_duplicates(subset=["t"], keep="first")
    if "rel_alt" in df.columns and df["rel_alt"].isna().all():
        df = df.drop(columns=["rel_alt"])
    return df.reset_index(drop=True)
