"""Keyframe-selection stage.

Three phases:
  1. analyse  - one sequential decode pass, sub-sampled to ``analysis_fps``;
                per analysed frame record sharpness, step displacement (LK flow)
                and interpolated GPS.  Written to ``report/keyframes_timeline.csv``.
  2. select   - pure function over the phase-1 arrays: blur gate, motion
                accumulation, hover gate, keyframe cap.  No I/O.
  3. extract  - second sequential pass, decode only the selected frames,
                resize to the preset, write ``frames/f_*.jpg`` + ``frames.csv``
                and a contact sheet.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd

from georecon.config import KeyframeCfg
from georecon.ingest.telemetry import sample_at_video_time

if TYPE_CHECKING:
    from georecon.pipeline import StageContext

ANALYSIS_W = 640
_EARTH_R = 6_371_000.0
_GFTT = dict(maxCorners=400, qualityLevel=0.01, minDistance=8)
_MIN_TRACKS = 20


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _find_video(input_dir: Path) -> Path:
    hits = sorted(p for p in input_dir.glob("video.*") if p.suffix.lower() != ".csv")
    if not hits:
        raise ValueError(f"no ingested video found in {input_dir}")
    return hits[0]


def _equirect_xy(lat: np.ndarray, lon: np.ndarray):
    """Local metres via equirectangular projection about the first valid fix."""
    ok = np.isfinite(lat) & np.isfinite(lon)
    x = np.full(lat.shape, np.nan)
    y = np.full(lat.shape, np.nan)
    if not ok.any():
        return x, y
    i0 = int(np.argmax(ok))
    lat0 = math.radians(float(lat[i0]))
    x[ok] = np.radians(lon[ok] - lon[i0]) * math.cos(lat0) * _EARTH_R
    y[ok] = np.radians(lat[ok] - lat[i0]) * _EARTH_R
    return x, y


# --------------------------------------------------------------------------- #
# phase 1
# --------------------------------------------------------------------------- #
def _analyse(video: Path, analysis_fps: int, tel_df, offset_s: float, log):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    k = max(1, round(fps / max(1, analysis_fps)))

    frame_idx = -1
    grabbed = 0
    prev_gray = None
    prev_pts = None
    lost = 0
    rows: list[tuple] = []
    a_h = 0

    t0 = time.time()
    while True:
        if not cap.grab():
            break
        frame_idx += 1
        grabbed += 1
        if frame_idx % k:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break

        h0, w0 = frame.shape[:2]
        a_h = max(1, round(h0 * ANALYSIS_W / w0))
        small = cv2.resize(frame, (ANALYSIS_W, a_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        disp = np.nan
        if prev_gray is not None:
            if prev_pts is not None and len(prev_pts) >= 1:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
                good = (st.ravel() == 1)
                if int(good.sum()) >= _MIN_TRACKS:
                    d = np.linalg.norm(
                        (nxt[good] - prev_pts[good]).reshape(-1, 2), axis=1)
                    disp = float(np.median(d))
                else:
                    lost += 1
            else:
                lost += 1
        # re-detect features on the current frame for the next step
        prev_pts = cv2.goodFeaturesToTrack(gray, **_GFTT)
        prev_gray = gray

        rows.append((frame_idx, frame_idx / fps, sharp, disp))

    cap.release()
    seconds = time.time() - t0

    arr = np.array(rows, dtype=float) if rows else np.zeros((0, 4))
    frame_idx_a = arr[:, 0].astype(int)
    t_a = arr[:, 1]
    sharp_a = arr[:, 2]
    disp_a = arr[:, 3]

    if tel_df is not None and len(t_a):
        lat, lon, alt, oob = sample_at_video_time(tel_df, t_a, offset_s)
        lat = np.where(oob, np.nan, lat)
        lon = np.where(oob, np.nan, lon)
        alt = np.where(oob, np.nan, alt)
    else:
        lat = lon = alt = np.full(len(t_a), np.nan)

    log.info("analysed %d frames (every %d of %.2f fps), %d grabbed, %d lost-tracking",
             len(t_a), k, fps, grabbed, lost)
    return {
        "frame_idx": frame_idx_a, "t": t_a, "sharpness": sharp_a, "disp": disp_a,
        "lat": lat, "lon": lon, "alt": alt,
        "a_h": a_h, "grabbed": grabbed, "lost": lost, "seconds": seconds,
    }


# --------------------------------------------------------------------------- #
# phase 2  (pure)
# --------------------------------------------------------------------------- #
def select(sharpness, disp, lat, lon, cfg: KeyframeCfg, flow_thr_px: float,
           max_keyframes: int):
    """Pick keyframe positions from the analysed arrays. Pure: no I/O.

    Returns ``(positions, reasons, final_thr)`` where ``positions`` indexes the
    input arrays and ``reasons`` counts
    ``rejected_blur / rejected_hover / forced / forced_lost``.
    """
    sharpness = np.asarray(sharpness, dtype=float)
    disp_raw = np.asarray(disp, dtype=float)
    lost_mask = ~np.isfinite(disp_raw)
    n = len(sharpness)
    _zero = {"rejected_blur": 0, "rejected_hover": 0, "forced": 0, "forced_lost": 0}
    if n == 0:
        return [], dict(_zero), flow_thr_px

    med = (pd.Series(sharpness)
           .rolling(cfg.blur_window, center=True, min_periods=1)
           .median().to_numpy())
    sharp_ok = sharpness >= cfg.blur_ratio * med
    x, y = _equirect_xy(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))
    has_xy = np.isfinite(x) & np.isfinite(y)
    step_move = np.full(n, np.nan)
    if n > 1:
        step_move[1:] = np.hypot(np.diff(x), np.diff(y))

    def _eff_disp(i: int, thr: float, reasons: dict) -> float:
        """Per-step displacement, resolving lost-tracking frames via GPS."""
        if not lost_mask[i]:
            return float(disp_raw[i])
        moved = (i > 0 and has_xy[i] and has_xy[i - 1]
                 and step_move[i] >= cfg.min_gps_move_m)
        no_gps = not (i > 0 and has_xy[i] and has_xy[i - 1])
        if moved or no_gps:                        # force a boundary
            reasons["forced_lost"] += 1
            return thr
        return 0.0                                 # lost but GPS says we hovered

    def _run(thr: float):
        idxs: list[int] = []
        reasons = dict(_zero)
        accum = 0.0
        win: list[tuple[int, float]] = []   # (pos, accum_at_pos)
        last: int | None = None
        last_sharp: int | None = None

        for i in range(n):
            eff = _eff_disp(i, thr, reasons) if last is not None else 0.0
            if not sharp_ok[i]:
                reasons["rejected_blur"] += 1
                if last is not None:
                    accum += eff
                continue
            last_sharp = i
            if last is None:                       # first sharp candidate: always in
                idxs.append(i)
                last = i
                accum = 0.0
                win = []
                continue

            accum += eff
            if accum >= 0.7 * thr:
                win.append((i, accum))
            if not (accum >= thr or (accum >= 1.5 * thr and not win)):
                continue

            if win:
                cpos, cacc = max(win, key=lambda p: (sharpness[p[0]], p[1]))
            else:
                cpos, cacc = i, accum

            if has_xy[cpos] and has_xy[last]:
                if math.hypot(x[cpos] - x[last], y[cpos] - y[last]) < cfg.min_gps_move_m:
                    reasons["rejected_hover"] += 1
                    win = []
                    continue

            idxs.append(cpos)
            last = cpos
            accum -= cacc
            win = [(p, a - cacc) for (p, a) in win if p > cpos]

        # make sure the last usable stretch ends on a keyframe
        if last is not None and last_sharp is not None and last_sharp > last:
            if accum >= 0.3 * thr:
                tail = [i for i in range(last + 1, n) if sharp_ok[i]]
                cpos = max(tail, key=lambda p: sharpness[p])
                idxs.append(cpos)
                reasons["forced"] += 1

        return sorted(set(idxs)), reasons

    idxs, reasons = _run(flow_thr_px)
    final_thr = flow_thr_px
    if len(idxs) > max_keyframes:
        lo, hi = flow_thr_px, flow_thr_px
        for _ in range(12):                         # bracket an upper bound
            hi *= 2
            if len(_run(hi)[0]) <= max_keyframes:
                break
        for _ in range(12):                         # binary search
            mid = 0.5 * (lo + hi)
            m_idx, m_rsn = _run(mid)
            if len(m_idx) <= max_keyframes:
                hi, idxs, reasons, final_thr = mid, m_idx, m_rsn, mid
            else:
                lo = mid
    return idxs, reasons, final_thr


# --------------------------------------------------------------------------- #
# phase 3
# --------------------------------------------------------------------------- #
def _extract(video: Path, positions, data, long_side: int, jpeg_q: int,
             frames_dir: Path, report_dir: Path, log):
    fi = data["frame_idx"]
    pos_by_fi = {int(fi[p]): p for p in positions}
    want = set(pos_by_fi)

    cap = cv2.VideoCapture(str(video))
    idx = -1
    grabbed = 0
    rows: list[dict] = []
    t0 = time.time()
    while want:
        if not cap.grab():
            break
        idx += 1
        grabbed += 1
        if idx not in want:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        want.discard(idx)

        h0, w0 = frame.shape[:2]
        s = min(1.0, long_side / max(h0, w0))
        if s < 1.0:
            frame = cv2.resize(frame, (round(w0 * s), round(h0 * s)),
                               interpolation=cv2.INTER_AREA)
        name = f"f_{idx:06d}.jpg"
        cv2.imwrite(str(frames_dir / name), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_q)])
        p = pos_by_fi[idx]
        rows.append({
            "name": name, "frame_idx": idx, "t": round(float(data["t"][p]), 4),
            "lat": data["lat"][p], "lon": data["lon"][p], "alt": data["alt"][p],
            "sharpness": round(float(data["sharpness"][p]), 3),
            "disp_px": data["disp"][p],
        })
    cap.release()

    rows.sort(key=lambda r: r["frame_idx"])
    df = pd.DataFrame(rows, columns=["name", "frame_idx", "t", "lat", "lon", "alt",
                                     "sharpness", "disp_px"])
    df.to_csv(frames_dir / "frames.csv", index=False)
    _contact_sheet(rows, frames_dir, report_dir)
    log.info("extracted %d keyframes (%d grabbed)", len(rows), grabbed)
    return rows, grabbed, time.time() - t0


def _contact_sheet(rows, frames_dir: Path, report_dir: Path):
    thumbs = []
    for r in rows[:12]:
        img = cv2.imread(str(frames_dir / r["name"]))
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        s = 240.0 / max(h0, w0)
        th = cv2.resize(img, (max(1, round(w0 * s)), max(1, round(h0 * s))))
        base = th.shape[0] - 6
        cv2.putText(th, r["name"], (4, base), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
        cv2.putText(th, r["name"], (4, base), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
        thumbs.append(th)
    if not thumbs:
        return
    cols = min(4, len(thumbs))
    rows_n = (len(thumbs) + cols - 1) // cols
    ch = max(t.shape[0] for t in thumbs)
    cw = max(t.shape[1] for t in thumbs)
    sheet = np.zeros((rows_n * ch, cols * cw, 3), np.uint8)
    for i, th in enumerate(thumbs):
        rr, cc = divmod(i, cols)
        sheet[rr * ch:rr * ch + th.shape[0], cc * cw:cc * cw + th.shape[1]] = th
    cv2.imwrite(str(report_dir / "keyframes_contact.jpg"), sheet)


# --------------------------------------------------------------------------- #
# stage entry point
# --------------------------------------------------------------------------- #
def run(ctx: "StageContext") -> dict:
    cfg = ctx.cfg
    kf: KeyframeCfg = cfg.keyframes
    paths = ctx.paths
    video = _find_video(paths.input)

    tel_path = paths.input / "telemetry_norm.csv"
    has_tel = tel_path.exists()
    tel_df = pd.read_csv(tel_path) if has_tel else None

    # ---- phase 1 -------------------------------------------------------------
    data = _analyse(video, kf.analysis_fps, tel_df, cfg.telemetry_offset_s, ctx.log)
    n = len(data["t"])
    if n == 0:
        raise ValueError("keyframes: analysis produced no frames")

    a_diag = math.hypot(ANALYSIS_W, data["a_h"])
    flow_thr_px = kf.flow_frac * a_diag

    pd.DataFrame({
        "frame_idx": data["frame_idx"], "t": data["t"],
        "sharpness": data["sharpness"], "step_disp_px": data["disp"],
        "lat": data["lat"], "lon": data["lon"], "alt": data["alt"],
    }).to_csv(paths.report / "keyframes_timeline.csv", index=False)

    # ---- phase 2 ----------------------------------------------------------
    t_sel = time.time()
    positions, reasons, final_thr = select(
        data["sharpness"], data["disp"], data["lat"], data["lon"],
        kf, flow_thr_px, ctx.preset.max_keyframes,
    )
    select_s = time.time() - t_sel
    ctx.log.info("selected %d keyframes (thr %.1f px, blur-rej %d, hover-rej %d, "
                 "forced %d, forced-lost %d)",
                 len(positions), final_thr, reasons["rejected_blur"],
                 reasons["rejected_hover"], reasons["forced"], reasons["forced_lost"])

    # ---- phase 3 ----------------------------------------------------------
    rows, grab3, extract_s = _extract(
        video, positions, data, ctx.preset.frame_long_side, kf.jpeg_quality,
        paths.frames, paths.report, ctx.log,
    )

    med = (pd.Series(data["sharpness"])
           .rolling(kf.blur_window, center=True, min_periods=1).median().to_numpy())
    candidates = int(np.sum(data["sharpness"] >= kf.blur_ratio * med))
    valid_disp = data["disp"][np.isfinite(data["disp"])]
    est_overlap = (float(np.clip(1.0 - valid_disp / data["a_h"], 0.0, 1.0).mean())
                   if valid_disp.size else 0.0)
    decode_fps = ((data["grabbed"] + grab3) / (data["seconds"] + extract_s)
                  if (data["seconds"] + extract_s) > 0 else 0.0)

    n_missing_gps = sum(1 for r in rows if not np.isfinite(r["lat"]))
    if has_tel and n_missing_gps:
        ctx.warn(f"{n_missing_gps}/{len(rows)} selected keyframes have no GPS fix")
    if len(rows) < 20:
        ctx.warn(f"only {len(rows)} keyframes selected (<20)")
    if est_overlap < 0.6:
        ctx.warn(f"estimated frame overlap {est_overlap:.0%} is low (<60%)")

    return {
        "analysed": n,
        "candidates": candidates,
        "keyframes": len(rows),
        "rejected_blur": reasons["rejected_blur"],
        "rejected_hover": reasons["rejected_hover"],
        "forced": reasons["forced"],
        "forced_lost": reasons["forced_lost"],
        "lost_tracking": data["lost"],
        "flow_thr_px": round(final_thr, 3),
        "est_overlap_mean": round(est_overlap, 4),
        "sharpness_mean": round(float(np.mean(data["sharpness"])), 3),
        "sharpness_min": round(float(np.min(data["sharpness"])), 3),
        "decode_fps": round(float(decode_fps), 2),
        "analyse_s": round(data["seconds"], 3),
        "select_s": round(select_s, 3),
        "extract_s": round(extract_s, 3),
    }
