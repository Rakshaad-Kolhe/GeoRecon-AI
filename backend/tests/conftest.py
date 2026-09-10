"""Shared test fixtures.

``synth_clip`` builds a small, textured, drone-like clip (a crop window panning
over a big synthetic image at constant speed, with a short hard-blur segment)
plus a matching DJI-style SRT. Reused by the keyframes tests and later PRs.
"""

import math

import cv2
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _reset_colmap_probe():
    """The COLMAP CLI probe is process-cached; keep tests independent."""
    from georecon.util import colmap_cli

    colmap_cli.reset_probe_cache()
    yield
    colmap_cli.reset_probe_cache()


def _ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@pytest.fixture
def synth_clip(tmp_path):
    def _make(name="clip", seconds=6.0, fps=20, w=640, h=360,
              speed_px=5.0, blur_seg=(2.0, 2.5), with_srt=True):
        rng = np.random.default_rng(42)
        n = int(round(seconds * fps))
        big_w = w + int(math.ceil(n * speed_px)) + 40
        big_h = h + 40

        base = rng.integers(0, 256, (big_h, big_w, 3), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (7, 7), 0)
        for _ in range(500):
            x1 = int(rng.integers(0, big_w - 2))
            y1 = int(rng.integers(0, big_h - 2))
            col = tuple(int(c) for c in rng.integers(0, 256, 3))
            if rng.random() < 0.5:
                x2 = min(big_w - 1, x1 + int(rng.integers(6, 70)))
                y2 = min(big_h - 1, y1 + int(rng.integers(6, 70)))
                cv2.rectangle(base, (x1, y1), (x2, y2), col, -1)
            else:
                cv2.circle(base, (x1, y1), int(rng.integers(3, 34)), col, -1)

        video = tmp_path / f"{name}.mp4"
        vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not vw.isOpened():
            pytest.skip("no mp4v encoder in this OpenCV build")
        for i in range(n):
            t = i / fps
            x = 20 + int(round(i * speed_px))
            y = 20 + int(round(4 * math.sin(i * 0.12)))
            crop = base[y:y + h, x:x + w].copy()
            if blur_seg and blur_seg[0] <= t < blur_seg[1]:
                crop = cv2.GaussianBlur(crop, (31, 31), 0)
            vw.write(crop)
        vw.release()
        if not video.exists() or video.stat().st_size == 0:
            pytest.skip("mp4v encoder produced no output")

        srt = None
        if with_srt:
            srt = tmp_path / f"{name}.srt"
            lat0, lon0 = 18.5000, 73.8500
            blocks = []
            for s in range(int(math.ceil(seconds)) + 1):
                lat = lat0 + s * 2.0e-5      # ~2.2 m/s north
                lon = lon0 + s * 1.0e-5
                blocks.append(
                    f"{s + 1}\n{_ts(s)} --> {_ts(s + 1)}\n"
                    f"[latitude: {lat:.6f}] [longitude: {lon:.6f}] "
                    f"[rel_alt: 40.000 abs_alt: 540.000]\n"
                )
            srt.write_text("\n".join(blocks), encoding="utf-8")
        return video, srt

    return _make
