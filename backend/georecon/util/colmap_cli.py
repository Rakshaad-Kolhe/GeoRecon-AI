"""Thin wrapper around a native COLMAP CLI (for CUDA feature extraction /
matching and dense MVS, which the pycolmap CPU wheel cannot do).

Set ``GEORECON_COLMAP_BIN`` to a ``colmap`` / ``colmap.bat`` (COLMAP 4.x).

Option names verified live against COLMAP 4.2.0 CUDA (`colmap <tool> -h`,
Commit be5e291) on 2026-09-10. In 4.2 the SIFT GPU toggles moved namespace:
``SiftExtraction.use_gpu`` -> ``FeatureExtraction.use_gpu`` and
``SiftMatching.use_gpu`` -> ``FeatureMatching.use_gpu`` (old names are rejected
with "unrecognised option"). Only these options are used:

  feature_extractor
    --database_path --image_path --image_list_path
    --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL
    --ImageReader.mask_path --ImageReader.camera_params "f,cx,cy,k"
    --SiftExtraction.max_num_features --FeatureExtraction.use_gpu 1
  sequential_matcher
    --database_path --SequentialMatching.overlap
    --SequentialMatching.quadratic_overlap --SequentialMatching.loop_detection 0
    --FeatureMatching.use_gpu 1
  image_undistorter
    --image_path --input_path (sparse/0) --output_path
    --output_type COLMAP --max_image_size
  patch_match_stereo
    --workspace_path --workspace_format COLMAP
    --PatchMatchStereo.geom_consistency 1 --PatchMatchStereo.max_image_size
  stereo_fusion
    --workspace_path --workspace_format COLMAP --input_type geometric --output_path
    --StereoFusion.min_num_pixels --StereoFusion.max_reproj_error
    --StereoFusion.max_depth_error --StereoFusion.max_normal_error

`<bin> -h` header lines used by probe(): "COLMAP <maj.min.patch>" and
"... with CUDA" (absent -> CPU-only build).
"""

from __future__ import annotations

import collections
import re
import shutil
import subprocess
from pathlib import Path

from georecon.config import settings

_probe_cache: dict | None = None


def build_args(opts: dict) -> list[str]:
    """{'A.b': 8192, 'flag': True, 'skip': None} -> ['--A.b','8192','--flag','1']."""
    args: list[str] = []
    for key, val in opts.items():
        if val is None:
            continue
        if isinstance(val, bool):
            args += [f"--{key}", "1" if val else "0"]
        else:
            args += [f"--{key}", str(val)]
    return args


def _bin() -> str:
    return settings.colmap_bin or ""


def _prefix(bin_path: str) -> list[str]:
    # .bat must go through cmd; list form keeps paths-with-spaces safe (no shell).
    if bin_path.lower().endswith(".bat"):
        return ["cmd", "/c", bin_path]
    return [bin_path]


def parse_help(text: str) -> dict:
    """version (maj.min[.patch]) + cuda flag from a `colmap -h` header."""
    m = re.search(r"COLMAP\s+(\d+\.\d+(?:\.\d+)?)", text)
    return {"version": m.group(1) if m else None, "cuda": "with CUDA" in text}


def probe() -> dict:
    """{'available', 'version', 'cuda'} for the configured CLI. Cached per process."""
    global _probe_cache
    if _probe_cache is not None:
        return _probe_cache

    res = {"available": False, "version": None, "cuda": False}
    b = _bin()
    if b and (Path(b).exists() or shutil.which(b)):
        try:
            out = subprocess.run(_prefix(b) + ["-h"], capture_output=True,
                                 text=True, timeout=30)
            res["available"] = True
            res.update(parse_help((out.stdout or "") + "\n" + (out.stderr or "")))
        except (OSError, subprocess.SubprocessError):
            pass
    _probe_cache = res
    return res


def reset_probe_cache() -> None:
    global _probe_cache
    _probe_cache = None


def check_version_compat(log) -> None:
    import pycolmap

    p = probe()
    if not p["available"] or not p["version"]:
        return
    cli_mm = ".".join(p["version"].split(".")[:2])
    py_mm = ".".join(pycolmap.__version__.split(".")[:2])
    if cli_mm != py_mm:
        log.warning("COLMAP CLI %s vs pycolmap %s (major.minor differ) — the "
                    "SfM database format may be incompatible", p["version"],
                    pycolmap.__version__)


def run(tool: str, opts: dict, log, timeout: float | None = None) -> None:
    """Run ``<bin> <tool> <args>``, streaming output to ``log``. Raises with the
    last 30 lines on a non-zero exit."""
    b = _bin()
    if not b:
        raise RuntimeError("colmap_cli.run: GEORECON_COLMAP_BIN is not set")
    cmd = _prefix(b) + [tool] + build_args(opts)
    log.info("colmap %s", tool)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    tail: collections.deque[str] = collections.deque(maxlen=30)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        tail.append(line)
        log.info("[colmap] %s", line)
    proc.wait(timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f"colmap {tool} exited {proc.returncode}\n" + "\n".join(tail))
