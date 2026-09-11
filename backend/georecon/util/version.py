"""Running-checkout git version stamp: short commit + dirty flag.

Used to tag jobs at creation (``config.json``), every stage-metrics rebuild
(``report/metrics.json``), and ``/api/health`` — so a result can always be
traced back to the code that produced it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_cache: dict | None = None


def code_version() -> dict:
    """{'commit': short-sha or None, 'dirty': bool or None}. Cached per process."""
    global _cache
    if _cache is not None:
        return _cache
    commit: str | None = None
    dirty: bool | None = None
    try:
        r = subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            commit = r.stdout.strip() or None
        # --untracked-files=no: dirty means "tracked source differs from HEAD",
        # not "there happen to be scratch/data directories lying around".
        s = subprocess.run(["git", "-C", str(_REPO_ROOT), "status", "--porcelain",
                            "--untracked-files=no"],
                           capture_output=True, text=True, timeout=5)
        if s.returncode == 0:
            dirty = bool(s.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    _cache = {"commit": commit, "dirty": dirty}
    return _cache


def reset_cache() -> None:
    global _cache
    _cache = None
