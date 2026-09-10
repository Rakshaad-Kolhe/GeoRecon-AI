"""Settings, presets and per-job configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide settings, overridable via GEORECON_* env vars."""

    model_config = SettingsConfigDict(env_prefix="GEORECON_", extra="ignore")

    jobs_dir: str = "jobs"
    max_upload_mb: int = 2048
    colmap_backend: Literal["pycolmap", "cli"] = "pycolmap"


settings = Settings()


@dataclass(frozen=True)
class Preset:
    max_keyframes: int
    frame_long_side: int
    mvs_max_image_size: int


PRESETS: dict[str, Preset] = {
    "fast": Preset(max_keyframes=150, frame_long_side=1280, mvs_max_image_size=800),
    "balanced": Preset(max_keyframes=300, frame_long_side=1600, mvs_max_image_size=1200),
    "accurate": Preset(max_keyframes=600, frame_long_side=2000, mvs_max_image_size=1600),
}


class JobConfig(BaseModel):
    """Per-job configuration, serialised to ``<job_dir>/config.json``."""

    preset: str = "fast"
    mask_dynamic: bool = True
    hfov_deg: Optional[float] = None
    force_from: Optional[str] = None
    # Source media, resolved to absolute paths by the CLI / API.
    video_path: str = ""
    telemetry_path: Optional[str] = None

    @property
    def resolved_preset(self) -> Preset:
        if self.preset not in PRESETS:
            raise ValueError(
                f"unknown preset {self.preset!r}; choose from {sorted(PRESETS)}"
            )
        return PRESETS[self.preset]

    def to_json(self, job_dir) -> Path:
        path = Path(job_dir) / "config.json"
        path.write_text(json.dumps(self.model_dump(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, job_dir) -> "JobConfig":
        path = Path(job_dir) / "config.json"
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
