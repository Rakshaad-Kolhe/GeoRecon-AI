"""Settings, presets and per-job configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide settings, overridable via GEORECON_* env vars."""

    model_config = SettingsConfigDict(
        env_prefix="GEORECON_", extra="ignore", env_file=".env", env_file_encoding="utf-8"
    )

    jobs_dir: str = "jobs"
    models_dir: str = "models"
    max_upload_mb: int = 2048
    colmap_backend: Literal["pycolmap", "cli"] = "pycolmap"
    colmap_bin: str = ""          # path to a CUDA COLMAP CLI (GEORECON_COLMAP_BIN)


settings = Settings()


@dataclass(frozen=True)
class Preset:
    max_keyframes: int
    frame_long_side: int
    mvs_max_image_size: int
    # dense MVS speed knobs (see DenseCfg); accurate widens them
    mvs_num_src_images: int = 8
    mvs_window_radius: int = 4
    mvs_num_iterations: int = 4
    mvs_ref_stride: int = 1
    # mesh (see MeshCfg)
    poisson_depth: int = 9
    mesh_max_tris: int = 200_000


PRESETS: dict[str, Preset] = {
    # dense.ref_stride stays 1 for every preset (halving depth maps thins the
    # fused cloud ~4x); keep it as an opt-in knob via ``--set dense.ref_stride``.
    "fast": Preset(max_keyframes=150, frame_long_side=1280, mvs_max_image_size=640,
                   mvs_num_iterations=3, mvs_num_src_images=6),
    "balanced": Preset(max_keyframes=300, frame_long_side=1600, mvs_max_image_size=1200,
                       poisson_depth=10, mesh_max_tris=400_000),
    "accurate": Preset(max_keyframes=600, frame_long_side=2000, mvs_max_image_size=1600,
                       mvs_num_src_images=12, mvs_window_radius=5, mvs_num_iterations=5,
                       poisson_depth=11, mesh_max_tris=800_000),
}


class KeyframeCfg(BaseModel):
    """Keyframe-selection tuning. ``flow_frac`` is a fraction of the diagonal
    of the 640px-wide analysis frame (the actual pixel threshold is derived
    by the stage)."""

    model_config = ConfigDict(frozen=True)

    analysis_fps: int = 10
    blur_ratio: float = 0.6
    blur_window: int = 31
    flow_frac: float = 0.10
    min_gps_move_m: float = 0.5
    jpeg_quality: int = 95


class MaskCfg(BaseModel):
    """Dynamic-object (YOLO segmentation) masking tuning."""

    model_config = ConfigDict(frozen=True)

    model: str = "yolo11n-seg.pt"
    imgsz: int = 1280
    conf: float = 0.25
    # COCO ids: person/bicycle/car/motorcycle/bus/train/truck/boat + common animals.
    classes: list[int] = Field(
        default_factory=lambda: [0, 1, 2, 3, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19]
    )
    dilate_frac: float = 0.01
    batch: int = 8
    max_masked_warn: float = 0.5


class SfmCfg(BaseModel):
    """Structure-from-Motion (pycolmap) tuning."""

    model_config = ConfigDict(frozen=True)

    camera_model: str = "SIMPLE_RADIAL"
    max_features: int = 8192
    seq_overlap: int = 12
    quadratic_overlap: bool = True
    mapper: str = "global"            # "global" (GLOMAP) | "incremental"
    min_model_size: int = 6
    use_gpu: Union[bool, str] = "auto"   # "auto" | true | false


class GeorefCfg(BaseModel):
    """Georeferencing (SfM -> local ENU) tuning."""

    model_config = ConfigDict(frozen=True)

    mode: str = "auto"                 # "auto" | "sim3" | "collinear"
    collinear_ratio: float = 0.1       # sigma2/sigma1 below this -> collinear branch
    ransac_thr_m: float = 5.0
    ransac_iters: int = 1000
    plane_thr_frac: float = 0.02       # x cloud extent -> ground-plane inlier band
    holdout_folds: int = 5
    seed: int = 0


class DenseCfg(BaseModel):
    """Dense MVS (patch-match) speed knobs. Defaults match the fast/balanced
    presets; ``accurate`` widens them. An explicit override here (e.g.
    ``--set dense.window_radius=6``) wins over the preset."""

    model_config = ConfigDict(frozen=True)

    num_src_images: int = 8      # image_undistorter --num_patch_match_src_images
    window_radius: int = 4       # PatchMatchStereo.window_radius
    num_iterations: int = 4      # PatchMatchStereo.num_iterations
    ref_stride: int = 1          # compute a depth map for every Nth keyframe only


class MeshCfg(BaseModel):
    """Meshing tuning. Defaults match ``fast``; presets widen depth / tri budget.
    An explicit override here wins over the preset."""

    model_config = ConfigDict(frozen=True)

    method: str = "auto"        # "auto" | "poisson" | "heightfield"
    poisson_depth: int = 9      # PoissonMeshing.depth
    poisson_trim: int = 7       # PoissonMeshing.trim
    max_tris: int = 200_000     # decimation target
    min_points_for_poisson: int = 50_000
    max_seconds: float = 300.0  # poisson over this -> heightfield fallback


class JobConfig(BaseModel):
    """Per-job configuration, serialised to ``<job_dir>/config.json``."""

    preset: str = "fast"
    mask_dynamic: bool = True
    hfov_deg: Optional[float] = None
    force_from: Optional[str] = None
    # Source media, resolved to absolute paths by the CLI / API.
    video_path: str = ""
    telemetry_path: Optional[str] = None
    # telemetry_t = video_t + telemetry_offset_s (applied when sampling telemetry).
    telemetry_offset_s: float = 0.0
    keyframes: KeyframeCfg = KeyframeCfg()
    masking: MaskCfg = MaskCfg()
    sfm: SfmCfg = SfmCfg()
    georef: GeorefCfg = GeorefCfg()
    dense: DenseCfg = DenseCfg()
    mesh: MeshCfg = MeshCfg()

    @property
    def resolved_preset(self) -> Preset:
        if self.preset not in PRESETS:
            raise ValueError(
                f"unknown preset {self.preset!r}; choose from {sorted(PRESETS)}"
            )
        return PRESETS[self.preset]

    @property
    def resolved_dense(self) -> DenseCfg:
        """Preset dense knobs unless the job overrode ``dense`` explicitly."""
        p = self.resolved_preset
        if self.dense != DenseCfg():
            return self.dense
        return DenseCfg(num_src_images=p.mvs_num_src_images,
                        window_radius=p.mvs_window_radius,
                        num_iterations=p.mvs_num_iterations,
                        ref_stride=p.mvs_ref_stride)

    @property
    def resolved_mesh(self) -> MeshCfg:
        """Preset mesh knobs unless the job overrode ``mesh`` explicitly."""
        p = self.resolved_preset
        if self.mesh != MeshCfg():
            return self.mesh
        return MeshCfg(poisson_depth=p.poisson_depth, max_tris=p.mesh_max_tris)

    def to_json(self, job_dir) -> Path:
        path = Path(job_dir) / "config.json"
        path.write_text(json.dumps(self.model_dump(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, job_dir) -> "JobConfig":
        path = Path(job_dir) / "config.json"
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
