// Mirrors the backend contract in CLAUDE.md (status.json + report/metrics.json).

export type StageName =
  | 'ingest'
  | 'keyframes'
  | 'masking'
  | 'sfm'
  | 'georef'
  | 'dense'
  | 'mesh'
  | 'export'
  | 'validate'

export const STAGE_NAMES: StageName[] = [
  'ingest',
  'keyframes',
  'masking',
  'sfm',
  'georef',
  'dense',
  'mesh',
  'export',
  'validate',
]

export type JobState = 'queued' | 'running' | 'done' | 'failed'

export interface JobStatus {
  job_id: string
  state: JobState
  stage: StageName
  progress: number // 0..1
  message: string
  warnings: string[]
  updated_at: string // ISO-8601
  // Not in status.json, but the `GET /api/jobs` list contract exposes created time.
  created_at: string // ISO-8601
}

export type StageMetrics = { seconds: number } & Record<string, unknown>

export interface Metrics {
  stages: Partial<Record<StageName, StageMetrics>>
  total_seconds: number
}

export type Preset = 'fast' | 'balanced' | 'accurate'

// merged status.json + metrics.json; `preset` echoed back by the job runner.
export type JobDetail = JobStatus & {
  metrics?: Metrics
  preset?: Preset
}

export interface CreateJobInput {
  video: File
  telemetry?: File
  preset: Preset
  mask_dynamic: boolean
  telemetry_offset_s: number
}

// GET /api/health (backend feat/10-api).
export interface HealthInfo {
  torch_cuda: boolean
  gpu_name: string
  colmap: { available: boolean; version: string; cuda: boolean }
  pycolmap_version: string
  queue_len: number
  running_job: string | null
}

// GET /api/jobs/{id}/files — paths relative to outputs/.
export interface FileEntry {
  path: string
  bytes: number
}

// metrics.stages.validate.summary (backend validate stage).
export interface ValidateSummary {
  accuracy: {
    branch?: string
    rmse_h?: number
    rmse_v?: number
    holdout_rmse_h?: number
    holdout_rmse_v?: number
    inliers?: number
    pairs?: number
    excluded_images?: number
    scale_drift_pct?: number
    mean_reproj_px?: number
  }
  completeness: {
    registered_pct?: number
    dense_points?: number
    points_per_m2?: number
    coverage_pct?: number
    mesh_surface_area_m2?: number
  }
  speed: {
    per_stage_s?: Record<string, number>
    total_seconds?: number
    video_duration_s?: number
    keyframes?: number
    kf10?: number
    projected_10min_s?: number
  }
}

// web/meta.json emitted by backend PR 08 alongside the downsampled web assets.
// Scene is local ENU in metres, +Z up.
export interface ViewerMeta {
  origin: { lat: number; lon: number; alt: number }
  utm_epsg: number
  bbox_enu: { min: [number, number, number]; max: [number, number, number] }
  points: number
  triangles: number
  median_spacing_m: number
  trajectory_enu: [number, number, number][]
  height_ref: string
}
