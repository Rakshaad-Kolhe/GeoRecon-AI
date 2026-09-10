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
