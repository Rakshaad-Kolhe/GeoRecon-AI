import type { Api } from './client'
import { generateMockSite, hashSeed } from './mockSite'
import type {
  CreateJobInput,
  JobDetail,
  Metrics,
  StageMetrics,
  StageName,
  ViewerMeta,
} from './types'
import { STAGE_NAMES } from './types'

// ---------------------------------------------------------------------------
// In-memory mock backend. Drives one advancing job per tick (1.5 s) plus two
// seeded jobs built from real pipeline numbers.
// ---------------------------------------------------------------------------

const TICK_MS = 1500
const SUBSTEPS = 3 // progress increments inside a stage before it completes

interface MockJob {
  detail: JobDetail
  createdMs: number
  stageIdx: number
  substep: number
  seeded: boolean
}

// Plausible per-stage metrics for freshly created jobs.
const SYNTH: Record<StageName, StageMetrics> = {
  ingest: { seconds: 2 },
  keyframes: { seconds: 8, keyframes: 60 },
  masking: { seconds: 6, frames: 60, masked_pct_mean: 0.6, device: 'cuda' },
  sfm: {
    seconds: 22,
    device: 'cuda-cli',
    registered: 57,
    frames: 60,
    registered_pct: 95.0,
    mean_reproj_px: 0.52,
    points3d: 12800,
  },
  georef: {
    seconds: 1,
    branch: 'sim3',
    rmse_h: 0.9,
    rmse_v: 1.1,
    holdout_rmse_h: 0.95,
    holdout_rmse_v: 1.2,
    inliers: 55,
    pairs: 57,
    scale_drift_pct: 0.12,
  },
  dense: {
    seconds: 300,
    dense: true,
    points: 350000,
    mean_views: 6.5,
    points_per_m2: 7.0,
    stereo_s: 285,
    fusion_s: 12,
  },
  mesh: { seconds: 35, vertices: 121000, faces: 239000 },
  export: { seconds: 5, formats: 6 },
  validate: { seconds: 3, holdout_folds: 5 },
}

const WARNING_POOL = [
  'masking: 3 frames had no confident detections — passed through unmasked',
  'georef: GPS track near-collinear (σ2/σ1=0.08) — used ground-plane branch',
  'dense: CUDA out-of-memory once, retried at max_image_size 640',
  'keyframes: telemetry gap 1.8 s around t=12 s — interpolated',
  'sfm: 2 frames dropped, insufficient inliers',
]

function nowIso(offsetMs = 0): string {
  return new Date(Date.now() + offsetMs).toISOString()
}

function buildMetrics(upTo: number, override?: Partial<Record<StageName, StageMetrics>>): Metrics {
  const stages: Partial<Record<StageName, StageMetrics>> = {}
  let total = 0
  for (let i = 0; i < upTo && i < STAGE_NAMES.length; i++) {
    const name = STAGE_NAMES[i]
    const m = override?.[name] ?? SYNTH[name]
    stages[name] = m
    total += typeof m.seconds === 'number' ? m.seconds : 0
  }
  return { stages, total_seconds: Math.round(total * 10) / 10 }
}

// ---- seeded jobs ----------------------------------------------------------

const REAL_DEMO_METRICS: Metrics = {
  stages: {
    ingest: { seconds: 2 },
    keyframes: { keyframes: 52, seconds: 9 },
    masking: { frames: 52, masked_pct_mean: 0.4, device: 'cuda', seconds: 6 },
    sfm: {
      device: 'cuda-cli',
      registered: 50,
      frames: 52,
      registered_pct: 96.2,
      mean_reproj_px: 0.48,
      points3d: 11695,
      extract_s: 2.7,
      match_s: 5.5,
      map_s: 16.0,
      seconds: 25,
    },
    georef: {
      branch: 'sim3',
      rmse_h: 0.83,
      rmse_v: 0.96,
      holdout_rmse_h: 0.87,
      holdout_rmse_v: 1.08,
      inliers: 49,
      pairs: 50,
      scale_drift_pct: 0.09,
      seconds: 1,
    },
    dense: {
      dense: true,
      points: 388574,
      mean_views: 6.94,
      points_per_m2: 7.66,
      stereo_s: 318.9,
      fusion_s: 10.2,
      seconds: 335,
    },
    mesh: { seconds: 41, vertices: 121334, faces: 239880 },
    export: { seconds: 5, formats: 6 },
    validate: { seconds: 3, holdout_folds: 5 },
  },
  total_seconds: 427,
}

function seed(store: Map<string, MockJob>) {
  const demoCreated = Date.now() - 1000 * 60 * 12
  store.set('real-demo', {
    createdMs: demoCreated,
    stageIdx: STAGE_NAMES.length,
    substep: 0,
    seeded: true,
    detail: {
      job_id: 'real-demo',
      state: 'done',
      stage: 'validate',
      progress: 1,
      message: 'pipeline complete — 9/9 stages',
      warnings: [],
      created_at: new Date(demoCreated).toISOString(),
      updated_at: new Date(demoCreated + 427_000).toISOString(),
      preset: 'balanced',
      metrics: REAL_DEMO_METRICS,
    },
  })

  const badCreated = Date.now() - 1000 * 60 * 5
  store.set('bad-telemetry', {
    createdMs: badCreated,
    stageIdx: STAGE_NAMES.indexOf('georef'),
    substep: 0,
    seeded: true,
    detail: {
      job_id: 'bad-telemetry',
      state: 'failed',
      stage: 'georef',
      progress: STAGE_NAMES.indexOf('georef') / STAGE_NAMES.length,
      message: 'georef: ValueError: <3 GPS pairs',
      warnings: ['telemetry: only 2 GPS fixes parsed from video.SRT'],
      created_at: new Date(badCreated).toISOString(),
      updated_at: new Date(badCreated + 41_000).toISOString(),
      preset: 'fast',
      metrics: buildMetrics(STAGE_NAMES.indexOf('georef')),
    },
  })
}

// ---- store + ticker -----------------------------------------------------

const store = new Map<string, MockJob>()
seed(store)

let seq = 0

function tick() {
  for (const job of store.values()) {
    if (job.seeded) continue
    const d = job.detail
    if (d.state === 'done' || d.state === 'failed') continue

    d.state = 'running'
    job.substep += 1

    if (job.substep >= SUBSTEPS) {
      job.substep = 0
      job.stageIdx += 1
      if (Math.random() < 0.25) {
        const w = WARNING_POOL[Math.floor(Math.random() * WARNING_POOL.length)]
        if (!d.warnings.includes(w)) d.warnings.push(w)
      }
    }

    if (job.stageIdx >= STAGE_NAMES.length) {
      d.state = 'done'
      d.stage = 'validate'
      d.progress = 1
      d.message = 'pipeline complete — 9/9 stages'
      d.metrics = buildMetrics(STAGE_NAMES.length)
    } else {
      d.stage = STAGE_NAMES[job.stageIdx]
      d.progress =
        Math.round(
          ((job.stageIdx + job.substep / SUBSTEPS) / STAGE_NAMES.length) * 1000,
        ) / 1000
      d.message = `${d.stage}: working (${job.substep + 1}/${SUBSTEPS})`
      d.metrics = buildMetrics(job.stageIdx)
    }
    d.updated_at = nowIso()
  }
}

setInterval(tick, TICK_MS)

// ---- log lines --------------------------------------------------------

const LOG_LINES: { stage: StageName; text: string }[] = [
  { stage: 'ingest', text: 'probe: 1920x1080 @ 29.97 fps, 00:31, h264' },
  { stage: 'ingest', text: 'telemetry: parsed 930 samples from video.SRT' },
  { stage: 'keyframes', text: 'scoring frames by Laplacian-variance sharpness' },
  { stage: 'keyframes', text: 'selected 52 keyframes (min gap 6, sharp >= 0.34)' },
  { stage: 'masking', text: 'YOLOv8x-seg @ cuda, batch 8' },
  { stage: 'masking', text: '52/52 frames masked, mean coverage 0.4%' },
  { stage: 'sfm', text: 'SIFT extract (cuda-cli): 52 imgs in 2.7 s' },
  { stage: 'sfm', text: 'exhaustive match: 1326 pairs in 5.5 s' },
  { stage: 'sfm', text: 'mapper: registered 50/52, 11695 pts, reproj 0.48 px' },
  { stage: 'georef', text: 'paired 50 camera centres with GPS ENU' },
  { stage: 'georef', text: 'sim3 RANSAC: 49/50 inliers, rmse_h 0.83 m rmse_v 0.96 m' },
  { stage: 'dense', text: 'patch-match stereo: 50 views @ max 800 px' },
  { stage: 'dense', text: 'fusion: 388574 points, mean 6.94 views/pt' },
  { stage: 'mesh', text: 'poisson_mesher depth 11: 121k verts / 240k faces' },
  { stage: 'export', text: 'wrote PLY, LAS, OBJ, GLB, dsm.tif, color.tif' },
  { stage: 'validate', text: 'holdout 5-fold: rmse_h 0.87 m rmse_v 1.08 m' },
]

function logFor(job: MockJob, tail: number): string {
  const maxIdx = job.detail.state === 'done' ? STAGE_NAMES.length : job.stageIdx + 1
  const lines = LOG_LINES.filter((l) => STAGE_NAMES.indexOf(l.stage) < maxIdx).map(
    (l) => `[${l.stage}] ${l.text}`,
  )
  if (job.detail.state === 'failed') {
    lines.push(`[${job.detail.stage}] ERROR ${job.detail.message}`)
  }
  return lines.slice(-tail).join('\n')
}

// ---- api impl -------------------------------------------------------

function delay<T>(value: T, ms = 120): Promise<T> {
  return new Promise((r) => setTimeout(() => r(value), ms))
}

function clone<T>(v: T): T {
  return JSON.parse(JSON.stringify(v)) as T
}

function slug(name: string): string {
  return (
    name
      .replace(/\.[^.]+$/, '')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 24) || 'job'
  )
}

// ---- viewer assets -------------------------------------------------

interface ViewerEntry {
  meta: ViewerMeta
  publicAssets: boolean
  urls: Record<string, string>
}

const viewerCache = new Map<string, ViewerEntry>()

function blobUrl(data: BlobPart, type: string): string {
  return URL.createObjectURL(new Blob([data], { type }))
}

async function prepareViewerEntry(jobId: string): Promise<ViewerEntry> {
  const hit = viewerCache.get(jobId)
  if (hit) return hit

  // public override: drop real files under frontend/public/mock/<jobId>/web/
  try {
    const res = await fetch(`/mock/${jobId}/web/meta.json`)
    if (res.ok && res.headers.get('content-type')?.includes('json')) {
      const meta = (await res.json()) as ViewerMeta
      const entry: ViewerEntry = { meta, publicAssets: true, urls: {} }
      viewerCache.set(jobId, entry)
      return entry
    }
  } catch {
    // fall through to procedural
  }

  const site = generateMockSite(hashSeed(jobId))
  const entry: ViewerEntry = {
    meta: site.meta,
    publicAssets: false,
    urls: {
      'web/meta.json': blobUrl(JSON.stringify(site.meta), 'application/json'),
      'web/pointcloud.ply': blobUrl(site.cloudPLY, 'application/octet-stream'),
      'web/mesh.ply': blobUrl(site.meshPLY, 'application/octet-stream'),
    },
  }
  viewerCache.set(jobId, entry)
  return entry
}

export const mockApi: Api = {
  async listJobs() {
    const jobs = [...store.values()]
      .sort((a, b) => b.createdMs - a.createdMs)
      .map((j) => clone(j.detail))
    return delay(jobs)
  },

  async getJob(id) {
    const job = store.get(id)
    if (!job) {
      const err = new Error(`job '${id}' not found`) as Error & { status: number }
      err.status = 404
      throw err
    }
    return delay(clone(job.detail))
  },

  createJob(input: CreateJobInput, onUploadProgress) {
    return new Promise((resolve) => {
      let p = 0
      const bump = setInterval(() => {
        p = Math.min(1, p + 0.2)
        onUploadProgress?.(p)
        if (p >= 1) {
          clearInterval(bump)
          const id = `${slug(input.video.name)}-${(++seq).toString(36)}`
          const createdMs = Date.now()
          store.set(id, {
            createdMs,
            stageIdx: 0,
            substep: 0,
            seeded: false,
            detail: {
              job_id: id,
              state: 'queued',
              stage: 'ingest',
              progress: 0,
              message: 'queued',
              warnings: input.telemetry
                ? []
                : ['no telemetry supplied — model will be unscaled'],
              created_at: new Date(createdMs).toISOString(),
              updated_at: new Date(createdMs).toISOString(),
              preset: input.preset,
              metrics: { stages: {}, total_seconds: 0 },
            },
          })
          setTimeout(() => resolve({ job_id: id }), 150)
        }
      }, 120)
    })
  },

  async getLog(id, tail = 200) {
    const job = store.get(id)
    if (!job) return delay('')
    return delay(logFor(job, tail))
  },

  async prepareViewer(id) {
    const entry = await prepareViewerEntry(id)
    return entry.meta
  },

  fileUrl(id, path) {
    const norm = path.replace(/^\/+/, '')
    const entry = viewerCache.get(id)
    if (!entry) return `/mock/${id}/${norm}`
    if (entry.publicAssets) return `/mock/${id}/${norm}`
    return entry.urls[norm] ?? `/mock/${id}/${norm}`
  },
}
