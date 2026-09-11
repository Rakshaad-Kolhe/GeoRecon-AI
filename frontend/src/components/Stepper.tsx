import type { JobDetail, Metrics, StageName } from '../api/types'
import { STAGE_NAMES } from '../api/types'
import { formatInt, formatPct, formatSeconds } from '../lib/format'
import { mnum, stageMetric } from '../lib/metrics'
import { scaleInfoFor } from '../lib/scale'

type StepStatus = 'pending' | 'running' | 'done' | 'failed' | 'skipped'

const DOT: Record<StepStatus, string> = {
  pending: 'bg-slate-700',
  running: 'bg-cyan-400 gr-pulse',
  done: 'bg-cyan-400',
  failed: 'bg-red-500',
  skipped: 'bg-slate-600 ring-1 ring-slate-500',
}

function keyMetric(stage: StageName, metrics: Metrics | undefined, georeferenced: boolean): string | null {
  const m = stageMetric(metrics, stage)
  if (!m) return null
  switch (stage) {
    case 'keyframes': {
      const v = mnum(m, 'keyframes')
      return v == null ? null : `${formatInt(v)} keyframes`
    }
    case 'masking': {
      const v = mnum(m, 'masked_pct_mean')
      return v == null ? null : `${formatPct(v)} masked`
    }
    case 'sfm': {
      const v = mnum(m, 'registered_pct')
      return v == null ? null : `${formatPct(v)} registered`
    }
    case 'georef': {
      if (!georeferenced) return 'unscaled'
      const v = mnum(m, 'holdout_rmse_h')
      return v == null ? null : `${v.toFixed(2)} m holdout RMSEh`
    }
    case 'dense': {
      const v = mnum(m, 'points')
      return v == null ? null : `${formatInt(v)} points`
    }
    default:
      return null
  }
}

function statusFor(job: JobDetail, i: number, curIdx: number): StepStatus {
  let s: StepStatus
  if (job.state === 'done') {
    s = 'done'
  } else if (job.state === 'failed') {
    s = i < curIdx ? 'done' : i === curIdx ? 'failed' : 'pending'
  } else {
    s = i < curIdx ? 'done' : i === curIdx && job.state === 'running' ? 'running' : 'pending'
  }
  if (s === 'done' && !stageMetric(job.metrics, STAGE_NAMES[i])) s = 'skipped'
  return s
}

export function Stepper({ job }: { job: JobDetail }) {
  const curIdx = Math.max(0, STAGE_NAMES.indexOf(job.stage))
  const { georeferenced } = scaleInfoFor(job)
  return (
    <ol className="space-y-0">
      {STAGE_NAMES.map((stage, i) => {
        const status = statusFor(job, i, curIdx)
        const m = stageMetric(job.metrics, stage)
        const secs = mnum(m, 'seconds')
        const km = status === 'done' ? keyMetric(stage, job.metrics, georeferenced) : null
        return (
          <li key={stage} className="flex gap-3">
            <div className="flex flex-col items-center">
              <span className={`mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full ${DOT[status]}`} />
              {i < STAGE_NAMES.length - 1 && (
                <span className="w-px flex-1 bg-slate-800" />
              )}
            </div>
            <div className="min-w-0 flex-1 pb-3">
              <div className="flex items-baseline justify-between gap-2">
                <span
                  className={`text-sm ${
                    status === 'pending'
                      ? 'text-slate-500'
                      : status === 'failed'
                        ? 'text-red-300'
                        : 'text-slate-200'
                  }`}
                >
                  {stage}
                </span>
                <span className="font-mono text-xs text-slate-500">
                  {status === 'running'
                    ? '···'
                    : status === 'skipped'
                      ? 'skipped'
                      : status === 'done' || status === 'failed'
                        ? formatSeconds(secs)
                        : ''}
                </span>
              </div>
              {km && <div className="font-mono text-xs text-cyan-400/90">{km}</div>}
            </div>
          </li>
        )
      })}
    </ol>
  )
}
