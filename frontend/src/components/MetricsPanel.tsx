import type { ReactNode } from 'react'
import type { Metrics, StageName } from '../api/types'
import { STAGE_NAMES } from '../api/types'
import { formatInt, formatMeters, formatPct, formatSeconds } from '../lib/format'
import { mnum, stageMetric } from '../lib/metrics'

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-slate-800 bg-slate-950 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="font-mono text-sm text-slate-100">{value}</div>
    </div>
  )
}

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
        {title}
      </h3>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">{children}</div>
    </div>
  )
}

export function MetricsPanel({ metrics }: { metrics?: Metrics }) {
  const g = stageMetric(metrics, 'georef')
  const s = stageMetric(metrics, 'sfm')
  const d = stageMetric(metrics, 'dense')

  const inl = mnum(g, 'inliers')
  const pairs = mnum(g, 'pairs')

  const speedRows = STAGE_NAMES.map((st) => ({
    st,
    sec: mnum(stageMetric(metrics, st), 'seconds'),
  })).filter((r): r is { st: StageName; sec: number } => r.sec != null)
  const maxSec = Math.max(1, ...speedRows.map((r) => r.sec))
  const total = metrics?.total_seconds

  return (
    <div className="space-y-5">
      <Group title="Accuracy">
        <Stat label="holdout RMSE h" value={formatMeters(mnum(g, 'holdout_rmse_h'))} />
        <Stat label="holdout RMSE v" value={formatMeters(mnum(g, 'holdout_rmse_v'))} />
        <Stat label="fit RMSE h" value={formatMeters(mnum(g, 'rmse_h'))} />
        <Stat label="fit RMSE v" value={formatMeters(mnum(g, 'rmse_v'))} />
        <Stat
          label="inliers / pairs"
          value={inl == null && pairs == null ? '—' : `${formatInt(inl)} / ${formatInt(pairs)}`}
        />
        <Stat label="scale drift" value={formatPct(mnum(g, 'scale_drift_pct'), 2)} />
      </Group>

      <Group title="Completeness">
        <Stat label="registered" value={formatPct(mnum(s, 'registered_pct'))} />
        <Stat label="dense points" value={formatInt(mnum(d, 'points'))} />
        <Stat label="points / m²" value={mnum(d, 'points_per_m2')?.toFixed(2) ?? '—'} />
      </Group>

      <div>
        <h3 className="mb-2 flex items-baseline justify-between text-xs font-semibold uppercase tracking-wider text-slate-400">
          <span>Speed</span>
          <span className="font-mono text-slate-500">total {formatSeconds(total)}</span>
        </h3>
        {speedRows.length === 0 ? (
          <p className="text-xs text-slate-600">no stage timings yet</p>
        ) : (
          <div className="space-y-1">
            {speedRows.map((r) => (
              <div key={r.st} className="flex items-center gap-2">
                <span className="w-20 shrink-0 text-xs text-slate-400">{r.st}</span>
                <div className="h-3 flex-1 overflow-hidden rounded-sm bg-slate-800">
                  <div
                    className="h-full bg-cyan-400/70"
                    style={{ width: `${(r.sec / maxSec) * 100}%` }}
                  />
                </div>
                <span className="w-16 shrink-0 text-right font-mono text-xs text-slate-400">
                  {formatSeconds(r.sec)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
