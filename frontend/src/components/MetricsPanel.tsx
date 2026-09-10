import { useQuery } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { api } from '../api/client'
import type { Metrics, ValidateSummary } from '../api/types'
import { STAGE_NAMES } from '../api/types'
import { formatInt, formatMeters, formatPct, formatSeconds } from '../lib/format'
import { getValidateSummary, mnum, stageMetric } from '../lib/metrics'

const PROJECTION_TARGET_S = 900

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

function SpeedChart({
  perStage,
  total,
}: {
  perStage: [string, number][]
  total: number | undefined
}) {
  const maxSec = Math.max(1, ...perStage.map(([, s]) => s))
  return (
    <div>
      <h3 className="mb-2 flex items-baseline justify-between text-xs font-semibold uppercase tracking-wider text-slate-400">
        <span>Speed</span>
        <span className="font-mono text-slate-500">total {formatSeconds(total)}</span>
      </h3>
      {perStage.length === 0 ? (
        <p className="text-xs text-slate-600">no stage timings yet</p>
      ) : (
        <div className="space-y-1">
          {perStage.map(([name, sec]) => (
            <div key={name} className="flex items-center gap-2">
              <span className="w-20 shrink-0 text-xs text-slate-400">{name}</span>
              <div className="h-3 flex-1 overflow-hidden rounded-sm bg-slate-800">
                <div
                  className="h-full bg-cyan-400/70"
                  style={{ width: `${(sec / maxSec) * 100}%` }}
                />
              </div>
              <span className="w-16 shrink-0 text-right font-mono text-xs text-slate-400">
                {formatSeconds(sec)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Projection({ summary }: { summary: ValidateSummary }) {
  const proj = summary.speed.projected_10min_s
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => api.getHealth(),
    staleTime: 30_000,
  })
  if (proj == null) return null
  const over = proj > PROJECTION_TARGET_S
  return (
    <div className="rounded border border-slate-800 bg-slate-950 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">
        projected 10-min clip <span className="text-slate-600">(estimate)</span>
      </div>
      <div className="font-mono text-sm">
        <span className={over ? 'text-red-400' : 'text-cyan-400'}>
          {formatSeconds(proj)}
        </span>
        <span className="text-slate-500"> vs {PROJECTION_TARGET_S} s target</span>
      </div>
      {health?.gpu_name && (
        <div className="text-[11px] text-slate-600">on {health.gpu_name}</div>
      )}
    </div>
  )
}

function SummaryView({ summary }: { summary: ValidateSummary }) {
  const a = summary.accuracy
  const c = summary.completeness
  const perStage: [string, number][] = Object.entries(summary.speed.per_stage_s ?? {})
  return (
    <div className="space-y-5">
      <Group title="Accuracy">
        <Stat label="holdout RMSE h" value={formatMeters(a.holdout_rmse_h)} />
        <Stat label="holdout RMSE v" value={formatMeters(a.holdout_rmse_v)} />
        <Stat label="fit RMSE h" value={formatMeters(a.rmse_h)} />
        <Stat label="fit RMSE v" value={formatMeters(a.rmse_v)} />
        <Stat
          label="inliers / pairs"
          value={
            a.inliers == null && a.pairs == null
              ? '—'
              : `${formatInt(a.inliers)} / ${formatInt(a.pairs)}`
          }
        />
        <Stat label="scale drift" value={formatPct(a.scale_drift_pct, 2)} />
        <Stat label="reproj error" value={a.mean_reproj_px?.toFixed(2) ?? '—'} />
        <Stat label="excluded frames" value={formatInt(a.excluded_images)} />
        <Stat label="branch" value={a.branch ?? '—'} />
      </Group>

      <Group title="Completeness">
        <Stat label="registered" value={formatPct(c.registered_pct)} />
        <Stat label="dense points" value={formatInt(c.dense_points)} />
        <Stat label="points / m²" value={c.points_per_m2?.toFixed(2) ?? '—'} />
        <Stat label="DSM coverage" value={formatPct(c.coverage_pct)} />
        <Stat
          label="mesh area"
          value={c.mesh_surface_area_m2 ? `${formatInt(c.mesh_surface_area_m2)} m²` : '—'}
        />
      </Group>

      <SpeedChart perStage={perStage} total={summary.speed.total_seconds} />
      <Projection summary={summary} />
    </div>
  )
}

function LegacyView({ metrics }: { metrics?: Metrics }) {
  const g = stageMetric(metrics, 'georef')
  const s = stageMetric(metrics, 'sfm')
  const d = stageMetric(metrics, 'dense')
  const inl = mnum(g, 'inliers')
  const pairs = mnum(g, 'pairs')
  const perStage: [string, number][] = []
  for (const st of STAGE_NAMES) {
    const sec = mnum(stageMetric(metrics, st), 'seconds')
    if (sec != null) perStage.push([st, sec])
  }

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

      <SpeedChart perStage={perStage} total={metrics?.total_seconds} />
    </div>
  )
}

export function MetricsPanel({ metrics }: { metrics?: Metrics }) {
  const summary = getValidateSummary(metrics)
  return summary ? <SummaryView summary={summary} /> : <LegacyView metrics={metrics} />
}
