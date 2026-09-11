import { formatInt } from '../../lib/format'
import type { SceneInfo } from './SceneCanvas'

export interface AssetStatus {
  label: string
  progress: number
  error: string | null
}

interface Props {
  points: number
  triangles: number
  info: SceneInfo | null
  unitsLabel: string
  assets: AssetStatus[]
  onRetry: (label: string) => void
}

function niceLength(raw: number): number {
  const p = 10 ** Math.floor(Math.log10(raw))
  const n = raw / p
  const m = n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10
  return m * p
}

export function SceneOverlay({ points, triangles, info, unitsLabel, assets, onRetry }: Props) {
  let barPx = 0
  let barLabel = '—'
  if (info && info.mppx > 0 && Number.isFinite(info.mppx)) {
    const len = niceLength(info.mppx * 90)
    // km rollup only makes sense for real metres, not relative/unscaled units.
    barLabel =
      unitsLabel === 'm' && len >= 1000 ? `${len / 1000} km` : `${len} ${unitsLabel}`
    barPx = Math.min(240, len / info.mppx)
  }

  const pending = assets.filter((a) => a.error || a.progress < 1)

  return (
    <div className="pointer-events-none absolute inset-0 select-none">
      <div className="absolute left-2 top-2 rounded bg-slate-950/70 px-2 py-1 font-mono text-[11px] text-slate-400">
        {formatInt(points)} pts · {formatInt(triangles)} tris
      </div>

      <div className="absolute right-2 top-2 flex flex-col items-center rounded bg-slate-950/70 px-2 py-1">
        <svg
          width="22"
          height="22"
          viewBox="0 0 24 24"
          style={{ transform: `rotate(${info?.northRad ?? 0}rad)` }}
        >
          <path d="M12 2 L16 20 L12 15 L8 20 Z" fill="#22d3ee" />
        </svg>
        <span className="font-mono text-[10px] text-slate-400">N</span>
      </div>

      <div className="absolute bottom-2 left-2 flex flex-col gap-0.5">
        <div
          className="h-1 border-x border-b border-slate-300 bg-slate-300/20"
          style={{ width: `${barPx}px` }}
        />
        <span className="font-mono text-[10px] text-slate-400">{barLabel}</span>
      </div>

      {pending.length > 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="space-y-2 rounded border border-slate-800 bg-slate-950/90 px-3 py-2 text-sm">
            {pending.map((a) => (
              <div key={a.label} className="pointer-events-auto flex items-center gap-2">
                {a.error ? (
                  <>
                    <span className="text-red-300">
                      {a.label} failed: {a.error}
                    </span>
                    <button
                      type="button"
                      onClick={() => onRetry(a.label)}
                      className="rounded border border-red-500/50 px-2 py-0.5 text-xs text-red-300 hover:bg-red-500/10"
                    >
                      Retry
                    </button>
                  </>
                ) : (
                  <span className="text-slate-400">
                    loading {a.label}… {Math.round(a.progress * 100)}%
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
