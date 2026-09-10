import { formatInt } from '../../lib/format'
import type { SceneInfo } from './SceneCanvas'

interface Props {
  points: number
  triangles: number
  info: SceneInfo | null
  loading: boolean
  error: string | null
}

function niceLength(raw: number): number {
  const p = 10 ** Math.floor(Math.log10(raw))
  const n = raw / p
  const m = n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10
  return m * p
}

export function SceneOverlay({ points, triangles, info, loading, error }: Props) {
  let barPx = 0
  let barLabel = '—'
  if (info && info.mppx > 0 && Number.isFinite(info.mppx)) {
    const len = niceLength(info.mppx * 90)
    barPx = Math.min(240, len / info.mppx)
    barLabel = len >= 1000 ? `${len / 1000} km` : `${len} m`
  }

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

      {(loading || error) && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div
            className={`rounded border px-3 py-2 text-sm ${
              error
                ? 'border-red-500/40 bg-red-500/10 text-red-300'
                : 'border-slate-800 bg-slate-950/90 text-slate-400'
            }`}
          >
            {error ? `viewer error: ${error}` : 'loading 3D model…'}
          </div>
        </div>
      )}
    </div>
  )
}
