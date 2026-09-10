import type { JobState } from '../api/types'

const STYLES: Record<JobState, string> = {
  queued: 'border-slate-700 bg-slate-800 text-slate-300',
  running: 'border-cyan-400/40 bg-cyan-400/10 text-cyan-300',
  done: 'border-cyan-400 bg-cyan-400 text-slate-950',
  failed: 'border-red-500/50 bg-red-500/15 text-red-300',
}

export function StateBadge({ state }: { state: JobState }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-xs font-semibold uppercase tracking-wide ${STYLES[state]}`}
    >
      {state === 'running' && (
        <span className="gr-pulse h-1.5 w-1.5 rounded-full bg-current" />
      )}
      {state}
    </span>
  )
}
