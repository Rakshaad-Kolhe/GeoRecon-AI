import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useRef } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, ApiError } from '../api/client'
import { LogPanel } from '../components/LogPanel'
import { Panel } from '../components/Panel'
import { ProgressBar } from '../components/ProgressBar'
import { ResultsTabs } from '../components/ResultsTabs'
import { StateBadge } from '../components/StateBadge'
import { Stepper } from '../components/Stepper'
import { elapsedSeconds, formatSeconds } from '../lib/format'
import { scaleInfoFor } from '../lib/scale'
import { formatCodeVersion, isStaleResult } from '../lib/version'

export function JobDetail() {
  const { id } = useParams<{ id: string }>()
  const logRef = useRef<HTMLDivElement>(null)
  const qc = useQueryClient()

  const { data: job, error, isLoading } = useQuery({
    queryKey: ['job', id],
    queryFn: () => api.getJob(id as string),
    enabled: !!id,
    refetchInterval: (q) => {
      const st = q.state.data?.state
      return st === 'queued' || st === 'running' ? 1500 : false
    },
  })

  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => api.getHealth(),
    staleTime: 5_000,
  })

  const cancelMut = useMutation({
    mutationFn: () => api.cancelJob(id as string),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['job', id] })
      void qc.invalidateQueries({ queryKey: ['jobs'] })
    },
  })

  if (isLoading) return <p className="text-sm text-slate-500">loading…</p>
  if (error) {
    const notFound = error instanceof ApiError && error.status === 404
    return (
      <Panel>
        <p className="text-sm text-red-400">
          {notFound ? `No job "${id}".` : (error as Error).message}
        </p>
        <Link to="/" className="mt-2 inline-block text-sm text-cyan-400">
          ← all jobs
        </Link>
      </Panel>
    )
  }
  if (!job) return null

  const active = job.state === 'queued' || job.state === 'running'
  const total = active
    ? elapsedSeconds(job.created_at)
    : elapsedSeconds(job.created_at, job.updated_at)
  const scale = scaleInfoFor(job)
  const unscaled = job.metrics?.stages?.georef !== undefined && !scale.georeferenced

  return (
    <div className="space-y-4">
      {unscaled && (
        <div className="rounded border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-300">
          {scale.reason}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <Link to="/" className="text-sm text-slate-500 hover:text-slate-300">
          ← jobs
        </Link>
        <h1 className="font-mono text-lg text-slate-100">{job.job_id}</h1>
        <StateBadge state={job.state} />
        {isStaleResult(job, health) && (
          <span
            className="rounded border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-xs text-amber-300"
            title={`job built with code_version ${formatCodeVersion(job.code_version)}, backend is now ${formatCodeVersion(health?.code_version)}`}
          >
            stale result
          </span>
        )}
        <span className="font-mono text-sm text-slate-400">
          {formatSeconds(total)} elapsed
        </span>
        {job.preset && (
          <span className="rounded border border-slate-800 px-2 py-0.5 text-xs text-slate-400">
            {job.preset}
          </span>
        )}
        {active && (
          <button
            type="button"
            disabled={cancelMut.isPending}
            onClick={() => {
              if (window.confirm('Cancel this job? Any completed stages are kept.')) {
                cancelMut.mutate()
              }
            }}
            className="ml-auto rounded border border-red-500/50 px-2 py-0.5 text-xs text-red-300 hover:bg-red-500/10 disabled:opacity-40"
          >
            {cancelMut.isPending ? 'cancelling…' : 'Cancel'}
          </button>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="lg:col-span-1">
          <Panel title="Stages">
            <Stepper job={job} />
          </Panel>
        </div>

        <div className="space-y-4 lg:col-span-2">
          <Panel title="Progress">
            <ProgressBar value={job.progress} />
            <p className="mt-2 font-mono text-xs text-slate-400">{job.message}</p>

            {job.warnings.length > 0 && (
              <ul className="mt-3 space-y-1">
                {job.warnings.map((w, i) => (
                  <li
                    key={i}
                    className="rounded border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-xs text-amber-300"
                  >
                    {w}
                  </li>
                ))}
              </ul>
            )}

            {job.state === 'failed' && (
              <div className="mt-3 rounded border border-red-500/40 bg-red-500/10 p-3">
                <p className="text-sm font-semibold text-red-300">
                  Failed at {job.stage}
                </p>
                <p className="mt-1 font-mono text-xs text-red-200/90">
                  {job.message}
                </p>
                <button
                  type="button"
                  onClick={() =>
                    logRef.current?.scrollIntoView({ behavior: 'smooth' })
                  }
                  className="mt-2 text-xs text-cyan-400 hover:underline"
                >
                  view log
                </button>
              </div>
            )}
          </Panel>

          <ResultsTabs job={job} />
        </div>
      </div>

      <div ref={logRef}>
        <LogPanel jobId={job.job_id} active={active} />
      </div>
    </div>
  )
}
