import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { Panel } from '../components/Panel'
import { ProgressBar } from '../components/ProgressBar'
import { StateBadge } from '../components/StateBadge'
import { elapsedSeconds, formatSeconds, relativeTime } from '../lib/format'

export function JobsList() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['jobs'],
    queryFn: () => api.listJobs(),
    refetchInterval: 3000,
  })

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold text-slate-100">Jobs</h1>
        <span className="text-xs text-slate-600">auto-refresh 3 s</span>
      </div>

      <Panel>
        {isLoading ? (
          <p className="text-sm text-slate-500">loading…</p>
        ) : error ? (
          <p className="text-sm text-red-400">
            failed to load jobs: {(error as Error).message}
          </p>
        ) : !data || data.length === 0 ? (
          <p className="text-sm text-slate-500">
            no jobs yet — <Link to="/new" className="text-cyan-400">create one</Link>
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-xs uppercase tracking-wide text-slate-500">
                <tr className="border-b border-slate-800">
                  <th className="py-2 pr-3 font-medium">Job</th>
                  <th className="py-2 pr-3 font-medium">State</th>
                  <th className="py-2 pr-3 font-medium">Stage</th>
                  <th className="py-2 pr-3 font-medium">Progress</th>
                  <th className="py-2 pr-3 font-medium">Created</th>
                  <th className="py-2 pr-3 font-medium">Updated</th>
                  <th className="py-2 font-medium">Total</th>
                </tr>
              </thead>
              <tbody>
                {data.map((job) => {
                  const running = job.state === 'queued' || job.state === 'running'
                  const total = running
                    ? elapsedSeconds(job.created_at)
                    : elapsedSeconds(job.created_at, job.updated_at)
                  return (
                    <tr
                      key={job.job_id}
                      className="border-b border-slate-800/60 hover:bg-slate-800/30"
                    >
                      <td className="py-2 pr-3">
                        <Link
                          to={`/jobs/${job.job_id}`}
                          className="font-mono text-cyan-400 hover:underline"
                        >
                          {job.job_id}
                        </Link>
                      </td>
                      <td className="py-2 pr-3">
                        <StateBadge state={job.state} />
                      </td>
                      <td className="py-2 pr-3 text-slate-300">{job.stage}</td>
                      <td className="py-2 pr-3">
                        <div className="flex items-center gap-2">
                          <ProgressBar value={job.progress} className="w-24" />
                          <span className="font-mono text-xs text-slate-500">
                            {Math.round((job.progress || 0) * 100)}%
                          </span>
                        </div>
                      </td>
                      <td className="py-2 pr-3 text-slate-400">
                        {relativeTime(job.created_at)}
                      </td>
                      <td className="py-2 pr-3 text-slate-400">
                        {relativeTime(job.updated_at)}
                      </td>
                      <td className="py-2 font-mono text-slate-400">
                        {formatSeconds(total)}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  )
}
