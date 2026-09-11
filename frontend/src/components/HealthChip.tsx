import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { formatCodeVersion } from '../lib/version'

export function HealthChip() {
  const { data, isError } = useQuery({
    queryKey: ['health'],
    queryFn: () => api.getHealth(),
    refetchInterval: 10_000,
    staleTime: 5_000,
  })

  if (isError || !data) {
    return (
      <span className="rounded border border-red-500/50 bg-red-500/15 px-2 py-0.5 font-mono text-xs text-red-300">
        API unreachable
      </span>
    )
  }

  const colmapOk = data.colmap.available && data.colmap.cuda
  return (
    <span className="flex items-center gap-2 rounded border border-slate-800 bg-slate-950 px-2 py-0.5 font-mono text-xs text-slate-400">
      <span className="truncate text-slate-300" title={data.gpu_name}>
        {data.gpu_name || 'no GPU'}
      </span>
      <span className={colmapOk ? 'text-cyan-400' : 'text-amber-400'}>
        COLMAP CUDA {colmapOk ? '✓' : '✗'}
      </span>
      <span>queue {data.queue_len}</span>
      {data.code_version && (
        <span className="text-slate-600">v{formatCodeVersion(data.code_version)}</span>
      )}
    </span>
  )
}
