import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef } from 'react'
import { api } from '../api/client'
import { Panel } from './Panel'

export function LogPanel({
  jobId,
  active,
  tail = 200,
}: {
  jobId: string
  active: boolean
  tail?: number
}) {
  const { data } = useQuery({
    queryKey: ['log', jobId],
    queryFn: () => api.getLog(jobId, tail),
    refetchInterval: active ? 1500 : false,
  })

  const preRef = useRef<HTMLPreElement>(null)
  useEffect(() => {
    const el = preRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [data])

  return (
    <Panel
      title="Log"
      right={<span className="font-mono text-xs text-slate-600">tail {tail}</span>}
    >
      <pre
        ref={preRef}
        className="max-h-72 overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-slate-400"
      >
        {data || 'no log output yet'}
      </pre>
    </Panel>
  )
}
