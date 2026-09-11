import type { HealthInfo, JobDetail } from '../api/types'

export function formatCodeVersion(v: { commit: string; dirty: boolean } | undefined): string {
  if (!v) return ''
  return v.dirty ? `${v.commit}*` : v.commit
}

/** true only once both sides report a code_version and the commits disagree. */
export function isStaleResult(job: JobDetail, health: HealthInfo | undefined): boolean {
  return (
    !!job.code_version && !!health?.code_version && job.code_version.commit !== health.code_version.commit
  )
}
