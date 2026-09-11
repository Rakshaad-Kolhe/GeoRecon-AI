import type { JobDetail, ViewerMeta } from '../api/types'

export interface ScaleInfo {
  georeferenced: boolean
  /** never plain "m" for an unscaled job — always says so. */
  unitsLabel: string
  reason: string
}

/**
 * The backend doesn't emit meta.georeferenced/units/scale_source yet — those
 * are read first (forward-compatible once it does), falling back to the
 * georeferenced flag the georef stage already reports in metrics.json.
 */
export function scaleInfoFor(job: JobDetail, meta?: ViewerMeta | null): ScaleInfo {
  const georefStage = job.metrics?.stages?.georef as { georeferenced?: boolean } | undefined
  const georeferenced = meta?.georeferenced ?? georefStage?.georeferenced ?? true
  const unitsLabel = meta?.units ?? (georeferenced ? 'm' : 'rel. units')
  const reason = 'Not georeferenced — no GPS telemetry'
  return { georeferenced, unitsLabel, reason }
}
