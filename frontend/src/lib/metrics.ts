import type { Metrics, StageMetrics, StageName, ValidateSummary } from '../api/types'

export function stageMetric(
  metrics: Metrics | undefined,
  stage: StageName,
): StageMetrics | undefined {
  return metrics?.stages?.[stage]
}

export function mnum(m: StageMetrics | undefined, key: string): number | undefined {
  const v = m?.[key]
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined
}

export function mstr(m: StageMetrics | undefined, key: string): string | undefined {
  const v = m?.[key]
  return typeof v === 'string' ? v : undefined
}

/** metrics.stages.validate.summary when the validate stage produced it. */
export function getValidateSummary(
  metrics: Metrics | undefined,
): ValidateSummary | undefined {
  const s = metrics?.stages?.validate?.summary
  return s && typeof s === 'object' ? (s as ValidateSummary) : undefined
}
