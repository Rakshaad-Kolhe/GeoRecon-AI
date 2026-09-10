export function formatSeconds(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return '—'
  if (s < 1) return `${Math.round(s * 1000)} ms`
  if (s < 60) return `${s < 10 ? s.toFixed(1) : Math.round(s)} s`
  const m = Math.floor(s / 60)
  const rem = Math.round(s % 60)
  if (m < 60) return `${m}m ${String(rem).padStart(2, '0')}s`
  const h = Math.floor(m / 60)
  return `${h}h ${String(m % 60).padStart(2, '0')}m`
}

export function formatMeters(m: number | null | undefined, digits = 2): string {
  if (m == null || !Number.isFinite(m)) return '—'
  return `${m.toFixed(digits)} m`
}

export function formatInt(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—'
  return Math.round(n).toLocaleString('en-US')
}

export function formatPct(frac: number | null | undefined, digits = 1): string {
  if (frac == null || !Number.isFinite(frac)) return '—'
  return `${frac.toFixed(digits)}%`
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes)) return '—'
  const units = ['B', 'KB', 'MB', 'GB']
  let v = bytes
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v < 10 && i > 0 ? v.toFixed(1) : Math.round(v)} ${units[i]}`
}

export function relativeTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  const t = Date.parse(iso)
  if (Number.isNaN(t)) return '—'
  const s = Math.round((Date.now() - t) / 1000)
  const a = Math.abs(s)
  const suffix = s >= 0 ? 'ago' : 'from now'
  if (a < 5) return 'just now'
  if (a < 60) return `${a}s ${suffix}`
  if (a < 3600) return `${Math.floor(a / 60)}m ${suffix}`
  if (a < 86400) return `${Math.floor(a / 3600)}h ${suffix}`
  return `${Math.floor(a / 86400)}d ${suffix}`
}

/** Wall-clock elapsed between two ISO timestamps, in seconds. */
export function elapsedSeconds(startIso: string, endIso?: string): number | null {
  const a = Date.parse(startIso)
  const b = endIso ? Date.parse(endIso) : Date.now()
  if (Number.isNaN(a) || Number.isNaN(b)) return null
  return Math.max(0, (b - a) / 1000)
}
