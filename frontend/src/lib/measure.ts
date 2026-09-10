import type { ViewerMeta } from '../api/types'

export type Vec3 = { x: number; y: number; z: number }
export type MeasureTool = 'distance' | 'area' | 'height' | 'inspect'

export interface Measurement {
  id: string
  tool: MeasureTool
  points: Vec3[]
  label: string // one-line summary for the sidebar
  detail: [string, string][] // label / formatted value, for copy-to-clipboard
}

export const TOOL_HINT: Record<MeasureTool, string> = {
  distance: 'click 2 points — 3D, horizontal and ΔZ',
  area: 'click vertices, then Enter or double-click to close',
  height: 'click 2 points — vertical difference',
  inspect: 'click a point — ENU and lat / lon / alt',
}

export const TOOL_POINTS: Record<MeasureTool, number> = {
  distance: 2,
  height: 2,
  inspect: 1,
  area: 0, // variable, closed explicitly
}

const R_EARTH = 6378137

const d3 = (a: Vec3, b: Vec3) =>
  Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z)
const dXY = (a: Vec3, b: Vec3) => Math.hypot(b.x - a.x, b.y - a.y)

function polygonAreaXY(pts: Vec3[]): number {
  let s = 0
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i]
    const b = pts[(i + 1) % pts.length]
    s += a.x * b.y - b.x * a.y
  }
  return Math.abs(s) / 2
}

function perimeterXY(pts: Vec3[]): number {
  let s = 0
  for (let i = 0; i < pts.length; i++) s += dXY(pts[i], pts[(i + 1) % pts.length])
  return s
}

export function enuToGeodetic(
  p: Vec3,
  origin: ViewerMeta['origin'],
): { lat: number; lon: number; alt: number } {
  const lat = origin.lat + (p.y / R_EARTH) * (180 / Math.PI)
  const lon =
    origin.lon +
    (p.x / (R_EARTH * Math.cos((origin.lat * Math.PI) / 180))) * (180 / Math.PI)
  return { lat, lon, alt: origin.alt + p.z }
}

/** The four bbox_enu XY corners as [lat, lon] (same tangent approx as Inspect). */
export function bboxFootprintLatLng(meta: ViewerMeta): [number, number][] {
  const { min, max } = meta.bbox_enu
  const z = (min[2] + max[2]) / 2
  const corners: Vec3[] = [
    { x: min[0], y: min[1], z },
    { x: max[0], y: min[1], z },
    { x: max[0], y: max[1], z },
    { x: min[0], y: max[1], z },
  ]
  return corners.map((c) => {
    const g = enuToGeodetic(c, meta.origin)
    return [g.lat, g.lon]
  })
}

const m = (v: number) => `${v.toFixed(3)} m`

export function buildMeasurement(
  tool: MeasureTool,
  points: Vec3[],
  meta: ViewerMeta,
): Measurement {
  const id = `${tool}-${Date.now().toString(36)}-${Math.floor(Math.random() * 1e4)}`
  let label = ''
  let detail: [string, string][] = []

  if (tool === 'distance') {
    const [a, b] = points
    const total = d3(a, b)
    detail = [
      ['3D distance', m(total)],
      ['Horizontal', m(dXY(a, b))],
      ['ΔZ', m(b.z - a.z)],
    ]
    label = `Distance ${total.toFixed(2)} m`
  } else if (tool === 'height') {
    const [a, b] = points
    const dz = Math.abs(b.z - a.z)
    detail = [
      ['Height difference', m(dz)],
      ['Z (a)', m(a.z)],
      ['Z (b)', m(b.z)],
    ]
    label = `Height ${dz.toFixed(2)} m`
  } else if (tool === 'area') {
    const area = polygonAreaXY(points)
    detail = [
      ['Area (XY)', `${area.toFixed(2)} m²`],
      ['Perimeter', m(perimeterXY(points))],
      ['Vertices', String(points.length)],
    ]
    label = `Area ${area.toFixed(1)} m²`
  } else {
    const p = points[0]
    const g = enuToGeodetic(p, meta.origin)
    detail = [
      ['E / N / U', `${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)} m`],
      ['Latitude', g.lat.toFixed(7)],
      ['Longitude', g.lon.toFixed(7)],
      ['Altitude', `${g.alt.toFixed(2)} m (${meta.height_ref})`],
    ]
    label = `Point ${p.x.toFixed(1)}, ${p.y.toFixed(1)}, ${p.z.toFixed(1)}`
  }

  return { id, tool, points, label, detail }
}

export function centroidXY(pts: Vec3[]): Vec3 {
  const c = pts.reduce(
    (acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y, z: acc.z + p.z }),
    { x: 0, y: 0, z: 0 },
  )
  const n = pts.length || 1
  return { x: c.x / n, y: c.y / n, z: c.z / n }
}

export function midpoint(a: Vec3, b: Vec3): Vec3 {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 }
}
