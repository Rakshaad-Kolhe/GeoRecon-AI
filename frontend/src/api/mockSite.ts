// Procedural stand-in for backend PR 08 web assets: a ~300x250 m site with a
// noisy heightfield, a few box buildings, a road strip, a coloured point cloud
// with varying confidence, a matching mesh, and a zig-zag flight path.
// Everything is emitted as real binary PLY so the viewer's PLYLoader path is
// exercised even with no files on disk.

import type { ViewerMeta } from './types'

export interface MockSite {
  meta: ViewerMeta
  cloudPLY: ArrayBuffer
  meshPLY: ArrayBuffer
}

const SITE_E = 300
const SITE_N = 250
const ORIGIN = { lat: 18.5204, lon: 73.8567, alt: 560 }

export function hashSeed(s: string): number {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

interface Building {
  e0: number
  e1: number
  n0: number
  n1: number
  h: number
}

interface SiteModel {
  height: (e: number, n: number) => { z: number; kind: 'terrain' | 'road' | 'building'; bh: number }
  buildings: Building[]
  maxZ: number
}

function buildModel(rng: () => number): SiteModel {
  // coarse value-noise grid, bilinearly sampled
  const gx = 24
  const gy = 20
  const grid = new Float32Array(gx * gy)
  for (let i = 0; i < grid.length; i++) grid[i] = rng()

  const noise = (e: number, n: number) => {
    const fx = (e / SITE_E) * (gx - 1)
    const fy = (n / SITE_N) * (gy - 1)
    const x0 = Math.floor(fx)
    const y0 = Math.floor(fy)
    const x1 = Math.min(x0 + 1, gx - 1)
    const y1 = Math.min(y0 + 1, gy - 1)
    const tx = fx - x0
    const ty = fy - y0
    const a = grid[y0 * gx + x0]
    const b = grid[y0 * gx + x1]
    const c = grid[y1 * gx + x0]
    const d = grid[y1 * gx + x1]
    return (
      a * (1 - tx) * (1 - ty) +
      b * tx * (1 - ty) +
      c * (1 - tx) * ty +
      d * tx * ty
    )
  }

  const terrain = (e: number, n: number) => {
    const rolling =
      3.0 * Math.sin(e * 0.018 + 1.3) +
      2.2 * Math.cos(n * 0.021) +
      1.4 * Math.sin((e + n) * 0.03)
    return 0.02 * e + 0.015 * n + rolling + 6 * (noise(e, n) - 0.5)
  }

  const buildings: Building[] = [
    { e0: 60, e1: 96, n0: 40, n1: 74, h: 18 },
    { e0: 120, e1: 150, n0: 150, n1: 186, h: 27 },
    { e0: 200, e1: 232, n0: 60, n1: 92, h: 12 },
    { e0: 168, e1: 190, n0: 30, n1: 52, h: 9 },
    { e0: 232, e1: 262, n0: 170, n1: 205, h: 21 },
  ]

  const height = (e: number, n: number) => {
    const base = terrain(e, n)
    for (const b of buildings) {
      if (e >= b.e0 && e <= b.e1 && n >= b.n0 && n <= b.n1) {
        return { z: base + b.h, kind: 'building' as const, bh: b.h }
      }
    }
    if (n >= 118 && n <= 130) {
      return { z: base - 0.4, kind: 'road' as const, bh: 0 }
    }
    return { z: base, kind: 'terrain' as const, bh: 0 }
  }

  let maxZ = -Infinity
  for (let e = 0; e <= SITE_E; e += 10)
    for (let n = 0; n <= SITE_N; n += 10) maxZ = Math.max(maxZ, height(e, n).z)

  return { height, buildings, maxZ }
}

function colourFor(
  kind: 'terrain' | 'road' | 'building',
  zFrac: number,
  jitter: number,
): [number, number, number] {
  if (kind === 'road') return [66 + jitter * 8, 68 + jitter * 8, 74 + jitter * 8]
  if (kind === 'building') {
    const v = 168 + zFrac * 40 + jitter * 12
    return [v, v - 4, v - 12]
  }
  // terrain: grass -> dirt -> rock by height
  const grass: [number, number, number] = [104, 128, 74]
  const dirt: [number, number, number] = [132, 110, 78]
  const rock: [number, number, number] = [150, 146, 138]
  const mix = (a: [number, number, number], b: [number, number, number], t: number) =>
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t] as [
      number,
      number,
      number,
    ]
  const c = zFrac < 0.5 ? mix(grass, dirt, zFrac * 2) : mix(dirt, rock, (zFrac - 0.5) * 2)
  return [c[0] + jitter * 14, c[1] + jitter * 14, c[2] + jitter * 14]
}

const clampByte = (v: number) => Math.max(0, Math.min(255, Math.round(v)))

function buildTrajectory(maxZ: number): [number, number, number][] {
  const u = maxZ + 45
  const pts: [number, number, number][] = []
  const passes = 6
  for (let k = 0; k < passes; k++) {
    const n = 22 + (k * (SITE_N - 44)) / (passes - 1)
    const forward = k % 2 === 0
    const steps = 9
    for (let s = 0; s <= steps; s++) {
      const f = s / steps
      const e = 16 + (forward ? f : 1 - f) * (SITE_E - 32)
      pts.push([e, n, u + Math.sin(f * Math.PI) * 2])
    }
  }
  return pts
}

// ---- PLY encoders ----------------------------------------------------------

function withHeader(header: string, body: ArrayBuffer): ArrayBuffer {
  const head = new TextEncoder().encode(header)
  const out = new Uint8Array(head.length + body.byteLength)
  out.set(head, 0)
  out.set(new Uint8Array(body), head.length)
  return out.buffer
}

function encodeCloudPLY(
  pos: Float32Array,
  col: Uint8Array,
  conf: Uint8Array,
): ArrayBuffer {
  const n = conf.length
  const header =
    'ply\nformat binary_little_endian 1.0\n' +
    `element vertex ${n}\n` +
    'property float x\nproperty float y\nproperty float z\n' +
    'property uchar red\nproperty uchar green\nproperty uchar blue\n' +
    'property uchar conf\nend_header\n'
  const body = new ArrayBuffer(n * 16)
  const dv = new DataView(body)
  for (let i = 0; i < n; i++) {
    const o = i * 16
    dv.setFloat32(o, pos[i * 3], true)
    dv.setFloat32(o + 4, pos[i * 3 + 1], true)
    dv.setFloat32(o + 8, pos[i * 3 + 2], true)
    dv.setUint8(o + 12, col[i * 3])
    dv.setUint8(o + 13, col[i * 3 + 1])
    dv.setUint8(o + 14, col[i * 3 + 2])
    dv.setUint8(o + 15, conf[i])
  }
  return withHeader(header, body)
}

function encodeMeshPLY(
  pos: Float32Array,
  col: Uint8Array,
  idx: Uint32Array,
): ArrayBuffer {
  const v = pos.length / 3
  const f = idx.length / 3
  const header =
    'ply\nformat binary_little_endian 1.0\n' +
    `element vertex ${v}\n` +
    'property float x\nproperty float y\nproperty float z\n' +
    'property uchar red\nproperty uchar green\nproperty uchar blue\n' +
    `element face ${f}\n` +
    'property list uchar uint vertex_indices\nend_header\n'
  const body = new ArrayBuffer(v * 15 + f * 13)
  const dv = new DataView(body)
  for (let i = 0; i < v; i++) {
    const o = i * 15
    dv.setFloat32(o, pos[i * 3], true)
    dv.setFloat32(o + 4, pos[i * 3 + 1], true)
    dv.setFloat32(o + 8, pos[i * 3 + 2], true)
    dv.setUint8(o + 12, col[i * 3])
    dv.setUint8(o + 13, col[i * 3 + 1])
    dv.setUint8(o + 14, col[i * 3 + 2])
  }
  let fo = v * 15
  for (let i = 0; i < f; i++) {
    dv.setUint8(fo, 3)
    dv.setUint32(fo + 1, idx[i * 3], true)
    dv.setUint32(fo + 5, idx[i * 3 + 1], true)
    dv.setUint32(fo + 9, idx[i * 3 + 2], true)
    fo += 13
  }
  return withHeader(header, body)
}

// ---- main ----------------------------------------------------------------

export function generateMockSite(seed: number): MockSite {
  const rng = mulberry32(seed)
  const model = buildModel(rng)

  // point cloud
  const nx = 520
  const ny = 390
  const count = nx * ny
  const pos = new Float32Array(count * 3)
  const col = new Uint8Array(count * 3)
  const conf = new Uint8Array(count)
  const bmin: [number, number, number] = [Infinity, Infinity, Infinity]
  const bmax: [number, number, number] = [-Infinity, -Infinity, -Infinity]
  const zSpan = model.maxZ + 30

  let p = 0
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++, p++) {
      const e = (i / (nx - 1)) * SITE_E
      const n = (j / (ny - 1)) * SITE_N
      const h = model.height(e, n)
      const z = h.z + (rng() - 0.5) * 0.06
      pos[p * 3] = e
      pos[p * 3 + 1] = n
      pos[p * 3 + 2] = z
      if (e < bmin[0]) bmin[0] = e
      if (n < bmin[1]) bmin[1] = n
      if (z < bmin[2]) bmin[2] = z
      if (e > bmax[0]) bmax[0] = e
      if (n > bmax[1]) bmax[1] = n
      if (z > bmax[2]) bmax[2] = z

      const zFrac = Math.max(0, Math.min(1, z / zSpan))
      const c = colourFor(h.kind, zFrac, rng() - 0.5)
      col[p * 3] = clampByte(c[0])
      col[p * 3 + 1] = clampByte(c[1])
      col[p * 3 + 2] = clampByte(c[2])

      const edge = Math.min(e, SITE_E - e, n, SITE_N - n)
      let cf = 0.92 - (h.kind === 'building' ? 0.15 : 0)
      if (edge < 14) cf -= (14 - edge) * 0.035
      cf += (rng() - 0.5) * 0.16
      conf[p] = clampByte(Math.max(0.05, Math.min(1, cf)) * 255)
    }
  }

  // mesh (coarser grid)
  const mx = 150
  const my = 125
  const mpos = new Float32Array(mx * my * 3)
  const mcol = new Uint8Array(mx * my * 3)
  for (let j = 0; j < my; j++) {
    for (let i = 0; i < mx; i++) {
      const k = j * mx + i
      const e = (i / (mx - 1)) * SITE_E
      const n = (j / (my - 1)) * SITE_N
      const h = model.height(e, n)
      mpos[k * 3] = e
      mpos[k * 3 + 1] = n
      mpos[k * 3 + 2] = h.z
      const zFrac = Math.max(0, Math.min(1, h.z / zSpan))
      const c = colourFor(h.kind, zFrac, 0)
      mcol[k * 3] = clampByte(c[0])
      mcol[k * 3 + 1] = clampByte(c[1])
      mcol[k * 3 + 2] = clampByte(c[2])
    }
  }
  const idx = new Uint32Array((mx - 1) * (my - 1) * 6)
  let t = 0
  for (let j = 0; j < my - 1; j++) {
    for (let i = 0; i < mx - 1; i++) {
      const a = j * mx + i
      const b = a + 1
      const c = a + mx
      const d = c + 1
      idx[t++] = a
      idx[t++] = c
      idx[t++] = b
      idx[t++] = b
      idx[t++] = c
      idx[t++] = d
    }
  }

  const trajectory = buildTrajectory(model.maxZ)

  const meta: ViewerMeta = {
    origin: { ...ORIGIN },
    utm_epsg: 32643,
    bbox_enu: { min: bmin, max: bmax },
    points: count,
    triangles: idx.length / 3,
    median_spacing_m: SITE_E / (nx - 1),
    trajectory_enu: trajectory,
    height_ref: 'ellipsoidal',
  }

  return {
    meta,
    cloudPLY: encodeCloudPLY(pos, col, conf),
    meshPLY: encodeMeshPLY(mpos, mcol, idx),
  }
}
