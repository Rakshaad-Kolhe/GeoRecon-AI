// Small LUT colormaps (0..1 in, [r,g,b] 0..1 out) for point-cloud colour modes.

type Stop = [number, number, number] // 0..255

const TURBO: Stop[] = [
  [48, 18, 59],
  [65, 69, 171],
  [57, 131, 228],
  [30, 187, 215],
  [54, 224, 152],
  [147, 246, 79],
  [224, 219, 49],
  [245, 133, 44],
  [122, 4, 3],
]

const VIRIDIS: Stop[] = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 74, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [109, 205, 89],
  [253, 231, 37],
]

function sample(lut: Stop[], t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0)) * (lut.length - 1)
  const i = Math.floor(x)
  const j = Math.min(i + 1, lut.length - 1)
  const f = x - i
  const a = lut[i]
  const b = lut[j]
  return [
    (a[0] + (b[0] - a[0]) * f) / 255,
    (a[1] + (b[1] - a[1]) * f) / 255,
    (a[2] + (b[2] - a[2]) * f) / 255,
  ]
}

export const turbo = (t: number) => sample(TURBO, t)
export const viridis = (t: number) => sample(VIRIDIS, t)
