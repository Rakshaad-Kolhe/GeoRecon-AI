// Small LUT colormaps (0..1 in, [r,g,b] LINEAR 0..1 out) for point-cloud colour
// modes. The stops below are authored in sRGB; we return linear so the values
// can be written straight into a three vertex-colour attribute without looking
// dark — equivalent to `new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace)`.

type Stop = [number, number, number] // 0..255, sRGB

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

// three.js SRGBToLinear (ColorManagement) — matches Color.setRGB(..., SRGBColorSpace)
export function srgbToLinear(c: number): number {
  return c < 0.04045
    ? c * 0.0773993808
    : Math.pow(c * 0.9478672986 + 0.0521327014, 2.4)
}

function sample(lut: Stop[], t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0)) * (lut.length - 1)
  const i = Math.floor(x)
  const j = Math.min(i + 1, lut.length - 1)
  const f = x - i
  const a = lut[i]
  const b = lut[j]
  return [
    srgbToLinear((a[0] + (b[0] - a[0]) * f) / 255),
    srgbToLinear((a[1] + (b[1] - a[1]) * f) / 255),
    srgbToLinear((a[2] + (b[2] - a[2]) * f) / 255),
  ]
}

export const turbo = (t: number) => sample(TURBO, t)
export const viridis = (t: number) => sample(VIRIDIS, t)
