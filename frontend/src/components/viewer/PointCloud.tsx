import type { ThreeEvent } from '@react-three/fiber'
import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { srgbToLinear, turbo, viridis } from '../../lib/colormaps'
import { usePlyGeometry } from '../../lib/loadPly'
import type { Vec3 } from '../../lib/measure'
import type { ColorMode } from './types'

export interface AssetLoadState {
  progress: number
  error: string | null
}

interface Props {
  url: string
  active: boolean
  retrySignal: number
  colorMode: ColorMode
  /** world-space point diameter (1.5 x median_spacing_m x slider multiplier). */
  worldSize: number
  visible: boolean
  pickable: boolean
  onPick: (p: Vec3) => void
  onHover: (p: Vec3 | null) => void
  onDoubleClick?: (p: Vec3) => void
  onReady: () => void
  onLoadState: (s: AssetLoadState) => void
}

// discard fragments outside the point's inscribed circle → round points instead
// of the default square sprite.
function roundPoints(shader: { fragmentShader: string }) {
  shader.fragmentShader = shader.fragmentShader.replace(
    '#include <clipping_planes_fragment>',
    `#include <clipping_planes_fragment>
    vec2 pc = gl_PointCoord - vec2( 0.5 );
    if ( dot( pc, pc ) > 0.25 ) discard;`,
  )
}

export function PointCloud({
  url,
  active,
  retrySignal,
  colorMode,
  worldSize,
  visible,
  pickable,
  onPick,
  onHover,
  onDoubleClick,
  onReady,
  onLoadState,
}: Props) {
  const state = usePlyGeometry(url, active, retrySignal, { conf: ['conf'] })
  const geom = state.status === 'ready' ? state.geometry : null

  useEffect(() => {
    if (state.status === 'loading') onLoadState({ progress: state.progress, error: null })
    else if (state.status === 'error') onLoadState({ progress: 0, error: state.message })
    else if (state.status === 'ready') onLoadState({ progress: 1, error: null })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state])

  useEffect(() => {
    if (geom) onReady()
  }, [geom, onReady])

  const ref = useRef<THREE.Points>(null)

  const material = useMemo(() => {
    const m = new THREE.PointsMaterial({ vertexColors: true, sizeAttenuation: true })
    m.onBeforeCompile = roundPoints
    return m
  }, [])
  useEffect(() => () => material.dispose(), [material])
  useLayoutEffect(() => {
    material.size = worldSize
  }, [material, worldSize])

  const palettes = useMemo(() => {
    if (!geom) return null
    const n = geom.attributes.position.count
    const pos = geom.attributes.position.array as Float32Array

    const rgb = new Float32Array(n * 3)
    const src = geom.attributes.color
    // PLY colours are sRGB 0..1; convert to linear so the renderer's own
    // linear->sRGB output pass doesn't double-encode them (128 must stay ~128).
    for (let i = 0; i < n; i++) {
      if (src) {
        rgb[i * 3] = srgbToLinear(src.getX(i))
        rgb[i * 3 + 1] = srgbToLinear(src.getY(i))
        rgb[i * 3 + 2] = srgbToLinear(src.getZ(i))
      } else {
        rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = srgbToLinear(0.8)
      }
    }

    const confAttr = geom.attributes.conf as THREE.BufferAttribute | undefined
    const conf = new Float32Array(n * 3)
    for (let i = 0; i < n; i++) {
      let c = confAttr ? confAttr.getX(i) : 0.5 + 0.45 * Math.sin(i * 0.37)
      if (c > 1) c /= 255
      const [r, g, b] = turbo(c)
      conf[i * 3] = r
      conf[i * 3 + 1] = g
      conf[i * 3 + 2] = b
    }

    let zmin = Infinity
    let zmax = -Infinity
    for (let i = 0; i < n; i++) {
      const z = pos[i * 3 + 2]
      if (z < zmin) zmin = z
      if (z > zmax) zmax = z
    }
    const span = zmax - zmin || 1
    const height = new Float32Array(n * 3)
    for (let i = 0; i < n; i++) {
      const [r, g, b] = viridis((pos[i * 3 + 2] - zmin) / span)
      height[i * 3] = r
      height[i * 3 + 1] = g
      height[i * 3 + 2] = b
    }

    return { rgb, conf, height }
  }, [geom])

  useLayoutEffect(() => {
    if (!geom || !palettes) return
    const arr =
      colorMode === 'rgb'
        ? palettes.rgb
        : colorMode === 'conf'
          ? palettes.conf
          : palettes.height
    // fresh BufferAttribute each time → three re-uploads without a needsUpdate flag
    geom.setAttribute('color', new THREE.BufferAttribute(arr, 3))
  }, [geom, palettes, colorMode])

  if (!geom) return null

  return (
    <points
      ref={ref}
      visible={visible}
      frustumCulled={false}
      onClick={
        pickable
          ? (e: ThreeEvent<MouseEvent>) => {
              e.stopPropagation()
              onPick({ x: e.point.x, y: e.point.y, z: e.point.z })
            }
          : undefined
      }
      onDoubleClick={
        onDoubleClick
          ? (e: ThreeEvent<MouseEvent>) => {
              e.stopPropagation()
              onDoubleClick({ x: e.point.x, y: e.point.y, z: e.point.z })
            }
          : undefined
      }
      onPointerMove={
        pickable
          ? (e: ThreeEvent<PointerEvent>) => {
              e.stopPropagation()
              onHover({ x: e.point.x, y: e.point.y, z: e.point.z })
            }
          : undefined
      }
      onPointerOut={pickable ? () => onHover(null) : undefined}
    >
      <primitive object={geom} attach="geometry" />
      <primitive object={material} attach="material" />
    </points>
  )
}
