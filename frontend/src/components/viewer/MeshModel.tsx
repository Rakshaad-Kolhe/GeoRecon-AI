import type { ThreeEvent } from '@react-three/fiber'
import { useEffect, useLayoutEffect, useMemo } from 'react'
import * as THREE from 'three'
import { srgbToLinear, viridis } from '../../lib/colormaps'
import { usePlyGeometry } from '../../lib/loadPly'
import type { Vec3 } from '../../lib/measure'
import type { AssetLoadState } from './PointCloud'
import type { ColorMode, MaterialMode } from './types'

interface Props {
  url: string
  active: boolean
  retrySignal: number
  colorMode: ColorMode
  material: MaterialMode
  /** offset the mesh's depth slightly so points drawn at the same surface don't z-fight. */
  polygonOffset: boolean
  visible: boolean
  pickable: boolean
  onPick: (p: Vec3) => void
  onHover: (p: Vec3 | null) => void
  onDoubleClick?: (p: Vec3) => void
  onReady: () => void
  onLoadState: (s: AssetLoadState) => void
}

export function MeshModel({
  url,
  active,
  retrySignal,
  colorMode,
  material,
  polygonOffset,
  visible,
  pickable,
  onPick,
  onHover,
  onDoubleClick,
  onReady,
  onLoadState,
}: Props) {
  const state = usePlyGeometry(url, active, retrySignal)
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

  const palettes = useMemo(() => {
    if (!geom) return null
    if (!geom.attributes.normal) geom.computeVertexNormals()
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
        rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = srgbToLinear(0.75)
      }
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
    // mesh has no per-vertex confidence — reuse height ramp so the toggle never breaks
    return { rgb, height, conf: height }
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

  const handlers = {
    onClick: pickable
      ? (e: ThreeEvent<MouseEvent>) => {
          e.stopPropagation()
          onPick({ x: e.point.x, y: e.point.y, z: e.point.z })
        }
      : undefined,
    onDoubleClick: onDoubleClick
      ? (e: ThreeEvent<MouseEvent>) => {
          e.stopPropagation()
          onDoubleClick({ x: e.point.x, y: e.point.y, z: e.point.z })
        }
      : undefined,
    onPointerMove: pickable
      ? (e: ThreeEvent<PointerEvent>) => {
          e.stopPropagation()
          onHover({ x: e.point.x, y: e.point.y, z: e.point.z })
        }
      : undefined,
    onPointerOut: pickable ? () => onHover(null) : undefined,
  }

  return (
    <mesh visible={visible} {...handlers}>
      <primitive object={geom} attach="geometry" />
      {material === 'photo' ? (
        <meshBasicMaterial
          vertexColors
          polygonOffset={polygonOffset}
          polygonOffsetFactor={1}
          polygonOffsetUnits={1}
        />
      ) : (
        <meshLambertMaterial
          vertexColors
          polygonOffset={polygonOffset}
          polygonOffsetFactor={1}
          polygonOffsetUnits={1}
        />
      )}
    </mesh>
  )
}
