import { useLoader } from '@react-three/fiber'
import type { ThreeEvent } from '@react-three/fiber'
import { useEffect, useLayoutEffect, useMemo } from 'react'
import * as THREE from 'three'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'
import { viridis } from '../../lib/colormaps'
import type { Vec3 } from '../../lib/measure'
import type { ColorMode } from './types'

interface Props {
  url: string
  colorMode: ColorMode
  visible: boolean
  pickable: boolean
  onPick: (p: Vec3) => void
  onHover: (p: Vec3 | null) => void
  onReady: () => void
}

export function MeshModel({
  url,
  colorMode,
  visible,
  pickable,
  onPick,
  onHover,
  onReady,
}: Props) {
  const geom = useLoader(PLYLoader, url) as THREE.BufferGeometry

  const palettes = useMemo(() => {
    if (!geom.attributes.normal) geom.computeVertexNormals()
    const n = geom.attributes.position.count
    const pos = geom.attributes.position.array as Float32Array

    const rgb = new Float32Array(n * 3)
    const src = geom.attributes.color
    for (let i = 0; i < n; i++) {
      if (src) {
        rgb[i * 3] = src.getX(i)
        rgb[i * 3 + 1] = src.getY(i)
        rgb[i * 3 + 2] = src.getZ(i)
      } else {
        rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = 0.75
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
    const arr =
      colorMode === 'rgb'
        ? palettes.rgb
        : colorMode === 'conf'
          ? palettes.conf
          : palettes.height
    // fresh BufferAttribute each time → three re-uploads without a needsUpdate flag
    geom.setAttribute('color', new THREE.BufferAttribute(arr, 3))
  }, [geom, palettes, colorMode])

  useEffect(() => {
    onReady()
    return () => {
      geom.dispose()
      useLoader.clear(PLYLoader, url)
    }
  }, [geom, url, onReady])

  return (
    <mesh
      visible={visible}
      onClick={
        pickable
          ? (e: ThreeEvent<MouseEvent>) => {
              e.stopPropagation()
              onPick({ x: e.point.x, y: e.point.y, z: e.point.z })
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
      <meshStandardMaterial vertexColors roughness={0.95} metalness={0} />
    </mesh>
  )
}
