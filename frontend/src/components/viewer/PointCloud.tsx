import { useLoader } from '@react-three/fiber'
import type { ThreeEvent } from '@react-three/fiber'
import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'
import { turbo, viridis } from '../../lib/colormaps'
import type { Vec3 } from '../../lib/measure'
import type { ColorMode } from './types'

interface Props {
  url: string
  colorMode: ColorMode
  size: number
  visible: boolean
  pickable: boolean
  onPick: (p: Vec3) => void
  onHover: (p: Vec3 | null) => void
  onReady: () => void
}

export function PointCloud({
  url,
  colorMode,
  size,
  visible,
  pickable,
  onPick,
  onHover,
  onReady,
}: Props) {
  const geom = useLoader(PLYLoader, url, (loader) => {
    loader.setCustomPropertyNameMapping({ conf: ['conf'] })
  }) as THREE.BufferGeometry

  const ref = useRef<THREE.Points>(null)

  const palettes = useMemo(() => {
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
        rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = 0.8
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
      <pointsMaterial vertexColors size={size} sizeAttenuation={false} />
    </points>
  )
}
