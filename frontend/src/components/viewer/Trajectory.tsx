import { Line } from '@react-three/drei'
import { useMemo } from 'react'
import * as THREE from 'three'

interface Props {
  points: [number, number, number][]
  visible: boolean
  dotRadius: number
}

export function Trajectory({ points, visible, dotRadius }: Props) {
  const pts = useMemo(
    () => points.map((p) => new THREE.Vector3(p[0], p[1], p[2])),
    [points],
  )

  if (!visible || pts.length < 2) return null

  return (
    <group>
      <Line points={pts} color="#f59e0b" lineWidth={1.5} />
      {pts.map((p, i) => (
        <mesh key={i} position={p}>
          <sphereGeometry args={[dotRadius, 8, 8]} />
          <meshBasicMaterial color="#f59e0b" />
        </mesh>
      ))}
    </group>
  )
}
