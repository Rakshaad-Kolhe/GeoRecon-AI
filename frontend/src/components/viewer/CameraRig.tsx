import { OrbitControls } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import { useCallback, useEffect, useRef } from 'react'
import * as THREE from 'three'
import type { CameraApi } from './types'

export interface CameraRigApi extends CameraApi {
  pivotTo: (p: THREE.Vector3) => void
}

interface Props {
  groundTarget: THREE.Vector3
  defaultPos: THREE.Vector3
  diag: number
  apiRef: React.RefObject<CameraRigApi | null>
}

const DURATION_S = 0.5

export function CameraRig({ groundTarget, defaultPos, diag, apiRef }: Props) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const controlsRef = useRef<any>(null)
  const { camera, invalidate } = useThree()
  const anim = useRef<{
    fromPos: THREE.Vector3
    toPos: THREE.Vector3
    fromTarget: THREE.Vector3
    toTarget: THREE.Vector3
    t: number
  } | null>(null)

  const animateTo = useCallback(
    (toPos: THREE.Vector3, toTarget: THREE.Vector3) => {
      const controls = controlsRef.current
      if (!controls) return
      anim.current = {
        fromPos: camera.position.clone(),
        toPos,
        fromTarget: controls.target.clone(),
        toTarget,
        t: 0,
      }
      invalidate()
    },
    [camera, invalidate],
  )

  useFrame((_, delta) => {
    const a = anim.current
    const controls = controlsRef.current
    if (!a || !controls) return
    a.t = Math.min(1, a.t + delta / DURATION_S)
    const e = 1 - (1 - a.t) ** 3 // ease-out cubic
    camera.position.lerpVectors(a.fromPos, a.toPos, e)
    controls.target.lerpVectors(a.fromTarget, a.toTarget, e)
    controls.update()
    if (a.t < 1) invalidate()
    else anim.current = null
  })

  useEffect(() => {
    const dist = defaultPos.distanceTo(groundTarget)

    const topPos = groundTarget
      .clone()
      .add(new THREE.Vector3(0.001, 0.001, 1).normalize().multiplyScalar(dist))
    const obliquePos = groundTarget
      .clone()
      .add(new THREE.Vector3(1, -1, Math.SQRT2).normalize().multiplyScalar(dist))

    apiRef.current = {
      reset: () => animateTo(defaultPos.clone(), groundTarget.clone()),
      top: () => animateTo(topPos, groundTarget.clone()),
      oblique: () => animateTo(obliquePos, groundTarget.clone()),
      pivotTo: (p: THREE.Vector3) => {
        const controls = controlsRef.current
        if (!controls) return
        const offset = camera.position.clone().sub(controls.target)
        animateTo(p.clone().add(offset), p.clone())
      },
    }
  }, [defaultPos, groundTarget, animateTo, camera, apiRef])

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      target={[groundTarget.x, groundTarget.y, groundTarget.z]}
      enableDamping
      dampingFactor={0.08}
      zoomToCursor
      minDistance={Math.max(0.1, diag * 0.02)}
      maxDistance={diag * 5}
      maxPolarAngle={0.49 * Math.PI}
    />
  )
}
