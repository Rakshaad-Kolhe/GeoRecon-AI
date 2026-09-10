import { OrbitControls } from '@react-three/drei'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Suspense, useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { ViewerMeta } from '../../api/types'
import type { MeasureTool, Measurement, Vec3 } from '../../lib/measure'
import { MeasureLayer } from './MeasureLayer'
import { MeshModel } from './MeshModel'
import { PointCloud } from './PointCloud'
import { Trajectory } from './Trajectory'
import type { ViewState } from './types'

export interface SceneInfo {
  mppx: number
  northRad: number
}

interface Props {
  meta: ViewerMeta
  cloudUrl: string
  meshUrl: string
  view: ViewState
  active: boolean
  measureTool: MeasureTool | null
  onPick: (p: Vec3) => void
  onHover: (p: Vec3 | null) => void
  hover: Vec3 | null
  draft: { tool: MeasureTool; points: Vec3[] } | null
  measurements: Measurement[]
  onSceneInfo: (info: SceneInfo) => void
  onReady: () => void
}

function bboxMetrics(meta: ViewerMeta) {
  const { min, max } = meta.bbox_enu
  const center = new THREE.Vector3(
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  )
  const diag = new THREE.Vector3(
    max[0] - min[0],
    max[1] - min[1],
    max[2] - min[2],
  ).length()
  return { center, diag }
}

function RaycasterConfig({ threshold }: { threshold: number }) {
  const raycaster = useThree((s) => s.raycaster)
  useEffect(() => {
    // shared scene raycaster — mutating params is the documented r3f approach
    // oxlint-disable-next-line react/immutability
    raycaster.params.Points = { threshold }
  }, [raycaster, threshold])
  return null
}

function SceneInfoReporter({ onSceneInfo }: { onSceneInfo: (i: SceneInfo) => void }) {
  const last = useRef(0)
  const prev = useRef<SceneInfo>({ mppx: 0, northRad: 0 })
  const a = useMemo(() => new THREE.Vector3(), [])
  const b = useMemo(() => new THREE.Vector3(), [])

  useFrame((state) => {
    const now = state.clock.elapsedTime
    if (now - last.current < 0.15) return
    last.current = now

    const cam = state.camera as THREE.PerspectiveCamera
    const controls = state.controls as unknown as { target?: THREE.Vector3 } | null
    const target = controls?.target ?? new THREE.Vector3()
    const dist = cam.position.distanceTo(target)
    const vfov = (cam.fov * Math.PI) / 180
    const mppx = (2 * Math.tan(vfov / 2) * dist) / state.size.height

    a.set(0, 0, 0).project(cam)
    b.set(0, 1, 0).project(cam)
    const northRad = Math.atan2(b.x - a.x, -(b.y - a.y))

    if (
      Math.abs(mppx - prev.current.mppx) > mppx * 0.02 ||
      Math.abs(northRad - prev.current.northRad) > 0.01
    ) {
      prev.current = { mppx, northRad }
      onSceneInfo({ mppx, northRad })
    }
  })
  return null
}

export function SceneCanvas({
  meta,
  cloudUrl,
  meshUrl,
  view,
  active,
  measureTool,
  onPick,
  onHover,
  hover,
  draft,
  measurements,
  onSceneInfo,
  onReady,
}: Props) {
  const { center, diag } = useMemo(() => bboxMetrics(meta), [meta])

  const camPos = useMemo(() => {
    const dir = new THREE.Vector3(0.55, -0.95, 0.7).normalize()
    return center.clone().add(dir.multiplyScalar(diag * 1.15))
  }, [center, diag])

  const markerRadius = Math.max(0.35, diag * 0.004)
  const pointThreshold = Math.max(0.3, 2 * meta.median_spacing_m)

  const meshVisible = view.display !== 'cloud'
  const cloudVisible = view.display !== 'mesh'
  const measuring = measureTool != null
  const meshPickable = measuring && meshVisible
  const cloudPickable = measuring && cloudVisible && !meshVisible

  return (
    <Canvas
      dpr={[1, 2]}
      frameloop={active ? 'always' : 'demand'}
      gl={{ antialias: true, powerPreference: 'high-performance' }}
      camera={{
        position: [camPos.x, camPos.y, camPos.z],
        up: [0, 0, 1],
        fov: 50,
        near: Math.max(0.1, diag / 1000),
        far: diag * 12,
      }}
      onCreated={(state) => {
        state.gl.setClearColor('#020617')
        state.camera.up.set(0, 0, 1)
      }}
    >
      <hemisphereLight args={[0xffffff, 0x30323a, 1.1]} />
      <directionalLight position={[diag, -diag, diag * 1.5]} intensity={1.3} />

      <RaycasterConfig threshold={pointThreshold} />
      <SceneInfoReporter onSceneInfo={onSceneInfo} />
      <OrbitControls makeDefault target={[center.x, center.y, center.z]} />

      <Suspense fallback={null}>
        <PointCloud
          url={cloudUrl}
          colorMode={view.colorMode}
          size={view.pointSize}
          visible={cloudVisible}
          pickable={cloudPickable}
          onPick={onPick}
          onHover={onHover}
          onReady={onReady}
        />
        <MeshModel
          url={meshUrl}
          colorMode={view.colorMode}
          visible={meshVisible}
          pickable={meshPickable}
          onPick={onPick}
          onHover={onHover}
          onReady={onReady}
        />
      </Suspense>

      <Trajectory
        points={meta.trajectory_enu}
        visible={view.showTrajectory}
        dotRadius={markerRadius * 1.3}
      />

      <MeasureLayer
        measurements={measurements}
        draft={draft}
        hover={measuring ? hover : null}
        markerRadius={markerRadius}
      />
    </Canvas>
  )
}
