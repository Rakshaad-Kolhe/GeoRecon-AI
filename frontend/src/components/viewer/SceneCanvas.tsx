import { AdaptiveDpr, PerformanceMonitor, Stats } from '@react-three/drei'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { ViewerMeta } from '../../api/types'
import type { MeasureTool, Measurement, Vec3 } from '../../lib/measure'
import { CameraRig, type CameraRigApi } from './CameraRig'
import { MeasureLayer } from './MeasureLayer'
import { MeshModel } from './MeshModel'
import { PointCloud, type AssetLoadState } from './PointCloud'
import { Trajectory } from './Trajectory'
import type { BackgroundMode, CameraApi, ViewState } from './types'

export interface SceneInfo {
  mppx: number
  northRad: number
}

/** Maps BackgroundMode to canvas clear colours used for colour evaluation. */
const BG_COLORS: Record<BackgroundMode, string> = {
  dark: '#0f172a',
  grey: '#1e2126',
  light: '#f1f5f9',
}

interface Props {
  meta: ViewerMeta
  cloudUrl: string
  meshUrl: string
  view: ViewState
  active: boolean
  debug: boolean
  measureTool: MeasureTool | null
  onPick: (p: Vec3) => void
  onHover: (p: Vec3 | null) => void
  hover: Vec3 | null
  draft: { tool: MeasureTool; points: Vec3[] } | null
  measurements: Measurement[]
  onSceneInfo: (info: SceneInfo) => void
  onReady: () => void
  cameraApiRef: React.RefObject<CameraApi | null>
  cloudRetry: number
  meshRetry: number
  onCloudLoadState: (s: AssetLoadState) => void
  onMeshLoadState: (s: AssetLoadState) => void
}

function bboxMetrics(meta: ViewerMeta) {
  // Prefer ROI bbox if available, fall back to full bbox
  const box = meta.roi_enu ?? meta.bbox_enu
  const { min, max } = box
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
  // Ground target: centre of bbox at lowest Z
  const groundTarget = new THREE.Vector3(center.x, center.y, min[2])
  return { diag, groundTarget, center }
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

/** Syncs renderer clear colour whenever the background prop changes. */
function RendererSetup({ bg }: { bg: BackgroundMode }) {
  const gl = useThree((s) => s.gl)
  useEffect(() => {
    gl.setClearColor(BG_COLORS[bg])
  }, [gl, bg])
  return null
}

/** invalidate the demand-frameloop canvas whenever view/measurement state changes. */
function InvalidateOnChange({ token }: { token: string }) {
  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => {
    invalidate()
  }, [token, invalidate])
  return null
}

export function SceneCanvas({
  meta,
  cloudUrl,
  meshUrl,
  view,
  active,
  debug,
  measureTool,
  onPick,
  onHover,
  hover,
  draft,
  measurements,
  onSceneInfo,
  onReady,
  cameraApiRef,
  cloudRetry,
  meshRetry,
  onCloudLoadState,
  onMeshLoadState,
}: Props) {
  const { diag, groundTarget } = useMemo(() => bboxMetrics(meta), [meta])

  // 45° oblique start: approach from south-west at 45° elevation.
  // Direction: 0.55 south, -0.95 west, 0.7 up — avoid near-grazing angles.
  const defaultPos = useMemo(() => {
    const dir = new THREE.Vector3(0.55, -0.95, 0.7).normalize()
    return groundTarget.clone().add(dir.multiplyScalar(diag * 1.15))
  }, [groundTarget, diag])

  const rigApiRef = useRef<CameraRigApi | null>(null)

  const markerRadius = Math.max(0.35, diag * 0.004)
  const pointThreshold = Math.max(0.3, 2 * meta.median_spacing_m)

  const meshVisible = view.display !== 'cloud'
  const cloudVisible = view.display !== 'mesh'
  const both = view.display === 'both'
  const measuring = measureTool != null
  const meshPickable = measuring && meshVisible
  const cloudPickable = measuring && cloudVisible && !meshVisible

  const pointWorldSize = 1.5 * meta.median_spacing_m * view.pointSize * (both ? 0.5 : 1)

  const onDoublePick = measuring
    ? undefined
    : (p: Vec3) => rigApiRef.current?.pivotTo(new THREE.Vector3(p.x, p.y, p.z))

  const invalidateToken = [
    view.display,
    view.colorMode,
    view.pointSize,
    view.showTrajectory,
    view.material,
    view.detail,
    view.background,
    view.exposure,
    measureTool,
    measurements.length,
    draft?.points.length ?? 0,
    hover ? `${hover.x},${hover.y},${hover.z}` : '',
  ].join('|')

  return (
    <Canvas
      dpr={[1, 2]}
      frameloop="demand"
      gl={{ antialias: true, powerPreference: 'high-performance' }}
      camera={{
        position: [defaultPos.x, defaultPos.y, defaultPos.z],
        up: [0, 0, 1],
        fov: 50,
        near: Math.max(0.01, diag / 2000),
        far: diag * 20,
      }}
      onCreated={(state) => {
        // Phase A: flat / unmodified colour pipeline
        state.gl.toneMapping = THREE.NoToneMapping
        state.gl.outputColorSpace = THREE.SRGBColorSpace
        console.log(
          '[SceneCanvas] toneMapping=%o outputColorSpace=%o',
          state.gl.toneMapping,
          state.gl.outputColorSpace,
        )
        state.gl.setClearColor(BG_COLORS[view.background])
        state.camera.up.set(0, 0, 1)
        cameraApiRef.current = {
          reset: () => rigApiRef.current?.reset(),
          top: () => rigApiRef.current?.top(),
          oblique: () => rigApiRef.current?.oblique(),
        }
      }}
    >
      <PerformanceMonitor bounds={() => [45, 60]} />
      <AdaptiveDpr pixelated />
      {debug && <Stats className="!absolute" />}

      <RendererSetup bg={view.background} />

      <ambientLight intensity={0.55} />
      <hemisphereLight args={[0xffffff, 0x384049, 0.9]} />
      <directionalLight position={[diag, -diag, diag * 1.5]} intensity={0.6} />

      <RaycasterConfig threshold={pointThreshold} />
      <SceneInfoReporter onSceneInfo={onSceneInfo} />
      <InvalidateOnChange token={invalidateToken} />
      <CameraRig
        groundTarget={groundTarget}
        defaultPos={defaultPos}
        diag={diag}
        apiRef={rigApiRef}
      />

      {cloudVisible && (
        <PointCloud
          url={cloudUrl}
          active={active}
          retrySignal={cloudRetry}
          colorMode={view.colorMode}
          worldSize={pointWorldSize}
          visible={cloudVisible}
          pickable={cloudPickable}
          exposure={view.exposure}
          onPick={onPick}
          onHover={onHover}
          onDoubleClick={onDoublePick}
          onReady={onReady}
          onLoadState={onCloudLoadState}
        />
      )}
      {meshVisible && (
        <MeshModel
          url={meshUrl}
          active={active}
          retrySignal={meshRetry}
          colorMode={view.colorMode}
          material={view.material}
          polygonOffset={both}
          visible={meshVisible}
          pickable={meshPickable}
          exposure={view.exposure}
          onPick={onPick}
          onHover={onHover}
          onDoubleClick={onDoublePick}
          onReady={onReady}
          onLoadState={onMeshLoadState}
        />
      )}

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
