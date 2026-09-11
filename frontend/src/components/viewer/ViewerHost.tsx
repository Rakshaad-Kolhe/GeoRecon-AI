import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../../api/client'
import type { JobDetail, ViewerMeta } from '../../api/types'
import {
  buildMeasurement,
  TOOL_POINTS,
  type MeasureTool,
  type Measurement,
  type Vec3,
} from '../../lib/measure'
import { scaleInfoFor } from '../../lib/scale'
import { ErrorBoundary } from './ErrorBoundary'
import { MeasureSidebar } from './MeasureSidebar'
import { MeasureToolbar } from './MeasureToolbar'
import type { AssetLoadState } from './PointCloud'
import { SceneCanvas, type SceneInfo } from './SceneCanvas'
import { SceneOverlay, type AssetStatus } from './SceneOverlay'
import { DEFAULT_VIEW_STATE, type CameraApi, type ViewState } from './types'
import { ViewControls } from './ViewControls'

interface Props {
  job: JobDetail
  mode: 'view' | 'measure'
  active: boolean
}

const CANVAS_H = 480
const NO_LOAD: AssetLoadState = { progress: 0, error: null }

function bboxDiag(meta: ViewerMeta): number {
  const { min, max } = meta.bbox_enu
  return Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2])
}

export function ViewerHost({ job, mode, active }: Props) {
  const [meta, setMeta] = useState<ViewerMeta | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [, setReady] = useState(false)

  const [view, setView] = useState<ViewState>(DEFAULT_VIEW_STATE)
  const [tool, setTool] = useState<MeasureTool | null>(null)
  const [draft, setDraft] = useState<Vec3[]>([])
  const [items, setItems] = useState<Measurement[]>([])
  const [hover, setHover] = useState<Vec3 | null>(null)
  const [info, setInfo] = useState<SceneInfo | null>(null)

  const [cloudLoad, setCloudLoad] = useState<AssetLoadState>(NO_LOAD)
  const [meshLoad, setMeshLoad] = useState<AssetLoadState>(NO_LOAD)
  const [cloudRetry, setCloudRetry] = useState(0)
  const [meshRetry, setMeshRetry] = useState(0)

  const cameraApiRef = useRef<CameraApi | null>(null)
  const debug = useMemo(() => new URLSearchParams(window.location.search).has('debug'), [])

  // ViewerHost is keyed by job id upstream, so this runs once per job.
  useEffect(() => {
    let alive = true
    api
      .prepareViewer(job.job_id)
      .then((m) => alive && setMeta(m))
      .catch((e: unknown) => alive && setError((e as Error).message || 'failed to load model'))
    return () => {
      alive = false
    }
  }, [job.job_id])

  // default mode: mesh if the export has triangles, cloud otherwise — only on
  // first load (meta arrives async), so it never fights a display choice the
  // user already made. Snapshotting into state (rather than deriving at
  // render time) keeps the ViewControls toolbar highlight in sync too.
  useEffect(() => {
    if (!meta || !view.displayAuto) return
    // oxlint-disable-next-line react/set-state-in-effect
    setView((v) => ({ ...v, display: meta.triangles > 0 ? 'mesh' : 'cloud', displayAuto: false }))
  }, [meta, view.displayAuto])

  const highUrl = meta?.lod?.points_hi != null && view.detail === 'high'
  const urls = useMemo(() => {
    if (!meta) return null
    return {
      cloud: api.fileUrl(job.job_id, highUrl ? 'web/pointcloud_hi.ply' : 'web/pointcloud.ply'),
      mesh: api.fileUrl(job.job_id, 'web/mesh.ply'),
    }
  }, [meta, job.job_id, highUrl])

  const dedupeDist = meta ? Math.max(0.5, bboxDiag(meta) * 0.006) : 1
  const scale = useMemo(() => scaleInfoFor(job, meta), [job, meta])

  const onReady = useCallback(() => setReady(true), [])
  const onSceneInfo = useCallback((i: SceneInfo) => setInfo(i), [])
  const onHover = useCallback((p: Vec3 | null) => setHover(p), [])
  const patchView = useCallback(
    (patch: Partial<ViewState>) => setView((v) => ({ ...v, ...patch })),
    [],
  )

  const closeArea = useCallback(() => {
    setDraft((d) => {
      if (tool === 'area' && d.length >= 3 && meta) {
        setItems((prev) => [...prev, buildMeasurement('area', d, meta, scale)])
        return []
      }
      return d
    })
  }, [tool, meta, scale])

  const closeAreaRef = useRef(closeArea)
  useEffect(() => {
    closeAreaRef.current = closeArea
  }, [closeArea])

  useEffect(() => {
    if (!active) return
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA') return

      if (mode === 'measure') {
        if (e.key === 'Escape') return setDraft([])
        if (e.key === 'Enter') return closeAreaRef.current()
      }
      if (e.key === 'r' || e.key === 'R') cameraApiRef.current?.reset()
      else if (e.key === 't' || e.key === 'T') cameraApiRef.current?.top()
      else if (e.key === '1') patchView({ display: 'cloud' })
      else if (e.key === '2') patchView({ display: 'mesh' })
      else if (e.key === '3') patchView({ display: 'both' })
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [mode, active, patchView])

  const onPick = useCallback(
    (p: Vec3) => {
      if (!tool || !meta) return
      if (tool === 'inspect') {
        setItems((prev) => [...prev, buildMeasurement('inspect', [p], meta, scale)])
        return
      }
      if (tool === 'area') {
        setDraft((d) => {
          const lastP = d[d.length - 1]
          if (lastP && Math.hypot(p.x - lastP.x, p.y - lastP.y, p.z - lastP.z) < dedupeDist) {
            return d
          }
          return [...d, p]
        })
        return
      }
      // distance | height — fixed vertex count
      const need = TOOL_POINTS[tool]
      setDraft((d) => {
        const next = [...d, p]
        if (next.length >= need) {
          setItems((prev) => [
            ...prev,
            buildMeasurement(tool, next.slice(0, need), meta, scale),
          ])
          return []
        }
        return next
      })
    },
    [tool, meta, dedupeDist, scale],
  )

  const onTool = useCallback((t: MeasureTool | null) => {
    setTool(t)
    setDraft([])
  }, [])

  const clearAll = useCallback(() => {
    setItems([])
    setDraft([])
  }, [])

  const deleteItem = useCallback(
    (id: string) => setItems((prev) => prev.filter((m) => m.id !== id)),
    [],
  )

  const measuring = mode === 'measure'
  const draftForScene = measuring && tool ? { tool, points: draft } : null

  const assets: AssetStatus[] = []
  if (view.display !== 'mesh') assets.push({ label: 'cloud', ...cloudLoad })
  if (view.display !== 'cloud') assets.push({ label: 'mesh', ...meshLoad })
  const onRetryAsset = useCallback((label: string) => {
    if (label === 'cloud') setCloudRetry((n) => n + 1)
    else setMeshRetry((n) => n + 1)
  }, [])

  return (
    <div className="flex flex-col gap-3">
      <ViewControls view={view} onChange={patchView} meta={meta} cameraApi={cameraApiRef} />

      {mode === 'measure' && (
        <MeasureToolbar
          tool={tool}
          onTool={onTool}
          canClose={draft.length >= 3}
          onClose={closeArea}
          onClear={clearAll}
          count={items.length}
        />
      )}

      <div className="flex gap-3">
        <div
          className="relative flex-1 overflow-hidden rounded border border-slate-800 bg-slate-950"
          style={{ height: CANVAS_H }}
          onDoubleClick={
            mode === 'measure' && tool === 'area' ? () => closeArea() : undefined
          }
        >
          {meta && urls && !error && (
            <ErrorBoundary onError={setError}>
              <SceneCanvas
                meta={meta}
                cloudUrl={urls.cloud}
                meshUrl={urls.mesh}
                view={view}
                active={active}
                debug={debug}
                measureTool={measuring ? tool : null}
                onPick={onPick}
                onHover={onHover}
                hover={measuring ? hover : null}
                draft={draftForScene}
                measurements={items}
                onSceneInfo={onSceneInfo}
                onReady={onReady}
                cameraApiRef={cameraApiRef}
                cloudRetry={cloudRetry}
                meshRetry={meshRetry}
                onCloudLoadState={setCloudLoad}
                onMeshLoadState={setMeshLoad}
              />
            </ErrorBoundary>
          )}
          {meta && (
            <SceneOverlay
              points={meta.points}
              triangles={meta.triangles}
              info={info}
              unitsLabel={scale.unitsLabel}
              assets={assets}
              onRetry={onRetryAsset}
            />
          )}
          {error && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="rounded border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-300">
                viewer error: {error}
              </div>
            </div>
          )}
          {!meta && !error && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="rounded border border-slate-800 bg-slate-950/90 px-3 py-2 text-sm text-slate-400">
                loading 3D model…
              </div>
            </div>
          )}
        </div>

        {mode === 'measure' && (
          <div className="w-64 shrink-0" style={{ height: CANVAS_H }}>
            <MeasureSidebar items={items} onDelete={deleteItem} />
          </div>
        )}
      </div>
    </div>
  )
}
