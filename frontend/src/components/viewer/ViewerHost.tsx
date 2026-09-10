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
import { ErrorBoundary } from './ErrorBoundary'
import { MeasureSidebar } from './MeasureSidebar'
import { MeasureToolbar } from './MeasureToolbar'
import { SceneCanvas, type SceneInfo } from './SceneCanvas'
import { SceneOverlay } from './SceneOverlay'
import { DEFAULT_VIEW_STATE, type ViewState } from './types'
import { ViewControls } from './ViewControls'

interface Props {
  job: JobDetail
  mode: 'view' | 'measure'
  active: boolean
}

const CANVAS_H = 480

function bboxDiag(meta: ViewerMeta): number {
  const { min, max } = meta.bbox_enu
  return Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2])
}

export function ViewerHost({ job, mode, active }: Props) {
  const [meta, setMeta] = useState<ViewerMeta | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [ready, setReady] = useState(false)

  const [view, setView] = useState<ViewState>(DEFAULT_VIEW_STATE)
  const [tool, setTool] = useState<MeasureTool | null>(null)
  const [draft, setDraft] = useState<Vec3[]>([])
  const [items, setItems] = useState<Measurement[]>([])
  const [hover, setHover] = useState<Vec3 | null>(null)
  const [info, setInfo] = useState<SceneInfo | null>(null)

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

  const urls = useMemo(() => {
    if (!meta) return null
    return {
      cloud: api.fileUrl(job.job_id, 'web/pointcloud.ply'),
      mesh: api.fileUrl(job.job_id, 'web/mesh.ply'),
    }
  }, [meta, job.job_id])

  const dedupeDist = meta ? Math.max(0.5, bboxDiag(meta) * 0.006) : 1

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
        setItems((prev) => [...prev, buildMeasurement('area', d, meta)])
        return []
      }
      return d
    })
  }, [tool, meta])

  const closeAreaRef = useRef(closeArea)
  useEffect(() => {
    closeAreaRef.current = closeArea
  }, [closeArea])

  useEffect(() => {
    if (mode !== 'measure') return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setDraft([])
      else if (e.key === 'Enter') closeAreaRef.current()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [mode])

  const onPick = useCallback(
    (p: Vec3) => {
      if (!tool || !meta) return
      if (tool === 'inspect') {
        setItems((prev) => [...prev, buildMeasurement('inspect', [p], meta)])
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
          setItems((prev) => [...prev, buildMeasurement(tool, next.slice(0, need), meta)])
          return []
        }
        return next
      })
    },
    [tool, meta, dedupeDist],
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

  return (
    <div className="flex flex-col gap-3">
      <ViewControls view={view} onChange={patchView} />

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
                measureTool={measuring ? tool : null}
                onPick={onPick}
                onHover={onHover}
                hover={measuring ? hover : null}
                draft={draftForScene}
                measurements={items}
                onSceneInfo={onSceneInfo}
                onReady={onReady}
              />
            </ErrorBoundary>
          )}
          <SceneOverlay
            points={meta?.points ?? 0}
            triangles={meta?.triangles ?? 0}
            info={info}
            loading={!ready && !error}
            error={error}
          />
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
