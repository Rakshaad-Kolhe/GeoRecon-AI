import { useState } from 'react'
import type { JobDetail } from '../api/types'
import { scaleInfoFor } from '../lib/scale'
import { DownloadsPanel } from './DownloadsPanel'
import { MetricsPanel } from './MetricsPanel'
import { Panel } from './Panel'
import { MapPanel } from './viewer/MapPanel'
import { ViewerHost } from './viewer/ViewerHost'

const TABS = ['3D', 'Measure', 'Map', 'Metrics', 'Downloads'] as const
type Tab = (typeof TABS)[number]

export function ResultsTabs({ job }: { job: JobDetail }) {
  const [tab, setTab] = useState<Tab>('Metrics')
  const done = job.state === 'done'
  const scale = scaleInfoFor(job)

  // Mount the WebGL viewer once (first time 3D/Measure is opened) and keep it
  // alive — toggling tabs only shows/hides it, so there is never a second context.
  const viewerActive = tab === '3D' || tab === 'Measure'
  const [viewerMounted, setViewerMounted] = useState(false)

  return (
    <Panel title="Results">
      <div className="mb-4 flex flex-wrap gap-1 border-b border-slate-800">
        {TABS.map((t) => {
          const mapBlocked = t === 'Map' && !scale.georeferenced
          const disabled = !done || mapBlocked
          return (
            <button
              key={t}
              type="button"
              disabled={disabled}
              title={mapBlocked ? scale.reason : undefined}
              onClick={() => {
                setTab(t)
                if (t === '3D' || t === 'Measure') setViewerMounted(true)
              }}
              className={`-mb-px border-b-2 px-3 py-1.5 text-sm transition-colors ${
                tab === t && done
                  ? 'border-cyan-400 text-cyan-300'
                  : 'border-transparent text-slate-500'
              } ${disabled ? 'cursor-not-allowed opacity-40' : 'hover:text-slate-300'}`}
            >
              {t}
            </button>
          )
        })}
      </div>

      {!done ? (
        <p className="text-sm text-slate-600">Results unlock when the job finishes.</p>
      ) : (
        <>
          {viewerMounted && (
            <div className={viewerActive ? '' : 'hidden'}>
              <ViewerHost
                key={job.job_id}
                job={job}
                mode={tab === 'Measure' ? 'measure' : 'view'}
                active={viewerActive}
              />
            </div>
          )}

          {tab === 'Metrics' && <MetricsPanel metrics={job.metrics} />}
          {tab === 'Map' && <MapPanel jobId={job.job_id} />}
          {tab === 'Downloads' && <DownloadsPanel jobId={job.job_id} />}
        </>
      )}
    </Panel>
  )
}
