import { useState } from 'react'
import type { JobDetail } from '../api/types'
import { MetricsPanel } from './MetricsPanel'
import { Panel } from './Panel'

const TABS = ['3D', 'Measure', 'Map', 'Metrics', 'Downloads'] as const
type Tab = (typeof TABS)[number]

const DOWNLOADS = [
  'outputs/pointcloud.ply',
  'outputs/pointcloud.las',
  'outputs/mesh.obj',
  'outputs/model.glb',
  'outputs/dsm.tif',
  'outputs/color.tif',
  'outputs/trajectory.geojson',
  'report/metrics.json',
]

export function ResultsTabs({ job }: { job: JobDetail }) {
  const [tab, setTab] = useState<Tab>('Metrics')
  const done = job.state === 'done'

  return (
    <Panel title="Results">
      <div className="mb-4 flex flex-wrap gap-1 border-b border-slate-800">
        {TABS.map((t) => (
          <button
            key={t}
            type="button"
            disabled={!done}
            onClick={() => setTab(t)}
            className={`-mb-px border-b-2 px-3 py-1.5 text-sm transition-colors ${
              tab === t && done
                ? 'border-cyan-400 text-cyan-300'
                : 'border-transparent text-slate-500'
            } ${done ? 'hover:text-slate-300' : 'cursor-not-allowed opacity-40'}`}
          >
            {t}
          </button>
        ))}
      </div>

      {!done ? (
        <p className="text-sm text-slate-600">Results unlock when the job finishes.</p>
      ) : tab === 'Metrics' ? (
        <MetricsPanel metrics={job.metrics} />
      ) : tab === 'Downloads' ? (
        <ul className="space-y-1 font-mono text-xs text-slate-400">
          {DOWNLOADS.map((f) => (
            <li
              key={f}
              className="flex justify-between rounded border border-slate-800 px-2 py-1"
            >
              <span>{f}</span>
              <span className="text-slate-600">stub</span>
            </li>
          ))}
        </ul>
      ) : (
        <div className="flex h-40 items-center justify-center rounded border border-dashed border-slate-800 text-sm text-slate-600">
          {tab} viewer — {tab === 'Map' ? 'PR 13' : 'PR 12'}
        </div>
      )}
    </Panel>
  )
}
