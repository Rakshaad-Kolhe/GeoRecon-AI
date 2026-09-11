import type { ViewerMeta } from '../../api/types'
import { formatInt } from '../../lib/format'
import { SegmentedControl } from '../SegmentedControl'
import { Toggle } from '../Toggle'
import type { BackgroundMode, CameraApi, ColorMode, DetailLevel, DisplayMode, MaterialMode, ViewState } from './types'

interface Props {
  view: ViewState
  onChange: (patch: Partial<ViewState>) => void
  meta: ViewerMeta | null
  cameraApi: React.RefObject<CameraApi | null>
}

export function ViewControls({ view, onChange, meta, cameraApi }: Props) {
  const hasHi = meta?.lod?.points_hi != null
  return (
    <div className="flex flex-wrap items-end gap-4">
      <SegmentedControl<DisplayMode>
        label="Show"
        value={view.display}
        onChange={(v) => onChange({ display: v, displayAuto: false })}
        options={[
          { value: 'cloud', label: 'Cloud' },
          { value: 'mesh', label: 'Mesh' },
          { value: 'both', label: 'Both' },
        ]}
      />

      <SegmentedControl<ColorMode>
        label="Colour"
        value={view.colorMode}
        onChange={(v) => onChange({ colorMode: v })}
        options={[
          { value: 'rgb', label: 'RGB' },
          { value: 'conf', label: 'Confidence' },
          { value: 'height', label: 'Height' },
        ]}
      />

      <SegmentedControl<MaterialMode>
        label="Mesh"
        value={view.material}
        onChange={(v) => onChange({ material: v })}
        options={[
          { value: 'photo', label: 'Photo' },
          { value: 'shaded', label: 'Shaded' },
        ]}
      />

      <div className="flex flex-col gap-1">
        <SegmentedControl<DetailLevel>
          label="Detail"
          value={view.detail}
          onChange={(v) => onChange({ detail: v })}
          options={[
            { value: 'standard', label: 'Standard' },
            { value: 'high', label: hasHi ? `High (${formatInt(meta?.lod?.points_hi)})` : 'High' },
          ]}
        />
        {view.detail === 'high' && !hasHi && (
          <span className="text-[11px] text-slate-600">no high-detail asset for this job</span>
        )}
      </div>

      <label className="flex flex-col gap-1">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Point size
        </span>
        <span className="flex items-center gap-2">
          <input
            type="range"
            min={0.5}
            max={3}
            step={0.1}
            value={view.pointSize}
            onChange={(e) => onChange({ pointSize: Number(e.target.value) })}
            className="w-32 accent-cyan-400"
          />
          <span className="w-8 font-mono text-xs text-slate-400">
            {view.pointSize.toFixed(1)}×
          </span>
        </span>
      </label>

      <label className="flex flex-col gap-1">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Exposure
        </span>
        <span className="flex items-center gap-2">
          <input
            type="range"
            min={0.6}
            max={1.6}
            step={0.05}
            value={view.exposure}
            onChange={(e) => onChange({ exposure: Number(e.target.value) })}
            className="w-28 accent-amber-400"
          />
          <span className="w-8 font-mono text-xs text-slate-400">
            {view.exposure.toFixed(2)}
          </span>
        </span>
      </label>

      <SegmentedControl<BackgroundMode>
        label="BG"
        value={view.background}
        onChange={(v) => onChange({ background: v })}
        options={[
          { value: 'dark', label: '⬛' },
          { value: 'grey', label: '⬜' },
          { value: 'light', label: '☀' },
        ]}
      />

      <Toggle
        label="Trajectory"
        checked={view.showTrajectory}
        onChange={(v) => onChange({ showTrajectory: v })}
      />

      <div className="flex flex-col gap-1">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Camera
        </span>
        <span className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => cameraApi.current?.reset()}
            className="rounded border border-slate-800 px-2 py-1 text-xs text-slate-300 hover:border-slate-700"
            title="Reset (R)"
          >
            Reset
          </button>
          <button
            type="button"
            onClick={() => cameraApi.current?.top()}
            className="rounded border border-slate-800 px-2 py-1 text-xs text-slate-300 hover:border-slate-700"
            title="Top (T)"
          >
            Top
          </button>
          <button
            type="button"
            onClick={() => cameraApi.current?.oblique()}
            className="rounded border border-slate-800 px-2 py-1 text-xs text-slate-300 hover:border-slate-700"
            title="Oblique 45°"
          >
            Oblique
          </button>
        </span>
      </div>
    </div>
  )
}
