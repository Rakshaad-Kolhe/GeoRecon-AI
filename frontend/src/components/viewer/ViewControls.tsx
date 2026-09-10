import { SegmentedControl } from '../SegmentedControl'
import { Toggle } from '../Toggle'
import type { ColorMode, DisplayMode, ViewState } from './types'

interface Props {
  view: ViewState
  onChange: (patch: Partial<ViewState>) => void
}

export function ViewControls({ view, onChange }: Props) {
  return (
    <div className="flex flex-wrap items-end gap-4">
      <SegmentedControl<DisplayMode>
        label="Show"
        value={view.display}
        onChange={(v) => onChange({ display: v })}
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

      <label className="flex flex-col gap-1">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Point size
        </span>
        <span className="flex items-center gap-2">
          <input
            type="range"
            min={0.5}
            max={4}
            step={0.1}
            value={view.pointSize}
            onChange={(e) => onChange({ pointSize: Number(e.target.value) })}
            className="w-32 accent-cyan-400"
          />
          <span className="w-8 font-mono text-xs text-slate-400">
            {view.pointSize.toFixed(1)}
          </span>
        </span>
      </label>

      <Toggle
        label="Trajectory"
        checked={view.showTrajectory}
        onChange={(v) => onChange({ showTrajectory: v })}
      />
    </div>
  )
}
