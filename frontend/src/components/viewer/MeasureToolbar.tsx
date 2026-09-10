import { TOOL_HINT, type MeasureTool } from '../../lib/measure'

interface Props {
  tool: MeasureTool | null
  onTool: (t: MeasureTool | null) => void
  canClose: boolean
  onClose: () => void
  onClear: () => void
  count: number
}

const TOOLS: { id: MeasureTool; label: string }[] = [
  { id: 'distance', label: 'Distance' },
  { id: 'area', label: 'Area' },
  { id: 'height', label: 'Height' },
  { id: 'inspect', label: 'Inspect' },
]

export function MeasureToolbar({
  tool,
  onTool,
  canClose,
  onClose,
  onClear,
  count,
}: Props) {
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap gap-1">
        {TOOLS.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => onTool(tool === t.id ? null : t.id)}
            className={`rounded border px-2.5 py-1 text-sm transition-colors ${
              tool === t.id
                ? 'border-cyan-400 bg-cyan-400/15 text-cyan-300'
                : 'border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            {t.label}
          </button>
        ))}

        {tool === 'area' && (
          <button
            type="button"
            disabled={!canClose}
            onClick={onClose}
            className="rounded border border-slate-800 px-2.5 py-1 text-sm text-slate-300 disabled:opacity-40"
          >
            Close polygon
          </button>
        )}

        <button
          type="button"
          disabled={count === 0}
          onClick={onClear}
          className="ml-auto rounded border border-slate-800 px-2.5 py-1 text-sm text-slate-400 hover:text-slate-200 disabled:opacity-40"
        >
          Clear all
        </button>
      </div>

      <p className="font-mono text-[11px] text-slate-500">
        {tool ? TOOL_HINT[tool] : 'pick a tool · Esc cancels current · Enter closes area'}
      </p>
    </div>
  )
}
