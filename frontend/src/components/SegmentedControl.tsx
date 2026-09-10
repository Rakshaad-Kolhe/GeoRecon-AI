export interface SegOption<T extends string> {
  value: T
  label: string
  hint?: string
}

export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  label,
}: {
  options: SegOption<T>[]
  value: T
  onChange: (v: T) => void
  label?: string
}) {
  const active = options.find((o) => o.value === value)
  return (
    <div>
      {label && (
        <div className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">
          {label}
        </div>
      )}
      <div className="inline-flex rounded-md border border-slate-800 bg-slate-900 p-0.5">
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            onClick={() => onChange(o.value)}
            className={`rounded px-3 py-1 text-sm font-medium transition-colors ${
              o.value === value
                ? 'bg-cyan-400 text-slate-950'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {o.label}
          </button>
        ))}
      </div>
      {active?.hint && (
        <p className="mt-1 text-xs text-slate-500">{active.hint}</p>
      )}
    </div>
  )
}
