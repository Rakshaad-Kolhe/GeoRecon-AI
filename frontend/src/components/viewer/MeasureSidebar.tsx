import { useState } from 'react'
import type { Measurement } from '../../lib/measure'

interface Props {
  items: Measurement[]
  onDelete: (id: string) => void
}

function CopyButton({ item }: { item: Measurement }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      type="button"
      onClick={() => {
        const text = item.detail.map(([k, v]) => `${k}: ${v}`).join('\n')
        void navigator.clipboard?.writeText(text)
        setCopied(true)
        setTimeout(() => setCopied(false), 1200)
      }}
      className="text-[11px] text-slate-500 hover:text-cyan-400"
    >
      {copied ? 'copied' : 'copy'}
    </button>
  )
}

export function MeasureSidebar({ items, onDelete }: Props) {
  return (
    <div className="flex h-full flex-col rounded border border-slate-800 bg-slate-950">
      <div className="border-b border-slate-800 px-3 py-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
        Measurements ({items.length})
      </div>
      <div className="flex-1 space-y-2 overflow-y-auto p-2">
        {items.length === 0 ? (
          <p className="px-1 text-xs text-slate-600">nothing measured yet</p>
        ) : (
          items.map((m) => (
            <div key={m.id} className="rounded border border-slate-800 bg-slate-900 p-2">
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-xs text-slate-200">{m.label}</span>
                <span className="flex shrink-0 items-center gap-2">
                  <CopyButton item={m} />
                  <button
                    type="button"
                    onClick={() => onDelete(m.id)}
                    className="text-[11px] text-slate-500 hover:text-red-400"
                  >
                    ✕
                  </button>
                </span>
              </div>
              <dl className="mt-1 space-y-0.5">
                {m.detail.map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-2 font-mono text-[11px]">
                    <dt className="text-slate-500">{k}</dt>
                    <dd className="text-right text-slate-300">{v}</dd>
                  </div>
                ))}
              </dl>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
