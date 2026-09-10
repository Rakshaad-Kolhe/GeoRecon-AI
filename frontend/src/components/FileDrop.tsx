import { useRef, useState } from 'react'
import { formatBytes } from '../lib/format'

export function FileDrop({
  label,
  accept, // e.g. ['.mp4', '.mov']
  maxBytes,
  file,
  onFile,
  error,
  optional,
}: {
  label: string
  accept: string[]
  maxBytes: number
  file: File | null
  onFile: (f: File | null, error?: string) => void
  error?: string
  optional?: boolean
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  function validate(f: File): string | undefined {
    const ext = '.' + (f.name.split('.').pop() ?? '').toLowerCase()
    if (!accept.includes(ext)) return `${label}: expected ${accept.join(' / ')}, got ${ext}`
    if (f.size > maxBytes) return `${label}: ${formatBytes(f.size)} exceeds ${formatBytes(maxBytes)} limit`
    return undefined
  }

  function take(f: File | undefined | null) {
    if (!f) return
    const err = validate(f)
    onFile(err ? null : f, err)
  }

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          {label}
        </span>
        {optional && <span className="text-xs text-slate-600">optional</span>}
      </div>
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          take(e.dataTransfer.files?.[0])
        }}
        onClick={() => inputRef.current?.click()}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed px-4 py-6 text-center text-sm transition-colors ${
          dragging
            ? 'border-cyan-400 bg-cyan-400/5'
            : error
              ? 'border-red-500/50 bg-red-500/5'
              : 'border-slate-700 bg-slate-950 hover:border-slate-600'
        }`}
      >
        {file ? (
          <>
            <span className="font-mono text-slate-200">{file.name}</span>
            <span className="mt-0.5 text-xs text-slate-500">
              {formatBytes(file.size)} · click to replace
            </span>
          </>
        ) : (
          <>
            <span className="text-slate-400">
              Drop file or <span className="text-cyan-400">browse</span>
            </span>
            <span className="mt-0.5 text-xs text-slate-600">
              {accept.join(', ')} · max {formatBytes(maxBytes)}
            </span>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={accept.join(',')}
          className="hidden"
          onChange={(e) => take(e.target.files?.[0])}
        />
      </div>
      {error && <p className="mt-1 text-xs text-red-400">{error}</p>}
    </div>
  )
}
