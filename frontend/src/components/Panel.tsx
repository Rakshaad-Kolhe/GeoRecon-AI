import type { ReactNode } from 'react'

export function Panel({
  title,
  right,
  children,
  className = '',
}: {
  title?: ReactNode
  right?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section
      className={`rounded-md border border-slate-800 bg-slate-900 ${className}`}
    >
      {(title || right) && (
        <header className="flex items-center justify-between border-b border-slate-800 px-4 py-2">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            {title}
          </h2>
          {right}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  )
}
