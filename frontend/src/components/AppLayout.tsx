import { Link, Outlet, useLocation } from 'react-router-dom'

export function AppLayout() {
  const { pathname } = useLocation()
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-800 bg-slate-900">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
          <Link to="/" className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-cyan-400" />
            <span className="text-sm font-semibold tracking-wide text-slate-100">
              GeoRecon<span className="text-cyan-400"> AI</span>
            </span>
          </Link>
          {pathname !== '/new' && (
            <Link
              to="/new"
              className="rounded bg-cyan-400 px-3 py-1 text-sm font-semibold text-slate-950 hover:bg-cyan-300"
            >
              New job
            </Link>
          )}
        </div>
      </header>
      <main className="mx-auto max-w-5xl px-4 py-6">
        <Outlet />
      </main>
    </div>
  )
}
