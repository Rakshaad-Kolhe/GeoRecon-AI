import { USE_MOCKS } from '../api/client'

export function MockBanner() {
  if (!USE_MOCKS) return null
  return (
    <div className="w-full bg-amber-500 px-4 py-1.5 text-center text-sm font-semibold text-slate-950">
      MOCK DATA — backend not used
    </div>
  )
}
