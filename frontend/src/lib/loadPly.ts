import { useEffect, useState } from 'react'
import * as THREE from 'three'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'

export type PlyState =
  | { status: 'idle' }
  | { status: 'loading'; progress: number }
  | { status: 'ready'; geometry: THREE.BufferGeometry }
  | { status: 'error'; message: string }

/**
 * Manual fetch + PLYLoader.parse (instead of r3f's Suspense-based useLoader) so
 * we get real per-asset download progress, an abort on tab-switch/unmount, and
 * a Retry that re-fetches from scratch.
 */
export function usePlyGeometry(
  url: string,
  active: boolean,
  retrySignal: number,
  customPropertyNameMapping?: Record<string, string[]>,
): PlyState {
  const [state, setState] = useState<PlyState>({ status: 'idle' })

  useEffect(() => {
    if (!active) {
      setState({ status: 'idle' })
      return
    }
    let cancelled = false
    const controller = new AbortController()
    setState({ status: 'loading', progress: 0 })

    void (async () => {
      try {
        const res = await fetch(url, { signal: controller.signal })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const total = Number(res.headers.get('content-length')) || 0
        const reader = res.body?.getReader()

        let buffer: ArrayBuffer
        if (!reader) {
          buffer = await res.arrayBuffer()
        } else {
          const chunks: Uint8Array[] = []
          let received = 0
          for (;;) {
            const { done, value } = await reader.read()
            if (done) break
            chunks.push(value)
            received += value.length
            if (total > 0 && !cancelled) {
              setState({ status: 'loading', progress: received / total })
            }
          }
          const merged = new Uint8Array(received)
          let offset = 0
          for (const c of chunks) {
            merged.set(c, offset)
            offset += c.length
          }
          buffer = merged.buffer
        }

        if (cancelled) return
        const loader = new PLYLoader()
        if (customPropertyNameMapping) {
          loader.setCustomPropertyNameMapping(customPropertyNameMapping)
        }
        const geometry = loader.parse(buffer) as THREE.BufferGeometry
        setState({ status: 'ready', geometry })
      } catch (e) {
        if (cancelled || controller.signal.aborted) return
        setState({ status: 'error', message: (e as Error).message || 'load failed' })
      }
    })()

    return () => {
      cancelled = true
      controller.abort()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, active, retrySignal])

  useEffect(() => {
    return () => {
      if (state.status === 'ready') state.geometry.dispose()
    }
  }, [state])

  return state
}
