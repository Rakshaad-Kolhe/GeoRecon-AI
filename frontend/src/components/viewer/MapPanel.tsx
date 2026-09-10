import { useQuery } from '@tanstack/react-query'
import * as L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { useEffect } from 'react'
import {
  CircleMarker,
  MapContainer,
  Polygon,
  Polyline,
  TileLayer,
  useMap,
} from 'react-leaflet'
import { api } from '../../api/client'
import type { ViewerMeta } from '../../api/types'
import { bboxFootprintLatLng, enuToGeodetic } from '../../lib/measure'

type LatLng = [number, number]

interface CameraPt {
  pos: LatLng
  inlier: boolean
  name?: string
}

interface TrajData {
  line: LatLng[]
  cameras: CameraPt[]
}

interface GeoFeature {
  geometry:
    | { type: 'LineString'; coordinates: number[][] }
    | { type: 'Point'; coordinates: number[] }
  properties?: { inlier?: boolean; name?: string; kind?: string }
}

function parseTrajectory(fc: { features?: GeoFeature[] }): TrajData {
  const line: LatLng[] = []
  const cameras: CameraPt[] = []
  for (const f of fc.features ?? []) {
    const g = f.geometry
    if (g.type === 'LineString') {
      for (const c of g.coordinates) line.push([c[1], c[0]])
    } else if (g.type === 'Point') {
      cameras.push({
        pos: [g.coordinates[1], g.coordinates[0]],
        inlier: f.properties?.inlier !== false,
        name: f.properties?.name,
      })
    }
  }
  return { line, cameras }
}

/** fallback when trajectory.geojson is missing: ENU track → lat/lon. */
function trajectoryFromMeta(meta: ViewerMeta): TrajData {
  const line = meta.trajectory_enu.map((p): LatLng => {
    const g = enuToGeodetic({ x: p[0], y: p[1], z: p[2] }, meta.origin)
    return [g.lat, g.lon]
  })
  return { line, cameras: [] }
}

function FitBounds({ points }: { points: LatLng[] }) {
  const map = useMap()
  useEffect(() => {
    // react-leaflet mounts before the container has size in a freshly shown tab
    const t = setTimeout(() => {
      map.invalidateSize()
      if (points.length) map.fitBounds(L.latLngBounds(points), { padding: [24, 24] })
    }, 0)
    return () => clearTimeout(t)
  }, [map, points])
  return null
}

export function MapPanel({ jobId }: { jobId: string }) {
  const metaQ = useQuery({
    queryKey: ['viewerMeta', jobId],
    queryFn: () => api.prepareViewer(jobId),
  })

  const trajQ = useQuery({
    queryKey: ['trajectory', jobId],
    queryFn: async (): Promise<TrajData | null> => {
      const res = await fetch(api.fileUrl(jobId, 'trajectory.geojson'))
      if (!res.ok) return null
      return parseTrajectory(await res.json())
    },
    retry: false,
  })

  if (metaQ.isLoading) return <p className="text-sm text-slate-500">loading map…</p>
  if (metaQ.error || !metaQ.data) {
    return (
      <p className="text-sm text-red-400">
        map unavailable: {(metaQ.error as Error)?.message ?? 'no metadata'}
      </p>
    )
  }

  const meta = metaQ.data
  const footprint = bboxFootprintLatLng(meta)
  const traj = trajQ.data ?? trajectoryFromMeta(meta)
  const allPts: LatLng[] = [...footprint, ...traj.line, ...traj.cameras.map((c) => c.pos)]
  const center: LatLng = footprint[0] ?? [meta.origin.lat, meta.origin.lon]

  return (
    <div>
      <MapContainer
        center={center}
        zoom={16}
        scrollWheelZoom
        style={{ height: 480, background: '#020617' }}
        className="rounded border border-slate-800"
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; OpenStreetMap'
        />
        <Polygon
          positions={footprint}
          pathOptions={{ color: '#22d3ee', weight: 1.5, fillOpacity: 0.08 }}
        />
        {traj.line.length > 1 && (
          <Polyline positions={traj.line} pathOptions={{ color: '#f59e0b', weight: 2 }} />
        )}
        {traj.cameras.map((c, i) => (
          <CircleMarker
            key={i}
            center={c.pos}
            radius={4}
            pathOptions={{
              color: c.inlier ? '#22d3ee' : '#ef4444',
              fillColor: c.inlier ? '#22d3ee' : '#ef4444',
              fillOpacity: 0.9,
              weight: 1,
            }}
          />
        ))}
        <FitBounds points={allPts} />
      </MapContainer>
      <p className="mt-1 text-[11px] text-slate-500">
        cyan = footprint / inlier camera · red = outlier camera · amber = flight path
        {trajQ.data == null && ' · track from meta (trajectory.geojson unavailable)'}
      </p>
    </div>
  )
}
