import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { api } from '../api/client'
import type { FileEntry } from '../api/types'
import { formatBytes } from '../lib/format'

const DESC: Record<string, string> = {
  'pointcloud.ply': 'PLY · binary, RGB + confidence · MeshLab / CloudCompare',
  'pointcloud.las': 'LAS 1.2 · UTM (EPSG from georef.json) · QGIS / CloudCompare',
  'mesh.obj': 'OBJ (+ MTL) · textured triangle mesh',
  'model.glb': 'glTF 2.0 binary · web / AR viewers',
  'mesh.ply': 'PLY mesh · per-vertex colour',
  'dsm.tif': 'GeoTIFF DSM · UTM, ~1 m cells',
  'color.tif': 'GeoTIFF orthomosaic · UTM',
  'trajectory.geojson': 'GeoJSON · camera track + inliers, WGS84',
  'georef.json': 'similarity transform · UTM EPSG · ENU origin',
  'meta.json': 'viewer metadata · ENU bbox, trajectory, origin',
}

const GROUPS: { title: string; match: (p: string) => boolean }[] = [
  { title: 'Point cloud', match: (p) => p === 'pointcloud.ply' || p.endsWith('.las') },
  {
    title: 'Mesh',
    match: (p) => p.endsWith('.obj') || p.endsWith('.glb') || p === 'mesh.ply',
  },
  { title: 'GIS', match: (p) => p.endsWith('.tif') || p.endsWith('.geojson') },
  { title: 'Metadata', match: (p) => p.endsWith('.json') },
]

const base = (p: string) => p.split('/').pop() ?? p

function group(files: FileEntry[]) {
  const buckets = GROUPS.map((g) => ({ title: g.title, files: [] as FileEntry[] }))
  const other: FileEntry[] = []
  for (const f of files) {
    const name = base(f.path)
    const gi = GROUPS.findIndex((g) => g.match(name))
    if (gi >= 0) buckets[gi].files.push(f)
    else other.push(f)
  }
  if (other.length) buckets.push({ title: 'Other', files: other })
  return buckets.filter((b) => b.files.length > 0)
}

function Row({ jobId, file }: { jobId: string; file: FileEntry }) {
  const name = base(file.path)
  return (
    <a
      href={api.fileUrl(jobId, file.path)}
      download
      className="flex items-baseline justify-between gap-3 rounded border border-slate-800 px-2 py-1.5 hover:border-slate-700 hover:bg-slate-800/30"
    >
      <span className="min-w-0">
        <span className="font-mono text-xs text-cyan-400">{name}</span>
        <span className="block truncate text-[11px] text-slate-500">
          {DESC[name] ?? file.path}
        </span>
      </span>
      <span className="shrink-0 font-mono text-[11px] text-slate-400">
        {formatBytes(file.bytes)}
      </span>
    </a>
  )
}

export function DownloadsPanel({ jobId }: { jobId: string }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['files', jobId],
    queryFn: () => api.getFiles(jobId),
  })
  const [thumbOk, setThumbOk] = useState(true)

  if (isLoading) return <p className="text-sm text-slate-500">loading file list…</p>
  if (error) {
    return (
      <p className="text-sm text-red-400">
        file list unavailable: {(error as Error).message}
      </p>
    )
  }

  const buckets = group(data ?? [])

  return (
    <div className="grid gap-4 md:grid-cols-[1fr_auto]">
      <div className="space-y-4">
        {buckets.map((b) => (
          <div key={b.title}>
            <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
              {b.title}
            </h3>
            <div className="space-y-1">
              {b.files.map((f) => (
                <Row key={f.path} jobId={jobId} file={f} />
              ))}
            </div>
          </div>
        ))}

        <div>
          <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Report
          </h3>
          <a
            href={api.reportUrl(jobId, 'summary.md')}
            target="_blank"
            rel="noreferrer"
            className="font-mono text-xs text-cyan-400 hover:underline"
          >
            report/summary.md
          </a>
        </div>
      </div>

      {thumbOk && (
        <figure className="w-full max-w-[220px]">
          <img
            src={api.reportUrl(jobId, 'mesh_preview.png')}
            alt="mesh preview"
            onError={() => setThumbOk(false)}
            className="rounded border border-slate-800"
          />
          <figcaption className="mt-1 text-[11px] text-slate-500">
            report/mesh_preview.png
          </figcaption>
        </figure>
      )}
    </div>
  )
}
