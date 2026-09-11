import type { FormEvent } from 'react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { Preset } from '../api/types'
import { FileDrop } from '../components/FileDrop'
import { Panel } from '../components/Panel'
import { ProgressBar } from '../components/ProgressBar'
import { SegmentedControl } from '../components/SegmentedControl'
import { Toggle } from '../components/Toggle'

const GB = 1024 ** 3
const MB = 1024 ** 2

const PRESETS: { value: Preset; label: string; hint: string }[] = [
  { value: 'fast', label: 'fast', hint: 'preview — ~minutes, 150 keyframes' },
  { value: 'balanced', label: 'balanced', hint: 'default — 300 keyframes' },
  { value: 'accurate', label: 'accurate', hint: 'slow, max detail — 600 keyframes' },
]

export function NewJob() {
  const navigate = useNavigate()

  const [video, setVideo] = useState<File | null>(null)
  const [videoErr, setVideoErr] = useState<string>()
  const [telemetry, setTelemetry] = useState<File | null>(null)
  const [telErr, setTelErr] = useState<string>()
  const [preset, setPreset] = useState<Preset>('balanced')
  const [maskDynamic, setMaskDynamic] = useState(false)
  const [offset, setOffset] = useState('0')
  const [altitude, setAltitude] = useState('')

  const [submitting, setSubmitting] = useState(false)
  const [uploadFrac, setUploadFrac] = useState(0)
  const [submitErr, setSubmitErr] = useState<string>()

  async function submit(e: FormEvent) {
    e.preventDefault()
    setSubmitErr(undefined)
    if (!video) {
      setSubmitErr('a video file is required')
      return
    }
    if (videoErr || telErr) return

    setSubmitting(true)
    setUploadFrac(0)
    try {
      const { job_id } = await api.createJob(
        {
          video,
          telemetry: telemetry ?? undefined,
          preset,
          mask_dynamic: maskDynamic,
          telemetry_offset_s: Number(offset) || 0,
          assumed_altitude_m: altitude.trim() ? Number(altitude) : undefined,
        },
        setUploadFrac,
      )
      navigate(`/jobs/${job_id}`)
    } catch (err) {
      setSubmitErr((err as Error).message || 'submit failed')
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <h1 className="text-lg font-semibold text-slate-100">New job</h1>

      <Panel>
        <div className="space-y-4">
          <FileDrop
            label="Drone video"
            accept={['.mp4', '.mov', '.mkv', '.avi']}
            maxBytes={20 * GB}
            file={video}
            error={videoErr}
            onFile={(f, err) => {
              setVideo(f)
              setVideoErr(err)
            }}
          />
          <FileDrop
            label="Telemetry"
            optional
            accept={['.srt', '.csv']}
            maxBytes={64 * MB}
            file={telemetry}
            error={telErr}
            onFile={(f, err) => {
              setTelemetry(f)
              setTelErr(err)
            }}
          />
        </div>
      </Panel>

      <Panel>
        <div className="space-y-4">
          <SegmentedControl
            label="Preset"
            options={PRESETS}
            value={preset}
            onChange={setPreset}
          />
          <Toggle
            label="Mask dynamic objects"
            hint="YOLO segmentation removes people / vehicles before SfM"
            checked={maskDynamic}
            onChange={setMaskDynamic}
          />
          <label className="block">
            <span className="mb-1 block text-xs font-semibold uppercase tracking-wider text-slate-400">
              Telemetry offset (s)
            </span>
            <input
              type="number"
              step="0.1"
              value={offset}
              onChange={(e) => setOffset(e.target.value)}
              className="w-32 rounded border border-slate-800 bg-slate-950 px-2 py-1 font-mono text-sm text-slate-100 outline-none focus:border-cyan-400"
            />
          </label>
          <label className="block">
            <span className="mb-1 block text-xs font-semibold uppercase tracking-wider text-slate-400">
              Approx. flight altitude (m)
            </span>
            <input
              type="number"
              step="1"
              min="0"
              value={altitude}
              onChange={(e) => setAltitude(e.target.value)}
              className="w-32 rounded border border-slate-800 bg-slate-950 px-2 py-1 font-mono text-sm text-slate-100 outline-none focus:border-cyan-400"
            />
            <span className="mt-1 block text-[11px] text-slate-500">
              used only when no telemetry is uploaded
            </span>
          </label>
        </div>
      </Panel>

      {submitting && (
        <div className="space-y-1">
          <ProgressBar value={uploadFrac} />
          <p className="font-mono text-xs text-slate-500">
            uploading… {Math.round(uploadFrac * 100)}%
          </p>
        </div>
      )}

      {submitErr && <p className="text-sm text-red-400">{submitErr}</p>}

      <button
        type="submit"
        disabled={submitting}
        className="rounded bg-cyan-400 px-4 py-1.5 text-sm font-semibold text-slate-950 hover:bg-cyan-300 disabled:opacity-50"
      >
        {submitting ? 'submitting…' : 'Start job'}
      </button>
    </form>
  )
}
