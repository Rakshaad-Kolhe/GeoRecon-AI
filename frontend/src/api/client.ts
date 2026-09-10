import type { CreateJobInput, JobDetail, JobStatus, ViewerMeta } from './types'
import { mockApi } from './mock'

export interface Api {
  listJobs(): Promise<JobStatus[]>
  getJob(id: string): Promise<JobDetail>
  createJob(
    input: CreateJobInput,
    onUploadProgress?: (fraction: number) => void,
  ): Promise<{ job_id: string }>
  getLog(id: string, tail?: number): Promise<string>
  /** Load web/meta.json; mocks synthesise a procedural site when no files exist. */
  prepareViewer(jobId: string): Promise<ViewerMeta>
  /** Resolve a served path under a job's outputs/ or report/ (or web/) to a URL. */
  fileUrl(jobId: string, path: string): string
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = (await res.json()) as { detail?: string }
      if (body?.detail) detail = body.detail
    } catch {
      // non-JSON error body
    }
    throw new ApiError(res.status, detail)
  }
  return (await res.json()) as T
}

const realApi: Api = {
  async listJobs() {
    return j<JobStatus[]>(await fetch('/api/jobs'))
  },

  async getJob(id) {
    return j<JobDetail>(await fetch(`/api/jobs/${encodeURIComponent(id)}`))
  },

  createJob(input, onUploadProgress) {
    // multipart via XHR so we get upload progress; field names == CreateJobInput keys.
    return new Promise((resolve, reject) => {
      const form = new FormData()
      form.append('video', input.video)
      if (input.telemetry) form.append('telemetry', input.telemetry)
      form.append('preset', input.preset)
      form.append('mask_dynamic', String(input.mask_dynamic))
      form.append('telemetry_offset_s', String(input.telemetry_offset_s))

      const xhr = new XMLHttpRequest()
      xhr.open('POST', '/api/jobs')
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onUploadProgress) {
          onUploadProgress(e.loaded / e.total)
        }
      }
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText) as { job_id: string })
          } catch {
            reject(new ApiError(xhr.status, 'malformed job response'))
          }
        } else {
          reject(new ApiError(xhr.status, xhr.statusText || 'upload failed'))
        }
      }
      xhr.onerror = () => reject(new ApiError(0, 'network error during upload'))
      xhr.send(form)
    })
  },

  async getLog(id, tail = 200) {
    const res = await fetch(
      `/api/jobs/${encodeURIComponent(id)}/log?tail=${tail}`,
    )
    if (!res.ok) throw new ApiError(res.status, res.statusText)
    return res.text()
  },

  async prepareViewer(id) {
    return j<ViewerMeta>(
      await fetch(`/api/jobs/${encodeURIComponent(id)}/files/web/meta.json`),
    )
  },

  fileUrl(id, path) {
    return `/api/jobs/${encodeURIComponent(id)}/files/${path.replace(/^\/+/, '')}`
  },
}

export const USE_MOCKS = import.meta.env.VITE_USE_MOCKS === '1'

export const api: Api = USE_MOCKS ? mockApi : realApi
