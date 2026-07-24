const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"
const USER_ID = 1 // Hardcoded for now

export interface Project {
  id: number
  project_name: string
  user_id: number
  run_id?: string
  created_at?: string
  updated_at?: string
  last_access?: string
}

export interface RunResponse {
  run_id: string
  project_id: number
  duration_minutes?: number | null
}

export interface PricingConfig {
  base_rate_per_min_cents: number
  fix_timing_addon_cents: number
  segment_regen_cost_cents: number
  volume_discounts: { min_minutes: number; discount_pct: number }[]
}

export interface ProjectStatus {
  project_id: number
  run_id: string | null
  status: "initial" | "uploaded" | "processing" | "finished" | "failed"
  file_name?: string
  duration_minutes?: number | null
  dst_language?: string | null
  ttsmodel?: number | null
  mix_gains?: [number, number, number, number] | null
  src_language?: string | null
  use_non_speech?: boolean | null
  is_dubbed?: boolean | null
  emotions_flag?: boolean | null
  trans_type?: string | null
  fix_timing?: boolean | null
}

export interface RunStatusEvent {
  run_id: string
  flow_run_id: string
  flow_state_name: string
  stages: Record<string, "not_started" | "in_progress" | "done">
  percent?: number
}

export interface RunErrorEvent {
  run_id: string
  error: string
}

export class InsufficientBalanceError extends Error {
  required_cents: number
  balance_cents: number
  constructor(required_cents: number, balance_cents: number) {
    super("Insufficient balance")
    this.name = "InsufficientBalanceError"
    this.required_cents = required_cents
    this.balance_cents = balance_cents
  }
}

// Health check
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { credentials: "include", signal: AbortSignal.timeout(5000) })
    return res.ok
  } catch {
    return false
  }
}

// Create project
export async function createProject(projectName: string): Promise<Project> {
  const res = await fetch(`${API_BASE_URL}/projects/create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({
      user_id: USER_ID,
      project_name: projectName,
    }),
  })
  if (!res.ok) throw new Error("Failed to create project")
  return res.json()
}

// List projects
export async function listProjects(): Promise<Project[]> {
  const res = await fetch(`${API_BASE_URL}/projects?user_id=${USER_ID}`, { credentials: "include", signal: AbortSignal.timeout(5000) })
  if (!res.ok) throw new Error("Failed to list projects")
  return res.json()
}

// Upload input video
export async function uploadRunVideo(
  projectId: number,
  file: File,
  onProgress?: (percent: number) => void
): Promise<RunResponse> {
  const formData = new FormData()
  formData.append("file", file)
  formData.append("project_id", String(projectId))

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.withCredentials = true
    xhr.open("POST", `${API_BASE_URL}/runs/upload?project_id=${projectId}`)

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText))
        } catch {
          reject(new Error("Invalid response from server"))
        }
      } else if (xhr.status === 413) {
        reject(new Error("File too large — ask your admin to increase the server upload limit"))
      } else {
        reject(new Error(`Upload failed: server returned ${xhr.status}`))
      }
    }

    xhr.onerror = () => reject(new Error("Upload failed: connection was dropped. If the file is large, the server may have a size limit configured."))
    xhr.onabort = () => reject(new Error("Upload cancelled"))

    xhr.send(formData)
  })
}

// Fetch a backend URL with credentials and return an object URL (blob)
export async function fetchBlobUrl(url: string): Promise<string> {
  const res = await fetch(url, { credentials: "include" })
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`)
  const blob = await res.blob()
  return URL.createObjectURL(blob)
}

// Get input video as a credentialed blob URL
export async function getInputVideoUrl(runId: string): Promise<string> {
  return fetchBlobUrl(`${API_BASE_URL}/runs/${runId}/input_video`)
}

// Get output video as a credentialed blob URL
export async function getOutputVideoUrl(runId: string): Promise<string> {
  return fetchBlobUrl(`${API_BASE_URL}/runs/${runId}/video`)
}

// Download a backend resource with credentials, triggering a browser download
export async function downloadWithCredentials(url: string, filename: string): Promise<void> {
  const blobUrl = await fetchBlobUrl(url)
  const a = document.createElement("a")
  a.href = blobUrl
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(blobUrl)
}

// --- Pricing ---

export async function getPricing(): Promise<PricingConfig> {
  const res = await fetch(`${API_BASE_URL}/pricing`, { credentials: "include" })
  if (!res.ok) throw new Error("Failed to fetch pricing")
  return res.json()
}

// --- Export ---

export async function startExport(runId: string, resolution: string): Promise<void> {
  const res = await fetch(
    `${API_BASE_URL}/runs/${runId}/export?resolution=${resolution}`,
    { method: "POST", credentials: "include" }
  )
  if (!res.ok) throw new Error(`Failed to start export: ${res.status}`)
}

export async function getExportStatus(runId: string, resolution: string): Promise<{ status: string }> {
  const res = await fetch(
    `${API_BASE_URL}/runs/${runId}/export/status?resolution=${resolution}`,
    { credentials: "include" }
  )
  if (!res.ok) throw new Error(`Failed to get export status: ${res.status}`)
  return res.json()
}

export async function downloadExportFile(runId: string, resolution: string, fileName?: string): Promise<void> {
  const base = fileName ? fileName.replace(/\.[^.]+$/, "") : "video"
  await downloadWithCredentials(
    `${API_BASE_URL}/runs/${runId}/export/file?resolution=${resolution}`,
    `verbox_${base}_${resolution}.mp4`
  )
}

export async function downloadAudio(runId: string, fileName?: string): Promise<void> {
  const base = fileName ? fileName.replace(/\.[^.]+$/, "") : "audio"
  await downloadWithCredentials(
    `${API_BASE_URL}/runs/${runId}/audio`,
    `verbox_${base}_audio.wav`
  )
}

// Speaker name mapping
export async function getSpeakerNames(runId: string): Promise<Record<string, string>> {
  try {
    const res = await fetch(`${API_BASE_URL}/runs/${runId}/speaker-names`, { credentials: "include" })
    if (!res.ok) return {}
    return res.json()
  } catch {
    return {}
  }
}

export async function saveSpeakerNames(runId: string, names: Record<string, string>): Promise<void> {
  await fetch(`${API_BASE_URL}/runs/${runId}/speaker-names`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify(names),
  })
}

// Get original subtitles
export async function getOriginalSubtitles(runId: string): Promise<string> {
  const res = await fetch(`${API_BASE_URL}/runs/${runId}/subtitles/original`, { credentials: "include", cache: "no-store", signal: AbortSignal.timeout(5000) })
  if (!res.ok) throw new Error("Failed to fetch original subtitles")
  return res.text()
}

// Get translated subtitles
export async function getTranslatedSubtitles(runId: string, lang?: string): Promise<string> {
  const url = lang
    ? `${API_BASE_URL}/runs/${runId}/subtitles/translated?lang=${lang}`
    : `${API_BASE_URL}/runs/${runId}/subtitles/translated`
  const res = await fetch(url, { credentials: "include", cache: "no-store", signal: AbortSignal.timeout(5000) })
  if (!res.ok) throw new Error("Failed to fetch translated subtitles")
  return res.text()
}

// Delete project
export async function deleteProject(projectId: number): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/projects/${projectId}`, {
    method: "DELETE",
    credentials: "include",
  })
  if (!res.ok) throw new Error("Failed to delete project")
}

// Start dubbing run
export async function startRun(params: {
  dst_language: string
  trans_type: string
  elevenlabs: boolean
  num_speakers: number | null
  emotions_flag: boolean
  elevenlabs_emotions: 0 | 1 | 2
  fix_timing: boolean
  changed_list: unknown[]
  is_dubbed: boolean
  ttsmodel: number
  run_id: string
  stage: number
}): Promise<string> {
  const res = await fetch(`${API_BASE_URL}/runs/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify(params),
    signal: AbortSignal.timeout(10000),
  })
  if (res.status === 402) {
    const body = await res.json().catch(() => ({}))
    const detail = body.detail || body
    throw new InsufficientBalanceError(detail.required_cents ?? 0, detail.balance_cents ?? 0)
  }
  if (!res.ok) throw new Error(`Failed to start run: ${res.status}`)
  return res.text()
}

// Get project status
export async function getProjectStatus(projectId: number): Promise<ProjectStatus> {
  const res = await fetch(`${API_BASE_URL}/projects/${projectId}/status`, { credentials: "include", signal: AbortSignal.timeout(5000) })
  if (!res.ok) throw new Error("Failed to fetch project status")
  return res.json()
}

// Regenerate dubbing after subtitle edits
export async function regenerateSubtitles(runId: string, subtitles: string, changedList: number[], ttsmodel?: number): Promise<{ run_id: string; flow_run_id: string }> {
  const res = await fetch(`${API_BASE_URL}/runs/${runId}/regenerate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({
      subtitles,
      changed_list: changedList,
      ...(ttsmodel !== undefined && { ttsmodel }),
    }),
    signal: AbortSignal.timeout(10000),
  })
  if (res.status === 402) {
    const body = await res.json().catch(() => ({}))
    const detail = body.detail || body
    throw new InsufficientBalanceError(detail.required_cents ?? 0, detail.balance_cents ?? 0)
  }
  if (!res.ok) throw new Error(`Failed to regenerate subtitles: ${res.status}`)
  return res.json()
}
// Get available stems for a run (returns { background?: url, dialog?: url, original?: url })
export async function getStemsInfo(runId: string): Promise<Record<string, string>> {
  const res = await fetch(`${API_BASE_URL}/runs/${runId}/stems`, { credentials: "include" })
  if (!res.ok) return {}
  return res.json()
}

// Fetch a stem as an AudioBuffer using the Web Audio API
export async function fetchStemBuffer(stemUrl: string, ctx: AudioContext): Promise<AudioBuffer> {
  const res = await fetch(`${API_BASE_URL}${stemUrl}`, { credentials: "include" })
  if (!res.ok) throw new Error(`Failed to fetch stem: ${res.status}`)
  const arrayBuffer = await res.arrayBuffer()
  return ctx.decodeAudioData(arrayBuffer)
}

// Apply mix and re-render on backend (runs COMBINE stage with new gains)
// mix_gains = [background_db, dialog_db, non_speech_db, original_underlay_db]
export async function remixAudio(runId: string, mixGains: [number, number, number, number]): Promise<{ run_id: string; flow_run_id: string }> {
  const res = await fetch(`${API_BASE_URL}/runs/${runId}/remix`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ mix_gains: mixGains }),
    signal: AbortSignal.timeout(10000),
  })
  if (!res.ok) throw new Error(`Failed to remix: ${res.status}`)
  return res.json()
}

// Update user profile (name + email for local, name only for Google)
export async function updateProfile(data: { first_name: string; last_name: string; email?: string }): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/users/profile`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify(data),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || "Failed to update profile")
  }
}

// Change password (local auth only)
export async function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/users/change-password`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || "Failed to change password")
  }
}

// Get current user's balance in cents
export async function getUserBalance(): Promise<number> {
  const res = await fetch(`${API_BASE_URL}/users/balance`, { credentials: "include" })
  if (!res.ok) throw new Error("Failed to fetch balance")
  const data = await res.json()
  return data.balance as number
}

// Create a Stripe Checkout session and return the hosted URL
export async function createCheckoutSession(amountCents: number): Promise<string> {
  const res = await fetch(`${API_BASE_URL}/payments/create-checkout`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ amount_cents: amountCents }),
  })
  if (!res.ok) throw new Error("Failed to create checkout session")
  const data = await res.json()
  return data.checkout_url as string
}

// Start YouTube download
export async function startYoutubeDownload(
  projectId: number,
  url: string
): Promise<{ run_id: string }> {
  const res = await fetch(`${API_BASE_URL}/runs/youtube`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, project_id: projectId }),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail?.detail || "Failed to start YouTube download")
  }
  return res.json()
}

export interface YtProgressEvent {
  status: "downloading" | "done" | "error"
  percent: number
  filename: string | null
  run_id: string
  error: string | null
  duration_minutes?: number | null
}

// Stream YouTube download progress via SSE
export function subscribeToYoutubeProgress(
  runId: string,
  onProgress: (data: YtProgressEvent) => void,
  onDone: (data: YtProgressEvent) => void,
  onError: (data: YtProgressEvent) => void
): () => void {
  const es = new EventSource(`${API_BASE_URL}/runs/${runId}/youtube-progress`, { withCredentials: true })

  es.onmessage = (event) => {
    try {
      const data: YtProgressEvent = JSON.parse(event.data)
      if (data.status === "done") {
        onDone(data)
        es.close()
      } else if (data.status === "error") {
        onError(data)
        es.close()
      } else {
        onProgress(data)
      }
    } catch (e) {
      console.error("[yt] Failed to parse progress event:", e)
    }
  }

  es.onerror = () => {
    es.close()
  }

  return () => es.close()
}

export function subscribeToRunStatus(
  runId: string,
  onStatus: (data: RunStatusEvent) => void,
  onDone: (data: RunStatusEvent) => void,
  onError: (data: RunErrorEvent) => void,
  onClose: () => void
): () => void {
  const eventSource = new EventSource(`${API_BASE_URL}/runs/${runId}/status`, { withCredentials: true })

  eventSource.addEventListener("status", (event) => {
    try {
      const data = JSON.parse(event.data)
      onStatus(data)
    } catch (e) {
      console.error("Failed to parse status event:", e)
    }
  })

  eventSource.addEventListener("done", (event) => {
    try {
      const data = JSON.parse(event.data)
      onDone(data)
      eventSource.close()
      onClose()
    } catch (e) {
      console.error("Failed to parse done event:", e)
    }
  })

  eventSource.addEventListener("error", (event) => {
    try {
      // Error events may not have data, only handle if data exists
      if (event.data && event.data !== "undefined") {
        const data = JSON.parse(event.data)
        onError(data)
      }
    } catch (e) {
      console.warn("Error event has no JSON data, ignoring:", e)
    }
  })

  eventSource.onerror = () => {
    console.warn("SSE connection error, retrying in 1 second...")
    eventSource.close()
    // Throttle retries to 1 second minimum
    setTimeout(() => {
      subscribeToRunStatus(runId, onStatus, onDone, onError, onClose)
    }, 1000)
  }

  // Return cleanup function
  return () => {
    eventSource.close()
  }
}
