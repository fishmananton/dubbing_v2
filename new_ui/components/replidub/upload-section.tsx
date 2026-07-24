"use client"

import { useCallback } from "react"
import { UploadCloud, FileVideo, RefreshCw, CheckCircle2, Youtube, Link } from "lucide-react"
import { Progress } from "@/components/ui/progress"
import { useState } from "react"

interface UploadSectionProps {
  onUploaded: (file: File) => void
  onYoutubeUrl?: (url: string) => void
  onYoutubeReset?: () => void
  uploadedFile?: { name: string; url: string } | null
  uploadError?: string
  uploadProgress?: number | null
  ytProgress?: number | null
  ytStatus?: "idle" | "downloading" | "done" | "error"
}

export function UploadSection({
  onUploaded,
  onYoutubeUrl,
  onYoutubeReset,
  uploadedFile,
  uploadError,
  uploadProgress,
  ytProgress,
  ytStatus = "idle",
}: UploadSectionProps) {
  const [tab, setTab] = useState<"file" | "youtube">("file")
  const [ytUrl, setYtUrl] = useState("")

  const fileUploading = uploadProgress !== null && uploadProgress !== undefined && uploadProgress < 100
  const isYtActive = ytStatus === "downloading"
  const isYtDone   = ytStatus === "done"
  const tabSwitchDisabled = fileUploading || isYtActive

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && (file.type.startsWith("video/") || file.name.endsWith(".mov"))) {
      onUploaded(file)
    }
  }, [onUploaded])

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) onUploaded(file)
  }

  const handleYoutubeSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = ytUrl.trim()
    if (!trimmed) return
    onYoutubeUrl?.(trimmed)
  }

  return (
    <div className="rounded-xl border border-border bg-card p-5 shadow-sm">
      {/* Header + tabs */}
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-foreground">Upload Video</h2>
        <div className="flex items-center gap-1 rounded-lg bg-secondary p-1 text-xs">
          <button
            type="button"
            onClick={() => !tabSwitchDisabled && setTab("file")}
            disabled={tabSwitchDisabled}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 font-medium transition-colors ${
              tab === "file"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            <UploadCloud className="h-3.5 w-3.5" />
            File
          </button>
          <button
            type="button"
            onClick={() => !tabSwitchDisabled && setTab("youtube")}
            disabled={tabSwitchDisabled}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 font-medium transition-colors ${
              tab === "youtube"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            <Youtube className="h-3.5 w-3.5" />
            YouTube
          </button>
        </div>
      </div>

      {/* ---- FILE TAB ---- */}
      {tab === "file" && (
        <>
          {!uploadedFile && !fileUploading && (
            <label
              htmlFor="video-upload"
              onDragOver={(e) => { e.preventDefault() }}
              onDrop={handleDrop}
              className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-border bg-secondary hover:border-primary/50 hover:bg-accent/40 py-10 transition-all"
            >
              <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-muted">
                <UploadCloud className="h-6 w-6 text-muted-foreground" />
              </div>
              <p className="text-sm font-medium text-foreground">Drag & drop your video here</p>
              <p className="mt-0.5 text-xs text-foreground/50">or click to browse</p>
              <p className="mt-2 text-xs text-muted-foreground">MP4, MOV, AVI, WebM, and MKV · max 2 GB</p>
              <p className="mt-3 text-[11px] text-muted-foreground/60">You must have rights to all content and voices used</p>
              <input
                id="video-upload"
                type="file"
                accept="video/*,.mov"
                className="sr-only"
                onChange={handleFileInput}
              />
            </label>
          )}

          {fileUploading && (
            <div className="rounded-xl border border-border bg-secondary p-5">
              <div className="mb-3 flex items-center gap-3">
                <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-accent">
                  <FileVideo className="h-5 w-5 text-accent-foreground" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="truncate text-sm font-medium text-foreground">Uploading…</p>
                  <p className="text-xs text-muted-foreground">{uploadProgress ?? 0}%</p>
                </div>
              </div>
              <Progress value={uploadProgress ?? 0} className="h-1.5" />
            </div>
          )}

          {uploadedFile && !fileUploading && (
            <div className="flex items-center gap-3 rounded-xl border border-border bg-secondary p-4">
              <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-success/10">
                <CheckCircle2 className="h-5 w-5 text-success" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="truncate text-sm font-medium text-foreground">{uploadedFile.name}</p>
                <p className="text-xs text-muted-foreground">Upload complete</p>
              </div>
              <label
                htmlFor="video-reupload"
                className="rounded-md p-1 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors cursor-pointer"
                title="Upload different video"
              >
                <RefreshCw className="h-4 w-4" />
                <input
                  id="video-reupload"
                  type="file"
                  accept="video/*,.mov"
                  className="sr-only"
                  onChange={handleFileInput}
                />
              </label>
            </div>
          )}
        </>
      )}

      {/* ---- YOUTUBE TAB ---- */}
      {tab === "youtube" && (
        <div className="space-y-3">
          {!isYtActive && !isYtDone && (
            <form onSubmit={handleYoutubeSubmit}>
              <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                YouTube URL
              </label>
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Link className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <input
                    type="url"
                    value={ytUrl}
                    onChange={(e) => setYtUrl(e.target.value)}
                    placeholder="https://www.youtube.com/watch?v=..."
                    className="w-full rounded-lg border border-border bg-secondary pl-9 pr-3 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                  />
                </div>
                <button
                  type="submit"
                  disabled={!ytUrl.trim()}
                  className="flex items-center gap-1.5 rounded-lg bg-gradient-to-r from-primary to-brand-end px-4 py-2.5 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
                >
                  <Youtube className="h-4 w-4" />
                  Import
                </button>
              </div>
              <p className="mt-1.5 text-[11px] text-muted-foreground">
                The video will be downloaded server-side directly from YouTube.
              </p>
              <p className="mt-1 text-[11px] text-muted-foreground/60">You must have rights to all content and voices used</p>
            </form>
          )}

          {isYtActive && (
            <div className="rounded-xl border border-border bg-secondary p-5">
              <div className="mb-3 flex items-center gap-3">
                <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-accent">
                  <Youtube className="h-5 w-5 text-accent-foreground" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="truncate text-sm font-medium text-foreground">Downloading from YouTube…</p>
                  <p className="text-xs text-muted-foreground">{Math.round(ytProgress ?? 0)}%</p>
                </div>
              </div>
              <Progress value={ytProgress ?? 0} className="h-1.5" />
            </div>
          )}

          {isYtDone && (
            <div className="flex items-center gap-3 rounded-xl border border-border bg-secondary p-4">
              <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-success/10">
                <CheckCircle2 className="h-5 w-5 text-success" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="truncate text-sm font-medium text-foreground">
                  {uploadedFile?.name ?? "YouTube video"}
                </p>
                <p className="text-xs text-muted-foreground">Download complete</p>
              </div>
              <button
                type="button"
                onClick={() => { setYtUrl(""); onYoutubeReset?.() }}
                className="rounded-md p-1 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors cursor-pointer"
                title="Import different video"
              >
                <RefreshCw className="h-4 w-4" />
              </button>
            </div>
          )}
        </div>
      )}

      {uploadError && (
        <div className="mt-3 rounded-lg border border-destructive/30 bg-destructive/10 p-3">
          <p className="text-xs font-medium text-destructive">{uploadError}</p>
        </div>
      )}
    </div>
  )
}
