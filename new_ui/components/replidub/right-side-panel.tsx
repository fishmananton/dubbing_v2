"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { Loader2, ChevronDown, ChevronUp, Pencil, Check, X, Video, Music, Download } from "lucide-react"
import { LANGUAGES, type ProjectSetupConfig } from "./project-setup-panel"
import { startExport, getExportStatus, downloadExportFile, downloadAudio, type PricingConfig } from "@/lib/api"
import { calculateCost, formatCents } from "@/lib/pricing"

type ExportStatus = "idle" | "processing" | "ready" | "error"

interface ProjectActionsPanelProps {
  projectName?: string
  onRenameProject?: (name: string) => void
  config?: ProjectSetupConfig
  completedLanguages?: string[]
  runId?: string | null
  durationMinutes?: number | null
  pricing?: PricingConfig | null
  fileName?: string
}

export function ProjectActionsPanel({
  projectName = "YouTube — Spanish Pack",
  onRenameProject,
  config,
  completedLanguages = ["es", "fr", "de"],
  runId,
  durationMinutes,
  pricing,
  fileName,
}: ProjectActionsPanelProps) {
  const [quality, setQuality] = useState("1080")
  const [exportStatus, setExportStatus] = useState<ExportStatus>("idle")
  const [isDownloadingAudio, setIsDownloadingAudio] = useState(false)
  const [isEditingName, setIsEditingName] = useState(false)
  const [editName, setEditName] = useState(projectName)
  const [showParams, setShowParams] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Check existing export status on mount and whenever quality or runId changes
  useEffect(() => {
    stopPolling()
    if (!runId) {
      setExportStatus("idle")
      return
    }
    getExportStatus(runId, quality)
      .then(({ status }) => {
        if (status === "ready") {
          setExportStatus("ready")
        } else if (status === "processing") {
          setExportStatus("processing")
          pollRef.current = setInterval(async () => {
            try {
              const { status: s } = await getExportStatus(runId, quality)
              if (s === "ready") { stopPolling(); setExportStatus("ready") }
              else if (s === "error") { stopPolling(); setExportStatus("error") }
            } catch { stopPolling(); setExportStatus("error") }
          }, 2000)
        } else {
          setExportStatus("idle")
        }
      })
      .catch(() => setExportStatus("idle"))
  }, [quality, runId])

  // Cleanup on unmount
  useEffect(() => () => stopPolling(), [])

  function stopPolling() {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  const handleExportVideo = async () => {
    if (!runId) return
    setExportStatus("processing")
    try {
      await startExport(runId, quality)
      pollRef.current = setInterval(async () => {
        try {
          const { status } = await getExportStatus(runId, quality)
          if (status === "ready") {
            stopPolling()
            setExportStatus("ready")
          } else if (status === "error") {
            stopPolling()
            setExportStatus("error")
          }
        } catch {
          stopPolling()
          setExportStatus("error")
        }
      }, 2000)
    } catch (e) {
      console.error("[export] Failed to start export:", e)
      setExportStatus("error")
    }
  }

  const handleDownloadExport = async () => {
    if (!runId) return
    try {
      await downloadExportFile(runId, quality, fileName)
    } catch (e) {
      console.error("[export] Download failed:", e)
    }
  }

  const handleDownloadAudio = async () => {
    if (!runId) return
    setIsDownloadingAudio(true)
    try {
      await downloadAudio(runId, fileName)
    } catch (e) {
      console.error("[export] Audio download failed:", e)
    } finally {
      setIsDownloadingAudio(false)
    }
  }

  const handleSaveName = () => {
    if (editName.trim() && onRenameProject) {
      onRenameProject(editName.trim())
    }
    setIsEditingName(false)
  }

  const completedLangsList = completedLanguages.map(code => {
    const lang = LANGUAGES.find(l => l.code === code)
    return lang || { code, label: code.toUpperCase(), flag: code.toUpperCase().slice(0, 2) }
  })

  return (
    <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {/* Column 1: Export */}
        <div className="space-y-2">
          <Label className="text-[10px] font-medium text-muted-foreground block uppercase tracking-wide">Export</Label>
          <Select value={quality} onValueChange={setQuality} disabled={exportStatus === "processing"}>
            <SelectTrigger className="h-8 text-xs border-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1080">1080p</SelectItem>
              <SelectItem value="720">720p</SelectItem>
              <SelectItem value="480">480p</SelectItem>
              <SelectItem value="original">Original</SelectItem>
            </SelectContent>
          </Select>
          <div className="flex gap-2">
            {exportStatus === "ready" ? (
              <Button
                variant="default"
                className="flex-1 h-8 gap-1.5 text-xs"
                onClick={handleDownloadExport}
              >
                <Download className="h-3.5 w-3.5" />
                Download
              </Button>
            ) : (
              <Button
                variant="outline"
                className="flex-1 h-8 gap-1.5 text-xs"
                onClick={exportStatus === "error" ? handleExportVideo : handleExportVideo}
                disabled={exportStatus === "processing" || !runId}
              >
                {exportStatus === "processing" ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Video className="h-3.5 w-3.5" />
                )}
                {exportStatus === "processing" ? "Exporting…" : exportStatus === "error" ? "Retry" : "Export Video"}
              </Button>
            )}
            <Button
              variant="outline"
              className="flex-1 h-8 gap-1.5 text-xs"
              onClick={handleDownloadAudio}
              disabled={isDownloadingAudio || !runId}
            >
              {isDownloadingAudio ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Music className="h-3.5 w-3.5" />
              )}
              Audio
            </Button>
          </div>
          {exportStatus === "error" && (
            <p className="text-[10px] text-destructive">Export failed. Please retry.</p>
          )}
        </div>

        {/* Column 2: Project Parameters */}
        <div>
          <Collapsible open={showParams} onOpenChange={setShowParams}>
            <CollapsibleTrigger className="flex w-full items-center justify-between mb-2">
              <Label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">Project Parameters</Label>
              {showParams ? (
                <ChevronUp className="h-3.5 w-3.5 text-muted-foreground" />
              ) : (
                <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
              )}
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="rounded-lg border border-border bg-secondary/50 p-2.5 space-y-1.5 text-[10px]">
                <div className="flex justify-between py-0.5 border-b border-border">
                  <span className="text-muted-foreground">Audio Mixl</span>
                  <span className="font-medium text-foreground">
                    {config?.ttsMode === "original_voice" ? "Original Voice" : "Natural"}
                  </span>
                </div>
                <div className="flex justify-between py-0.5 border-b border-border">
                  <span className="text-muted-foreground">Audio Mix</span>
                  <span className="font-medium text-foreground">
                    {config?.voiceMode === "overlay" ? "Overlay" : "New Voice"}
                  </span>
                </div>
                <div className="flex justify-between py-0.5 border-b border-border">
                  <span className="text-muted-foreground">Emotions</span>
                  <span className="font-medium text-foreground">
                    {config?.useEmotions ? "On" : "Off"}
                  </span>
                </div>
                <div className="flex justify-between py-0.5 border-b border-border">
                  <span className="text-muted-foreground">Transcription</span>
                  <span className="font-medium text-foreground capitalize">{config?.transcribeMode || "Transcribe"}</span>
                </div>
                <div className="flex justify-between py-0.5 border-b border-border">
                  <span className="text-muted-foreground">Source Language</span>
                  <span className="font-medium text-foreground capitalize">{config?.sourceLang === "auto" ? "Auto-detect" : config?.sourceLang || "Auto-detect"}</span>
                </div>
                <div className="flex justify-between py-0.5 border-b border-border">
                  <span className="text-muted-foreground">Fix Timing</span>
                  <span className="font-medium text-foreground">{config?.fixTiming !== false ? "On" : "Off"}</span>
                </div>
                {config?.voiceMode !== "overlay" && (
                  <div className="flex justify-between py-0.5 border-b border-border">
                    <span className="text-muted-foreground">Ambient Sounds</span>
                    <span className="font-medium text-foreground">{config?.useNonSpeech !== false ? "On" : "Off"}</span>
                  </div>
                )}
                {durationMinutes && pricing && (
                  <div className="flex justify-between py-0.5 border-b border-border">
                    <span className="text-muted-foreground">Run cost</span>
                    <span className="font-medium text-destructive">
                      {formatCents(calculateCost(durationMinutes, { fixTiming: config?.fixTiming ?? true }, pricing))}
                    </span>
                  </div>
                )}
                <div className="flex justify-between py-0.5">
                  <span className="text-muted-foreground">Translated to</span>
                  <div className="flex flex-wrap gap-1 justify-end">
                    {completedLangsList.map((lang) => (
                      <div
                        key={lang.code}
                        className="flex items-center gap-0.5 rounded-full border border-border bg-card px-1.5 py-0.5 text-[9px] font-medium text-foreground"
                      >
                        <span className="text-[8px]">{lang.flag}</span>
                        {lang.label}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CollapsibleContent>
          </Collapsible>

          {!showParams && (
            <div className="rounded-lg border border-border bg-secondary/50 p-2.5 text-[10px] space-y-1">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Audio Mixl</span>
                <span className="font-medium text-foreground">{config?.ttsMode === "original_voice" ? "Original Voice" : "Natural"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Audio Mix</span>
                <span className="font-medium text-foreground">{config?.voiceMode === "overlay" ? "Overlay" : "New Voice"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Source Language</span>
                <span className="font-medium text-foreground">{config?.sourceLang === "auto" ? "Auto-detect" : config?.sourceLang || "Auto-detect"}</span>
              </div>
              {config?.voiceMode !== "overlay" && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Ambient Sounds</span>
                  <span className="font-medium text-foreground">{config?.useNonSpeech !== false ? "On" : "Off"}</span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-muted-foreground">Translated to</span>
                <span className="font-medium text-foreground">{completedLangsList.map(l => l.label).join(", ") || "—"}</span>
              </div>
            </div>
          )}
        </div>

        {/* Column 3: Project Name */}
        <div>
          <Label className="text-[10px] font-medium text-muted-foreground mb-1 block uppercase tracking-wide">Project Name</Label>
          {isEditingName ? (
            <div className="flex items-center gap-1.5">
              <Input
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                className="h-8 text-sm"
                autoFocus
                onKeyDown={(e) => e.key === "Enter" && handleSaveName()}
              />
              <button
                onClick={handleSaveName}
                className="flex h-8 w-8 items-center justify-center rounded-md bg-gradient-to-r from-primary to-brand-end text-primary-foreground hover:opacity-90"
              >
                <Check className="h-3.5 w-3.5" />
              </button>
              <button
                onClick={() => { setIsEditingName(false); setEditName(projectName) }}
                className="flex h-8 w-8 items-center justify-center rounded-md border border-border bg-secondary text-muted-foreground hover:bg-muted"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-1.5">
              <p className="text-sm font-semibold text-foreground truncate flex-1">{projectName}</p>
              <button
                onClick={() => setIsEditingName(true)}
                className="flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
              >
                <Pencil className="h-3.5 w-3.5" />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export function RightSidePanel(props: ProjectActionsPanelProps) {
  return <ProjectActionsPanel {...props} />
}

export function ProjectParametersPanel({ config, completedLanguages = [] }: { config?: ProjectSetupConfig; completedLanguages?: string[] }) {
  return null
}
