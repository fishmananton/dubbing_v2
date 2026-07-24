"use client"

import { useState, useCallback, useEffect, useRef, Suspense } from "react"
import { useSearchParams } from "next/navigation"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { useRouter } from "next/navigation"
import { AppHeader, type Project } from "@/components/replidub/app-header"
import { UploadSection } from "@/components/replidub/upload-section"
import { ProjectSetupPanel, clearProjectConfig, type ProjectSetupConfig } from "@/components/replidub/project-setup-panel"
import { ProcessingTimeline } from "@/components/replidub/processing-timeline"
import { VideoResultPanel } from "@/components/replidub/video-result-panel"
import { SubtitleEditor } from "@/components/replidub/subtitle-editor"
import { ProjectActionsPanel } from "@/components/replidub/right-side-panel"
import { cn } from "@/lib/utils"
import {
  createProject,
  listProjects,
  uploadRunVideo,
  startRun,
  subscribeToRunStatus,
  getProjectStatus,
  regenerateSubtitles,
  deleteProject,
  getUserBalance,
  getPricing,
  startYoutubeDownload,
  subscribeToYoutubeProgress,
  InsufficientBalanceError,
  type RunStatusEvent,
  type RunErrorEvent,
  type PricingConfig,
  type YtProgressEvent,
} from "@/lib/api"

type AppPhase = "setup" | "processing" | "results"

function configFromStatus(status: import("@/lib/api").ProjectStatus, prev?: ProjectSetupConfig | null): ProjectSetupConfig {
  return {
    selectedLang: status.dst_language ?? prev?.selectedLang ?? "es",
    ttsMode: status.ttsmodel === 3 ? "original_voice" : "natural",
    voiceMode: status.is_dubbed != null ? (status.is_dubbed ? "overlay" : "new_voice") : (prev?.voiceMode ?? "new_voice"),
    transcribeMode: status.trans_type != null ? (status.trans_type === "ocr" ? "ocr" : "transcribe") : (prev?.transcribeMode ?? "transcribe"),
    sourceLang: status.src_language ?? prev?.sourceLang ?? "auto",
    useEmotions: status.emotions_flag ?? prev?.useEmotions ?? true,
    fixTiming: status.fix_timing ?? prev?.fixTiming ?? true,
    numSpeakers: prev?.numSpeakers ?? "auto",
    useNonSpeech: status.use_non_speech ?? prev?.useNonSpeech ?? true,
  }
}

export interface ProjectState {
  id: number
  name: string
  run_id?: string
  status: "initial" | "uploaded" | "processing" | "finished" | "failed"
}

function DashboardPageInner() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [phase, setPhase] = useState<AppPhase>("setup")
  const [currentTime, setCurrentTime] = useState(0)
  const [darkMode, setDarkMode] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("theme") === "dark"
    }
    return false
  })

  // Projects state
  const [projects, setProjects] = useState<ProjectState[]>([])
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(null)
  const [projectConfig, setProjectConfig] = useState<ProjectSetupConfig | null>(null)
  const [selectedSubtitleLang, setSelectedSubtitleLang] = useState("es")
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoadingAuth, setIsLoadingAuth] = useState(true)
  const [isLoadingProjects, setIsLoadingProjects] = useState(true)
  const [userBalance, setUserBalance] = useState<number | undefined>(undefined)
  const [pricing, setPricing] = useState<PricingConfig | null>(null)
  const [videoDurationMinutes, setVideoDurationMinutes] = useState<number | null>(null)
  const [paymentSuccessBanner, setPaymentSuccessBanner] = useState(false)
  const [insufficientBalanceError, setInsufficientBalanceError] = useState<{ required_cents: number; balance_cents: number } | null>(null)
  const [showBillingModal, setShowBillingModal] = useState(false)
  const [currentUser, setCurrentUser] = useState<{
    firstName: string
    lastName: string
    email: string
    authProvider: string
  } | undefined>(undefined)

  // File upload state
  const [uploadedFile, setUploadedFile] = useState<{ name: string; url: string } | null>(null)
  const [uploadError, setUploadError] = useState("")
  const [uploadProgress, setUploadProgress] = useState<number | null>(null)
  const [inputVideoUrl, setInputVideoUrl] = useState<string | null>(null)

  // YouTube import state
  const [ytProgress, setYtProgress] = useState<number | null>(null)
  const [ytStatus, setYtStatus] = useState<"idle" | "downloading" | "done" | "error">("idle")

  // Processing state
  const [currentRunId, setCurrentRunId] = useState<string | null>(null)
  const [currentTtsModel, setCurrentTtsModel] = useState<number>(1)
  const [savedMixGains, setSavedMixGains] = useState<[number, number, number, number] | null>(null)
  const [stages, setStages] = useState<Record<string, "not_started" | "in_progress" | "done">>({})
  const [processingError, setProcessingError] = useState("")
  const [lastFailedError, setLastFailedError] = useState("")
  const [processingPercent, setProcessingPercent] = useState<number>(0)
  
  // Video playback control ref
  const [shouldPlayAfterSeek, setShouldPlayAfterSeek] = useState(false)
  const [isVideoPlaying, setIsVideoPlaying] = useState(false)
  const videoTogglePlayRef = useRef<(() => void) | null>(null)
  const sseUnsubscribeRef = useRef<(() => void) | null>(null)

  // Check authentication on mount
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const res = await fetch(
          `${process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"}/auth/me`,
          { credentials: "include" }
        )
        if (res.ok) {
          const data = await res.json()
          const u = data.user
          setCurrentUser({
            firstName: u.first_name || "",
            lastName: u.last_name || "",
            email: u.email || "",
            authProvider: u.auth_provider || "local",
          })
          setIsAuthenticated(true)
        } else {
          router.push("/login")
        }
      } catch (e) {
        console.error("Auth check failed:", e)
        router.push("/login")
      } finally {
        setIsLoadingAuth(false)
      }
    }
    checkAuth()
  }, [router])

  const fetchBalance = async () => {
    try {
      const balance = await getUserBalance()
      setUserBalance(balance)
    } catch (e) {
      console.error("Failed to fetch balance:", e)
    }
  }

  // Load projects, balance and pricing on auth
  useEffect(() => {
    if (isAuthenticated) {
      loadProjects()
      fetchBalance()
      getPricing().then(setPricing).catch(console.error)
    }
  }, [isAuthenticated])

  // Handle Stripe payment success redirect
  useEffect(() => {
    if (!isAuthenticated) return
    const params = new URLSearchParams(window.location.search)
    if (params.get("payment") === "success") {
      setPaymentSuccessBanner(true)
      fetchBalance()
      window.history.replaceState({}, "", window.location.pathname)
      setTimeout(() => setPaymentSuccessBanner(false), 5000)
    }
  }, [isAuthenticated])

  // Set first project as selected by default, or restore from URL param
  useEffect(() => {
    if (projects.length === 0 || selectedProjectId) return
    const paramId = searchParams.get("project")
    const idToSelect = paramId ? projects.find(p => p.id === Number(paramId))?.id : null
    handleSelectProject(idToSelect ?? projects[0].id).finally(() => setIsLoadingProjects(false))
  }, [projects, selectedProjectId])

  // Keep ?project= URL param in sync with selected project
  useEffect(() => {
    if (!isAuthenticated || !selectedProjectId) return
    const params = new URLSearchParams(window.location.search)
    params.set("project", String(selectedProjectId))
    window.history.replaceState({}, "", `${window.location.pathname}?${params}`)
  }, [selectedProjectId, isAuthenticated])

  // Apply dark mode and persist preference
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark")
      localStorage.setItem("theme", "dark")
    } else {
      document.documentElement.classList.remove("dark")
      localStorage.setItem("theme", "light")
    }
  }, [darkMode])

  // Global Space key handler for video play/pause (only when not editing)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle Space when not in an editable element
      if (e.code === "Space" && 
          !(e.target instanceof HTMLInputElement) && 
          !(e.target instanceof HTMLTextAreaElement) &&
          !(e.target instanceof HTMLSelectElement)) {
        e.preventDefault()
        videoTogglePlayRef.current?.()
      }
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [])

  const loadProjects = async () => {
    try {
      const projectsList = await listProjects()
      const mapped = projectsList.map((p) => ({
        id: p.id,
        name: p.project_name,
        run_id: p.run_id,
        status: "initial" as const,
      }))
      setProjects(mapped)
      if (mapped.length === 0) setIsLoadingProjects(false)
    } catch (e) {
      console.error("Failed to load projects:", e)
      setIsLoadingProjects(false)
    }
  }

  const handleNewProject = async (name: string) => {
    try {
      const newProject = await createProject(name)
      const projectState: ProjectState = {
        id: newProject.id,
        name: newProject.project_name,
        status: "initial",
      }
      setProjects((prev) => [projectState, ...prev])
      setSelectedProjectId(newProject.id)
      setPhase("setup")
      setUploadedFile(null)
      setUploadError("")
      setProjectConfig(null)
      setCurrentRunId(null)
      setYtStatus("idle")
      setYtProgress(null)
      setVideoDurationMinutes(null)
    } catch (e) {
      console.error("Failed to create project:", e)
    }
  }

  const handleSelectProject = async (projectId: number) => {
    setSelectedProjectId(projectId)
    setInputVideoUrl(null)
    setLastFailedError("")
    setVideoDurationMinutes(null)
    try {
      const status = await getProjectStatus(projectId)
      if (status.duration_minutes) setVideoDurationMinutes(status.duration_minutes)
      if (status.dst_language) setSelectedSubtitleLang(status.dst_language)
      if (status.mix_gains) setSavedMixGains(status.mix_gains)
      if (status.ttsmodel != null) {
        setCurrentTtsModel(status.ttsmodel)
        setProjectConfig((prev) => configFromStatus(status, prev))
      }
      if (status.status === "finished" && status.run_id) {
        setCurrentRunId(status.run_id)
        setPhase("results")
      } else if (status.status === "processing" && status.run_id) {
        setCurrentRunId(status.run_id)
        setPhase("processing")
        subscribeToProcessing(status.run_id)
      } else if (status.status === "uploaded" && status.run_id) {
        setCurrentRunId(status.run_id)
        setPhase("setup")
        setUploadedFile({ name: status.file_name || "uploaded_video.mp4", url: status.run_id })
        setInputVideoUrl(`${process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"}/runs/${status.run_id}/input_video`)
      } else if (status.status === "failed" && status.run_id) {
        setCurrentRunId(status.run_id)
        setPhase("setup")
        setUploadedFile({ name: status.file_name || "uploaded_video.mp4", url: status.run_id })
        setInputVideoUrl(`${process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"}/runs/${status.run_id}/input_video`)
        setLastFailedError("Previous processing run failed. You can reconfigure and try again.")
      } else {
        // initial status or unknown — treat same as a fresh new project
        setPhase("setup")
        setUploadedFile(null)
        setCurrentRunId(null)
        setInputVideoUrl(null)
      }
    } catch (e) {
      console.error("Failed to get project status:", e)
      setPhase("setup")
    }
  }

  const handleFileUpload = async (file: File) => {
    if (!selectedProjectId) {
      setUploadError("No project selected")
      return
    }

    try {
      setUploadError("")
      setUploadProgress(0)
      const result = await uploadRunVideo(selectedProjectId, file, setUploadProgress)
      setUploadProgress(null)
      setCurrentRunId(result.run_id)
      if (result.duration_minutes) setVideoDurationMinutes(result.duration_minutes)
      setUploadedFile({
        name: file.name,
        url: result.run_id,
      })
      setInputVideoUrl(`${process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"}/runs/${result.run_id}/input_video`)
      // Update project with new run_id
      setProjects((prev) =>
        prev.map((p) =>
          p.id === selectedProjectId ? { ...p, run_id: result.run_id, status: "uploaded" as const } : p
        )
      )
    } catch (e) {
      setUploadProgress(null)
      setUploadError(e instanceof Error ? e.message : "Upload failed")
      console.error("Upload error:", e)
    }
  }

  const handleYoutubeReset = () => {
    setYtStatus("idle")
    setYtProgress(null)
    setUploadedFile(null)
    setCurrentRunId(null)
    setInputVideoUrl(null)
  }

  const handleYoutubeUrl = async (url: string) => {
    if (!selectedProjectId) {
      setUploadError("No project selected")
      return
    }
    try {
      setUploadError("")
      setUploadedFile(null)
      setInputVideoUrl(null)
      setYtStatus("downloading")
      setYtProgress(0)

      const result = await startYoutubeDownload(selectedProjectId, url)
      const run_id = result?.run_id
      if (!run_id) throw new Error("Server did not return a run_id")
      setCurrentRunId(run_id)

      subscribeToYoutubeProgress(
        run_id,
        (data: YtProgressEvent) => {
          setYtProgress(data.percent)
        },
        (data: YtProgressEvent) => {
          setYtProgress(100)
          setYtStatus("done")
          if (data.duration_minutes) setVideoDurationMinutes(data.duration_minutes)
          const name = data.filename ?? "youtube_video.mp4"
          setUploadedFile({ name, url: run_id })
          setInputVideoUrl(`${process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"}/runs/${run_id}/input_video`)
          setProjects((prev) =>
            prev.map((p) =>
              p.id === selectedProjectId ? { ...p, run_id, status: "uploaded" as const } : p
            )
          )
        },
        (data: YtProgressEvent) => {
          setYtStatus("error")
          setUploadError(data.error ?? "YouTube download failed")
        }
      )
    } catch (e) {
      setYtStatus("error")
      setUploadError(e instanceof Error ? e.message : "YouTube download failed")
    }
  }

  const subscribeToProcessing = (runId: string) => {
    // Close any existing SSE connection before opening a new one
    sseUnsubscribeRef.current?.()
    sseUnsubscribeRef.current = null

    const unsubscribe = subscribeToRunStatus(
      runId,
      (data: RunStatusEvent) => {
        setStages(data.stages)
        if (data.percent !== undefined) setProcessingPercent(data.percent)
      },
      async (data: RunStatusEvent) => {
        setStages(data.stages)
        if (data.percent !== undefined) setProcessingPercent(data.percent)
        setLastFailedError("")
        setProcessingError("")
        if (selectedProjectId) {
          const status = await getProjectStatus(selectedProjectId)
          if (status.mix_gains) setSavedMixGains(status.mix_gains)
          setProjectConfig((prev) => configFromStatus(status, prev))
        }
        setPhase("results")
        fetchBalance()
      },
      (data: RunErrorEvent) => {
        setProcessingError(data.error)
        setLastFailedError(data.error || "Processing failed. You can reconfigure and try again.")
        setPhase("setup")
      },
      () => {
        console.log("SSE connection closed")
      }
    )

    sseUnsubscribeRef.current = unsubscribe
  }

  // Close SSE connection when the page unmounts
  useEffect(() => {
    return () => { sseUnsubscribeRef.current?.() }
  }, [])

  const handleStart = async (config: ProjectSetupConfig) => {
    if (!currentRunId || !selectedProjectId) {
      console.error("Missing run_id or project_id")
      return
    }

    setInsufficientBalanceError(null)

    try {
      setProcessingError("")
      setLastFailedError("")
      setProcessingPercent(0)
      setProjectConfig(config)
      setSelectedSubtitleLang(config.selectedLang)

      const ttsmodel = config.ttsMode === "original_voice" ? 3 : 1
      setCurrentTtsModel(ttsmodel)

      const flowRunId = await startRun({
        dst_language: config.selectedLang,
        trans_type: config.transcribeMode === "ocr" ? "ocr" : "default",
        elevenlabs: false,
        num_speakers: config.numSpeakers === "auto" ? null : config.numSpeakers,
        emotions_flag: config.useEmotions,
        elevenlabs_emotions: 1,
        fix_timing: config.fixTiming,
        changed_list: [],
        is_dubbed: config.voiceMode === "overlay",
        use_non_speech: config.voiceMode === "new_voice" ? (config.useNonSpeech ?? true) : false,
        ttsmodel,
        run_id: currentRunId,
        stage: 0,
      })

      setPhase("processing")
      console.log("Flow run started:", flowRunId)
      subscribeToProcessing(currentRunId)
    } catch (e) {
      if (e instanceof InsufficientBalanceError) {
        setInsufficientBalanceError({ required_cents: e.required_cents, balance_cents: e.balance_cents })
      } else {
        setProcessingError(e instanceof Error ? e.message : "Failed to start processing")
        console.error("Start run error:", e)
      }
    }
  }

  const handleDeleteProject = async (projectId: number) => {
    try {
      await deleteProject(projectId)
      clearProjectConfig(projectId)
      setProjects((prev) => prev.filter((p) => p.id !== projectId))
      if (selectedProjectId === projectId) {
        setSelectedProjectId(null)
        setPhase("setup")
        setUploadedFile(null)
        setCurrentRunId(null)
        setInputVideoUrl(null)
      }
    } catch (e) {
      console.error("Failed to delete project:", e)
    }
  }

  const handleRenameProject = (newName: string) => {
    if (!selectedProjectId) return
    setProjects((prev) =>
      prev.map((p) => (p.id === selectedProjectId ? { ...p, name: newName } : p))
    )
  }

  const handleTimeChangeWithPlay = (time: number, shouldPlay?: boolean) => {
    setCurrentTime(time)
    if (shouldPlay) {
      setShouldPlayAfterSeek(true)
    }
  }

  const handleRegenerate = async (subtitles: string, changedList: number[]) => {
    if (!currentRunId) return

    setInsufficientBalanceError(null)

    try {
      setProcessingError("")
      setProcessingPercent(0)
      setStages({})

      await regenerateSubtitles(currentRunId, subtitles, changedList, currentTtsModel)
      setPhase("processing")
      subscribeToProcessing(currentRunId)
    } catch (e) {
      if (e instanceof InsufficientBalanceError) {
        setInsufficientBalanceError({ required_cents: e.required_cents, balance_cents: e.balance_cents })
        setPhase("results")
      } else {
        setProcessingError(e instanceof Error ? e.message : "Failed to regenerate")
        setPhase("results")
        console.error("Regenerate error:", e)
      }
    }
  }

  const currentProject = projects.find((p) => p.id === selectedProjectId)

  const handleLogout = async () => {
    try {
      await fetch(
        `${process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"}/auth/logout`,
        { method: "POST", credentials: "include" }
      )
    } catch (e) {
      console.error("Logout error:", e)
    }
    router.push("/login")
  }

  return (
    <div className="flex flex-col h-screen bg-background">
      {isLoadingAuth || !isAuthenticated || isLoadingProjects ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="mx-auto mb-4 h-8 w-8 rounded-full border-2 border-[#7247ED] border-t-transparent animate-spin" />
            <p className="text-sm font-medium text-foreground">Loading...</p>
          </div>
        </div>
      ) : (
        <>
          <AppHeader
            projects={projects}
            selectedProjectId={selectedProjectId}
            onSelectProject={handleSelectProject}
            onNewProject={handleNewProject}
            onDeleteProject={handleDeleteProject}
            transferring={(uploadProgress !== null && uploadProgress < 100) || ytStatus === "downloading"}
            darkMode={darkMode}
            onDarkModeChange={setDarkMode}
            onLogout={handleLogout}
            balance={userBalance}
            currentUser={currentUser}
            billingOpen={showBillingModal}
            onBillingOpenChange={setShowBillingModal}
          />

      <main className="flex-1 px-4 py-4">
        {/* Payment success banner */}
        {paymentSuccessBanner && (
          <div className="mb-4 rounded-xl border border-green-500/30 bg-green-500/10 p-3 text-sm text-green-700 dark:text-green-400">
            Payment successful! Your balance has been updated.
          </div>
        )}

        {/* Insufficient balance modal */}
        <Dialog open={!!insufficientBalanceError} onOpenChange={(open) => { if (!open) setInsufficientBalanceError(null) }}>
          <DialogContent className="sm:max-w-xs">
            <DialogHeader>
              <DialogTitle className="text-destructive text-base">Insufficient Balance</DialogTitle>
            </DialogHeader>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Required</span>
                <span className="font-semibold">${((insufficientBalanceError?.required_cents ?? 0) / 100).toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Your balance</span>
                <span className="font-semibold">${((insufficientBalanceError?.balance_cents ?? 0) / 100).toFixed(2)}</span>
              </div>
            </div>
            <DialogFooter className="gap-2 sm:gap-0">
              <Button variant="outline" size="sm" onClick={() => setInsufficientBalanceError(null)}>
                Cancel
              </Button>
              <Button size="sm" onClick={() => { setInsufficientBalanceError(null); setShowBillingModal(true) }}>
                Add Funds
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* EMPTY STATE - No projects */}
        {projects.length === 0 && phase === "setup" && (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-accent">
                <svg className="h-8 w-8 text-accent-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                </svg>
              </div>
              <h2 className="mb-2 text-2xl font-semibold text-foreground">Create Your First Project</h2>
              <p className="mb-8 text-sm text-muted-foreground">Start by creating a new project to begin translating and dubbing your videos</p>
              <button
                onClick={() => {
                  const dialog = document.createElement("div")
                  dialog.innerHTML = `
                    <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
                      <div class="rounded-lg bg-card p-6 shadow-lg border border-border">
                        <h3 class="text-lg font-semibold mb-4 text-foreground">New Project</h3>
                        <input type="text" id="project-name" placeholder="Project name" class="w-full px-3 py-2 border border-border rounded-md mb-4 text-foreground bg-background" />
                        <div class="flex gap-3">
                          <button id="cancel-btn" class="px-4 py-2 text-sm rounded-md border border-border text-foreground hover:bg-secondary">Cancel</button>
                          <button id="create-btn" style="background:linear-gradient(to right,#7247ED,#20B2E1)" class="px-4 py-2 text-sm rounded-md text-white">Create</button>
                        </div>
                      </div>
                    </div>
                  `
                  document.body.appendChild(dialog)
                  const input = dialog.querySelector("#project-name") as HTMLInputElement
                  const createBtn = dialog.querySelector("#create-btn") as HTMLButtonElement
                  const cancelBtn = dialog.querySelector("#cancel-btn") as HTMLButtonElement
                  input.focus()
                  const create = () => {
                    if (input.value.trim()) {
                      handleNewProject(input.value.trim())
                      dialog.remove()
                    }
                  }
                  createBtn.addEventListener("click", create)
                  cancelBtn.addEventListener("click", () => dialog.remove())
                  input.addEventListener("keydown", (e) => e.key === "Enter" && create())
                }}
                className="rounded-lg bg-gradient-to-r from-primary to-brand-end px-6 py-3 text-sm font-medium text-primary-foreground hover:opacity-90 transition-opacity"
              >
                New Project
              </button>
            </div>
          </div>
        )}

        {/* SETUP PHASE */}
        {phase === "setup" && projects.length > 0 && (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_340px] lg:h-[calc(100vh-180px)]">
            <div className="flex flex-col gap-4 min-h-0">
              {lastFailedError && (
                <div className="rounded-xl border border-destructive/30 bg-destructive/10 p-4">
                  <p className="text-sm font-medium text-destructive">Processing Failed</p>
                  <p className="text-xs text-destructive/90 mt-1">{lastFailedError}</p>
                </div>
              )}
              <UploadSection
                onUploaded={handleFileUpload}
                onYoutubeUrl={handleYoutubeUrl}
                onYoutubeReset={handleYoutubeReset}
                uploadedFile={uploadedFile}
                uploadError={uploadError}
                uploadProgress={uploadProgress}
                ytProgress={ytProgress}
                ytStatus={ytStatus}
              />
              {uploadedFile && inputVideoUrl && (
                <div className="aspect-video lg:aspect-auto lg:flex-1 rounded-xl border border-border bg-zinc-900 overflow-hidden lg:min-h-0">
                  <video
                    key={inputVideoUrl}
                    src={inputVideoUrl}
                    className="w-full h-full object-contain"
                    controls
                    playsInline
                  />
                </div>
              )}
            </div>
            <div className="overflow-y-auto">
              <ProjectSetupPanel
                onStart={handleStart}
                isLoading={!uploadedFile}
                durationMinutes={videoDurationMinutes}
                pricing={pricing}
                projectId={selectedProjectId}
              />
            </div>
          </div>
        )}

        {/* PROCESSING PHASE */}
        {phase === "processing" && (
          <div className="space-y-4">
            <ProcessingTimeline
              stages={stages}
              onComplete={() => setPhase("results")}
            />
            {processingError && (
              <div className="rounded-xl border border-destructive/30 bg-destructive/10 p-4">
                <p className="text-sm font-medium text-destructive">Processing Error</p>
                <p className="text-xs text-destructive/90 mt-1">{processingError}</p>
              </div>
            )}
            {!processingError && (
              <div className="rounded-xl border border-dashed border-border bg-card/50 py-16 flex items-center justify-center">
                <div className="text-center">
                  <div className="relative mx-auto mb-3 h-16 w-16">
                    <svg className="h-16 w-16 -rotate-90" viewBox="0 0 64 64">
                      <defs>
                        <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                          <stop offset="0%" stopColor="#7247ED" />
                          <stop offset="100%" stopColor="#20B2E1" />
                        </linearGradient>
                      </defs>
                      <circle cx="32" cy="32" r="28" fill="none" stroke="currentColor" strokeWidth="4" className="text-border" />
                      <circle
                        cx="32" cy="32" r="28" fill="none" stroke="url(#progressGradient)" strokeWidth="4"
                        className="transition-all duration-500"
                        strokeDasharray={`${2 * Math.PI * 28}`}
                        strokeDashoffset={`${2 * Math.PI * 28 * (1 - processingPercent / 100)}`}
                        strokeLinecap="round"
                      />
                    </svg>
                    <span className="absolute inset-0 flex items-center justify-center text-sm font-semibold text-foreground">
                      {processingPercent}%
                    </span>
                  </div>
                  <p className="text-sm font-medium text-foreground">Processing your video...</p>
                  <p className="text-[10px] text-muted-foreground mt-0.5">This usually takes 2-5 minutes</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* RESULTS PHASE */}
        {phase === "results" && (
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <div className="space-y-4">
              <VideoResultPanel
                currentTime={currentTime}
                onTimeChange={handleTimeChangeWithPlay}
                runId={currentRunId}
                selectedLang={selectedSubtitleLang}
                shouldPlayAfterSeek={shouldPlayAfterSeek}
                onPlayStateChange={() => setShouldPlayAfterSeek(false)}
                onTogglePlayRef={videoTogglePlayRef}
                onPlayingChange={setIsVideoPlaying}
                initialMixGains={savedMixGains}
                onApplyStart={(_flowRunId) => {
                  setStages({})
                  setProcessingPercent(0)
                  setProcessingError("")
                  setPhase("processing")
                  subscribeToProcessing(currentRunId!)
                }}
              />
              <ProjectActionsPanel
                projectName={currentProject?.name || "Untitled Project"}
                onRenameProject={handleRenameProject}
                config={projectConfig || undefined}
                completedLanguages={[selectedSubtitleLang]}
                runId={currentRunId}
                durationMinutes={videoDurationMinutes}
                pricing={pricing}
                fileName={uploadedFile?.name}
              />
            </div>

            <div className="flex flex-col" style={{ minHeight: "500px", maxHeight: "calc(100vh - 180px)" }}>
              <SubtitleEditor
                currentTime={currentTime}
                onSeek={handleTimeChangeWithPlay}
                selectedLanguages={[selectedSubtitleLang]}
                selectedLang={selectedSubtitleLang}
                onLanguageChange={setSelectedSubtitleLang}
                runId={currentRunId}
                onRegenerate={handleRegenerate}
                isPlaying={isVideoPlaying}
                pricing={pricing}
                ttsMode={projectConfig?.ttsMode}
                durationMinutes={videoDurationMinutes}
              />
            </div>
          </div>
        )}
      </main>
        </>
      )}
    </div>
  )
}

export default function DashboardPage() {
  return (
    <Suspense fallback={null}>
      <DashboardPageInner />
    </Suspense>
  )
}
