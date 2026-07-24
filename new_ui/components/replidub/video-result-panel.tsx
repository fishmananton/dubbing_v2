"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { Slider } from "@/components/ui/slider"
import { Play, Pause, Volume2, VolumeX, Subtitles, Maximize2, ChevronDown, ChevronUp, Loader2, Check } from "lucide-react"
import { cn } from "@/lib/utils"
import { getTranslatedSubtitles, getOutputVideoUrl, getStemsInfo, fetchStemBuffer, remixAudio } from "@/lib/api"

interface SubtitleCue {
  startTime: number
  endTime: number
  text: string
}

// Parse SRT time format to seconds
function parseSRTTime(timeStr: string): number {
  const match = timeStr.match(/(\d{2}):(\d{2}):(\d{2})[,.](\d{3})/)
  if (!match) return 0
  const [, h, m, s, ms] = match
  return parseInt(h) * 3600 + parseInt(m) * 60 + parseInt(s) + parseInt(ms) / 1000
}

// Parse SRT content into cues
function parseSRTToCues(srt: string): SubtitleCue[] {
  const blocks = srt.trim().split(/\n\n+/)
  const cues: SubtitleCue[] = []
  
  for (const block of blocks) {
    const lines = block.split("\n")
    if (lines.length < 3) continue
    
    const timeMatch = lines[1].match(/(.+?)\s*-->\s*(.+)/)
    if (!timeMatch) continue
    
    const startTime = parseSRTTime(timeMatch[1])
    const endTime = parseSRTTime(timeMatch[2])
    
    // Get text, strip speaker prefix if present
    const textLine = lines.slice(2).join(" ")
    const speakerMatch = textLine.match(/^Speaker \d+:\s*(.*)$/)
    const text = speakerMatch ? speakerMatch[1] : textLine
    
    cues.push({ startTime, endTime, text })
  }
  
  return cues
}

interface VideoResultPanelProps {
  currentTime: number
  onTimeChange: (t: number, shouldPlay?: boolean) => void
  runId?: string | null
  selectedLang?: string
  shouldPlayAfterSeek?: boolean
  onPlayStateChange?: () => void
  onTogglePlayRef?: React.MutableRefObject<(() => void) | null>
  onPlayingChange?: (playing: boolean) => void
  initialMixGains?: [number, number, number, number] | null
  onApplyStart?: (flowRunId: string) => void
}

// dB to linear gain
function dbToLinear(db: number) { return Math.pow(10, db / 20) }


export function VideoResultPanel({ currentTime, onTimeChange, runId, selectedLang = "es", shouldPlayAfterSeek, onPlayStateChange, onTogglePlayRef, onPlayingChange, initialMixGains, onApplyStart }: VideoResultPanelProps) {
  const [playing, setPlaying] = useState(false)

  // Notify parent of playing state changes
  useEffect(() => {
    onPlayingChange?.(playing)
  }, [playing, onPlayingChange])

  // --- Mix section ---
  const [mixOpen, setMixOpen] = useState(false)
  // gains in dB: [background, dialog, non_speech, original_underlay]
  const defaultGains: [number, number, number, number] = initialMixGains ?? [-6, -4, -4, -22]
  const [mixGains, setMixGains] = useState<[number, number, number, number]>(defaultGains)
  const mixGainsRef = useRef<[number, number, number, number]>(defaultGains)

  useEffect(() => {
    const gains = initialMixGains ?? [-6, -4, -4, -22]
    setMixGains(gains)
    mixGainsRef.current = gains
  }, [initialMixGains])
  const [stems, setStems] = useState<Record<string, string>>({})
  const stemsReadyRef = useRef(false)
  const [stemsReady, setStemsReady] = useState(false)
  const [stemsLoading, setStemsLoading] = useState(false)
  const [applyStatus, setApplyStatus] = useState<"idle" | "applying" | "done" | "error">("idle")

  const audioCtxRef = useRef<AudioContext | null>(null)
  const stemBuffers = useRef<Record<string, AudioBuffer>>({})
  const stemSources = useRef<Record<string, AudioBufferSourceNode>>({})
  const gainNodes = useRef<Record<string, GainNode>>({})
  const mixActiveRef = useRef(false)

  // Load stems as soon as runId is available — guard with ref so this never fires twice per runId
  const stemsLoadStartedRef = useRef<string | null>(null)
  useEffect(() => {
    if (!runId || stemsLoadStartedRef.current === runId) return
    stemsLoadStartedRef.current = runId
    setStems({})
    setStemsReady(false)
    stemsReadyRef.current = false
    setStemsLoading(true)
    getStemsInfo(runId).then(info => {
      setStems(info)
      setStemsLoading(false)
      const ctx = audioCtxRef.current ?? new AudioContext()
      audioCtxRef.current = ctx
      const loads = Object.entries(info).map(([name, url]) =>
        fetchStemBuffer(url, ctx).then(buf => { stemBuffers.current[name] = buf })
      )
      Promise.all(loads).then(() => {
        stemsReadyRef.current = true
        setStemsReady(true)
      }).catch(console.error)
    }).catch(() => { stemsLoadStartedRef.current = null; setStemsLoading(false) })
  }, [runId])

  const stopMixPlayback = useCallback(() => {
    Object.values(stemSources.current).forEach(s => { try { s.stop() } catch {} })
    stemSources.current = {}
    gainNodes.current = {}
    mixActiveRef.current = false
  }, [])

  const startMixPlayback = useCallback((offset: number) => {
    if (!stemsReadyRef.current) return
    const ctx = audioCtxRef.current!
    if (ctx.state === "suspended") ctx.resume()
    // stop any existing sources first
    Object.values(stemSources.current).forEach(s => { try { s.stop() } catch {} })
    stemSources.current = {}
    gainNodes.current = {}

    const gainMap: Record<string, number> = {
      background: mixGainsRef.current[0],
      dialog: mixGainsRef.current[1],
      non_speech: mixGainsRef.current[1],  // always tracks dialog
      original: mixGainsRef.current[3],
    }
    Object.entries(stemBuffers.current).forEach(([name, buf]) => {
      const gain = ctx.createGain()
      gain.gain.value = dbToLinear((gainMap[name] ?? 0) )
      gain.connect(ctx.destination)
      gainNodes.current[name] = gain
      const src = ctx.createBufferSource()
      src.buffer = buf
      src.connect(gain)
      src.start(0, Math.max(0, offset))
      stemSources.current[name] = src
    })
    mixActiveRef.current = true
  }, [])

  // Update gain nodes directly when slider moves — no re-render side effects
  const handleGainChange = useCallback((index: number, value: number) => {
    const next: [number, number, number, number] = [...mixGainsRef.current] as any
    next[index] = value
    // non_speech (index 2) always tracks dialog (index 1)
    if (index === 1) next[2] = value
    mixGainsRef.current = next
    setMixGains(next)
    const nameMap = ["background", "dialog", "non_speech", "original"]
    const node = gainNodes.current[nameMap[index]]
    if (node) node.gain.value = dbToLinear(value )
    if (index === 1) {
      const nsNode = gainNodes.current["non_speech"]
      if (nsNode) nsNode.gain.value = dbToLinear(value )
    }
  }, [])

  // When stems finish loading while video is already playing, kick off mix immediately
  useEffect(() => {
    if (stemsReady && videoRef.current && !videoRef.current.paused) {
      startMixPlayback(videoRef.current.currentTime)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stemsReady])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopMixPlayback()
      audioCtxRef.current?.close()
    }
  }, [stopMixPlayback])

  const handleApply = async () => {
    if (!runId) return
    setApplyStatus("applying")
    try {
      const { flow_run_id } = await remixAudio(runId, mixGainsRef.current)
      setApplyStatus("done")
      setTimeout(() => setApplyStatus("idle"), 3000)
      onApplyStart?.(flow_run_id)
    } catch {
      setApplyStatus("error")
      setTimeout(() => setApplyStatus("idle"), 3000)
    }
  }
  
  const [volume, setVolume] = useState([80])
  const [muted, setMuted] = useState(false)
  const isIOS = typeof navigator !== "undefined" && /iPad|iPhone|iPod/.test(navigator.userAgent)
  const [showSubs, setShowSubs] = useState(true)
  const [duration, setDuration] = useState(60) // Default, will be updated when video loads
  const [subtitleCues, setSubtitleCues] = useState<SubtitleCue[]>([])
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const videoContainerRef = useRef<HTMLDivElement>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const progressBarRef = useRef<HTMLDivElement>(null)
  const isDraggingRef = useRef(false)

  // Fetch output video blob (requires credentials)
  useEffect(() => {
    let objectUrl: string | null = null
    if (!runId) { setVideoUrl(null); return }
    getOutputVideoUrl(runId)
      .then((url) => { objectUrl = url; setVideoUrl(url) })
      .catch((e) => console.error("Failed to fetch output video:", e))
    return () => { if (objectUrl) URL.revokeObjectURL(objectUrl) }
  }, [runId])

  // Fetch subtitles for overlay
  const fetchSubtitleCues = useCallback(async () => {
    if (!runId) {
      setSubtitleCues([])
      return
    }
    try {
      const srt = await getTranslatedSubtitles(runId, selectedLang)
      setSubtitleCues(parseSRTToCues(srt))
    } catch (e) {
      console.error("Failed to fetch subtitle cues:", e)
      setSubtitleCues([])
    }
  }, [runId, selectedLang])

  // Fetch subtitles when runId or language changes
  useEffect(() => {
    fetchSubtitleCues()
  }, [fetchSubtitleCues])

  // Sync video position when currentTime prop changes while paused (e.g. subtitle click)
  useEffect(() => {
    if (!playing && videoRef.current) {
      videoRef.current.currentTime = currentTime
    }
  }, [currentTime, playing])

  // Handle play after seek from subtitle click
  useEffect(() => {
    if (shouldPlayAfterSeek) {
      if (videoRef.current) videoRef.current.currentTime = currentTime
      if (mixOpen && stemsReadyRef.current) startMixPlayback(currentTime)
      setPlaying(true)
      onPlayStateChange?.()
    }
  }, [shouldPlayAfterSeek, currentTime, onPlayStateChange, mixOpen, startMixPlayback])

  // Clean up interval on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [])

  // Sync video element with playing state
  useEffect(() => {
    if (videoRef.current && videoUrl) {
      if (playing) {
        videoRef.current.play().catch(console.error)
      } else {
        videoRef.current.pause()
      }
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [playing, currentTime, onTimeChange, videoUrl])

  // Update volume on video element
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.volume = volume[0] / 100
      videoRef.current.muted = muted || volume[0] === 0
    }
  }, [volume, muted])

  // Apply initial volume when video loads
  const handleVideoLoaded = () => {
    if (videoRef.current) {
      videoRef.current.volume = volume[0] / 100
      videoRef.current.muted = muted || volume[0] === 0
    }
  }

  // Stop at end
  useEffect(() => {
    if (currentTime >= duration) {
      setPlaying(false)
      stopMixPlayback()
      onTimeChange(duration)
    }
  }, [currentTime, duration, onTimeChange, stopMixPlayback])

  const togglePlay = () => {
    const atEnd = currentTime >= duration
    if (atEnd) {
      onTimeChange(0)
      if (videoRef.current) videoRef.current.currentTime = 0
    }
    setPlaying(p => {
      const nextPlaying = !p
      if (mixOpen && stemsReadyRef.current) {
        if (nextPlaying) {
          startMixPlayback(atEnd ? 0 : (videoRef.current?.currentTime ?? 0))
        } else {
          stopMixPlayback()
        }
      }
      return nextPlaying
    })
  }

  // Expose togglePlay to parent via ref for global Space key handling
  useEffect(() => {
    if (onTogglePlayRef) {
      onTogglePlayRef.current = togglePlay
    }
    return () => {
      if (onTogglePlayRef) {
        onTogglePlayRef.current = null
      }
    }
  })

  const handleSeek = (time: number) => {
    onTimeChange(time)
    if (videoRef.current) videoRef.current.currentTime = time
    if (mixOpen && mixActiveRef.current) startMixPlayback(time)
  }

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, "0")}`
  }

  const progressPercent = Math.min((currentTime / duration) * 100, 100)

  // Get current subtitle based on time
  const currentSubtitle = subtitleCues.find(
    sub => currentTime >= sub.startTime && currentTime < sub.endTime
  )

  return (
    <div className="rounded-xl border border-border bg-card shadow-sm overflow-hidden">
      {/* Video area */}
      <div ref={videoContainerRef} className="relative bg-zinc-900 aspect-video flex items-center justify-center">
        {videoUrl ? (
          <video
            ref={videoRef}
            src={videoUrl}
            className="w-full h-full object-contain"
            onLoadedMetadata={(e) => { setDuration(e.currentTarget.duration); handleVideoLoaded() }}
            onTimeUpdate={(e) => onTimeChange(e.currentTarget.currentTime)}
            onEnded={() => { setPlaying(false); stopMixPlayback() }}
            crossOrigin="anonymous"
            playsInline
            muted={mixOpen}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-900">
            <p className="text-xs text-zinc-500">No video available</p>
          </div>
        )}

        {/* Subtitle overlay */}
        {showSubs && currentSubtitle && (
          <div className="absolute bottom-8 left-1/2 -translate-x-1/2 rounded-md bg-black/80 px-3 py-1.5 text-center max-w-[90%]">
            <p className="text-xs font-medium text-white leading-relaxed">
              {currentSubtitle.text}
            </p>
          </div>
        )}

        {/* Top controls */}
        <div className="absolute top-2 right-2 flex gap-1.5">
          <button
            onClick={() => {
              const vid = videoRef.current as any
              if (document.fullscreenElement) {
                document.exitFullscreen()
              } else if (videoContainerRef.current?.requestFullscreen) {
                videoContainerRef.current.requestFullscreen()
              } else if (vid?.webkitEnterFullscreen) {
                vid.webkitEnterFullscreen()
              }
            }}
            className="flex h-6 w-6 items-center justify-center rounded-md bg-black/50 text-white/70 backdrop-blur-sm hover:text-white transition-colors"
          >
            <Maximize2 className="h-3 w-3" />
          </button>
        </div>
      </div>

      {/* Controls bar */}
      <div className="border-t border-border bg-card px-3 pt-2 pb-3" onClick={() => { if (audioCtxRef.current?.state === "suspended") audioCtxRef.current.resume() }}>
        {/* Progress bar */}
        <div className="mb-2 flex items-center gap-2">
          <span className="w-8 text-right text-[10px] tabular-nums text-muted-foreground">{formatTime(currentTime)}</span>
          <div
            ref={progressBarRef}
            className="relative flex-1 h-3 cursor-pointer rounded-full bg-border flex items-center"
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId)
              isDraggingRef.current = true
              const rect = e.currentTarget.getBoundingClientRect()
              const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
              handleSeek(pct * duration)
            }}
            onPointerMove={(e) => {
              if (!isDraggingRef.current) return
              const rect = e.currentTarget.getBoundingClientRect()
              const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
              handleSeek(pct * duration)
            }}
            onPointerUp={(e) => {
              if (!isDraggingRef.current) return
              isDraggingRef.current = false
              e.currentTarget.releasePointerCapture(e.pointerId)
              const rect = e.currentTarget.getBoundingClientRect()
              const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
              handleSeek(pct * duration)
            }}
          >
            <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-1 rounded-full bg-border pointer-events-none" />
            <div
              className="absolute top-1/2 -translate-y-1/2 h-1 left-0 rounded-full bg-gradient-to-r from-primary to-brand-end pointer-events-none"
              style={{ width: `${progressPercent}%` }}
            />
            <div
              className="absolute top-1/2 -translate-y-1/2 h-3 w-3 rounded-full border-2 border-[#7247ED] bg-card shadow pointer-events-none"
              style={{ left: `calc(${progressPercent}% - 6px)` }}
            />
          </div>
          <span className="w-8 text-[10px] tabular-nums text-muted-foreground">{formatTime(duration)}</span>
        </div>

        {/* Buttons row */}
        <div className="flex items-center gap-2">
          <button
            onClick={togglePlay}
            className="flex h-7 w-7 items-center justify-center rounded-full bg-gradient-to-r from-primary to-brand-end text-primary-foreground hover:opacity-90 transition-opacity"
          >
            {playing ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3 ml-0.5" />}
          </button>

          <div className="flex items-center gap-1 ml-1">
            <button onClick={() => setMuted(m => !m)} className="text-muted-foreground hover:text-foreground transition-colors">
              {(muted || volume[0] === 0)
                ? <VolumeX className="h-3.5 w-3.5" />
                : <Volume2 className="h-3.5 w-3.5" />
              }
            </button>
            {!isIOS && (
              <Slider
                value={volume}
                onValueChange={setVolume}
                min={0}
                max={100}
                step={1}
                className="w-16"
              />
            )}
          </div>

          <div className="ml-auto flex items-center gap-1.5">
            <button
              onClick={() => setShowSubs((s) => !s)}
              className={cn(
                "flex items-center gap-1 rounded-md border px-2 py-1 text-[10px] font-medium transition-colors",
                showSubs
                  ? "border-[#7247ED] bg-gradient-to-r from-primary/10 to-brand-end/10 text-primary"
                  : "border-border bg-secondary text-muted-foreground hover:bg-muted"
              )}
            >
              <Subtitles className="h-3 w-3" />
              CC
            </button>
            {runId && (
              <button
                disabled={!stemsReady && !mixOpen}
                onClick={() => setMixOpen(o => {
                  if (o) {
                    stopMixPlayback()
                  } else if (stemsReadyRef.current && videoRef.current && !videoRef.current.paused) {
                    startMixPlayback(videoRef.current.currentTime)
                  }
                  return !o
                })}
                className={cn(
                  "flex items-center gap-1 rounded-md border px-2 py-1 text-[10px] font-medium transition-colors",
                  mixOpen
                    ? "border-[#7247ED] bg-gradient-to-r from-primary/10 to-brand-end/10 text-primary"
                    : !stemsReady
                    ? "border-border bg-secondary text-muted-foreground opacity-40 cursor-not-allowed"
                    : "border-border bg-secondary text-muted-foreground hover:bg-muted"
                )}
              >
                {stemsLoading ? <Loader2 className="h-3 w-3 animate-spin" /> : mixOpen ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                Audio Balance
              </button>
            )}
          </div>
        </div>

        {/* Mix panel */}
        {runId && mixOpen && (
          <div className="mt-2 border-t border-border pt-2 space-y-2">
            {stemsLoading && (
              <p className="text-[10px] text-muted-foreground flex items-center gap-1">
                <Loader2 className="h-3 w-3 animate-spin" /> Loading stems…
              </p>
            )}
            {!stemsLoading && !Object.keys(stems).length && (
              <p className="text-[10px] text-muted-foreground">Stems not available yet. Run the pipeline first.</p>
            )}
            {!stemsLoading && Object.keys(stems).length > 0 && (
              <>
                {(["background", "dialog"] as const).map((name, i) => (
                  <div key={name} className="flex items-center gap-2">
                    <span className="w-20 text-[10px] text-muted-foreground capitalize">{name}</span>
                    <Slider
                      value={[mixGains[i]]}
                      onValueChange={([v]) => handleGainChange(i, v)}
                      min={-40} max={6} step={0.5}
                      className="flex-1"
                    />
                    <span className="w-8 text-right text-[10px] tabular-nums text-muted-foreground">{mixGains[i] > 0 ? "+" : ""}{mixGains[i]}dB</span>
                  </div>
                ))}
                {"original" in stems && (
                  <div className="flex items-center gap-2">
                    <span className="w-20 text-[10px] text-muted-foreground">Original</span>
                    <Slider
                      value={[mixGains[3]]}
                      onValueChange={([v]) => handleGainChange(3, v)}
                      min={-40} max={6} step={0.5}
                      className="flex-1"
                    />
                    <span className="w-8 text-right text-[10px] tabular-nums text-muted-foreground">{mixGains[3] > 0 ? "+" : ""}{mixGains[3]}dB</span>
                  </div>
                )}
                <div className="flex items-center justify-between pt-1">
                  <p className="text-[9px] text-muted-foreground">Preview is live · Apply bakes into video</p>
                  <button
                    onClick={handleApply}
                    disabled={applyStatus === "applying"}
                    className={cn(
                      "flex items-center gap-1 rounded-md px-2.5 py-1 text-[10px] font-medium transition-colors",
                      applyStatus === "done"
                        ? "bg-green-500/20 text-green-600"
                        : applyStatus === "error"
                        ? "bg-destructive/20 text-destructive"
                        : "bg-gradient-to-r from-primary to-brand-end text-primary-foreground hover:opacity-90"
                    )}
                  >
                    {applyStatus === "applying" && <Loader2 className="h-3 w-3 animate-spin" />}
                    {applyStatus === "done" && <Check className="h-3 w-3" />}
                    {applyStatus === "applying" ? "Applying…" : applyStatus === "done" ? "Applied" : applyStatus === "error" ? "Failed" : "Apply"}
                  </button>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
