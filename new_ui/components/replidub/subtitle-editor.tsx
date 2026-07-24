"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { cn } from "@/lib/utils"
import { RefreshCw, Loader2, Pencil, Undo2 } from "lucide-react"
import { getOriginalSubtitles, getTranslatedSubtitles, getSpeakerNames, saveSpeakerNames, type PricingConfig } from "@/lib/api"
import { formatCents, calculateRegenCost } from "@/lib/pricing"
import { useIsMobile } from "@/hooks/use-mobile"

interface SubtitleSegment {
  id: number
  startTime: number
  endTime: number
  original: string
  translated: string
  speaker: string
  changed?: boolean
}

// Parse SRT time format to seconds
function parseSRTTime(timeStr: string): number {
  const match = timeStr.match(/(\d{2}):(\d{2}):(\d{2})[,.](\d{3})/)
  if (!match) return 0
  const [, h, m, s, ms] = match
  return parseInt(h) * 3600 + parseInt(m) * 60 + parseInt(s) + parseInt(ms) / 1000
}

// Parse SRT content into segments
// Format: text line is "SPEAKER:TEXT" where SPEAKER can be any string before the first colon
function parseSRT(srt: string): Array<{ id: number; startTime: number; endTime: number; speaker: string; text: string }> {
  const blocks = srt.trim().split(/\n\n+/)
  const segments: Array<{ id: number; startTime: number; endTime: number; speaker: string; text: string }> = []

  for (const block of blocks) {
    const lines = block.split("\n")
    if (lines.length < 3) continue

    const id = parseInt(lines[0])
    const timeMatch = lines[1].match(/(.+?)\s*-->\s*(.+)/)
    if (!timeMatch) continue

    const startTime = parseSRTTime(timeMatch[1])
    const endTime = parseSRTTime(timeMatch[2])

    // Parse speaker from text line (format: "SPEAKER:TEXT" - any text before first colon is speaker)
    const textLine = lines.slice(2).join(" ")
    const colonIdx = textLine.indexOf(":")

    let speaker = "Speaker 1"
    let text = textLine

    if (colonIdx > 0) {
      speaker = textLine.substring(0, colonIdx).trim()
      text = textLine.substring(colonIdx + 1).trim()
    }

    segments.push({ id, startTime, endTime, speaker, text })
  }

  return segments
}

const LANGUAGES_MAP: Record<string, { label: string; flag: string }> = {
  en: { label: "English",    flag: "EN" },
  es: { label: "Spanish",    flag: "ES" },
  fr: { label: "French",     flag: "FR" },
  de: { label: "German",     flag: "DE" },
  ko: { label: "Korean",     flag: "KO" },
  zh: { label: "Chinese",    flag: "ZH" },
  ja: { label: "Japanese",   flag: "JA" },
  ar: { label: "Arabic",     flag: "AR" },
  hi: { label: "Hindi",      flag: "HI" },
  he: { label: "Hebrew",     flag: "HE" },
  pt: { label: "Portuguese", flag: "PT" },
  it: { label: "Italian",    flag: "IT" },
  nl: { label: "Dutch",      flag: "NL" },
  pl: { label: "Polish",     flag: "PL" },
  ru: { label: "Russian",    flag: "RU" },
}

// Speakers list is dynamically built from original subtitles
const DEFAULT_SPEAKERS = ["Speaker 1", "Speaker 2", "Speaker 3"]

function formatTime(s: number) {
  const m = Math.floor(s / 60)
  const sec = s % 60
  return `${m}:${sec.toFixed(1).padStart(4, "0")}`
}

interface SubtitleEditorProps {
  currentTime: number
  onSeek: (t: number, shouldPlay?: boolean) => void
  selectedLanguages?: string[]
  onLanguageChange?: (lang: string) => void
  selectedLang?: string
  runId?: string | null
  onRegenerate?: (subtitles: string, changedList: number[]) => void
  isPlaying?: boolean
  pricing?: PricingConfig | null
  ttsMode?: "natural" | "original_voice"
  durationMinutes?: number | null
}

export function SubtitleEditor({
  currentTime,
  onSeek,
  selectedLanguages = ["es", "fr", "de"],
  onLanguageChange,
  selectedLang = "es",
  runId,
  onRegenerate,
  isPlaying = false,
  pricing,
  ttsMode,
  durationMinutes,
}: SubtitleEditorProps) {
  // Store original values to track actual changes
  const originalSegmentsRef = useRef<Record<string, SubtitleSegment[]>>({})
  const [segments, setSegments] = useState<SubtitleSegment[]>([])
  const [speakers, setSpeakers] = useState<string[]>(DEFAULT_SPEAKERS)
  const [speakerNames, setSpeakerNames] = useState<Record<string, string>>({})
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState("")
  const [editingId, setEditingId] = useState<number | null>(null)
  const [isRegenerating, setIsRegenerating] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const isMobile = useIsMobile()
  const fetchedLangsRef = useRef<Set<string>>(new Set())

  // Fetch subtitles from API
  const fetchSubtitles = useCallback(async (lang: string) => {
    if (!runId) {
      setSegments([])
      return
    }

    // Check if already fetched this language
    if (fetchedLangsRef.current.has(`${runId}-${lang}`) && originalSegmentsRef.current[lang]) {
      setSegments(originalSegmentsRef.current[lang].map(s => ({ ...s, changed: false })))
      return
    }

    setIsLoading(true)
    try {
      const [originalSRT, translatedSRT] = await Promise.all([
        getOriginalSubtitles(runId),
        getTranslatedSubtitles(runId, lang)
      ])

      const originalParsed = parseSRT(originalSRT)
      const translatedParsed = parseSRT(translatedSRT)

      // Extract unique speakers from original subtitles
      const uniqueSpeakers = [...new Set(originalParsed.map(s => s.speaker))].filter(Boolean)
      if (uniqueSpeakers.length > 0) {
        setSpeakers(uniqueSpeakers)
      }

      // Combine original and translated into segments
      // Text is displayed WITHOUT speaker prefix, speaker is stored separately
      const combinedSegments: SubtitleSegment[] = translatedParsed.map((t, idx) => {
        const orig = originalParsed[idx]
        return {
          id: t.id,
          startTime: t.startTime,
          endTime: t.endTime,
          original: orig?.text || "",  // Text without speaker prefix (already parsed)
          translated: t.text,           // Text without speaker prefix (already parsed)
          speaker: t.speaker,           // Speaker from translated SRT
          changed: false
        }
      })

      originalSegmentsRef.current[lang] = combinedSegments.map(s => ({ ...s }))
      fetchedLangsRef.current.add(`${runId}-${lang}`)
      setSegments(combinedSegments)
    } catch (e) {
      console.error("Failed to fetch subtitles:", e)
      setSegments([])
    } finally {
      setIsLoading(false)
    }
  }, [runId])

  // Fetch subtitles when language or runId changes
  useEffect(() => {
    fetchSubtitles(selectedLang)
  }, [selectedLang, runId, fetchSubtitles])

  // Load speaker names when runId changes
  useEffect(() => {
    if (!runId) return
    getSpeakerNames(runId).then(setSpeakerNames)
  }, [runId])

  const displayName = (id: string) => speakerNames[id] || id

  const commitRename = async (speakerId: string) => {
    const trimmed = renameValue.trim()
    setRenamingId(null)
    if (!trimmed || trimmed === displayName(speakerId)) return
    const updated = { ...speakerNames, [speakerId]: trimmed }
    setSpeakerNames(updated)
    if (runId) await saveSpeakerNames(runId, updated)
  }

  // Auto-scroll to active subtitle (only when not editing, desktop only)
  useEffect(() => {
    if (isMobile) return
    // Don't auto-scroll while user is editing
    if (editingId !== null) return

    const activeSegment = segments.find(s => currentTime >= s.startTime && currentTime < s.endTime)
    if (activeSegment && scrollContainerRef.current) {
      const activeElement = scrollContainerRef.current.querySelector(`[data-segment-id="${activeSegment.id}"]`)
      if (activeElement) {
        activeElement.scrollIntoView({ behavior: "smooth", block: "center" })
      }
    }
  }, [currentTime, segments, editingId, isMobile])

  const hasChanges = segments.some(s => s.changed)

  const updateTranslated = (id: number, value: string) => {
    const original = originalSegmentsRef.current[selectedLang]?.find(s => s.id === id)
    setSegments((prev) =>
      prev.map((s) => {
        if (s.id !== id) return s
        // Only mark as changed if value is actually different from original
        const isChanged = original ? value !== original.translated || s.speaker !== original.speaker : false
        return { ...s, translated: value, changed: isChanged }
      })
    )
  }

  const updateSpeaker = (id: number, value: string) => {
    const original = originalSegmentsRef.current[selectedLang]?.find(s => s.id === id)
    setSegments((prev) =>
      prev.map((s) => {
        if (s.id !== id) return s
        // Only mark as changed if value is actually different from original
        const isChanged = original ? s.translated !== original.translated || value !== original.speaker : false
        return { ...s, speaker: value, changed: isChanged }
      })
    )
  }

  // Build SRT from segments - format: <SPEAKER>:<TEXT>
  const buildSRT = () => {
    return segments.map((seg, idx) => {
      const startSRT = formatTimeSRT(seg.startTime)
      const endSRT = formatTimeSRT(seg.endTime)
      // Format text line as SPEAKER:TEXT (no space after colon to match expected format)
      return `${idx + 1}\n${startSRT} --> ${endSRT}\n${seg.speaker}: ${seg.translated}\n`
    }).join("\n")
  }

  const formatTimeSRT = (s: number) => {
    const h = Math.floor(s / 3600)
    const m = Math.floor((s % 3600) / 60)
    const sec = Math.floor(s % 60)
    const ms = Math.floor((s % 1) * 1000)
    return `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}:${sec.toString().padStart(2, "0")},${ms.toString().padStart(3, "0")}`
  }

  const handleReset = async () => {
    const original = originalSegmentsRef.current[selectedLang]
    if (original) {
      setSegments(original.map(s => ({ ...s, changed: false })))
    }
    if (runId) {
      const fresh = await getSpeakerNames(runId)
      setSpeakerNames(fresh)
    }
    setEditingId(null)
    setRenamingId(null)
  }

  const handleSaveAndRegenerate = () => {
    if (!onRegenerate) return
    setIsRegenerating(true)

    // Build full SRT text
    const srtText = buildSRT()

    // Get list of changed segment indexes (1-indexed as per SRT)
    const changedList = segments
      .filter(seg => seg.changed)
      .map(seg => seg.id)

    // Call regenerate callback
    onRegenerate(srtText, changedList)

    // Update original refs to current values
    originalSegmentsRef.current[selectedLang] = segments.map(s => ({ ...s }))
    setSegments(prev => prev.map(s => ({ ...s, changed: false })))
    setIsRegenerating(false)
  }

  const isActive = (seg: SubtitleSegment) =>
    currentTime >= seg.startTime && currentTime < seg.endTime

  const availableLangs = selectedLanguages.map(code => ({
    code,
    ...LANGUAGES_MAP[code]
  })).filter(l => l.label)

  const currentLangInfo = LANGUAGES_MAP[selectedLang] || { label: "Unknown", flag: "??" }

  return (
    <div className="rounded-xl border border-border bg-card shadow-sm flex flex-col h-full">
      {/* Header with language selector */}
      <div className="border-b border-border px-3 py-2 flex items-center justify-between shrink-0">
        <div>
          <h2 className="text-sm font-semibold text-foreground">Subtitles</h2>
          <p className="text-[10px] text-muted-foreground">{segments.length} segments</p>
        </div>
        {availableLangs.length > 1 ? (
          <Select value={selectedLang} onValueChange={onLanguageChange}>
            <SelectTrigger className="h-7 w-32 text-xs border-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {availableLangs.map((l) => (
                <SelectItem key={l.code} value={l.code}>
                  <span className="mr-1 text-[10px] font-medium bg-muted px-1 py-0.5 rounded">{l.flag}</span>
                  {l.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        ) : (
          <div className="flex items-center gap-1 text-xs text-foreground">
            <span className="text-[10px] font-medium bg-muted px-1 py-0.5 rounded">{currentLangInfo.flag}</span>
            <span>{currentLangInfo.label}</span>
          </div>
        )}
      </div>

      {/* Speakers rename bar */}
      {speakers.length > 0 && (
        <div className="border-b border-border px-3 py-2 shrink-0 bg-secondary/30">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide mb-1.5">Rename Speakers</p>
          <div className="flex items-center gap-2 flex-wrap">
            {speakers.map((sp) => (
              <div key={sp}>
                {renamingId === sp ? (
                  <input
                    autoFocus
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                    onBlur={() => commitRename(sp)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") commitRename(sp)
                      if (e.key === "Escape") setRenamingId(null)
                    }}
                    className="h-7 w-32 rounded border border-[#7247ED] bg-background px-2 text-sm text-foreground outline-none ring-1 ring-[#7247ED]"
                  />
                ) : (
                  <button
                    onClick={() => { setRenamingId(sp); setRenameValue(displayName(sp)) }}
                    className="flex items-center gap-1.5 rounded-md border border-border bg-card px-2.5 py-1 text-sm text-foreground hover:border-primary hover:bg-accent/40 transition-colors"
                  >
                    <Pencil className="h-3 w-3 text-muted-foreground" />
                    <span>{displayName(sp)}</span>
                    <span className="font-mono text-[9px] text-muted-foreground">({sp})</span>
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Scrollable subtitle list */}
      <div ref={scrollContainerRef} className="flex-1 overflow-y-auto min-h-0">
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <Loader2 className="h-6 w-6 animate-spin text-primary mx-auto mb-2" />
              <p className="text-xs text-muted-foreground">Loading subtitles...</p>
            </div>
          </div>
        ) : segments.length === 0 ? (
          <div className="flex items-center justify-center py-12">
            <p className="text-xs text-muted-foreground">No subtitles available</p>
          </div>
        ) : (
          <div className="divide-y divide-border">
            {segments.map((seg) => {
              const active = isActive(seg)
              const editing = editingId === seg.id

              return (
                <div
                  key={seg.id}
                  data-segment-id={seg.id}
                  onClick={() => onSeek(seg.startTime, isPlaying)}
                  className={cn(
                    "cursor-pointer px-2 py-1 transition-colors",
                    seg.changed ? "bg-warning/10 border-l-2 border-l-warning" : "",
                    active && !seg.changed ? "bg-accent/60" : "",
                    !active && !seg.changed ? "hover:bg-secondary/60" : ""
                  )}
                >
                  {/* Row layout: Time+Speaker | Translated | Original - all left aligned */}
                  <div className="flex items-start gap-3">
                    {/* Time + Speaker column - wider for speaker names */}
                    <div className="shrink-0 w-32 flex flex-col items-start gap-0.5">
                      <div className="flex items-center gap-1 text-[10px] tabular-nums font-mono">
                        <span className={cn(active ? "text-primary font-semibold" : "text-muted-foreground")}>
                          {formatTime(seg.startTime)}
                        </span>
                        <span className="text-muted-foreground">-</span>
                        <span className={cn(active ? "text-primary font-semibold" : "text-muted-foreground")}>
                          {formatTime(seg.endTime)}
                        </span>
                        {active && (
                          <span className="inline-flex items-center rounded bg-primary/10 px-1 text-[8px] font-medium text-primary">
                            LIVE
                          </span>
                        )}
                      </div>
                      <select
                        value={seg.speaker}
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => { e.stopPropagation(); updateSpeaker(seg.id, e.target.value) }}
                        className="rounded border border-border bg-secondary px-1 py-0.5 text-sm text-foreground outline-none focus:border-primary w-full"
                      >
                        {speakers.map((sp) => (
                          <option key={sp} value={sp}>{displayName(sp)}</option>
                        ))}
                      </select>
                    </div>

                    {/* Translated text - editable, takes available space */}
                    <div className="flex-1 min-w-0" onClick={(e) => { e.stopPropagation(); setEditingId(seg.id) }}>
                      {editing ? (
                        <textarea
                          autoFocus
                          defaultValue={seg.translated}
                          ref={(el) => { if (el) { el.style.height = "auto"; el.style.height = el.scrollHeight + "px" } }}
                          onInput={(e) => { const el = e.currentTarget; el.style.height = "auto"; el.style.height = el.scrollHeight + "px" }}
                          onBlur={(e) => {
                            updateTranslated(seg.id, e.target.value)
                            setEditingId(null)
                          }}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" && !e.shiftKey) {
                              e.preventDefault()
                              updateTranslated(seg.id, e.currentTarget.value)
                              setEditingId(null)
                            }
                          }}
                          className="w-full resize-none rounded border border-[#7247ED] bg-background p-1.5 text-sm leading-normal text-foreground outline-none ring-1 ring-[#7247ED]"
                          rows={1}
                        />
                      ) : (
                        <p className={cn(
                          "text-sm leading-normal cursor-text rounded px-1.5 py-1 -mx-1.5 -my-1 transition-colors break-words whitespace-pre-wrap",
                          "hover:bg-accent/40 text-foreground",
                          seg.changed && "font-medium"
                        )}>
                          {seg.translated}
                        </p>
                      )}
                    </div>

                    {/* Original text - read only, muted */}
                    <div className="flex-1 min-w-0">
                      <p className="text-xs leading-normal text-muted-foreground break-words whitespace-pre-wrap">
                        {seg.original}
                      </p>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Save buttons when there are changes */}
      {hasChanges && (
        <div className="border-t border-border px-3 py-2 flex items-center justify-end gap-2 bg-warning/5 shrink-0">
          <span className="text-[10px] text-muted-foreground mr-auto">
            {segments.filter(s => s.changed).length} segment{segments.filter(s => s.changed).length !== 1 ? "s" : ""} changed
            {pricing && (
              <span className="ml-1.5 font-medium text-destructive">
                · {formatCents(calculateRegenCost(segments.filter(s => s.changed), pricing, ttsMode, durationMinutes ?? (segments.length > 0 ? segments[segments.length - 1].endTime / 60 : 0)))}
              </span>
            )}
          </span>
          <Button
            size="sm"
            variant="outline"
            className="h-7 gap-1 text-[10px]"
            onClick={handleReset}
            disabled={isRegenerating}
          >
            <Undo2 className="h-3 w-3" />
            Reset
          </Button>
          <Button
            size="sm"
            className="h-7 gap-1 text-[10px] bg-gradient-to-r from-primary to-brand-end text-primary-foreground hover:opacity-90"
            onClick={handleSaveAndRegenerate}
            disabled={isRegenerating}
          >
            {isRegenerating ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
            Save & Regenerate
          </Button>
        </div>
      )}
    </div>
  )
}
