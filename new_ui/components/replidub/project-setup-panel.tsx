"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import { cn } from "@/lib/utils"
import { Globe, ChevronDown, ChevronUp, Sparkles, ChevronsUpDown, Check } from "lucide-react"
import { calculateCost, formatCents, type PricingConfig } from "@/lib/pricing"

export const LANGUAGES = [
  { code: "en", label: "English",    flag: "EN" },
  { code: "es", label: "Spanish",    flag: "ES" },
  { code: "fr", label: "French",     flag: "FR" },
  { code: "de", label: "German",     flag: "DE" },
  { code: "ko", label: "Korean",     flag: "KO" },
  { code: "zh", label: "Chinese",    flag: "ZH" },
  { code: "ja", label: "Japanese",   flag: "JA" },
  { code: "ar", label: "Arabic",     flag: "AR" },
  { code: "hi", label: "Hindi",      flag: "HI" },
  { code: "he", label: "Hebrew",     flag: "HE" },
  { code: "pt", label: "Portuguese", flag: "PT" },
  { code: "it", label: "Italian",    flag: "IT" },
  { code: "nl", label: "Dutch",      flag: "NL" },
  { code: "pl", label: "Polish",     flag: "PL" },
  { code: "ru", label: "Russian",    flag: "RU" },
]

export interface ProjectSetupConfig {
  selectedLang: string
  ttsMode: "natural" | "original_voice"
  voiceMode: "new_voice" | "overlay"
  transcribeMode: "transcribe" | "ocr"
  sourceLang: string
  useEmotions: boolean
  fixTiming: boolean
  numSpeakers: number | "auto"
  useNonSpeech: boolean
}

interface ProjectSetupPanelProps {
  onStart: (config: ProjectSetupConfig) => void
  isLoading?: boolean
  durationMinutes?: number | null
  pricing?: PricingConfig | null
  projectId?: number | null
}

export function clearProjectConfig(projectId: number) {
  localStorage.removeItem(`project_config_${projectId}`)
}

const SOURCE_LANGUAGES = [
  { code: "auto", label: "Auto-detect" },
  { code: "af", label: "Afrikaans" }, { code: "am", label: "Amharic" }, { code: "ar", label: "Arabic" },
  { code: "as", label: "Assamese" }, { code: "az", label: "Azerbaijani" }, { code: "ba", label: "Bashkir" },
  { code: "be", label: "Belarusian" }, { code: "bg", label: "Bulgarian" }, { code: "bn", label: "Bengali" },
  { code: "bo", label: "Tibetan" }, { code: "br", label: "Breton" }, { code: "bs", label: "Bosnian" },
  { code: "ca", label: "Catalan" }, { code: "cs", label: "Czech" }, { code: "cy", label: "Welsh" },
  { code: "da", label: "Danish" }, { code: "de", label: "German" }, { code: "el", label: "Greek" },
  { code: "en", label: "English" }, { code: "es", label: "Spanish" }, { code: "et", label: "Estonian" },
  { code: "eu", label: "Basque" }, { code: "fa", label: "Persian" }, { code: "fi", label: "Finnish" },
  { code: "fo", label: "Faroese" }, { code: "fr", label: "French" }, { code: "gl", label: "Galician" },
  { code: "gu", label: "Gujarati" }, { code: "ha", label: "Hausa" }, { code: "haw", label: "Hawaiian" },
  { code: "he", label: "Hebrew" }, { code: "hi", label: "Hindi" }, { code: "hr", label: "Croatian" },
  { code: "ht", label: "Haitian Creole" }, { code: "hu", label: "Hungarian" }, { code: "hy", label: "Armenian" },
  { code: "id", label: "Indonesian" }, { code: "is", label: "Icelandic" }, { code: "it", label: "Italian" },
  { code: "ja", label: "Japanese" }, { code: "jw", label: "Javanese" }, { code: "ka", label: "Georgian" },
  { code: "kk", label: "Kazakh" }, { code: "km", label: "Khmer" }, { code: "kn", label: "Kannada" },
  { code: "ko", label: "Korean" }, { code: "la", label: "Latin" }, { code: "lb", label: "Luxembourgish" },
  { code: "ln", label: "Lingala" }, { code: "lo", label: "Lao" }, { code: "lt", label: "Lithuanian" },
  { code: "lv", label: "Latvian" }, { code: "mg", label: "Malagasy" }, { code: "mi", label: "Maori" },
  { code: "mk", label: "Macedonian" }, { code: "ml", label: "Malayalam" }, { code: "mn", label: "Mongolian" },
  { code: "mr", label: "Marathi" }, { code: "ms", label: "Malay" }, { code: "mt", label: "Maltese" },
  { code: "my", label: "Burmese" }, { code: "ne", label: "Nepali" }, { code: "nl", label: "Dutch" },
  { code: "nn", label: "Nynorsk" }, { code: "no", label: "Norwegian" }, { code: "oc", label: "Occitan" },
  { code: "pa", label: "Punjabi" }, { code: "pl", label: "Polish" }, { code: "ps", label: "Pashto" },
  { code: "pt", label: "Portuguese" }, { code: "ro", label: "Romanian" }, { code: "ru", label: "Russian" },
  { code: "sa", label: "Sanskrit" }, { code: "sd", label: "Sindhi" }, { code: "si", label: "Sinhala" },
  { code: "sk", label: "Slovak" }, { code: "sl", label: "Slovenian" }, { code: "sn", label: "Shona" },
  { code: "so", label: "Somali" }, { code: "sq", label: "Albanian" }, { code: "sr", label: "Serbian" },
  { code: "su", label: "Sundanese" }, { code: "sv", label: "Swedish" }, { code: "sw", label: "Swahili" },
  { code: "ta", label: "Tamil" }, { code: "te", label: "Telugu" }, { code: "tg", label: "Tajik" },
  { code: "th", label: "Thai" }, { code: "tk", label: "Turkmen" }, { code: "tl", label: "Filipino" },
  { code: "tr", label: "Turkish" }, { code: "tt", label: "Tatar" }, { code: "uk", label: "Ukrainian" },
  { code: "ur", label: "Urdu" }, { code: "uz", label: "Uzbek" }, { code: "vi", label: "Vietnamese" },
  { code: "yi", label: "Yiddish" }, { code: "yo", label: "Yoruba" }, { code: "zh", label: "Chinese" },
]

export function ProjectSetupPanel({ onStart, isLoading = false, durationMinutes, pricing, projectId }: ProjectSetupPanelProps) {
  const [selectedLang, setSelectedLang] = useState<string>("es")
  const [ttsMode, setTtsMode] = useState<"natural" | "original_voice">("natural")
  const [voiceMode, setVoiceMode] = useState<"new_voice" | "overlay">("new_voice")
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [transcribeMode, setTranscribeMode] = useState<"transcribe" | "ocr">("transcribe")
  const [sourceLang, setSourceLang] = useState("auto")
  const [sourceLangOpen, setSourceLangOpen] = useState(false)
  const [useEmotions, setUseEmotions] = useState(true)
  const [fixTiming, setFixTiming] = useState(true)
  const [numSpeakers, setNumSpeakers] = useState<number | "auto">("auto")
  const [useNonSpeech, setUseNonSpeech] = useState(true)

  // Load saved config when projectId changes
  useEffect(() => {
    if (!projectId) return
    const saved = localStorage.getItem(`project_config_${projectId}`)
    if (!saved) return
    try {
      const c = JSON.parse(saved)
      if (c.selectedLang) setSelectedLang(c.selectedLang)
      if (c.ttsMode) setTtsMode(c.ttsMode)
      if (c.voiceMode) setVoiceMode(c.voiceMode)
      if (c.transcribeMode) setTranscribeMode(c.transcribeMode)
      if (c.sourceLang) setSourceLang(c.sourceLang)
      if (c.useEmotions !== undefined) setUseEmotions(c.useEmotions)
      if (c.fixTiming !== undefined) setFixTiming(c.fixTiming)
      if (c.numSpeakers !== undefined) setNumSpeakers(c.numSpeakers)
      if (c.useNonSpeech !== undefined) setUseNonSpeech(c.useNonSpeech)
    } catch {}
  }, [projectId])

  // Save config to localStorage on any change
  useEffect(() => {
    if (!projectId) return
    localStorage.setItem(`project_config_${projectId}`, JSON.stringify({
      selectedLang, ttsMode, voiceMode, transcribeMode, sourceLang, useEmotions, fixTiming, numSpeakers, useNonSpeech,
    }))
  }, [projectId, selectedLang, ttsMode, voiceMode, transcribeMode, sourceLang, useEmotions, fixTiming, numSpeakers, useNonSpeech])

  const handleStart = () => {
    if (projectId) clearProjectConfig(projectId)
    onStart({
      selectedLang,
      ttsMode,
      voiceMode,
      transcribeMode,
      sourceLang,
      useEmotions,
      fixTiming,
      numSpeakers,
      useNonSpeech: voiceMode === "new_voice" ? useNonSpeech : false,
    })
  }

  return (
    <div className="rounded-xl border border-border bg-card p-5 shadow-sm">
      <h2 className="mb-4 text-sm font-semibold text-foreground">Project Setup</h2>

      {/* Target Language - Single select dropdown */}
      <div className="mb-5">
        <div className="mb-2 flex items-center gap-2">
          <Globe className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Target Language</span>
        </div>
        
        <Select value={selectedLang} onValueChange={(lang) => { setSelectedLang(lang); if (lang === "ru") setTtsMode("natural") }}>
          <SelectTrigger className="w-full bg-secondary border-border">
            <SelectValue placeholder="Select language..." />
          </SelectTrigger>
          <SelectContent>
            {LANGUAGES.map((lang) => (
              <SelectItem key={lang.code} value={lang.code}>
                <span className="mr-2 text-[10px] font-medium bg-muted px-1 py-0.5 rounded">{lang.flag}</span>
                {lang.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Voice Mode: Natural vs Original Voice */}
      <div className="mb-4">
        <div className="mb-2 flex items-center gap-2">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Voice Mode</span>
        </div>
        <div className="flex gap-1 rounded-lg border border-border bg-secondary p-0.5">
          <button
            onClick={() => setTtsMode("natural")}
            className={cn(
              "flex flex-1 flex-col items-center rounded-md px-3 py-2 transition-all",
              ttsMode === "natural"
                ? "bg-card text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <span className="text-xs font-medium">Natural</span>
          </button>
          <button
            onClick={() => setTtsMode("original_voice")}
            disabled={selectedLang === "ru"}
            className={cn(
              "flex flex-1 flex-col items-center rounded-md px-3 py-2 transition-all",
              selectedLang === "ru"
                ? "opacity-40 cursor-not-allowed text-muted-foreground"
                : ttsMode === "original_voice"
                  ? "bg-card text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
            )}
          >
            <span className="text-xs font-medium">Original Voice</span>
            {ttsMode === "original_voice" && selectedLang !== "ru" && (
              <span className="text-[10px] text-muted-foreground leading-tight text-center">more similar, may include accent</span>
            )}
          </button>
        </div>
      </div>

      {/* Audio Mix: New Voice vs Overlay */}
      <div className="mb-5">
        <div className="mb-2 flex items-center gap-2">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Audio Mix</span>
        </div>
        <div className="flex gap-1 rounded-lg border border-border bg-secondary p-0.5">
          <button
            onClick={() => setVoiceMode("new_voice")}
            className={cn(
              "flex flex-1 items-center justify-center rounded-md px-3 py-1.5 text-xs font-medium transition-all",
              voiceMode === "new_voice"
                ? "bg-card text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            New Voice
          </button>
          <button
            onClick={() => setVoiceMode("overlay")}
            className={cn(
              "flex flex-1 items-center justify-center rounded-md px-3 py-1.5 text-xs font-medium transition-all",
              voiceMode === "overlay"
                ? "bg-card text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            Overlay
          </button>
        </div>
      </div>

      {/* Use Emotions toggle */}
      <div className="mb-5 flex items-center justify-between rounded-lg border border-border bg-secondary px-3 py-2.5">
        <div>
          <p className="text-sm font-medium text-foreground">Use Emotions</p>
          <p className="text-xs text-muted-foreground">Add emotional inflections to voice</p>
        </div>
        <Switch checked={useEmotions} onCheckedChange={setUseEmotions} />
      </div>

      {/* Advanced Options */}
      <div className="mb-5">
        <button
          onClick={() => setShowAdvanced((v) => !v)}
          className="flex w-full items-center justify-between rounded-lg border border-border bg-secondary px-3 py-2.5 text-sm font-medium text-foreground hover:bg-muted transition-colors"
        >
          Additional Options
          {showAdvanced ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}
        </button>

        {showAdvanced && (
          <div className="mt-3 space-y-3 rounded-lg border border-border bg-secondary/50 p-3">
            {/* Number of speakers */}
            <div className="flex items-center justify-between">
              <Label className="text-xs font-medium text-foreground">Number of Speakers</Label>
              <div className="flex gap-1 rounded-lg border border-border bg-card p-0.5">
                {(["auto", "1", "2", "3", "4"] as const).map((num) => (
                  <button
                    key={num}
                    onClick={() => setNumSpeakers(num === "auto" ? "auto" : parseInt(num))}
                    className={cn(
                      "rounded-md px-2.5 py-1 text-xs font-medium transition-all",
                      (num === "auto" ? numSpeakers === "auto" : numSpeakers === parseInt(num))
                        ? "bg-gradient-to-r from-primary to-brand-end text-primary-foreground"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    {num === "auto" ? "Auto" : num}
                  </button>
                ))}
              </div>
            </div>

            {/* Transcribe mode */}
            <div className="flex items-center justify-between">
              <Label className="text-xs font-medium text-foreground">Transcription Mode</Label>
              <div className="flex gap-1 rounded-lg border border-border bg-card p-0.5">
                {(["transcribe", "ocr"] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setTranscribeMode(m)}
                    className={cn(
                      "rounded-md px-2.5 py-1 text-xs font-medium transition-all",
                      transcribeMode === m ? "bg-gradient-to-r from-primary to-brand-end text-primary-foreground" : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    {m === "transcribe" ? "Transcribe" : "OCR"}
                  </button>
                ))}
              </div>
            </div>

            {/* Source language */}
            <div className="flex items-center justify-between">
              <Label className="text-xs font-medium text-foreground">Source Language</Label>
              <Popover open={sourceLangOpen} onOpenChange={setSourceLangOpen}>
                <PopoverTrigger asChild>
                  <Button variant="outline" role="combobox" className="h-7 w-36 justify-between text-xs bg-card border-border px-2 font-normal">
                    {SOURCE_LANGUAGES.find(l => l.code === sourceLang)?.label ?? "Auto-detect"}
                    <ChevronsUpDown className="ml-1 h-3 w-3 shrink-0 opacity-50" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-48 p-0" align="end">
                  <Command>
                    <CommandInput placeholder="Search language..." className="h-8 text-xs" />
                    <CommandList className="max-h-48">
                      <CommandEmpty>No language found.</CommandEmpty>
                      <CommandGroup>
                        {SOURCE_LANGUAGES.map(l => (
                          <CommandItem key={l.code} value={l.label} onSelect={() => { setSourceLang(l.code); setSourceLangOpen(false) }} className="text-xs">
                            <Check className={cn("mr-2 h-3 w-3", sourceLang === l.code ? "opacity-100" : "opacity-0")} />
                            {l.label}
                          </CommandItem>
                        ))}
                      </CommandGroup>
                    </CommandList>
                  </Command>
                </PopoverContent>
              </Popover>
            </div>

            {/* Fix Timing */}
            <div className="flex items-center justify-between">
              <Label className="text-xs font-medium text-foreground">Fix Timing</Label>
              <Switch checked={fixTiming} onCheckedChange={setFixTiming} />
            </div>

            {/* Non-speech layer — only for New Voice mode */}
            {voiceMode === "new_voice" && (
              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-xs font-medium text-foreground">Keep Ambient Sounds</Label>
                  <p className="text-[10px] text-muted-foreground">Preserve background noise between speech</p>
                </div>
                <Switch checked={useNonSpeech} onCheckedChange={setUseNonSpeech} />
              </div>
            )}
          </div>
        )}
      </div>

      {durationMinutes && pricing && (
        <div className="mb-4 flex items-center justify-between rounded-lg border border-border bg-secondary/50 px-3 py-2 text-xs">
          <span className="text-muted-foreground">Estimated cost</span>
          <span className="font-semibold text-destructive">
            {formatCents(calculateCost(durationMinutes, { fixTiming }, pricing))}
          </span>
        </div>
      )}

      <Button
        className="w-full gap-2 bg-gradient-to-r from-primary to-brand-end text-primary-foreground hover:opacity-90 h-10"
        onClick={handleStart}
        disabled={isLoading || !selectedLang}
      >
        <Sparkles className="h-4 w-4" />
        Start Translation
      </Button>
    </div>
  )
}
