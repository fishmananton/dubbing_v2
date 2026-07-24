"use client"

import { useEffect, useState } from "react"
import { CheckCircle2, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import type { RunStatusEvent } from "@/lib/api"

type StageStatus = "not_started" | "in_progress" | "done"

interface Stage {
  backendId: string
  label: string
}

const STAGES: Stage[] = [
  { backendId: "PREPARE", label: "Preparing Video" },
  { backendId: "TRANSCRIBE", label: "Understanding Speech" },
  { backendId: "TRANSLATE", label: "Translating Content" },
  { backendId: "GENERATE", label: "Generating Voices" },
  { backendId: "SYNC", label: "Syncing Audio" },
  { backendId: "RENDER", label: "Finalizing Video" },
]

interface ProcessingTimelineProps {
  runId?: string
  stages?: Record<string, StageStatus>
  onComplete?: () => void
}

export function ProcessingTimeline({ runId, stages: initialStages, onComplete }: ProcessingTimelineProps) {
  const [statuses, setStatuses] = useState<Record<string, StageStatus>>(
    initialStages || Object.fromEntries(STAGES.map((s) => [s.backendId, "not_started"]))
  )

  // Update stages from prop
  useEffect(() => {
    if (initialStages) {
      setStatuses(initialStages)
    }
  }, [initialStages])

  const allDone = STAGES.every((s) => statuses[s.backendId] === "done")
  const isProcessing = Object.values(statuses).some((s) => s === "in_progress")

  return (
    <div className="rounded-xl border border-border bg-card p-5 shadow-sm">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-foreground">Processing Pipeline</h2>
        {allDone && (
          <span className="flex items-center gap-1 rounded-full bg-success/10 px-2.5 py-1 text-xs font-medium text-success">
            <CheckCircle2 className="h-3.5 w-3.5" />
            Complete
          </span>
        )}
        {!allDone && isProcessing && (
          <span className="flex items-center gap-1 rounded-full bg-gradient-to-r from-primary/10 to-brand-end/10 px-2.5 py-1 text-xs font-medium text-primary">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Running
          </span>
        )}
      </div>

      <div className="flex items-start gap-0">
        {STAGES.map((stage, idx) => {
          const status = statuses[stage.backendId]
          const isLast = idx === STAGES.length - 1

          return (
            <div key={stage.backendId} className="flex flex-1 flex-col items-center">
              <div className="flex w-full items-center">
                {/* Connector left */}
                <div className={cn(
                  "h-0.5 flex-1 transition-colors duration-500",
                  idx === 0 ? "opacity-0" : status === "done" ? "bg-gradient-to-r from-primary to-brand-end" : "bg-border"
                )} />

                {/* Icon circle */}
                <div className={cn(
                  "relative flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full border-2 transition-all duration-300",
                  status === "done"
                    ? "border-primary bg-gradient-to-br from-primary to-brand-end text-primary-foreground"
                    : status === "in_progress"
                    ? "border-primary bg-accent text-primary"
                    : "border-border bg-secondary text-muted-foreground"
                )}>
                  {status === "in_progress" ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : status === "done" ? (
                    <CheckCircle2 className="h-4 w-4" />
                  ) : (
                    <div className="h-2 w-2 rounded-full bg-current" />
                  )}
                </div>

                {/* Connector right */}
                <div className={cn(
                  "h-0.5 flex-1 transition-colors duration-500",
                  isLast ? "opacity-0" : status === "done" ? "bg-gradient-to-r from-primary to-brand-end" : "bg-border"
                )} />
              </div>

              {/* Label */}
              <p className={cn(
                "mt-2 text-center text-[11px] font-medium leading-tight transition-colors",
                status === "done"
                  ? "text-primary"
                  : status === "in_progress"
                  ? "text-foreground"
                  : "text-muted-foreground"
              )}>
                {stage.label}
              </p>

            </div>
          )
        })}
      </div>
    </div>
  )
}
