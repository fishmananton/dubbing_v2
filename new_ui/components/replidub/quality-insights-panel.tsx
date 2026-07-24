"use client"

import { useState } from "react"
import { AlertTriangle, CheckCircle2, ChevronRight, Lightbulb, TrendingUp } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

interface Warning {
  id: string
  type: "error" | "warning" | "info"
  segment: string
  message: string
  fix: string
}

const WARNINGS: Warning[] = [
  { id: "w1", type: "warning", segment: "0:22 – 0:28", message: "Segment too long", fix: "Split segment at natural pause" },
  { id: "w2", type: "error", segment: "0:37 – 0:44", message: "Voice mismatch detected", fix: "Re-assign to Speaker 1 voice profile" },
  { id: "w3", type: "warning", segment: "0:44 – 0:51", message: "High pitch deviation", fix: "Reduce emotion level for this segment" },
  { id: "w4", type: "info", segment: "0:09 – 0:15", message: "Timing gap detected (0.2s)", fix: "Enable Fix Timing and re-run" },
]

const SCORE = 87

export function QualityInsightsPanel() {
  const [dismissed, setDismissed] = useState<string[]>([])
  const [applied, setApplied] = useState<string[]>([])

  const visible = WARNINGS.filter((w) => !dismissed.includes(w.id))

  const applyFix = (id: string) => {
    setApplied((prev) => [...prev, id])
    setTimeout(() => setDismissed((prev) => [...prev, id]), 600)
  }

  const errorCount = visible.filter((w) => w.type === "error").length
  const warnCount = visible.filter((w) => w.type === "warning").length

  return (
    <div className="rounded-xl border border-border bg-card shadow-sm">
      <div className="border-b border-border px-4 py-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-foreground">Quality Insights</h2>
          <div className="flex items-center gap-1.5">
            <TrendingUp className="h-4 w-4 text-success" />
            <span className="text-sm font-bold text-success">{SCORE}</span>
            <span className="text-xs text-muted-foreground">/ 100</span>
          </div>
        </div>
        <div className="mt-1 flex gap-3 text-xs text-muted-foreground">
          {errorCount > 0 && (
            <span className="flex items-center gap-1 text-destructive">
              <AlertTriangle className="h-3 w-3" />
              {errorCount} error{errorCount > 1 ? "s" : ""}
            </span>
          )}
          {warnCount > 0 && (
            <span className="flex items-center gap-1 text-warning-foreground">
              <AlertTriangle className="h-3 w-3" />
              {warnCount} warning{warnCount > 1 ? "s" : ""}
            </span>
          )}
          {visible.length === 0 && (
            <span className="flex items-center gap-1 text-success">
              <CheckCircle2 className="h-3 w-3" />
              All issues resolved
            </span>
          )}
        </div>
      </div>

      <div className="divide-y divide-border">
        {visible.length === 0 ? (
          <div className="flex flex-col items-center gap-2 py-8">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-success/10">
              <CheckCircle2 className="h-5 w-5 text-success" />
            </div>
            <p className="text-sm font-medium text-foreground">Looking great!</p>
            <p className="text-xs text-muted-foreground">No quality issues detected</p>
          </div>
        ) : (
          visible.map((w) => (
            <div
              key={w.id}
              className={cn(
                "flex items-start gap-3 px-4 py-3 transition-opacity",
                applied.includes(w.id) && "opacity-50"
              )}
            >
              <div className={cn(
                "mt-0.5 flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full",
                w.type === "error" ? "bg-destructive/10" : w.type === "warning" ? "bg-warning/15" : "bg-accent"
              )}>
                {w.type === "info"
                  ? <Lightbulb className="h-3.5 w-3.5 text-accent-foreground" />
                  : <AlertTriangle className={cn("h-3.5 w-3.5", w.type === "error" ? "text-destructive" : "text-warning-foreground")} />
                }
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <p className="text-xs font-medium text-foreground">{w.message}</p>
                  <span className="rounded bg-secondary px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground">{w.segment}</span>
                </div>
                <p className="mt-0.5 text-xs text-muted-foreground">{w.fix}</p>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 gap-1 px-2 text-xs text-primary hover:bg-accent"
                onClick={() => applyFix(w.id)}
                disabled={applied.includes(w.id)}
              >
                {applied.includes(w.id) ? "Applied" : "Fix"}
                {!applied.includes(w.id) && <ChevronRight className="h-3 w-3" />}
              </Button>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
