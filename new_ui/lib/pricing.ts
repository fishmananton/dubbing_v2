export interface PricingConfig {
  base_rate_per_min_cents: number
  fix_timing_addon_cents: number
  segment_regen_cost_cents: number
  volume_discounts: { min_minutes: number; discount_pct: number }[]
}

export function calculateCost(
  durationMinutes: number,
  options: { fixTiming: boolean },
  pricing: PricingConfig
): number {
  const rate = pricing.base_rate_per_min_cents + (options.fixTiming ? pricing.fix_timing_addon_cents : 0)
  const subtotal = durationMinutes * rate

  const discount_pct =
    pricing.volume_discounts
      .filter((t) => durationMinutes >= t.min_minutes)
      .sort((a, b) => b.min_minutes - a.min_minutes)[0]?.discount_pct ?? 0

  return Math.round(subtotal * (1 - discount_pct / 100))
}

export function calculateRegenCost(
  changedSegments: { startTime: number; endTime: number }[],
  pricing: PricingConfig,
  ttsMode?: "natural" | "original_voice",
  videoDurationMinutes?: number | null
): number {
  if (ttsMode === "original_voice") {
    // Flat cost = full video duration × rate/4, regardless of how many segments changed
    const mins = videoDurationMinutes ?? 0
    return Math.round(mins * pricing.base_rate_per_min_cents / 4)
  }
  const totalMinutes = changedSegments.reduce((sum, s) => sum + (s.endTime - s.startTime), 0) / 60
  return Math.round(totalMinutes * pricing.base_rate_per_min_cents)
}

export function formatCents(cents: number): string {
  return `$${(cents / 100).toFixed(2)}`
}
