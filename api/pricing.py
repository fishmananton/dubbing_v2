import math

PRICING = {
    "base_rate_per_min_cents": 100,   # $1.00 / min
    "fix_timing_addon_cents":   20,   # $0.20 / min extra
    "segment_regen_cost_cents":  3,   # $0.03 / segment
    "volume_discounts": [
        {"min_minutes":  5, "discount_pct": 10},
        {"min_minutes": 10, "discount_pct": 15},
    ],
}


def calculate_run_cost_cents(duration_minutes: float, fix_timing: bool) -> int:
    if duration_minutes <= 0:
        return 0
    rate = PRICING["base_rate_per_min_cents"]
    if fix_timing:
        rate += PRICING["fix_timing_addon_cents"]
    subtotal = duration_minutes * rate
    best_discount = 0
    for tier in sorted(PRICING["volume_discounts"], key=lambda t: t["min_minutes"], reverse=True):
        if duration_minutes >= tier["min_minutes"]:
            best_discount = tier["discount_pct"]
            break
    return round(subtotal * (1 - best_discount / 100))


def calculate_regen_cost_cents(duration_minutes: float, ttsmodel: int = 1) -> int:
    if duration_minutes <= 0:
        return 0
    # ttsmodel 3 = Original Voice — cheaper (1/4 rate); others = Natural Voice — full rate
    if ttsmodel == 3:
        return round(duration_minutes * PRICING["base_rate_per_min_cents"] / 4)
    return round(duration_minutes * PRICING["base_rate_per_min_cents"])
