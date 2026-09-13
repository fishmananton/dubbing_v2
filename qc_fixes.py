from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Per-primitive auto-apply thresholds (spec Auto-Apply Gate).
# Reflect blast radius; all primitives are reversible.
PRIMITIVE_THRESHOLDS: dict[str, float] = {
    "edit_text": 0.70,
    "change_timing": 0.70,
    "set_emotion": 0.80,
    "change_speaker": 0.85,
    "drop_line": 0.90,
}

APPLIABLE_PRIMITIVES = set(PRIMITIVE_THRESHOLDS.keys())

# Higher tier = higher risk; used to break same-idx collisions.
COLLISION_TIER: dict[str, int] = {
    "drop_line": 4,
    "change_speaker": 3,
    "set_emotion": 2,
    "change_timing": 1,
    "edit_text": 1,
}


@dataclass
class Decision:
    idx: int
    primitive: str
    confidence: float
    params: dict[str, Any] = field(default_factory=dict)
    diagnosis: str = ""
    playbook_matched: bool = False
    playbook_boost: float = 0.0
    needs_audio: dict | None = None  # {"idx": int, "window": [start_s, end_s]}
    auto_apply: bool = False
    outcome: str = ""  # filled after re-QC: "fixed" | "regression" | "proposed"


def effective_confidence(d: Decision) -> float:
    boost = d.playbook_boost if d.playbook_matched else 0.0
    return min(1.0, d.confidence + boost)


def apply_gate(decisions: list[Decision]) -> None:
    """Set d.auto_apply in place. propose never auto-applies."""
    for d in decisions:
        if d.primitive not in APPLIABLE_PRIMITIVES:
            d.auto_apply = False
            continue
        threshold = PRIMITIVE_THRESHOLDS[d.primitive]
        d.auto_apply = effective_confidence(d) >= threshold
