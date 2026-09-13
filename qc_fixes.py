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


# Which SRT/artifact field each primitive writes. Primitives touching different
# fields on the same line can coexist; same-field conflicts must be downgraded.
PRIMITIVE_FIELD: dict[str, str] = {
    "edit_text": "text",
    "drop_line": "line",       # removes the whole line -> conflicts with everything
    "change_timing": "timing",
    "change_speaker": "speaker",
    "set_emotion": "emotion",
}


def _downgrade(d: Decision) -> None:
    d.auto_apply = False
    d.outcome = "proposed"


def resolve_collisions(auto: list[Decision]) -> list[Decision]:
    """Dedupe auto-apply decisions by idx. Same-field conflict -> downgrade both.
    Different fields -> keep both. drop_line removes the line, so it conflicts with
    any other primitive on that idx and wins by tier. Done in code, not the model."""
    by_idx: dict[int, list[Decision]] = {}
    for d in auto:
        by_idx.setdefault(d.idx, []).append(d)

    kept: list[Decision] = []
    for idx, group in by_idx.items():
        if len(group) == 1:
            kept.append(group[0])
            continue

        # drop_line on this idx supersedes everything else on the line.
        drops = [d for d in group if d.primitive == "drop_line"]
        if drops:
            winner = max(drops, key=lambda d: effective_confidence(d))
            for d in group:
                if d is not winner:
                    _downgrade(d)
            kept.append(winner)
            continue

        # Group remaining by field; same-field conflict downgrades all in that field.
        by_field: dict[str, list[Decision]] = {}
        for d in group:
            by_field.setdefault(PRIMITIVE_FIELD[d.primitive], []).append(d)

        for field_decisions in by_field.values():
            if len(field_decisions) == 1:
                kept.append(field_decisions[0])
            else:
                for d in field_decisions:
                    _downgrade(d)
    return kept
