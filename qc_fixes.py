from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import srt as _srt

from post_build_fix import apply_retranslation

# Per-primitive auto-apply thresholds (spec Auto-Apply Gate).
# Reflect blast radius; all primitives are reversible.
PRIMITIVE_THRESHOLDS: dict[str, float] = {
    "edit_text": 0.70,
    "change_timing": 0.70,
    "restore_line": 0.70,
    "set_emotion": 0.80,
    "change_speaker": 0.85,
    "drop_line": 0.90,
}

APPLIABLE_PRIMITIVES = set(PRIMITIVE_THRESHOLDS.keys())

# Params each primitive's writer requires. A decision missing any of these cannot be
# auto-applied — the writers index these keys directly, so a partial set would KeyError
# mid-loop and leave the SRT/emotions file half-mutated. drop_line needs no params.
REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {
    "edit_text": ("new_text",),
    "change_speaker": ("new_speaker",),
    "change_timing": ("new_start_ms", "new_end_ms"),
    "set_emotion": ("tag", "category", "vector"),
    "drop_line": (),
    "restore_line": (),  # resolves by window param or the decision's own idx
}


def has_required_params(d: "Decision") -> bool:
    return all(k in d.params for k in REQUIRED_PARAMS.get(d.primitive, ()))

# Higher tier = higher risk; used to break same-idx collisions.
COLLISION_TIER: dict[str, int] = {
    "drop_line": 4,
    "change_speaker": 3,
    "set_emotion": 2,
    "change_timing": 1,
    "edit_text": 1,
    "restore_line": 1,
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
        if not has_required_params(d):
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
    "restore_line": "line",
}


def _downgrade(d: Decision) -> None:
    d.auto_apply = False
    d.outcome = "proposed"


def resolve_collisions(auto: list[Decision]) -> list[Decision]:
    """Dedupe auto-apply decisions by idx. Same-field conflict -> downgrade both.
    Different fields -> keep both. drop_line removes the line, so it conflicts with
    any other primitive on that idx and wins by tier. Done in code, not the model."""
    by_idx: dict[int, list[Decision]] = {}
    kept: list[Decision] = []
    for d in auto:
        if d.idx < 0:
            # idx<0 means "not a specific line" (e.g. a window restore); it targets a
            # time span, not a subtitle index, so it can never collide with anything.
            kept.append(d)
            continue
        by_idx.setdefault(d.idx, []).append(d)

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


def _drop_line(idx: int, subtitles_file: str) -> list[int]:
    subs = list(_srt.parse(open(subtitles_file, encoding="utf-8").read()))
    remaining = [s for s in subs if s.index != idx]
    if len(remaining) == len(subs):
        return []
    with open(subtitles_file, "w", encoding="utf-8") as f:
        f.write(_srt.compose(sorted(remaining, key=lambda x: x.start), reindex=False))
    return [idx]


def _change_speaker(idx: int, new_speaker: str, subtitles_file: str) -> list[int]:
    subs = list(_srt.parse(open(subtitles_file, encoding="utf-8").read()))
    for sub in subs:
        if sub.index == idx:
            text = sub.content.split(":", 1)[1].strip() if ":" in sub.content else sub.content
            sub.content = f"{new_speaker}: {text}"
            break
    else:
        return []
    with open(subtitles_file, "w", encoding="utf-8") as f:
        f.write(_srt.compose(sorted(subs, key=lambda x: x.start), reindex=False))
    return [idx]


def _change_timing(idx: int, new_start_ms: int, new_end_ms: int,
                   subtitles_file: str) -> list[int]:
    subs = list(_srt.parse(open(subtitles_file, encoding="utf-8").read()))
    for sub in subs:
        if sub.index == idx:
            sub.start = timedelta(milliseconds=new_start_ms)
            sub.end = timedelta(milliseconds=new_end_ms)
            break
    else:
        return []
    with open(subtitles_file, "w", encoding="utf-8") as f:
        f.write(_srt.compose(sorted(subs, key=lambda x: x.start), reindex=False))
    return [idx]


def _set_emotion(idx: int, tag: str, category: str, vector: list[float],
                 emotions_file: str) -> list[int]:
    data = {}
    try:
        data = json.loads(open(emotions_file, encoding="utf-8").read())
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data[str(idx)] = {"emotion_tag": tag, "category": category, "emo_vector": vector}
    with open(emotions_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return [idx]


def _restore_line(d: Decision, subtitles_file: str,
                  dropped_lines_file: str | None) -> list[int]:
    """Re-insert lines that song detection soft-dropped. Resolves targets against the
    sidecar stash: a window [start_s, end_s] restores every stashed block whose time
    span overlaps it; otherwise the decision's own idx restores that single block.
    Blocks are re-inserted verbatim (already-translated, incl. speaker prefix) with
    their original idx/times, so no re-translation happens."""
    if not dropped_lines_file:
        return []
    try:
        stash = json.loads(open(dropped_lines_file, encoding="utf-8").read())
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    window = d.params.get("window")
    targets: list[str] = []
    if window:
        lo_ms, hi_ms = window[0] * 1000, window[1] * 1000
        for k, row in stash.items():
            if row["start_ms"] < hi_ms and row["end_ms"] > lo_ms:  # overlap
                targets.append(k)
    elif d.idx >= 0 and str(d.idx) in stash:
        targets.append(str(d.idx))

    if not targets:
        return []

    subs = list(_srt.parse(open(subtitles_file, encoding="utf-8").read()))
    present = {s.index for s in subs}
    restored: list[int] = []
    for k in targets:
        idx = int(k)
        if idx in present:
            continue
        row = stash[k]
        subs.append(_srt.Subtitle(
            index=idx,
            start=timedelta(milliseconds=row["start_ms"]),
            end=timedelta(milliseconds=row["end_ms"]),
            content=row["content"],
        ))
        restored.append(idx)
    if not restored:
        return []
    with open(subtitles_file, "w", encoding="utf-8") as f:
        f.write(_srt.compose(sorted(subs, key=lambda x: x.start), reindex=False))
    return restored


def apply_fixes(decisions: list[Decision], subtitles_file: str,
                emotions_file: str, dropped_lines_file: str | None = None) -> list[int]:
    """Write each auto-apply decision to the frozen retranslated SRT (text/speaker/
    timing/drop) or emotions_tags.json (emotion). Returns changed indices for regen."""
    changed: list[int] = []
    for d in decisions:
        if not d.auto_apply or not has_required_params(d):
            continue
        p, prm = d.primitive, d.params
        if p == "edit_text":
            changed += apply_retranslation(d.idx, prm["new_text"], subtitles_file)
        elif p == "drop_line":
            changed += _drop_line(d.idx, subtitles_file)
        elif p == "change_speaker":
            changed += _change_speaker(d.idx, prm["new_speaker"], subtitles_file)
        elif p == "change_timing":
            changed += _change_timing(d.idx, prm["new_start_ms"],
                                      prm["new_end_ms"], subtitles_file)
        elif p == "set_emotion":
            changed += _set_emotion(d.idx, prm["tag"], prm["category"],
                                    prm["vector"], emotions_file)
        elif p == "restore_line":
            changed += _restore_line(d, subtitles_file, dropped_lines_file)
    return sorted(set(changed))


def load_playbook(path: str = "config/qc_playbook.json") -> list[dict]:
    return _load_playbook_json(path)


def _load_playbook_json(path: str) -> list[dict]:
    try:
        return json.loads(open(path, encoding="utf-8").read())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def match_playbook(entries: list[dict], primitive: str) -> float:
    """Return the max confidence_boost among entries whose primitive matches.
    The agent supplies the human-readable pattern match; this rewards alignment
    of the chosen primitive with a known pattern."""
    boosts = [float(e.get("confidence_boost", 0.0))
              for e in entries if e.get("primitive") == primitive]
    return max(boosts) if boosts else 0.0
