from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from qc_fixes import Decision, apply_gate, match_playbook

AGENT_SYSTEM_PROMPT = """You are a dubbing QC-fix agent. For each flagged issue you \
receive the QC symptom plus a deterministic evidence bundle (text layers, diarization \
spans, original-vocal energy, a precomputed long-activity/short-text mismatch signal, \
emotion tag, timing/fit row). Reason from evidence to exactly one primitive per issue:

- edit_text {new_text}: fix wording/normalization (numbers, units, times).
- drop_line {}: remove an invented line (STT hallucination on non-speech, e.g. a scream \
transcribed as words — look for long_activity_short_text).
- set_emotion {tag, category, vector}: fix a clearly wrong emotion.
- change_timing {new_start_ms, new_end_ms}: fix spill/misalignment.
- change_speaker {new_speaker}: fix a misattributed voice. Prefer 'propose' if adjacent \
lines look mislabeled too (diarization-block error).
- propose {suggested_fix}: anything low-confidence, ambiguous, or risky.

Emit an explicit confidence in [0,1]. If you cannot decide without hearing the audio, \
emit needs_audio {idx, window:[start_s,end_s]} instead of a final primitive; the window \
may extend past the SRT box (e.g. a scream that spills). Set playbook_pattern_matched \
true when your diagnosis matches a provided playbook pattern."""


@dataclass
class RawDecision:
    idx: int
    primitive: str
    confidence: float
    diagnosis: str
    params: dict
    playbook_pattern_matched: bool = False
    needs_audio: dict | None = None


def _to_decision(raw: dict, playbook: list[dict]) -> Decision:
    primitive = raw.get("primitive", "propose")
    matched = bool(raw.get("playbook_pattern_matched", False))
    boost = match_playbook(playbook, primitive) if matched else 0.0
    return Decision(
        idx=raw["idx"],
        primitive=primitive,
        confidence=float(raw.get("confidence", 0.0)),
        params=raw.get("params", {}) or {},
        diagnosis=raw.get("diagnosis", ""),
        playbook_matched=matched,
        playbook_boost=boost,
        needs_audio=raw.get("needs_audio"),
    )


def _build_prompt(issues, bundle, playbook) -> str:
    return json.dumps({
        "playbook": playbook,
        "issues": [
            {"sub_index": i.sub_index, "symptom": i.symptom,
             "mismatch": i.mismatch, "start": i.start, "end": i.end,
             "evidence": bundle.get(i.sub_index, {})}
            for i in issues
        ],
    }, ensure_ascii=False, default=str)


def _all_proposals(issues) -> list[Decision]:
    out = []
    for i in issues:
        d = Decision(idx=i.sub_index if i.sub_index is not None else -1,
                     primitive="propose", confidence=0.0,
                     params={"suggested_fix": "agent unavailable — human review"},
                     diagnosis="agent/LLM failure")
        d.auto_apply = False
        out.append(d)
    return out


def decide(issues, bundle, playbook,
           model_call: Callable[[str, str, object], dict],
           audio_provider: Callable[[list[dict]], dict] | None = None) -> list[Decision]:
    """Two-phase agent. Phase 1: text only. Phase 2 (optional): re-invoke with audio
    clips for issues that requested them. Any exception -> all proposals (never blocks
    the pipeline). Returns Decisions with the gate applied; collision resolution is the
    orchestrator's job."""
    try:
        prompt = _build_prompt(issues, bundle, playbook)
        phase1 = model_call(AGENT_SYSTEM_PROMPT, prompt, None)
        raws = {r["idx"]: r for r in phase1.get("decisions", [])}

        audio_reqs = [r["needs_audio"] for r in raws.values()
                      if r.get("needs_audio")]
        if audio_reqs and audio_provider is not None:
            clips = audio_provider(audio_reqs)  # {idx: opus_bytes}
            phase2_prompt = json.dumps({
                "resolve_with_audio": [r["needs_audio"] for r in raws.values()
                                       if r.get("needs_audio")],
                "note": "audio clips attached separately; finalize these issues",
            }, default=str)
            phase2 = model_call(AGENT_SYSTEM_PROMPT, phase2_prompt, None)
            for r in phase2.get("decisions", []):
                raws[r["idx"]] = r  # phase-2 answer overrides the pending one

        decisions = [_to_decision(r, playbook) for r in raws.values()]
        apply_gate(decisions)
        return decisions
    except Exception as e:  # noqa: BLE001 — agent must never block the dub
        print(f"⚠️  QC agent failed ({e}); emitting all issues as proposals")
        return _all_proposals(issues)
