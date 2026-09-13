from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from qc_fixes import Decision, apply_gate, match_playbook

AGENT_SYSTEM_PROMPT = """You are a dubbing QC-fix agent. For each flagged issue you \
receive the QC symptom plus a deterministic evidence bundle (text layers, diarization \
spans, original-vocal energy, a precomputed long-activity/short-text mismatch signal, \
emotion tag, timing/fit row). Reason from evidence to exactly one primitive per issue:

- edit_text {new_text}: fix wording/normalization (numbers, units, times). new_text is \
the SPOKEN words only — never include the "Speaker:" label prefix (it is preserved \
automatically). To change who speaks, use change_speaker instead.
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
true when your diagnosis matches a provided playbook pattern.

Commit to the action primitive whenever you can name a concrete fix: if you know the \
corrected text, emit edit_text (not propose); if you know the right speaker, emit \
change_speaker; and so on. When the evidence shows a non-speech hallucination (a scream/ \
laugh/reaction transcribed as words — long_activity_short_text true, or the symptom \
describes a shout/scream replacing the scripted words), emit drop_line, not propose. \
All primitives are reversible and re-checked after regeneration, so a confident \
diagnosis should always be the action itself, never a propose. Reserve propose ONLY for \
cases where you genuinely cannot determine the fix at all (truly ambiguous or where a \
block of adjacent lines is systemically mislabeled)."""


@dataclass
class RawDecision:
    idx: int
    primitive: str
    confidence: float
    diagnosis: str
    params: dict
    playbook_pattern_matched: bool = False
    needs_audio: dict | None = None


def _raw_idx(raw: dict) -> int:
    val = raw.get("idx", raw.get("sub_index"))
    return int(val) if val is not None else -1


# Param keys each primitive needs; the model may nest them under "params" or emit
# them flat at the top level of the decision object. Collect from both.
_PARAM_KEYS = {
    "new_text", "new_speaker", "new_start_ms", "new_end_ms",
    "tag", "category", "vector", "suggested_fix",
}


def _collect_params(raw: dict) -> dict:
    params = dict(raw.get("params") or {})
    for k in _PARAM_KEYS:
        if k not in params and k in raw:
            params[k] = raw[k]
    return params


def _is_hallucination(evidence: dict) -> bool:
    """The deterministic non-speech-hallucination signature: long continuous vocal
    activity with little/no transcribed text (spec Evidence Bundle mismatch signal)."""
    sig = (evidence or {}).get("mismatch_signal", {})
    return bool(sig.get("long_activity_short_text"))


# When the model hedges to `propose` but the diagnosis is actionable, promote to the
# matching action primitive and let the per-primitive confidence gate decide. Concrete
# fix params imply text/speaker/timing/emotion edits; drop_line is inferred only from the
# hallucination evidence signal (it carries no param), so it is never promoted blindly.
def _promote_primitive(primitive: str, params: dict, evidence: dict) -> str:
    if primitive != "propose":
        return primitive
    if params.get("new_text"):
        return "edit_text"
    if params.get("new_speaker"):
        return "change_speaker"
    if params.get("new_start_ms") is not None and params.get("new_end_ms") is not None:
        return "change_timing"
    if params.get("tag") or params.get("vector") or params.get("category"):
        return "set_emotion"
    if _is_hallucination(evidence):
        return "drop_line"
    return primitive


def _to_decision(raw: dict, playbook: list[dict], evidence: dict | None = None) -> Decision:
    params = _collect_params(raw)
    primitive = _promote_primitive(raw.get("primitive", "propose"), params, evidence or {})
    matched = bool(raw.get("playbook_pattern_matched", False))
    boost = match_playbook(playbook, primitive) if matched else 0.0
    return Decision(
        idx=_raw_idx(raw),
        primitive=primitive,
        confidence=float(raw.get("confidence", 0.0)),
        params=params,
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


def _decisions_from_response(resp) -> list[dict]:
    """The model may return {"decisions": [...]}, a bare [...] list, or a single
    decision object. Normalize all three to a list of raw decision dicts."""
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        if isinstance(resp.get("decisions"), list):
            return resp["decisions"]
        if "idx" in resp or "sub_index" in resp:
            return [resp]
    return []


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
        raws = {_raw_idx(r): r for r in _decisions_from_response(phase1)}

        audio_reqs = [r["needs_audio"] for r in raws.values()
                      if r.get("needs_audio")]
        if audio_reqs and audio_provider is not None:
            clips = audio_provider(audio_reqs)  # {idx: opus_bytes}
            phase2_prompt = json.dumps({
                "resolve_with_audio": audio_reqs,
                "note": "audio clips for each idx are attached below, in the same "
                        "order; listen and finalize these issues",
            }, default=str)
            # Forward as a contents list: the prompt text, then one audio-bytes clip
            # per requested idx. model_call wraps raw bytes into audio Parts.
            contents = [phase2_prompt]
            for req in audio_reqs:
                clip = clips.get(req.get("idx"))
                if clip:
                    contents.append(clip)
            phase2 = model_call(AGENT_SYSTEM_PROMPT, contents, None)
            for r in _decisions_from_response(phase2):
                raws[_raw_idx(r)] = r  # phase-2 answer overrides the pending one

        decisions = [_to_decision(r, playbook, bundle.get(idx, {}))
                     for idx, r in raws.items()]
        apply_gate(decisions)
        return decisions
    except Exception as e:  # noqa: BLE001 — agent must never block the dub
        print(f"⚠️  QC agent failed ({e}); emitting all issues as proposals")
        return _all_proposals(issues)
