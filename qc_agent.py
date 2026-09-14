from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from qc_fixes import Decision, apply_gate, match_playbook

AGENT_SYSTEM_PROMPT = """You are a dubbing QC-fix agent. The dub speaks a translated \
transcript with TTS over the original video. You get the whole clip's timeline once: \
retranslated subtitles (the spoken text, with boxes), original subtitles (source \
reference), diarization spans (where real speech actually is), timing rows (box vs. \
actual TTS start/end), and the speaker list — plus a small per-issue evidence slice \
(emotion, vocal energy, a hallucination mismatch signal). Reason across the whole \
timeline, not just the flagged box.

Key mechanism: TTS is placed inside subtitle boxes; any original audio OUTSIDE every \
box is mixed back as background. So a box that starts late or ends early lets the \
original words bleed through the dub.

Your job is to fix each issue, not just describe it. Return a list of decisions. An \
issue usually maps to one action, but a single root cause can require several — fix all \
of them in one go. Each decision names a target line by `idx` and carries a one-sentence \
`diagnosis` (what the evidence showed, why this action). Emit as many decisions as the \
fix needs, on whatever lines it touches (they need not be the flagged line).

- edit_text {idx, new_text}: the spoken words are wrong or mis-normalized (numbers, \
units, phrasing). Always put the corrected spoken text in `new_text` (no "Speaker:" \
prefix) — never leave it only in the diagnosis. TTS speaks exactly the box text and \
nothing more, so extra or foreign words heard in the dub are always original-audio bleed \
(use change_timing), never text to edit.
- drop_line {idx}: the line is an STT hallucination — non-speech (scream/laugh/reaction) \
transcribed as words. The evidence signal is long continuous vocal activity with \
little/no text (long_activity_short_text).
- change_timing {idx, new_start_ms, new_end_ms}: the box is misaligned — TTS spills past \
it, or a foreign/original word bleeds beside the line. The QC timestamp is approximate \
and points at the line, not the leak; the leaked audio sits in the gap just outside the \
box, so find its diarization span there and move the boundary out past it. Restating the \
box's existing edge is a no-op, not a fix; never cross into a neighbor's box.
- set_emotion {idx, tag, category, vector}: the delivered emotion is clearly wrong for \
the line.
- change_speaker {idx, new_speaker}: the voice is misattributed. When a whole block is \
mislabeled, emit one change_speaker per affected line — and add edit_text on any line \
whose wording (e.g. gender agreement) must follow the reassignment.
- propose {suggested_fix}: only when you genuinely cannot name a concrete action — the \
evidence is too ambiguous to decide. `suggested_fix` states the human action.

Commit to concrete actions. Reach for `propose` only when you are truly unsure what the \
fix is, not merely because a fix spans multiple lines or fields — in that case emit all \
the actions. If an issue has no sub_index it's audio in a gap: locate it on the \
timeline; if a real box sits next to the leak, extend that box's timing to cover it \
rather than proposing; only if no box is near do you propose. Emit `confidence` in \
[0,1]; emit needs_audio {idx, window:[start_s,end_s]} if you must hear it to decide \
(the window may extend past the box). Set playbook_pattern_matched true when it matches \
a provided playbook pattern."""


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


# Cue words the model uses right before it quotes the corrected line ("correct to
# '...'", "should read \"...\""). We lift the quoted span ONLY when it follows such a
# cue — never the first stray quote — so we don't feed a random fragment to regen.
_CORRECTION_CUE = re.compile(
    r"(?:correct(?:ed)?\s+to|should\s+(?:be|read)|change\s+(?:it\s+)?to|"
    r"rewrite\s+(?:it\s+)?(?:as|to)|update\s+(?:the\s+line\s+)?to|fix(?:ed)?\s+to)"
    r"\s*[:]?\s*['\"‘’“”]([^'\"‘’“”]+)"
    r"['\"‘’“”]",
    re.IGNORECASE)


def _lift_edit_text(params: dict, diagnosis: str) -> None:
    """edit_text with no new_text is dead on arrival at the gate. If the model quoted
    the corrected line after a correction cue in its diagnosis, lift it into new_text
    so the fix can auto-apply. No cue -> leave it empty (stays a safe proposal)."""
    if params.get("new_text"):
        return
    m = _CORRECTION_CUE.search(diagnosis or "")
    if m:
        params["new_text"] = m.group(1).strip()


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
    if primitive == "edit_text":
        _lift_edit_text(params, raw.get("diagnosis", ""))
    matched = bool(raw.get("playbook_pattern_matched", False))
    boost = match_playbook(playbook, primitive) if matched else 0.0
    return Decision(
        idx=_raw_idx(raw),
        primitive=primitive,
        # A committed primitive IS the certainty signal; an omitted confidence is a
        # formatting slip, not doubt. Default to 1.0 (apply) — the model self-downgrades
        # only by emitting an explicit low number.
        confidence=float(raw.get("confidence", 1.0)),
        params=params,
        diagnosis=raw.get("diagnosis", ""),
        playbook_matched=matched,
        playbook_boost=boost,
        needs_audio=raw.get("needs_audio"),
    )


def _build_prompt(issues, bundle, playbook, context=None) -> str:
    return json.dumps({
        "playbook": playbook,
        "timeline": context or {},
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
           audio_provider: Callable[[list[dict]], dict] | None = None,
           context: dict | None = None) -> list[Decision]:
    """Two-phase agent. Phase 1: text only. Phase 2 (optional): re-invoke with audio
    clips for issues that requested them. Any exception -> all proposals (never blocks
    the pipeline). Returns Decisions with the gate applied; collision resolution is the
    orchestrator's job. `context` is the shared whole-clip timeline (subs/diarization/
    timing/speakers) the agent reasons across."""
    try:
        prompt = _build_prompt(issues, bundle, playbook, context)
        phase1 = model_call(AGENT_SYSTEM_PROMPT, prompt, None)
        # Keep a flat list, not a dict-by-idx: one issue may fan out to several actions
        # (same line: text+timing on a bleed; or several lines: a mislabeled speaker
        # block). Collision resolution downstream handles same-field conflicts.
        raws = list(_decisions_from_response(phase1))

        audio_reqs = [r["needs_audio"] for r in raws if r.get("needs_audio")]
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
            # Phase-2 answers finalize the idxs that requested audio: drop those pending
            # raws and replace them with the resolved ones.
            resolved_idxs = {req.get("idx") for req in audio_reqs}
            raws = [r for r in raws if _raw_idx(r) not in resolved_idxs]
            raws += _decisions_from_response(phase2)

        decisions = [_to_decision(r, playbook, bundle.get(_raw_idx(r), {}))
                     for r in raws]
        apply_gate(decisions)
        return decisions
    except Exception as e:  # noqa: BLE001 — agent must never block the dub
        print(f"⚠️  QC agent failed ({e}); emitting all issues as proposals")
        return _all_proposals(issues)
