"""
Post-build fix module for lines that still overflow after two-pass TTS + per-speaker factor.

Applies three strategies in order:
1. Merge same-speaker adjacent subs (if not visible during gap)
2. Expand window into available gap
3. Retranslate:
   - Short phrases (≤3 words or sub < 500ms): word-count constraint
   - Long phrases (4+ words and sub ≥ 500ms): 2 candidates + TTS measurement
"""

from __future__ import annotations

import json
import os
import srt
from datetime import timedelta
from openai import OpenAI


ATEMPO_THRESHOLD = 1.5
MIN_GAP_FOR_EXPAND_MS = 500
SHORT_PHRASE_MAX_WORDS = 3
SHORT_PHRASE_MAX_DUR_MS = 500

MILD_OVERFLOW_RATIO = 1.5
HARD_OVERFLOW_RATIO = 1.8


def identify_overflow_lines(stats: list[dict], segment_meta: dict) -> list[dict]:
    """Find lines where atempo > ATEMPO_THRESHOLD after regen."""
    overflows = []
    for stat in stats:
        idx = stat["index"]
        if idx not in segment_meta:
            continue
        raw_len = segment_meta[idx]["raw_len"]
        available = stat["actual_available_duration_ms"]
        if available <= 0:
            continue
        atempo = raw_len / available
        if atempo > ATEMPO_THRESHOLD:
            overflows.append({
                "idx": idx,
                "speaker": stat["speaker"],
                "atempo": round(atempo, 2),
                "raw_len_ms": raw_len,
                "available_ms": available,
                "subtitle_duration_ms": stat["subtitle_duration_ms"],
                "subtitle_start_ms": stat["subtitle_start_ms"],
                "subtitle_end_ms": stat["subtitle_end_ms"],
                "next_subtitle_start_ms": stat.get("next_subtitle_start_ms"),
            })
    if overflows:
        print(f"📋 Post-build fix: {len(overflows)} overflow lines: " + " ".join(f"#{o['idx']}({o['atempo']:.1f}x)" for o in overflows))
    else:
        print("📋 Post-build fix: no overflow lines found")
    return overflows


def classify_fix_action(
    overflow: dict,
    stats: list[dict],
    segment_meta: dict,
    visibility: dict | None = None,
) -> str:
    """Decide fix action for an overflow line: merge / expand / retranslate_short / retranslate_long / accept."""
    idx = overflow["idx"]
    meta = segment_meta[idx]
    vis = meta.get("visibility", {})

    stats_by_idx = {s["index"]: s for s in stats}

    # Check if can merge with adjacent same-speaker
    all_indices = sorted(stats_by_idx.keys())
    pos = all_indices.index(idx) if idx in all_indices else -1

    if pos >= 0 and pos + 1 < len(all_indices):
        next_idx = all_indices[pos + 1]
        next_stat = stats_by_idx[next_idx]
        next_meta = segment_meta.get(next_idx, {})
        next_vis = next_meta.get("visibility", {})

        if (next_stat["speaker"] == overflow["speaker"]
                and not vis.get("has_visible_speaking", False)):
            gap = next_stat["subtitle_start_ms"] - overflow["subtitle_end_ms"]
            if gap < 1000:
                return "merge"

    # Check if can expand into gap after
    next_sub_start = overflow.get("next_subtitle_start_ms")
    if next_sub_start:
        gap_after = next_sub_start - overflow["subtitle_end_ms"]
        if gap_after > MIN_GAP_FOR_EXPAND_MS and not vis.get("has_visible_speaking", False):
            return "expand"

    # Retranslate: short vs long
    sub_dur = overflow["subtitle_duration_ms"]
    text = _get_text_for_idx(idx, segment_meta)
    word_count = len(text.split()) if text else 0

    if word_count <= 1:
        return "accept"
    elif word_count <= SHORT_PHRASE_MAX_WORDS or sub_dur < SHORT_PHRASE_MAX_DUR_MS:
        return "retranslate_short"
    else:
        return "retranslate_long"


def _get_text_for_idx(idx: int, segment_meta: dict) -> str:
    """Extract translated text from segment metadata."""
    meta = segment_meta.get(idx, {})
    return meta.get("text", "")


def retranslate_short(
    text: str,
    context_before: str,
    context_after: str,
    target_language: str,
    client: OpenAI,
    model: str = "gpt-4o",
    available_ms: int = 0,
) -> str:
    """Shorten a short phrase using word-count constraint."""
    word_count = len(text.split())
    target_words = max(1, word_count - 1)

    urgency = ""
    if available_ms > 0 and available_ms < 600:
        urgency = (
            f"\nThis line must fit in only {available_ms}ms of speech — extremely tight. "
            f"Minimize syllable count. Use the shortest-sounding words possible."
        )

    prompt = (
        f"Shorten this dubbed line to max {target_words} words while preserving core meaning.\n"
        f"Language: {target_language}\n"
        f"Context before: \"{context_before}\"\n"
        f"Line to shorten: \"{text}\"\n"
        f"Context after: \"{context_after}\"\n"
        f"{urgency}\n"
        f"Reply with ONLY the shortened line, nothing else."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip().strip('"')


def retranslate_long(
    text: str,
    context_before: str,
    context_after: str,
    target_language: str,
    available_ms: int,
    client: OpenAI,
    model: str = "gpt-4o",
) -> list[str]:
    """Generate 2 progressively shorter alternatives for TTS measurement."""
    prompt = (
        f"Generate exactly 2 progressively shorter alternatives for this dubbed line.\n"
        f"Language: {target_language}\n"
        f"Context before: \"{context_before}\"\n"
        f"Line: \"{text}\"\n"
        f"Context after: \"{context_after}\"\n\n"
        f"The line must fit in {available_ms}ms of speech. Make each alternative shorter than the previous.\n"
        f"Preserve the core meaning as much as possible.\n\n"
        f"Reply as a JSON array of 2 strings, shortest last. Example: [\"alt1\", \"alt2\"]"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.5,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data[:2]
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v[:2]
    except (json.JSONDecodeError, TypeError):
        pass
    return [text]



def retranslate_timing_fix(
    overflow_requests: dict[int, dict],
    underflow_requests: dict[int, dict],
    openai_client: OpenAI,
    target_language: str,
    model: str = "gpt-4o",
) -> dict[int, list[str]]:
    """
    Single batched LLM call to fix both overflow (shorten) and underflow (expand) lines.
    Returns {idx: [original, variant1, variant2, variant3]} — original is always first
    so TTS candidate selection will keep it if no alternative fits better.
    """
    shorten_eligible: dict[int, dict] = {}
    for idx, req in overflow_requests.items():
        if len(req["text"].split()) > 1:
            shorten_eligible[idx] = req

    expand_eligible: dict[int, dict] = dict(underflow_requests)

    if not shorten_eligible and not expand_eligible:
        return {}

    # Tier overflow lines by severity: mild (2 variants), hard/extreme (3 variants)
    mild_lines: dict[int, dict] = {}
    hard_lines: dict[int, dict] = {}
    extreme_lines: dict[int, dict] = {}
    for idx, req in shorten_eligible.items():
        avail = req.get("available_window_ms", 1)
        own_ratio = req["current_duration_ms"] / avail if avail > 0 else 2.0
        if own_ratio <= MILD_OVERFLOW_RATIO:
            mild_lines[idx] = req
        elif own_ratio <= HARD_OVERFLOW_RATIO:
            hard_lines[idx] = req
        else:
            extreme_lines[idx] = req

    # Determine per-index variant count for response parsing
    idx_num_variants: dict[int, int] = {}
    for idx in mild_lines:
        idx_num_variants[idx] = 2
    for idx in hard_lines:
        idx_num_variants[idx] = 3
    for idx in extreme_lines:
        idx_num_variants[idx] = 3
    for idx in expand_eligible:
        idx_num_variants[idx] = 3

    if mild_lines or hard_lines or extreme_lines:
        print(f"  📊 Overflow tiers: {len(mild_lines)} mild (2v), {len(hard_lines)} hard (3v), {len(extreme_lines)} extreme (3v)")

    prompt_parts = [
        f"You are a professional dubbing translator re-translating dialogue lines that don't fit their time slots.\n"
        f"Target language: {target_language}\n\n"
        f"Each line has an original source text and a current translation that is too long or too short.\n"
        f"Re-translate from the source text to produce alternatives that fit the time slot better.\n"
        f"This is TRANSLATION, not free rewriting — every alternative must faithfully convey what the source text says.\n"
        f"Do not invent new meaning, do not borrow meaning from neighboring lines, do not paraphrase the context.\n"
        f"Each line carries a `target_length` — a rough length target measured in the target language's own units "
        f"(syllables for alphabetic scripts, characters for Chinese/Japanese/Korean, letters for Arabic). "
        f"Land close to it; the goal is to JUST fit, not to be as short as possible.\n"
        f"IMPORTANT: Always write numbers as fully spelled-out words (e.g., 'two forty-seven' not '247', 'twenty-five' not '25'). Digits cause pauses in TTS.\n"
    ]

    def _format_lines_block(lines: dict[int, dict]) -> str:
        entries = []
        for idx, req in lines.items():
            entry = {"source": req.get("source_text", ""), "current_translation": req["text"],
                     "target_length": req.get("target_length"),
                     "context_before": req.get("context_before", ""), "context_after": req.get("context_after", "")}
            entries.append(f"  {idx}: {json.dumps(entry, ensure_ascii=False)}")
        return "{\n" + ",\n".join(entries) + "\n}"

    # Floor is tier-dependent: mild lines barely overflow, so keep them near the
    # target to avoid dead air; hard/extreme lines genuinely need to shrink, so
    # give the model room to cut below target rather than pinning it at the ceiling.
    if mild_lines:
        prompt_parts.append(
            f"\nSHORTEN — MILD (produce exactly 2 alternatives):\n"
            "These lines are only slightly over-length. Produce the LONGEST faithful translation at or just "
            "under each line's `target_length`. Do NOT go below 0.9x the target — undershooting leaves audible silence. "
            "Variants should BRACKET the target (one at ~target, one slightly under), not march monotonically shorter.\n"
            + _format_lines_block(mild_lines) + "\n"
        )

    if hard_lines:
        prompt_parts.append(
            f"\nSHORTEN — HARD (produce exactly 3 alternatives):\n"
            "These lines are noticeably over-length and MUST shrink. Aim at or below each line's `target_length`; "
            "you may go down to about 0.75x the target if needed for natural phrasing. Err on the shorter side — "
            "a slightly short line is better than one that spills into the next. "
            "Variants should span from ~target down toward the lower bound.\n"
            + _format_lines_block(hard_lines) + "\n"
        )

    if extreme_lines:
        prompt_parts.append(
            f"\nSHORTEN — EXTREME (produce exactly 3 alternatives):\n"
            "These lines are severely over-length and MUST be cut aggressively. Hit each line's `target_length` "
            "or go shorter — down to about 0.65x the target — dropping filler words and picking short synonyms. "
            "Prioritize fitting over completeness, but keep the result sounding like natural spoken dialogue, "
            "not a telegram or bullet list. Variants should span from ~target down to the lower bound.\n"
            + _format_lines_block(extreme_lines) + "\n"
        )

    if expand_eligible:
        prompt_parts.append(
            f"\nEXPAND these lines — produce exactly 3 alternatives (current translation is too short, causing awkward silence).\n"
            f"Re-translate from the source text using fuller phrasing while preserving the original meaning. "
            f"Aim for each line's `target_length`; do NOT exceed 1.1x the target. "
            f"Variants should bracket the target, not grow without bound:\n"
            + _format_lines_block(expand_eligible) + "\n"
        )

    prompt_parts.append(
        f"\nReply as a single JSON object mapping index to an array of strings (translations only, no source text).\n"
        f"IMPORTANT: produce exactly the number of alternatives requested per section (2 or 3).\n"
        f"Example: {{\"5\": [\"shorter\", \"shortest\"], \"12\": [\"alt1\", \"alt2\", \"alt3\"]}}"
    )

    prompt = "\n".join(prompt_parts)
    all_eligible = {**shorten_eligible, **expand_eligible}

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
            temperature=0.5,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  ⚠️ Batched timing retranslation failed: {e}")
        return {}

    candidate_texts: dict[int, list[str]] = {}
    for idx_str, variants in data.items():
        try:
            idx = int(idx_str)
        except (ValueError, TypeError):
            continue
        if idx not in all_eligible:
            continue
        original = all_eligible[idx]["text"]

        # Normalize: accept both list and single string responses
        if isinstance(variants, str):
            variants = [variants]
        if not isinstance(variants, list):
            continue

        # Deduplicate and filter empty/identical
        seen = {original}
        valid_variants = []
        for v in variants:
            if isinstance(v, str) and v and v not in seen:
                seen.add(v)
                valid_variants.append(v)

        max_v = idx_num_variants.get(idx, 3)
        candidate_texts[idx] = [original] + valid_variants[:max_v]

    if candidate_texts:
        shorten_count = sum(1 for idx in candidate_texts if idx in shorten_eligible)
        expand_count = sum(1 for idx in candidate_texts if idx in expand_eligible)
        total_variants = sum(len(v) - 1 for v in candidate_texts.values())
        print(f"  🔄 Timing retranslation: {shorten_count} shortened, {expand_count} expanded — {total_variants} total variants + originals")

    return candidate_texts


def apply_merge(
    idx: int,
    next_idx: int,
    subtitles_file: str,
) -> list[int]:
    """Merge two adjacent subs in the SRT file. Returns list of changed indices."""
    with open(subtitles_file, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))

    subs_by_idx = {s.index: s for s in subs}
    if idx not in subs_by_idx or next_idx not in subs_by_idx:
        return []

    sub = subs_by_idx[idx]
    next_sub = subs_by_idx[next_idx]

    # Merge text
    speaker = sub.content.split(":", 1)[0].strip() if ":" in sub.content else ""
    text1 = sub.content.split(":", 1)[1].strip() if ":" in sub.content else sub.content
    text2 = next_sub.content.split(":", 1)[1].strip() if ":" in next_sub.content else next_sub.content

    sub.end = next_sub.end
    sub.content = f"{speaker}: {text1} {text2}" if speaker else f"{text1} {text2}"

    # Remove next_sub
    subs = [s for s in subs if s.index != next_idx]

    with open(subtitles_file, "w", encoding="utf-8") as f:
        f.write(srt.compose(sorted(subs, key=lambda x: x.start), reindex=False))

    return [idx]


def apply_expand(
    idx: int,
    expand_ms: int,
    subtitles_file: str,
) -> list[int]:
    """Expand a subtitle's end time into the gap after it."""
    with open(subtitles_file, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))

    for sub in subs:
        if sub.index == idx:
            sub.end = sub.end + timedelta(milliseconds=expand_ms)
            break
    else:
        return []

    with open(subtitles_file, "w", encoding="utf-8") as f:
        f.write(srt.compose(sorted(subs, key=lambda x: x.start), reindex=False))

    return [idx]


def apply_retranslation(
    idx: int,
    new_text: str,
    subtitles_file: str,
) -> list[int]:
    """Replace the text of a subtitle line."""
    with open(subtitles_file, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))

    for sub in subs:
        if sub.index == idx:
            speaker = sub.content.split(":", 1)[0].strip() if ":" in sub.content else ""
            text = new_text.strip()
            # The model sometimes echoes the speaker label into new_text
            # ("Paulo: ..."). Strip it so we don't double the prefix ("Paulo: Paulo:
            # ..."), which regen would speak aloud. Only strip when the leading label
            # equals this line's speaker — a genuine "he said:" in dialogue survives.
            if speaker and text.split(":", 1)[0].strip() == speaker:
                text = text.split(":", 1)[1].strip()
            sub.content = f"{speaker}: {text}" if speaker else text
            break
    else:
        return []

    with open(subtitles_file, "w", encoding="utf-8") as f:
        f.write(srt.compose(sorted(subs, key=lambda x: x.start), reindex=False))

    return [idx]


def post_build_fix(
    stats: list[dict],
    segment_meta: dict,
    subtitles_file: str,
    openai_client: OpenAI,
    target_language: str,
    model: str = "gpt-4o",
) -> tuple[list[int], dict[int, list[str]]]:
    """
    Run post-build fixes on overflow lines. Called BEFORE regen so text changes
    get regenerated with proper duration factors in the single regen pass.

    Args:
        stats: from tts_build_final testing pass
        segment_meta: from tts_build_final
        subtitles_file: path to the translated srt (will be modified in place)
        openai_client: for retranslation calls
        target_language: e.g. "en"
        model: OpenAI model for retranslation

    Returns:
        (changed_indices, candidate_texts)
        - changed_indices: indices where text was modified (merge/expand/retranslate_short)
        - candidate_texts: {idx: [variant1, variant2, ...]} for long-phrase lines
          that need TTS measurement to pick the best. These are NOT written to the SRT —
          TTS generates all variants, picks the winner, then the winner text gets applied.
    """
    overflows = identify_overflow_lines(stats, segment_meta)
    if not overflows:
        return [], {}

    # Load subs for context
    with open(subtitles_file, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))
    subs_by_idx = {s.index: s for s in subs}

    stats_by_idx = {s["index"]: s for s in stats}
    changed = []
    candidate_texts: dict[int, list[str]] = {}

    for overflow in overflows:
        idx = overflow["idx"]
        action = classify_fix_action(overflow, stats, segment_meta)
        print(f"  🔧 idx={idx} atempo={overflow['atempo']:.1f}x → action={action}")

        if action == "accept":
            continue

        elif action == "merge":
            all_indices = sorted(stats_by_idx.keys())
            pos = all_indices.index(idx)
            next_idx = all_indices[pos + 1]
            merged = apply_merge(idx, next_idx, subtitles_file)
            changed.extend(merged)

        elif action == "expand":
            next_sub_start = overflow.get("next_subtitle_start_ms")
            sub_end = overflow["subtitle_end_ms"]
            if next_sub_start:
                gap = next_sub_start - sub_end
                expand_by = min(gap - 120, overflow["raw_len_ms"] - overflow["available_ms"])
                expand_by = max(0, int(expand_by))
                if expand_by > 0:
                    expanded = apply_expand(idx, expand_by, subtitles_file)
                    changed.extend(expanded)

        elif action == "retranslate_short":
            sub = subs_by_idx.get(idx)
            if not sub:
                continue
            text = sub.content.split(":", 1)[1].strip() if ":" in sub.content else sub.content
            ctx_before = _get_context(subs_by_idx, idx, -1)
            ctx_after = _get_context(subs_by_idx, idx, 1)

            shortened = retranslate_short(text, ctx_before, ctx_after, target_language, openai_client, model, available_ms=overflow["available_ms"])
            if shortened and shortened != text:
                print(f"  ✂️ idx={idx}: '{text}' → '{shortened}'")
                retranslated = apply_retranslation(idx, shortened, subtitles_file)
                changed.extend(retranslated)
            else:
                print(f"  ⚠️ idx={idx}: retranslate_short unchanged: '{text}' → '{shortened}'")


        elif action == "retranslate_long":
            sub = subs_by_idx.get(idx)
            if not sub:
                continue
            text = sub.content.split(":", 1)[1].strip() if ":" in sub.content else sub.content
            ctx_before = _get_context(subs_by_idx, idx, -1)
            ctx_after = _get_context(subs_by_idx, idx, 1)

            candidates = retranslate_long(
                text, ctx_before, ctx_after, target_language,
                overflow["available_ms"], openai_client, model,
            )

            if candidates:
                # Include original text as first candidate (it might fit after regen with speaker factor)
                all_variants = [text] + [c for c in candidates if c != text]
                candidate_texts[idx] = all_variants

    return changed, candidate_texts


def _get_context(subs_by_idx: dict, idx: int, direction: int) -> str:
    """Get adjacent subtitle text for context."""
    target_idx = idx + direction
    sub = subs_by_idx.get(target_idx)
    if sub:
        text = sub.content.split(":", 1)[1].strip() if ":" in sub.content else sub.content
        return text
    return ""
