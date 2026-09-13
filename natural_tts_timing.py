"""
Natural TTS timing: per-speaker uniform atempo approach.

Instead of IndexTTS duration_factor (which causes choppy word-by-word output),
generate at factor=1.0 and use WSOLA atempo for speed adjustment. Atempo
preserves natural prosody up to ~1.35x.

Feature flag: use_natural_atempo in main_prefect_dag.py
"""
from __future__ import annotations

import json
import os
import re
from datetime import timedelta

import srt


# Alphabetic scripts (Latin + Cyrillic): a syllable ≈ a vowel group.
# Includes accented vowels so ES/DE/IT/PT (and Polish/Nordic) count correctly.
_VOWEL_GROUP_RE = re.compile(
    r"[aeiouy"
    r"àáâãäåāăą"
    r"èéêëēĕėęě"
    r"ìíîïĩīĭįı"
    r"òóôõöøōŏő"
    r"ùúûüũūŭůűų"
    r"ýÿ"
    r"аеёиоуыэюяіїєўї]+",
    re.IGNORECASE,
)

# Per-character scripts where each glyph ≈ one syllable/mora: CJK Han, kana, Hangul.
_CJK_RE = re.compile(r"[一-鿿぀-ゟ゠-ヿ가-힣]")

# Arabic is an abjad (short vowels unwritten); letter count is the best cheap proxy.
# Excludes tatweel (ـ) and diacritics.
_ARABIC_RE = re.compile(r"[ء-غف-ي]")


def estimate_speech_units(text: str) -> int:
    """Rough per-language speech-length metric for an LLM retranslation target.

    - Alphabetic (Latin/Cyrillic): vowel groups ≈ syllables
    - CJK (Chinese/Japanese/Korean): each glyph ≈ one syllable/mora
    - Arabic: letter count (short vowels are unwritten, so syllables aren't countable)

    Scripts sum, so a line is measured by whichever script dominates it. Only a rough
    target — the downstream TTS measure-and-pick loop refines the actual fit.
    """
    text = text or ""
    return (
        len(_VOWEL_GROUP_RE.findall(text))
        + len(_CJK_RE.findall(text))
        + len(_ARABIC_RE.findall(text))
    )


SPEAKER_BASE_ATEMPO_CAP = 1.35
SPEAKER_MIN_SUB_MS = 1000
OVERFLOW_ATEMPO_THRESHOLD = 1.35
UNDERFLOW_ATEMPO_THRESHOLD = 0.92
MIN_GAP_MS = 120
ULTRA_SHORT_MAX_WORDS = 1
ULTRA_SHORT_MAX_SUB_MS = 500
BRIEF_THRESHOLD_MS = 800


def compute_speaker_base_atempo(
    stats: list[dict],
    segment_meta: dict,
    speaker_base_cap: float = SPEAKER_BASE_ATEMPO_CAP,
) -> dict[str, float]:
    """
    After first TTS at factor=1.0, compute the minimum uniform atempo per speaker
    so that ~80% of their lines (>1s) fit within their own subtitle window.

    Returns: {speaker_name: base_atempo} (only speakers needing > 1.0)
    """
    speaker_to_ratios: dict[str, list[float]] = {}

    for stat in stats:
        idx = stat["index"]
        meta = segment_meta.get(idx)
        if not meta:
            continue
        sub_dur = meta["subtitle_duration_ms"]
        if sub_dur < SPEAKER_MIN_SUB_MS:
            continue

        raw_len = meta["raw_len"]
        if sub_dur <= 0:
            continue

        needed_atempo = raw_len / sub_dur
        speaker = stat["speaker"]
        speaker_to_ratios.setdefault(speaker, []).append(needed_atempo)

    speaker_base: dict[str, float] = {}
    for speaker, ratios in speaker_to_ratios.items():
        if not ratios:
            continue
        sorted_ratios = sorted(ratios)
        # Pick the 80th percentile — base atempo that makes 80% of lines fit
        p80_idx = int(len(sorted_ratios) * 0.80)
        p80_ratio = sorted_ratios[min(p80_idx, len(sorted_ratios) - 1)]

        if p80_ratio > 1.0:
            base = min(p80_ratio, speaker_base_cap)
            speaker_base[speaker] = round(base, 3)

    return speaker_base


def classify_lines(
    stats: list[dict],
    segment_meta: dict,
    speaker_base_atempo: dict[str, float],
    overflow_threshold: float = OVERFLOW_ATEMPO_THRESHOLD,
) -> dict:
    """
    After computing speaker base atempo, classify each line:
    - 'ok': fits with base atempo
    - 'overflow': needs retranslation (shorter text)
    - 'underflow': should reduce atempo toward 1.0

    Overflow is determined by whether the line fits its OWN subtitle window
    (not available_window_ms which is affected by previous lines' cascade).
    Final build uses available_window_ms for actual atempo calculation.

    Returns: {
        "per_line_atempo": {idx: float},
        "overflow_indices": [idx, ...],
        "underflow_indices": [idx, ...],
    }
    """
    per_line_atempo: dict[int, float] = {}
    overflow_indices: list[int] = []
    underflow_indices: list[int] = []

    for stat in stats:
        idx = stat["index"]
        meta = segment_meta.get(idx)
        if not meta:
            continue

        speaker = stat["speaker"]
        base = speaker_base_atempo.get(speaker, 1.0)
        raw_len = meta["raw_len"]
        available = stat["actual_available_duration_ms"]
        sub_dur = meta["subtitle_duration_ms"]
        text = meta.get("text", "")
        word_count = len(text.split()) if text else 0

        if available <= 0:
            per_line_atempo[idx] = base
            continue

        needed_atempo = raw_len / available

        # Ultra-short single-word lines: unlimited atempo, no retranslation
        if word_count <= ULTRA_SHORT_MAX_WORDS and sub_dur <= ULTRA_SHORT_MAX_SUB_MS:
            per_line_atempo[idx] = needed_atempo if needed_atempo > 1.0 else base
            continue

        # Brief lines (< 800ms) were already optimized at translation time, skip retranslation
        if sub_dur < BRIEF_THRESHOLD_MS:
            per_line_atempo[idx] = needed_atempo if needed_atempo > 1.0 else base
            continue

        # Retranslation decision: can speaker_base (1.25) make it fit its own window?
        own_ratio = raw_len / sub_dur if sub_dur > 0 else 1.0

        if own_ratio > overflow_threshold:
            overflow_indices.append(idx)
        elif own_ratio < UNDERFLOW_ATEMPO_THRESHOLD and sub_dur >= SPEAKER_MIN_SUB_MS:
            underflow_indices.append(idx)

        # Atempo assignment: based on available (actual sequential placement)
        if needed_atempo > overflow_threshold:
            per_line_atempo[idx] = overflow_threshold
        elif needed_atempo > base:
            per_line_atempo[idx] = needed_atempo
        elif needed_atempo < 1.0:
            per_line_atempo[idx] = max(needed_atempo, UNDERFLOW_ATEMPO_THRESHOLD)
        else:
            per_line_atempo[idx] = base

    return {
        "per_line_atempo": per_line_atempo,
        "overflow_indices": overflow_indices,
        "underflow_indices": underflow_indices,
    }


def build_retranslation_request(
    overflow_indices: list[int],
    stats: list[dict],
    segment_meta: dict,
    speaker_base_atempo: dict[str, float],
    subtitles_file: str,
    source_subtitles_file: str | None = None,
    overflow_threshold: float = OVERFLOW_ATEMPO_THRESHOLD,
) -> dict[int, dict]:
    """
    For overflow lines, compute how much shorter the text needs to be
    and prepare retranslation instructions.

    Returns: {idx: {"text": current_text, "source_text": original, "max_duration_ms": target, "speaker": ..., "context": ...}}
    """
    with open(subtitles_file, "r", encoding="utf-8") as f:
        subs = {s.index: s for s in srt.parse(f.read())}

    source_subs: dict[int, srt.Subtitle] = {}
    if source_subtitles_file:
        with open(source_subtitles_file, "r", encoding="utf-8") as f:
            source_subs = {s.index: s for s in srt.parse(f.read())}

    stats_by_idx = {s["index"]: s for s in stats}
    requests: dict[int, dict] = {}

    for idx in overflow_indices:
        stat = stats_by_idx.get(idx)
        meta = segment_meta.get(idx)
        if not stat or not meta:
            continue

        sub = subs.get(idx)
        if not sub:
            continue

        speaker = stat["speaker"]
        sub_dur = meta["subtitle_duration_ms"]

        # Target: audio that fits at the overflow cap with a 90% safety margin
        max_audio_ms = int(sub_dur * overflow_threshold * 0.90)

        text = sub.content.split(":", 1)[1].strip() if ":" in sub.content else sub.content.strip()

        # Scale the current length metric by the duration we need to hit, so the
        # LLM gets a concrete absolute target rather than a vague "shorter".
        current_ms = max(1, meta["raw_len"])
        current_length = estimate_speech_units(text)
        target_length = max(1, round(current_length * max_audio_ms / current_ms))

        source_text = ""
        src_sub = source_subs.get(idx)
        if src_sub:
            source_text = src_sub.content.split(":", 1)[1].strip() if ":" in src_sub.content else src_sub.content.strip()

        # Gather surrounding context for better retranslation
        context_before = ""
        context_after = ""
        for other_sub in sorted(subs.values(), key=lambda s: s.start):
            if other_sub.index == idx:
                continue
            other_text = other_sub.content.split(":", 1)[1].strip() if ":" in other_sub.content else other_sub.content.strip()
            if other_sub.end <= sub.start and other_sub.start >= sub.start - timedelta(seconds=10):
                context_before = other_text
            elif other_sub.start >= sub.end and not context_after:
                context_after = other_text

        requests[idx] = {
            "text": text,
            "source_text": source_text,
            "speaker": speaker,
            "max_duration_ms": max_audio_ms,
            "current_duration_ms": meta["raw_len"],
            "current_length": current_length,
            "target_length": target_length,
            "available_window_ms": sub_dur,
            "context_before": context_before,
            "context_after": context_after,
        }

    return requests


def build_underflow_retranslation_request(
    underflow_indices: list[int],
    stats: list[dict],
    segment_meta: dict,
    speaker_base_atempo: dict[str, float],
    subtitles_file: str,
    source_subtitles_file: str | None = None,
) -> dict[int, dict]:
    """
    For underflow lines, compute how much longer the text needs to be.
    Target: TTS audio that fills at least UNDERFLOW_ATEMPO_THRESHOLD of sub_dur.

    Returns: {idx: {"text": current_text, "source_text": original, "target_duration_ms": target, ...}}
    """
    with open(subtitles_file, "r", encoding="utf-8") as f:
        subs = {s.index: s for s in srt.parse(f.read())}

    source_subs: dict[int, srt.Subtitle] = {}
    if source_subtitles_file:
        with open(source_subtitles_file, "r", encoding="utf-8") as f:
            source_subs = {s.index: s for s in srt.parse(f.read())}

    stats_by_idx = {s["index"]: s for s in stats}
    requests: dict[int, dict] = {}

    for idx in underflow_indices:
        stat = stats_by_idx.get(idx)
        meta = segment_meta.get(idx)
        if not stat or not meta:
            continue

        sub = subs.get(idx)
        if not sub:
            continue

        speaker = stat["speaker"]
        sub_dur = meta["subtitle_duration_ms"]

        # Target: TTS output should be ~92% of sub_dur so it fills the window
        target_audio_ms = int(sub_dur * UNDERFLOW_ATEMPO_THRESHOLD)

        text = sub.content.split(":", 1)[1].strip() if ":" in sub.content else sub.content.strip()

        # Scale current length metric up to the fuller duration we want to fill.
        current_ms = max(1, meta["raw_len"])
        current_length = estimate_speech_units(text)
        target_length = max(1, round(current_length * target_audio_ms / current_ms))

        source_text = ""
        src_sub = source_subs.get(idx)
        if src_sub:
            source_text = src_sub.content.split(":", 1)[1].strip() if ":" in src_sub.content else src_sub.content.strip()

        context_before = ""
        context_after = ""
        for other_sub in sorted(subs.values(), key=lambda s: s.start):
            if other_sub.index == idx:
                continue
            other_text = other_sub.content.split(":", 1)[1].strip() if ":" in other_sub.content else other_sub.content.strip()
            if other_sub.end <= sub.start and other_sub.start >= sub.start - timedelta(seconds=10):
                context_before = other_text
            elif other_sub.start >= sub.end and not context_after:
                context_after = other_text

        requests[idx] = {
            "text": text,
            "source_text": source_text,
            "speaker": speaker,
            "target_duration_ms": target_audio_ms,
            "current_duration_ms": meta["raw_len"],
            "current_length": current_length,
            "target_length": target_length,
            "available_window_ms": sub_dur,
            "context_before": context_before,
            "context_after": context_after,
        }

    return requests


def save_natural_timing_data(
    output_dir: str,
    speaker_base_atempo: dict[str, float],
    per_line_atempo: dict[int, float],
    overflow_indices: list[int],
    underflow_indices: list[int],
):
    """Save timing analysis results for debugging and the final build pass."""
    data = {
        "speaker_base_atempo": speaker_base_atempo,
        "per_line_atempo": {str(k): v for k, v in per_line_atempo.items()},
        "overflow_indices": overflow_indices,
        "underflow_indices": underflow_indices,
    }
    path = os.path.join(output_dir, "natural_timing.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path
