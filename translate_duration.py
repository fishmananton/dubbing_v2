"""
Translation module for dubbing pipeline.

Single-pass contextual translation optimized for spoken dialogue quality.
Timing adjustments are handled separately in the TIMING_FIX stage.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import srt


# ---------------------------------------------------------------------------
# SRT utilities
# ---------------------------------------------------------------------------

def _extract_speaker_label(content: str) -> tuple[str | None, str]:
    if ":" not in content:
        return None, content.strip()
    label, text = content.split(":", 1)
    return label.strip(), text.strip()


def _strip_label_from_sub(sub: srt.Subtitle) -> srt.Subtitle:
    _, body = _extract_speaker_label(sub.content)
    return srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=body)


# ---------------------------------------------------------------------------
# Translation prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a professional dubbing translator. Your translations will be spoken aloud by AI voices as a dubbed audio track over the original video.

Task: Translate subtitles from {source_language} to {target_language}.

Translation approach:
- This is DUBBING, not subtitling. Translate how a voice actor would speak the line — natural spoken dialogue.
- Adapt idioms, slang, humor, and cultural references so they land naturally in {target_language}. Do not translate literally.
- Preserve each character's voice: if they're sarcastic, witty, formal, casual, angry — the translation must carry that same energy.
- Keep the register: street talk stays street talk, formal stays formal, playful stays playful.
- Dialogue should sound like real people talking, not like translated text.

Brevity constraints:
- Lines marked `brief` have very short time slots (under 1 second). Keep them to 1-3 words maximum. These are interjections, reactions, or short phrases — translate as the shortest natural equivalent.

Structure rules:
- Preserve subtitle indices and timestamps EXACTLY as given.
- Do NOT add, remove, merge, split, reorder, or renumber entries.
- Output only valid .srt content. No markdown fences, no commentary.
- Do NOT include speaker labels in your output. Translate only the spoken text.
- Each subtitle line has a [META] tag showing the speaker name and gender. Use this for grammatical gender agreement but do not output it.

{speaker_genders_block}

Grammar:
- Adjust 1st-person self-references (adjectives, verbs, past participles) to match the speaker's gender.
- Adjust 2nd-person direct address to match the listener's gender when clear from context.
- Convert numbers, dates, currencies to their spoken forms.

Priority order:
1. Natural spoken dialogue that preserves meaning, tone, and character voice.
2. Valid SRT structure with exact indices/timestamps.
3. Correct grammatical gender."""


def _build_speaker_genders_block(speakers_data: dict | None) -> str:
    if not speakers_data:
        return ""
    lines = ["SPEAKER GENDER METADATA:"]
    for speaker, data in speakers_data.items():
        gender = data.get("gender", "unknown") if isinstance(data, dict) else "unknown"
        lines.append(f"  {speaker}: {gender}")
    return "\n".join(lines)


def _build_system_prompt(source_language: str, target_language: str, speakers_data: dict | None = None) -> str:
    return SYSTEM_PROMPT.format(
        source_language=source_language,
        target_language=target_language,
        speaker_genders_block=_build_speaker_genders_block(speakers_data),
    )


def _build_user_content(
    subs: list[srt.Subtitle],
    labels_map: dict[int, str | None],
    speakers_data: dict | None,
    brief_indices: set[int],
    start: int,
    end: int,
    context_size: int = 5,
) -> str:
    context_subs = subs[max(0, start - context_size):start]
    translate_subs = subs[start:end]

    parts: list[str] = []

    if context_subs:
        parts.append("<<CONTEXT ONLY — do not translate, do not output>>")
        stripped_context = [_strip_label_from_sub(s) for s in context_subs]
        parts.append(srt.compose(stripped_context, reindex=False).strip())
        parts.append("")

    parts.append("<<TRANSLATE the following>>")
    for sub in translate_subs:
        label = labels_map.get(sub.index)
        gender = None
        if label and speakers_data:
            spk_data = speakers_data.get(label)
            if spk_data and isinstance(spk_data, dict):
                gender = spk_data.get("gender")
        meta_parts = []
        if label and gender:
            meta_parts.append(f"speaker={label} ({gender})")
        elif label:
            meta_parts.append(f"speaker={label}")
        if sub.index in brief_indices:
            meta_parts.append("brief")
        if meta_parts:
            meta = f"[META {' | '.join(meta_parts)}]"
            parts.append(meta)
        stripped = _strip_label_from_sub(sub)
        parts.append(srt.compose([stripped], reindex=False).strip())
        parts.append("")

    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _extract_text_from_response(response) -> str:
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def _clean_llm_srt_response(text: str) -> str:
    text = text.strip()
    match = re.match(r"```(?:srt)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    lines = [
        line for line in text.splitlines()
        if not line.strip().startswith("[META")
    ]
    return "\n".join(lines).strip()


async def _translate_batch(
    client: anthropic.AsyncAnthropic,
    model: str,
    system_prompt: str,
    user_content: str,
    batch_id: int,
    sem: asyncio.Semaphore,
    max_retries: int = 3,
) -> tuple[int, str]:
    async with sem:
        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=8192,
                    thinking={"type": "adaptive"},
                    output_config={"effort": "low"},
                    system=[{
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    messages=[{"role": "user", "content": user_content}],
                )
                return batch_id, _extract_text_from_response(response)
            except (anthropic.APITimeoutError, anthropic.APIConnectionError) as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Translation batch {batch_id} failed after {max_retries} retries: {e}")
                await asyncio.sleep(2 ** attempt)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
    return batch_id, ""


# ---------------------------------------------------------------------------
# Brief post-processing
# ---------------------------------------------------------------------------

BRIEF_COMPRESS_SYSTEM = """You shorten translated subtitle lines that are too long for their time slot.

You receive a JSON array. Each entry has:
- "idx": subtitle index
- "max_words": hard word limit for this line
- "current": the current translation (too long)
- "context_before": the 2 lines before this one (for continuity)
- "context_after": the 1 line after this one (for continuity)

Rules:
- Return a JSON object mapping idx (as string) to the shortened translation.
- Stay within max_words. This is a hard limit.
- Preserve meaning as much as possible, but brevity wins. Use fragments, contractions, shortest synonyms.
- The result must sound natural as spoken dialogue — not telegraphic or robotic. A real person would say it out loud.
- The shortened line must flow naturally with context_before/context_after.
- Do not include speaker labels.
- Output ONLY the JSON object, no markdown fences, no commentary."""


def _word_count(text: str) -> int:
    return len(text.split())


async def _compress_brief_lines(
    client: anthropic.AsyncAnthropic,
    model: str,
    translated_map: dict[int, srt.Subtitle],
    source_subs: list[srt.Subtitle],
    labels_map: dict[int, str | None],
    brief_indices: set[int],
    sem: asyncio.Semaphore,
) -> int:
    ordered_indices = [s.index for s in source_subs]
    idx_pos = {idx: pos for pos, idx in enumerate(ordered_indices)}

    candidates = []
    for idx in brief_indices:
        sub = translated_map.get(idx)
        if not sub:
            continue
        _, body = _extract_speaker_label(sub.content)
        dur_ms = int((sub.end - sub.start).total_seconds() * 1000)
        max_words = max(1, dur_ms // 350)
        if _word_count(body) <= max_words:
            continue
        pos = idx_pos.get(idx, 0)
        ctx_before = []
        for p in range(max(0, pos - 2), pos):
            s = translated_map.get(ordered_indices[p])
            if s:
                _, b = _extract_speaker_label(s.content)
                ctx_before.append(b)
        ctx_after = []
        if pos + 1 < len(ordered_indices):
            s = translated_map.get(ordered_indices[pos + 1])
            if s:
                _, b = _extract_speaker_label(s.content)
                ctx_after.append(b)

        candidates.append({
            "idx": idx,
            "max_words": max_words,
            "current": body,
            "context_before": ctx_before,
            "context_after": ctx_after,
        })

    if not candidates:
        return 0

    print(f"📝 Brief compress: {len(candidates)} lines over word limit")

    async with sem:
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=4096,
                thinking={"type": "disabled"},
                system=[{
                    "type": "text",
                    "text": BRIEF_COMPRESS_SYSTEM,
                }],
                messages=[{"role": "user", "content": json.dumps(candidates, ensure_ascii=False)}],
            )
            raw = _extract_text_from_response(response)
            raw = raw.strip()
            if raw.startswith("```"):
                match = re.match(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
                if match:
                    raw = match.group(1)
            replacements = json.loads(raw)
        except Exception as e:
            print(f"⚠️ Brief compress failed: {e}")
            return 0

    # The model is asked for a {idx: text} object but sometimes returns a list of
    # {"idx"/"index": ..., "text"/"current"/...: ...} entries. Normalize to a dict.
    if isinstance(replacements, list):
        normalized = {}
        for entry in replacements:
            if not isinstance(entry, dict):
                continue
            key = entry.get("idx", entry.get("index"))
            val = entry.get("text", entry.get("new_text", entry.get("current")))
            if key is not None and val is not None:
                normalized[key] = val
        replacements = normalized
    elif not isinstance(replacements, dict):
        print(f"⚠️ Brief compress: unexpected response type {type(replacements).__name__}")
        return 0

    fixed = 0
    for idx_str, new_text in replacements.items():
        try:
            idx = int(idx_str)
        except (ValueError, TypeError):
            continue
        if idx not in translated_map or not isinstance(new_text, str) or not new_text.strip():
            continue
        sub = translated_map[idx]
        label = labels_map.get(idx)
        content = f"{label}: {new_text.strip()}" if label else new_text.strip()
        translated_map[idx] = srt.Subtitle(
            index=sub.index, start=sub.start, end=sub.end, content=content,
        )
        fixed += 1
        print(f"  ✂️ idx={idx}: '{_extract_speaker_label(sub.content)[1]}' → '{new_text.strip()}'")

    return fixed


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

@dataclass
class TranslationResult:
    total_lines: int
    pass1_ok: int
    pass2_attempted: int = 0
    pass2_fixed: int = 0
    final_out_of_bounds: int = 0
    brief_compressed: int = 0
    duration_factors: dict[int, float] = field(default_factory=dict)


async def translate_duration_async(
    anthropic_api_key: str,
    subtitles_file: str,
    source_language: str,
    target_language: str,
    pass1_model: str,
    pass2_model: str,
    result_file: str,
    duration_factors_file: str | None = None,
    speakers_data: dict | None = None,
    batch_size: int = 70,
    context_size: int = 5,
    concurrency: int = 10,
) -> TranslationResult:
    source_text = Path(subtitles_file).read_text(encoding="utf-8")
    source_subs = list(srt.parse(source_text))

    if not source_subs:
        Path(result_file).write_text("", encoding="utf-8")
        return TranslationResult(0, 0)

    src_code = source_language[:2].lower()
    tgt_code = target_language[:2].lower()
    if src_code == tgt_code:
        Path(result_file).write_text(source_text, encoding="utf-8")
        return TranslationResult(len(source_subs), len(source_subs))

    BRIEF_THRESHOLD_MS = 800

    labels_map: dict[int, str | None] = {}
    brief_indices: set[int] = set()
    for sub in source_subs:
        label, _ = _extract_speaker_label(sub.content)
        labels_map[sub.index] = label
        sub_dur_ms = int((sub.end - sub.start).total_seconds() * 1000)
        if sub_dur_ms < BRIEF_THRESHOLD_MS:
            brief_indices.add(sub.index)

    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
    sem = asyncio.Semaphore(concurrency)

    system_prompt = _build_system_prompt(source_language, target_language, speakers_data)
    tasks: list[asyncio.Task] = []

    for start in range(0, len(source_subs), batch_size):
        end = min(start + batch_size, len(source_subs))
        user_content = _build_user_content(
            source_subs, labels_map, speakers_data, brief_indices, start, end, context_size=context_size,
        )
        tasks.append(
            asyncio.ensure_future(
                _translate_batch(client, pass1_model, system_prompt, user_content, start, sem)
            )
        )

    batch_results = await asyncio.gather(*tasks)
    batch_results.sort(key=lambda x: x[0])

    merged_srt = "\n\n".join(
        _clean_llm_srt_response(text) for _, text in batch_results
    ).strip() + "\n"

    try:
        translated_subs = list(srt.parse(merged_srt))
    except Exception:
        Path(result_file).write_text(merged_srt, encoding="utf-8")
        return TranslationResult(len(source_subs), 0)

    translated_map: dict[int, srt.Subtitle] = {}
    for s in translated_subs:
        label = labels_map.get(s.index)
        content = f"{label}: {s.content}" if label else s.content
        translated_map[s.index] = srt.Subtitle(
            index=s.index, start=s.start, end=s.end, content=content,
        )

    for src_sub in source_subs:
        if src_sub.index not in translated_map:
            translated_map[src_sub.index] = srt.Subtitle(
                index=src_sub.index, start=src_sub.start, end=src_sub.end, content=src_sub.content,
            )

    # --- Brief post-processing: compress overlong BRIEF translations ---
    brief_fixed = await _compress_brief_lines(
        client, pass1_model, translated_map, source_subs, labels_map, brief_indices, sem,
    )

    final_subs = [translated_map[s.index] for s in source_subs]
    output = srt.compose(final_subs, reindex=False)
    Path(result_file).write_text(output, encoding="utf-8")

    return TranslationResult(
        total_lines=len(source_subs),
        pass1_ok=len(translated_subs),
        brief_compressed=brief_fixed,
    )


# ---------------------------------------------------------------------------
# Synchronous entry point
# ---------------------------------------------------------------------------

def translate_duration(
    anthropic_api_key: str,
    subtitles_file: str,
    source_language: str,
    target_language: str,
    pass1_model: str,
    pass2_model: str,
    result_file: str,
    duration_factors_file: str | None = None,
    speakers_data: dict | None = None,
    batch_size: int = 70,
    concurrency: int = 10,
) -> TranslationResult:
    return asyncio.run(
        translate_duration_async(
            anthropic_api_key=anthropic_api_key,
            subtitles_file=subtitles_file,
            source_language=source_language,
            target_language=target_language,
            pass1_model=pass1_model,
            pass2_model=pass2_model,
            result_file=result_file,
            duration_factors_file=duration_factors_file,
            speakers_data=speakers_data,
            batch_size=batch_size,
            concurrency=concurrency,
        )
    )
