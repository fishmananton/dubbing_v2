import json
import pysrt
from openai import OpenAI, AsyncOpenAI
import re
from datetime import timedelta
import asyncio
from pathlib import Path
import srt


def merge_subtitles(file_name: str):
    # Load your original SRT file
    subs = pysrt.open(file_name)

    # Merge logic — adjust thresholds as needed
    merged = []
    buffer = subs[0]
    for next_sub in subs[1:]:
        gap = next_sub.start.ordinal - buffer.end.ordinal  # gap in ms
        if gap < 300 and (len(buffer.text) + len(next_sub.text) < 200):
            buffer.text += " " + next_sub.text
            buffer.end = next_sub.end
        else:
            merged.append(buffer)
            buffer = next_sub
    merged.append(buffer)

    # Save the merged subtitles to a new file
    merged_subs = pysrt.SubRipFile(items=merged)
    merged_subs.save(file_name, encoding='utf-8')


def clean_subtitle_response(text):
    text = text.strip()

    # Match and remove ```srt ... ```
    match = re.match(r"```(?:srt)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # Safety: remove helper metadata if the model accidentally copied it.
    lines = [
        line for line in text.splitlines()
        if not line.strip().startswith("[TIMING ")
    ]

    return "\n".join(lines).strip()







def build_translation_prompt(source_language: str,
                             target_language: str,
                             punctuation: bool = False,
                             batched: bool = False) -> str:
    return f"""
    You are a professional subtitle translator and adaptive voiceover editor.

    Task:
    Translate subtitles from **{source_language}** to **{target_language}** for dubbing and voiceover.

    Important:
    - Subtitles may include speaker labels (e.g., "SPEAKER_01:", "Person_1:").
    - Keep speaker labels exactly as they appear. Do not translate, rename, reformat, or remove them.
    - Translate only the spoken text after each label.
    - Use speaker labels and surrounding text to infer tone and phrasing.
    - Infer context from the subtitles and choose meanings that best fit it.
    - If a term is ambiguous or has no clear equivalent, use the shortest natural transliteration or equivalent. Add clarification only if it is essential and timing allows.
    - Do not replace unclear or domain-specific terms with generic substitutes.
    - Translate non-speech elements naturally (e.g., "Hi!", "(laughs)", "(applause)").

    SRT Structure:
    - {'Preserve subtitle structure exactly for blocks marked <<TRANSLATE>>.' if batched else 'Preserve subtitle structure exactly.'}
    - Do NOT modify subtitle indices or timestamps.
    - Do not add, remove, merge, split, reorder, or renumber subtitle entries.
    - Keep the same number of text lines in each translated subtitle block.
    - Output only valid `.srt` subtitles and nothing else.
    - The user may include [TIMING ...] helper lines before subtitle blocks.
    - [TIMING ...] lines are metadata, not subtitles.
    - Use [TIMING ...] lines as timing guidance, but never include them in the output.
    - Do not wrap the output in markdown code fences.
    
    {'''Batch Context:
    - Some leading subtitle blocks may be marked **<<CONTEXT ONLY>>**.
    - <<CONTEXT ONLY>> blocks are provided only for translation continuity, terminology, and tone.
    - Do NOT output <<CONTEXT ONLY>> blocks.
    - Translate only blocks marked **<<TRANSLATE>>**.
    - The first translated subtitle index may not start at 1. This is expected. Do NOT renumber subtitles.''' 
    if batched else ''}
    
        
    Dubbing Timing:
    - This translation is for AI dubbing, not written subtitles.
    - Each subtitle has a fixed time window.
    - If [TIMING ...] helper lines are present, use their pressure value as soft timing guidance.
    - If no timing helper is present, estimate timing from the SRT timestamp.
    - Timing pressure is not a hard word-count rule.
    - Preserve essential meaning, speaker intent, tone, names, numbers, negation, technical terms, and safety-critical details.
    - When timing is tight, compress wording instead of translating literally.
    - Remove filler, repetition, unnecessary intensifiers, parenthetical clarifications, and written-style phrasing.
    - Prefer short, natural, speakable target-language dubbing copy.
    - Very short subtitles should usually become very short utterances or fragments if meaning allows.
    - Longer subtitles may use natural speech, but should not expand beyond the source meaning.
    
    Priority order:
    1. Preserve valid SRT structure, indices, timestamps, and speaker labels.
    2. Preserve core meaning, tone, and essential details.
    3. Make the line natural and speakable in the target language.
    4. Compress wording to fit the timestamp as much as possible.

    Translation Style:
    - Translate for meaning, tone, and timing rather than literal wording.
    - Make phrasing fluent, natural, and easy to speak aloud.
    - Actively adapt phrasing to fit the original timing while preserving essential meaning.
    - Simplify or rephrase when needed for clarity and pacing.
    - Convert numbers, currencies, dates, percentages, and symbols into fully spoken forms appropriate for natural speech in the target language.
    - Use the correct grammatical form required by the language (including declensions, gender, or case if applicable).
    - If a sentence continues across subtitle blocks, make it flow naturally.
    - Improve grammar or style only when it improves natural speech flow without making the line longer unnecessarily.

    {""
    if punctuation else
    '''
    Voiceover & Timing Constraints (Very Important):
    - This translation will be used for speech synthesis and audio alignment.
    - Do NOT add sentence-ending punctuation unless it clearly exists in the source text.
    - Avoid introducing new ".", "?", or "!" at the end of lines.
    - Prefer commas or no punctuation over sentence-ending punctuation.
    - Preserve a continuous spoken-flow style rather than written prose.
    - Do not split a single flowing sentence into multiple sentences.
    - Keep pauses minimal and natural for dubbing.
    '''
    }

    If unsure about a translation, translate conservatively and preserve the original structure.
    """.strip()


def build_batch_user_content(
    subs,
    start: int,
    end: int,
    target_language: str,
    context_size: int = 10,
) -> str:
    context_subs = subs[max(0, start - context_size):start]
    translate_subs = subs[start:end]

    parts = []

    parts.append(
        f"Target language: {target_language}\n"
        f"Translate for AI dubbing using concise, natural spoken {target_language}.\n"
        "Use [TIMING ...] helper lines as soft timing guidance only.\n"
        "Do not output [TIMING ...] lines.\n"
        "Very short subtitles should remain very short unless essential meaning requires more."
    )
    parts.append("")

    if context_subs:
        parts.append("<<CONTEXT ONLY>>")
        parts.append(srt.compose(context_subs, reindex=False).strip())
        parts.append("")

    parts.append("<<TRANSLATE>>")
    parts.append(
        compose_with_timing_hints(
            translate_subs=translate_subs,
            all_subs=subs,
            start_pos=start,
        )
    )

    return "\n".join(parts).strip()


async def translate_batch(client: AsyncOpenAI, model: str, prompt: str, batch_text: str, batch_id: int, sem: asyncio.Semaphore):
    async with sem:
        response = await client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": batch_text},
            ],
        )
    return batch_id, response.choices[0].message.content.strip()



async def translate_batched_async(
    api_key: str,
    subtitles_file: str,
    source_language: str,
    target_language: str,
    model: str,
    result_file: str,
    punctuation: bool = False,
    batch_size: int = 80,
    context_size: int = 10,
):
    text = Path(subtitles_file).read_text(encoding="utf-8")
    subs = list(srt.parse(text))

    prompt = build_translation_prompt(source_language, target_language, punctuation, batched=True)
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(4)
    tasks = []
    batch_id = 0

    for start in range(0, len(subs), batch_size):
        end = min(start + batch_size, len(subs))
        batch_text = build_batch_user_content(
            subs,
            start,
            end,
            target_language=target_language,
            context_size=context_size,
        )
        tasks.append(translate_batch(client, model, prompt, batch_text, batch_id, sem))
        batch_id += 1

    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x[0])

    merged = "\n\n".join(text.strip() for _, text in results).strip() + "\n"
    Path(result_file).write_text(clean_subtitle_response(merged), encoding="utf-8")




def duration_constraint(ms: int, gap_after_ms: int | None = None) -> str:
    if ms < 900:
        pressure = "ULTRA_SHORT: shortest natural utterance; fragment preferred if meaning allows"
    elif ms < 1500:
        pressure = "VERY_SHORT: very short phrase; avoid extra clauses if possible"
    elif ms < 2500:
        pressure = "SHORT: one compact spoken idea if possible"
    elif ms < 4000:
        pressure = "NORMAL: concise natural spoken sentence"
    else:
        pressure = "RELAXED: natural speech allowed, but do not expand beyond source meaning"

    if gap_after_ms is not None and gap_after_ms < 180:
        pressure += "; next subtitle starts immediately, avoid spillover"

    return pressure


def timing_hint_for_sub(sub: srt.Subtitle, next_sub: srt.Subtitle | None = None) -> str:
    duration_ms = max(
        1,
        int(round((sub.end - sub.start).total_seconds() * 1000))
    )

    if next_sub is None:
        gap_after_ms = None
        gap_after_text = "none"
    else:
        gap_after_ms = max(
            0,
            int(round((next_sub.start - sub.end).total_seconds() * 1000))
        )
        gap_after_text = str(gap_after_ms)

    pressure = duration_constraint(duration_ms, gap_after_ms)

    return (
        f'[TIMING idx={sub.index} '
        f'duration_ms={duration_ms} '
        f'gap_after_ms={gap_after_text} '
        f'pressure="{pressure}"]'
    )


def compose_with_timing_hints(
    translate_subs: list[srt.Subtitle],
    all_subs: list[srt.Subtitle],
    start_pos: int,
) -> str:
    blocks = []

    for offset, sub in enumerate(translate_subs):
        absolute_pos = start_pos + offset
        next_sub = all_subs[absolute_pos + 1] if absolute_pos + 1 < len(all_subs) else None

        blocks.append(
            timing_hint_for_sub(sub, next_sub)
            + "\n"
            + srt.compose([sub], reindex=False).strip()
        )

    return "\n\n".join(blocks)



def translate(client: OpenAI, subtitles_file: str, source_language: str, target_language: str, model: str, result_file: str, punctuation: bool = False):
    if source_language == target_language:
        Path(result_file).write_text(
            Path(subtitles_file).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return

    text = Path(subtitles_file).read_text(encoding="utf-8")
    subs = list(srt.parse(text))

    prompt = build_translation_prompt(
        source_language,
        target_language,
        punctuation,
        batched=True,
    )

    # Do not split if total subtitles <= 120
    if len(subs) <= 120:
        user_content = build_batch_user_content(
            subs,
            0,
            len(subs),
            target_language=target_language,
            context_size=0,
        )

        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
        )
        Path(result_file).write_text(
            clean_subtitle_response(response.choices[0].message.content.strip()),
            encoding="utf-8",
        )
        return

    asyncio.run(
        translate_batched_async(
            api_key=client.api_key,
            subtitles_file=subtitles_file,
            source_language=source_language,
            target_language=target_language,
            model=model,
            result_file=result_file,
            punctuation=punctuation,
            batch_size=80,
            context_size=10,
        )
    )







def format_timestamp(seconds: float) -> str:
    """Convert seconds (float) to SRT time format."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def create_srt_for_speakers(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If your file has the structure "results" -> "audio_segments"
    segments = data.get("results", {}).get("audio_segments", [])
    if not segments:
        raise ValueError("No 'audio_segments' found in JSON structure.")

    # Group segments by speaker
    speakers = {}
    for seg in segments:
        speaker = seg.get("speaker_label", "unknown")
        speakers.setdefault(speaker, []).append(seg)

    # Write one SRT per speaker
    for speaker, segs in speakers.items():
        srt_filename = f"{speaker}.srt"
        with open(srt_filename, "w", encoding="utf-8") as srt_file:
            for idx, seg in enumerate(segs, 1):
                start = seg.get("start_time", 0)
                end = seg.get("end_time", 0)
                text = seg.get("transcript", "").strip()

                start_ts = format_timestamp(float(start))
                end_ts = format_timestamp(float(end))

                srt_file.write(f"{idx}\n")
                srt_file.write(f"{start_ts} --> {end_ts}\n")
                srt_file.write(f"{text}\n\n")

        print(f"✅ Created {srt_filename} with {len(segs)} segments")
