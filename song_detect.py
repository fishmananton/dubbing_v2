"""Detect sung subtitle lines from original audio and drop them before GENERATE.

Sung lyrics must not be translated/dubbed. This module chunks long audio by
inter-sub silence, asks Gemini per chunk which subtitle IDs are sung, unions the
result, and drops those IDs from the translated SRT. The pipeline's existing
no-sub -> original-vocal fallback then plays the original singing over the music.

The chunker is shared with the QA script (test_dub_qc.py).
"""
from __future__ import annotations

import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta

import srt
from google import genai
from pydub import AudioSegment

from test_dub_qc import build_script, compress_audio


@dataclass
class Chunk:
    subs: list  # srt.Subtitle in this chunk, original indices, absolute times
    offset_s: float  # chunk start in seconds; subtract to rebase to 0-based audio
    end_s: float  # chunk end in seconds


def _gaps(subs: list) -> list[tuple[float, float, int]]:
    """Return (gap_seconds, midpoint_seconds, index_of_left_sub) for each
    adjacent pair, in sub order."""
    out = []
    for i in range(len(subs) - 1):
        left_end = subs[i].end.total_seconds()
        right_start = subs[i + 1].start.total_seconds()
        gap = right_start - left_end
        mid = left_end + gap / 2.0
        out.append((gap, mid, i))
    return out


def chunk_subs(subs: list, target_min: float = 10.0,
               min_split_min: float = 12.0, gap_min_s: float = 2.0) -> list[Chunk]:
    """Split subs into chunks. <= min_split_min total -> single chunk. Else split
    near each target_min mark at the >gap_min_s inter-sub gap nearest the mark
    (cut at gap midpoint). If no qualifying gap near/after a mark, the chunk runs
    on to the next qualifying gap (oversized allowed)."""
    if not subs:
        return []
    subs = sorted(subs, key=lambda s: s.start)
    total = subs[-1].end.total_seconds() - subs[0].start.total_seconds()
    start0 = subs[0].start.total_seconds()
    if total <= min_split_min * 60:
        return [Chunk(subs=list(subs), offset_s=0.0,
                      end_s=subs[-1].end.total_seconds())]

    target = target_min * 60
    chunks: list[Chunk] = []
    cur: list = []
    chunk_start = start0
    next_mark = chunk_start + target
    gaps = _gaps(subs)
    qualifying = {i: mid for (g, mid, i) in gaps if g > gap_min_s}

    for i, s in enumerate(subs):
        cur.append(s)
        if s.end.total_seconds() >= next_mark and i in qualifying and i < len(subs) - 1:
            mid = qualifying[i]
            chunks.append(Chunk(subs=cur, offset_s=chunk_start, end_s=mid))
            cur = []
            chunk_start = mid
            next_mark = chunk_start + target
    if cur:
        chunks.append(Chunk(subs=cur, offset_s=chunk_start,
                            end_s=subs[-1].end.total_seconds()))
    return chunks


def build_chunk_script(chunk: Chunk) -> str:
    """Indexed script for a chunk with times rebased to the 0-based audio slice.
    IDs stay original (we delete by original ID). Format matches build_script:
    '[index] start-end  content'."""
    rebased = []
    for s in chunk.subs:
        rebased.append(srt.Subtitle(
            index=s.index,
            start=timedelta(seconds=max(0.0, s.start.total_seconds() - chunk.offset_s)),
            end=timedelta(seconds=max(0.0, s.end.total_seconds() - chunk.offset_s)),
            content=s.content,
        ))
    lines = []
    for s in rebased:
        start = s.start.total_seconds()
        end = s.end.total_seconds()
        lines.append(f"[{s.index}] {start:.2f}-{end:.2f}  {s.content}")
    return "\n".join(lines)


def drop_sub_ids(srt_path: str, ids: list[int]) -> list[int]:
    """Remove subtitles whose index is in `ids` from the SRT file, in place.
    reindex=False keeps surviving indices stable. Returns the IDs actually
    removed."""
    id_set = set(ids)
    if not id_set:
        return []
    subs = list(srt.parse(open(srt_path, encoding="utf-8").read()))
    present = {s.index for s in subs}
    remaining = [s for s in subs if s.index not in id_set]
    removed = sorted(id_set & present)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(sorted(remaining, key=lambda x: x.start),
                            reindex=False))
    return removed


SONG_SYSTEM_PROMPT = (
    "You are an audio analyst for a film dubbing pipeline. You are given a slice "
    "of the original-language film audio and the subtitle script transcribed from "
    "it (format: [index] start-end  Speaker: text), with timestamps relative to "
    "this audio slice. Some subtitle lines are SUNG song lyrics (a musical number, "
    "rap, or singing), not spoken dialogue. Sung lines must NOT be dubbed. "
    "Dialogue spoken over background music is NOT sung and must be kept. Listen to "
    "the audio and identify which subtitle lines are sung lyrics rather than "
    "spoken dialogue. Return only the integer subtitle indices that are sung."
)


def _song_schema():
    return genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "sung_indices": genai.types.Schema(
                type=genai.types.Type.ARRAY,
                items=genai.types.Schema(type=genai.types.Type.INTEGER),
            )
        },
        required=["sung_indices"],
    )


def slice_audio_opus(audio_path: str, offset_s: float, end_s: float) -> bytes:
    """Cut [offset_s, end_s] from audio_path to 0-based mono opus bytes."""
    seg = AudioSegment.from_file(audio_path).set_channels(1)
    clip = seg[int(offset_s * 1000):int(end_s * 1000)]
    buf = io.BytesIO()
    clip.export(buf, format="ogg", codec="libopus", bitrate="48k")
    return buf.getvalue()


def detect_songs_in_chunk(chunk: Chunk, audio_path: str, client, model: str,
                          thinking_level: str = "MEDIUM") -> list[int]:
    """One Gemini call: return the sung subtitle IDs in this chunk."""
    audio_bytes = slice_audio_opus(audio_path, chunk.offset_s, chunk.end_s)
    script = build_chunk_script(chunk)
    prompt = (
        "Here is the subtitle script for this audio slice "
        "(format: [index] start-end  Speaker: text):\n\n"
        f"{script}\n\n"
        "Now listen to the audio and return the indices of lines that are SUNG "
        "song lyrics, not spoken dialogue."
    )
    contents = [
        genai.types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"),
        prompt,
    ]
    config = genai.types.GenerateContentConfig(
        system_instruction=SONG_SYSTEM_PROMPT,
        temperature=0.4,
        thinking_config=genai.types.ThinkingConfig(thinking_level=thinking_level),
        response_mime_type="application/json",
        response_schema=_song_schema(),
    )
    r = client.models.generate_content(model=model, contents=contents, config=config)
    valid = {s.index for s in chunk.subs}
    got = json.loads(r.text).get("sung_indices", [])
    # keep only IDs that actually belong to this chunk (guard against drift)
    return [int(i) for i in got if int(i) in valid]


def detect_songs(audio_path: str, subtitles_path: str, client, model: str,
                 thinking_level: str = "MEDIUM",
                 target_min: float = 10.0, min_split_min: float = 12.0,
                 gap_min_s: float = 2.0) -> list[int]:
    """Return sorted unique subtitle IDs that are sung lyrics. One Gemini call per
    chunk, unioned. Original IDs (never rebased)."""
    subs = list(srt.parse(open(subtitles_path, encoding="utf-8").read()))
    if not subs:
        return []
    chunks = chunk_subs(subs, target_min=target_min,
                        min_split_min=min_split_min, gap_min_s=gap_min_s)
    sung: set[int] = set()
    with ThreadPoolExecutor(max_workers=max(1, len(chunks))) as pool:
        results = pool.map(
            lambda c: detect_songs_in_chunk(c, audio_path, client, model,
                                            thinking_level),
            chunks)
        for r in results:
            sung.update(r)
    return sorted(sung)
