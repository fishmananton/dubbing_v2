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
from dataclasses import dataclass, field

import srt
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
