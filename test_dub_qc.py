"""Exploratory: hand the FINAL dubbed audio + the subtitles it was built from to
Gemini and ask it to LISTEN and flag anomalies (mistranscribed screams, clipped
lines, wrong emotion, timing that doesn't match the words, etc.).

This is a probe to see what the model notices on its own — not a pipeline stage.

    python test_dub_qc.py --run output/20260903_barkoni_10
"""
from __future__ import annotations

import argparse
import io
import os
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Optional

import srt
from pydantic import BaseModel, Field
from pydub import AudioSegment
from dotenv import load_dotenv
from google import genai

load_dotenv()


QC_SYSTEM_PROMPT = """You are a professional dubbing quality-control reviewer.

You are given:
1. The FINAL dubbed audio track of a short video (the synthesized dub, one language).
2. The subtitle script it was generated from: each line has an index, a start/end
   timestamp, a speaker label, and the exact text that was supposed to be spoken.

The script contains ONLY real spoken dialogue. A line ending in "—" or "..." is a
line the speaker is interrupted on or trails off — it is NOT a stage direction, sound
effect, or caption. Never flag script text as if it were a direction read aloud.

Work through the audio line by line against the script. For EVERY subtitle, listen to
the corresponding region of audio and evaluate it on all the dimensions below. Do not
stop at content match — a line whose words are correct can still be a defect if it is
delivered flat, rushed, mis-stressed, oddly paced, or emotionally wrong. Judge the dub
the way a demanding reviewer would before it ships to a paying client.

Report anything on this (non-exhaustive) list:
- Content mismatch: audio doesn't say the scripted words — wrong/missing/extra words,
  gibberish, or a non-speech sound (scream, laugh, cough, held vowel, music, noise)
  where words were expected. This includes STT hallucinations: text that was invented
  from a shout, a held vowel, or noise in the source and does not belong in the dub.
- Text normalization: numbers, times, dates, acronyms, or abbreviations spoken wrong
  (e.g. a clock time read as a plain cardinal number, an acronym mangled into a word).
- Pronunciation: names or terms mispronounced or given wrong stress.
- Timing: a line's audio runs past its subtitle window, is cut off short, or is so
  rushed/dragged that the pacing sounds unnatural.
- Emotion / delivery: tone contradicts or fails to match the line's intent; robotic,
  flat, monotone, or affect-less delivery where the line calls for feeling; wrong
  emphasis; unnatural intonation that doesn't sound like a real person talking.
- Audio artifacts: clipping, clicks, truncated words, or silence where speech is due.

Be thorough and honest. It is better to surface a real weakness than to let a mediocre
line pass. Flag genuine problems across the whole track, not just the worst one or two.
Do not fabricate defects that aren't there, but do not withhold ones that are — a track
with several flat or awkward lines should produce several issues, not an empty list.

For EACH problem you find, return:
- "start" and "end": timestamp in seconds of the region in the audio
- "sub_index": the subtitle index it relates to (or null if none applies)
- "symptom": concretely what you HEAR that is wrong
- "mismatch": why it fails relative to the script or to good dubbing quality
- "severity": "high" | "medium" | "low"

Respond with a strict JSON object: {"issues": [ ... ]}. If the dub is good, return
{"issues": []}. No other text."""


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Issue(BaseModel):
    start: float = Field(description="start of the audio region in seconds")
    end: float = Field(description="end of the audio region in seconds")
    sub_index: Optional[int] = Field(
        default=None, description="subtitle index this relates to, or null")
    symptom: str = Field(description="what you HEAR that is wrong")
    mismatch: str = Field(description="why it fails")
    severity: Severity


class QCReport(BaseModel):
    issues: list[Issue]


_SEV_RANK = {Severity.low: 0, Severity.medium: 1, Severity.high: 2}


def dedup_issues(issues: list[Issue]) -> list[Issue]:
    """Union issues from multiple passes. Two issues collapse when they share a
    sub_index, or (when either lacks one) when their time windows overlap. On a
    collision keep the higher-severity, longer-symptom copy."""
    kept: list[Issue] = []
    for cand in issues:
        for i, k in enumerate(kept):
            same_sub = (cand.sub_index is not None
                        and cand.sub_index == k.sub_index)
            overlap = cand.start < k.end and k.start < cand.end
            if same_sub or overlap:
                better = (_SEV_RANK[cand.severity], len(cand.symptom)) > (
                    _SEV_RANK[k.severity], len(k.symptom))
                if better:
                    kept[i] = cand
                break
        else:
            kept.append(cand)
    kept.sort(key=lambda x: x.start)
    return kept


def build_script(srt_path: str) -> str:
    with open(srt_path, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))
    lines = []
    for s in subs:
        start = s.start.total_seconds()
        end = s.end.total_seconds()
        lines.append(f"[{s.index}] {start:.2f}-{end:.2f}  {s.content}")
    return "\n".join(lines)


def compress_audio(wav_path: str) -> bytes:
    """Downmix to mono opus so a 3-min track fits in an inline request."""
    seg = AudioSegment.from_file(wav_path).set_channels(1)
    buf = io.BytesIO()
    seg.export(buf, format="ogg", codec="libopus", bitrate="48k")
    return buf.getvalue()


def qc_check(
    audio_bytes: bytes,
    script: str,
    client,
    model: str,
    passes: int = 3,
    temperature: float = 0.4,
    thinking_level: str = "MEDIUM",
) -> list[Issue]:
    """Run `passes` independent Gemini review passes over the compressed audio +
    script and return the deduped union of issues. Shared by the CLI and the
    pipeline's post-COMBINE QC-fix stage."""
    prompt = (
        "Here is the subtitle script the dub was built from "
        "(format: [index] start-end  Speaker: text):\n\n"
        f"{script}\n\n"
        "Now listen to the final dubbed audio and report any defects."
    )
    contents = [
        genai.types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"),
        prompt,
    ]
    gen_config = genai.types.GenerateContentConfig(
        system_instruction=QC_SYSTEM_PROMPT,
        temperature=temperature,
        thinking_config=genai.types.ThinkingConfig(thinking_level=thinking_level),
        response_mime_type="application/json",
        response_schema=QCReport,
    )

    def run_pass(p: int) -> list[Issue]:
        r = client.models.generate_content(
            model=model, contents=contents, config=gen_config)
        report = r.parsed
        found = report.issues if report else []
        print(f"pass {p + 1}/{passes}: {len(found)} issue(s)")
        return found

    all_issues: list[Issue] = []
    with ThreadPoolExecutor(max_workers=passes) as pool:
        for found in pool.map(run_pass, range(passes)):
            all_issues.extend(found)
    return dedup_issues(all_issues)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to output/<run_id>")
    ap.add_argument("--audio", default="audio/final_audio.wav")
    ap.add_argument("--subs", default=None,
                    help="Subs path relative to run. Default: retranslated if it "
                         "exists (what the final audio was built from), else translated.")
    ap.add_argument("--out", default=None, help="Where to write the JSON report")
    ap.add_argument("--model", default=None,
                    help="Override GEMINI_MODEL (e.g. gemini-3.6-pro)")
    ap.add_argument("--passes", type=int, default=3,
                    help="Number of independent review passes to union (recall)")
    ap.add_argument("--temperature", type=float, default=0.4,
                    help="Sampling temp; >0 so passes explore different defects")
    ap.add_argument("--thinking-level", default="MEDIUM",
                    choices=["MINIMAL", "LOW", "MEDIUM", "HIGH"],
                    help="Thinking budget; higher = deeper per-line attention")
    args = ap.parse_args()

    audio_path = os.path.join(args.run, args.audio)
    if args.subs:
        subs_path = os.path.join(args.run, args.subs)
    else:
        # Match the pipeline's combine step: retranslated is the real spoken text.
        retrans = os.path.join(args.run, "data", "subtitles_retranslated.srt")
        trans = os.path.join(args.run, "data", "subtitles_translated.srt")
        subs_path = retrans if os.path.exists(retrans) else trans
    print(f"subs: {os.path.basename(subs_path)}")

    script = build_script(subs_path)
    audio_bytes = compress_audio(audio_path)
    print(f"audio: {len(audio_bytes)/1e6:.2f} MB opus | script: {len(script)} chars")

    api_key = os.getenv("GEMINI_API_KEY")
    model = args.model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    print(f"model: {model}")
    client = genai.Client(api_key=api_key)

    merged = qc_check(
        audio_bytes=audio_bytes,
        script=script,
        client=client,
        model=model,
        passes=args.passes,
        temperature=args.temperature,
        thinking_level=args.thinking_level,
    )
    print(f"\n===== {len(merged)} ISSUE(S) (union of {args.passes} passes) =====")
    for it in merged:
        print(f"\n[{it.severity.value.upper()}] "
              f"{it.start}-{it.end}s  sub={it.sub_index}")
        print(f"  hear:     {it.symptom}")
        print(f"  mismatch: {it.mismatch}")

    out = args.out or os.path.join(args.run, "data", "qc_report.json")
    with open(out, "w", encoding="utf-8") as f:
        f.write(QCReport(issues=merged).model_dump_json(indent=2))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
