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
import random
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Callable, Optional

import srt
from pydantic import BaseModel, Field
from pydub import AudioSegment
from dotenv import load_dotenv
from google import genai

from song_detect import chunk_subs, build_chunk_script, slice_audio_opus

load_dotenv()


QC_SYSTEM_PROMPT = """You are a professional dubbing quality-control reviewer.

You are given:
1. The FINAL dubbed audio track of a short video (the synthesized dub, one language).
2. The subtitle script it was generated from: each line has an index, a start/end
   timestamp, a speaker label, and the exact text that was supposed to be spoken.

### RULES FOR EVALUATION
Work through the audio line by line against the script. Do not stop at a simple word
match — evaluate delivery, pacing, and emotion. Judge the dub the way a demanding
reviewer would before it ships to a paying client.

The script contains ONLY real spoken dialogue.
- A line ending in "—" or "..." is an interruption or trail-off. It is NOT a stage
  direction. Never flag script text as if it were a direction read aloud.
- Songs are intentionally left in their original language. Do NOT flag original-language
  singing (any sung or melodic vocal passage, even if the script has a line for it).
  Only evaluate SPOKEN dialogue.

### DEFECT CATEGORIES TO REPORT (not exhaustive — report any other genuine defect too)
- Untranslated / original-language audio: the dub plays SPOKEN speech in the SOURCE
  language instead of the target — a whole line left undubbed, or original-language audio
  bleeding through beside or under a line. Listen for a SWITCH OF LANGUAGE, not just
  wrong words. This is a common, high-priority defect. (Singing is exempt — see above.)
- Missing / dropped line: a scripted line is not spoken at all — only silence or
  background where the target-language line should be.
- Content mismatch: wrong/missing/extra words, gibberish, or non-speech sounds (scream,
  laugh, noise) where words were expected. Includes STT hallucinations.
- Text normalization: numbers, times, dates, or acronyms spoken incorrectly.
- Pronunciation: names or terms mispronounced or given the wrong stress.
- Timing: audio runs past its subtitle window, is cut off short, or is unnaturally
  rushed/dragged.
- Emotion / delivery: tone contradicts the line's intent; robotic, flat, or monotone
  delivery where the line calls for feeling.
- Audio artifacts: clipping, clicks, truncated words, or unnatural silence.

### SEVERITY RUBRIC
- "high": completely ruins the line (missing words, original language spoken instead of
  the dub, severe glitch, totally wrong emotion).
- "medium": noticeable errors that distract the listener (slight mispronunciation,
  awkward timing).
- "low": minor nitpicks (slightly flat delivery, tiny artifact).

### OUTPUT
Be thorough and honest — do not fabricate defects, but do not withhold real ones. For
each issue return: "start"/"end" (seconds, as numbers), "sub_index" (integer, or null
if none applies), "symptom" (concretely what you HEAR), "mismatch" (why it fails vs the
script or good dubbing), and "severity". Example issue:
  {"start": 12.5, "end": 15.0, "sub_index": 3,
   "symptom": "the speaker says 'one two three' instead of 'one hundred twenty three'",
   "mismatch": "text normalization failure", "severity": "high"}
If the dub is perfect, return an empty issues array."""


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


# Gemini returns these when it's overloaded or we're rate-limited — transient, worth
# retrying. Client errors (400/404) are permanent and reraise immediately.
_TRANSIENT_CODES = {429, 500, 503}


def _is_transient_error(exc: Exception) -> bool:
    return getattr(exc, "code", None) in _TRANSIENT_CODES


def _retry_call(fn: Callable[[], object], max_attempts: int = 4,
                base_delay: float = 1.0,
                sleep: Callable[[float], None] = time.sleep,
                jitter: Callable[[], float] = random.random):
    """Call fn, retrying transient failures with exponential backoff + jitter. Non-
    transient errors reraise at once; transient ones reraise only after max_attempts."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — reraised below unless retryable
            if not _is_transient_error(e) or attempt == max_attempts - 1:
                raise
            sleep(base_delay * (2 ** attempt) + jitter())


def _qc_single_pass(
    audio_bytes: bytes,
    script: str,
    client,
    model: str,
    temperature: float = 0.4,
    thinking_level: str = "MEDIUM",
    max_retries: int = 4,
) -> list[Issue]:
    """One Gemini review pass over one audio slice + its script. Returned issue
    timestamps are LOCAL to the slice (the caller rebases when chunking)."""
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
    r = _retry_call(
        lambda: client.models.generate_content(
            model=model, contents=contents, config=gen_config),
        max_attempts=max_retries)
    report = r.parsed
    return report.issues if report else []


def _union_survivors(futures, total: int, label: str) -> list[Issue]:
    """Drain futures, union issues, tolerate partial failures. Only raise if EVERY
    task failed (so the caller's outer guard ships pass 1 rather than a partial review)."""
    all_issues: list[Issue] = []
    errors: list[Exception] = []
    for f in futures:
        try:
            all_issues.extend(f.result())
        except Exception as e:  # noqa: BLE001 — collect; reraise only if all fail
            errors.append(e)
    if errors and len(errors) == total:
        raise errors[0]
    if errors:
        print(f"⚠️  QC: {len(errors)}/{total} {label} failed; "
              f"using {total - len(errors)} survivor(s)")
    return dedup_issues(all_issues)


def qc_check(
    audio_bytes: bytes,
    script: str,
    client,
    model: str,
    passes: int = 3,
    temperature: float = 0.4,
    thinking_level: str = "MEDIUM",
    max_retries: int = 4,
) -> list[Issue]:
    """Run `passes` independent Gemini review passes over the whole compressed audio +
    script and return the deduped union of issues. Recall is a union across passes, so a
    single pass dying after retries doesn't sink the check — union the survivors."""
    def run_pass(p: int) -> list[Issue]:
        found = _qc_single_pass(audio_bytes, script, client, model, temperature,
                                thinking_level, max_retries)
        print(f"pass {p + 1}/{passes}: {len(found)} issue(s)")
        return found

    with ThreadPoolExecutor(max_workers=passes) as pool:
        futures = [pool.submit(run_pass, p) for p in range(passes)]
        return _union_survivors(futures, passes, "pass(es)")


def qc_check_chunked(
    seg,
    subs: list,
    client,
    model: str,
    passes: int = 3,
    temperature: float = 0.4,
    thinking_level: str = "MEDIUM",
    max_retries: int = 4,
    target_min: float = 10.0,
    min_split_min: float = 12.0,
    gap_min_s: float = 2.0,
) -> list[Issue]:
    """Chunk long audio (by inter-sub silence) and review it. ALL (chunk x pass) Gemini
    calls run concurrently; each chunk is sliced to its own 0-based audio + rebased
    script, so issues come back LOCAL and are rebased by the chunk offset before the
    union. `seg` is a preloaded mono AudioSegment. Partial failures are tolerated
    (survivor union); per-call transient errors are retried inside _qc_single_pass."""
    if not subs:
        return []
    chunks = chunk_subs(subs, target_min=target_min, min_split_min=min_split_min,
                        gap_min_s=gap_min_s)
    tasks = [(ci, chunk, p) for ci, chunk in enumerate(chunks) for p in range(passes)]

    def run(task):
        ci, chunk, p = task
        script = build_chunk_script(chunk)
        audio_bytes = slice_audio_opus(seg, chunk.offset_s, chunk.end_s)
        found = _qc_single_pass(audio_bytes, script, client, model, temperature,
                                thinking_level, max_retries)
        for it in found:  # rebase local chunk time -> absolute clip time
            it.start += chunk.offset_s
            it.end += chunk.offset_s
        print(f"chunk {ci + 1}/{len(chunks)} pass {p + 1}/{passes}: "
              f"{len(found)} issue(s)")
        return found

    with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as pool:
        futures = [pool.submit(run, t) for t in tasks]
        return _union_survivors(futures, len(tasks), "chunk-pass(es)")


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

    import srt as _srt
    from pydub import AudioSegment
    from song_detect import chunk_subs, build_chunk_script, slice_audio_opus

    subs = list(_srt.parse(open(subs_path, encoding="utf-8").read()))
    chunks = chunk_subs(subs)
    seg = AudioSegment.from_file(audio_path).set_channels(1)  # decode once
    print(f"audio: {len(chunks)} chunk(s)")

    api_key = os.getenv("GEMINI_API_KEY")
    model = args.model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    print(f"model: {model}")
    client = genai.Client(api_key=api_key)

    all_issues: list[Issue] = []
    for ci, chunk in enumerate(chunks):
        chunk_script = build_chunk_script(chunk)
        audio_bytes = slice_audio_opus(seg, chunk.offset_s, chunk.end_s)
        print(f"  chunk {ci+1}/{len(chunks)}: {len(audio_bytes)/1e6:.2f} MB opus")
        issues = qc_check(
            audio_bytes=audio_bytes,
            script=chunk_script,
            client=client,
            model=model,
            passes=args.passes,
            temperature=args.temperature,
            thinking_level=args.thinking_level,
        )
        # rebase issue timestamps back to absolute for the report
        for it in issues:
            it.start += chunk.offset_s
            it.end += chunk.offset_s
        all_issues.extend(issues)

    merged = dedup_issues(all_issues)
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
