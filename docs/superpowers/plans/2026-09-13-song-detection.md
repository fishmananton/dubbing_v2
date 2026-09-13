# Song Detection & Drop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect sung subtitle lines from the original audio via one Gemini call per chunk and drop them from the translated SRT before GENERATE, so songs play untranslated over the instrumental.

**Architecture:** A new `song_detect.py` module holds a deterministic subtitle-audio chunker (shared with the QA script) and a `detect_songs()` function that sends each chunk's 0-based audio + rebased indexed script to Gemini and unions the returned sung IDs. A Prefect task `t_detect_songs` runs in the existing EMOTION-stage parallel fan-out; its result drops IDs from the translated SRT right after TRANSLATE completes, before GENERATE. The existing no-sub→original-vocal mix fallback then plays the original singing for those spans — no mix changes.

**Tech Stack:** Python, Prefect, `srt`, `pydub` (AudioSegment), Google `genai` SDK, pytest.

**Spec:** `docs/superpowers/specs/2026-09-13-song-detection-design.md`

---

## File Structure

- **Create `song_detect.py`** — the chunker (`Chunk`, `chunk_subs`), per-chunk script builder with timestamp rebasing (`build_chunk_script`), audio slicing (`slice_audio_opus`), the Gemini call (`detect_songs_in_chunk`), the orchestrator (`detect_songs`), and the SRT drop helper (`drop_sub_ids`).
- **Create `tests/test_song_detect.py`** — unit tests for the deterministic pieces (chunker, rebasing, drop helper).
- **Modify `test_dub_qc.py`** — replace the single `compress_audio(full)` call in `main()` with the shared chunker so QA handles long audio too.
- **Modify `main_prefect_dag.py`** — add `t_detect_songs` task, submit it in the EMOTION fan-out, drop its result from the translated SRT after TRANSLATE.

**Reused as-is (do not modify):** `build_script` and `compress_audio` in `test_dub_qc.py`; `config.audio_file`, `config.subtitles`, `config.subtitles_translated_file`, `config.gemini_api_key`, `config.gemini_model` in `config.py`.

---

## Task 1: Chunker — the `Chunk` dataclass and `chunk_subs`

Splits a list of subs into chunks: ≤12min → one chunk; else target ~10min, split at the inter-sub gap >2s nearest each target, at the gap midpoint; if no qualifying gap, let the chunk run to the next >2s gap (oversized allowed).

**Files:**
- Create: `song_detect.py`
- Test: `tests/test_song_detect.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_song_detect.py
from datetime import timedelta

import srt

from song_detect import Chunk, chunk_subs


def _sub(idx, start_s, end_s, text="x"):
    return srt.Subtitle(index=idx,
                        start=timedelta(seconds=start_s),
                        end=timedelta(seconds=end_s),
                        content=f"Speaker: {text}")


def test_short_input_is_single_chunk():
    # total span 8 min < 12 min floor -> one chunk holding all subs
    subs = [_sub(1, 0, 2), _sub(2, 100, 102), _sub(3, 470, 480)]
    chunks = chunk_subs(subs, target_min=10, min_split_min=12, gap_min_s=2.0)
    assert len(chunks) == 1
    assert chunks[0].offset_s == 0.0
    assert [s.index for s in chunks[0].subs] == [1, 2, 3]


def test_splits_long_input_at_nearest_gap_over_threshold():
    # subs every ~60s out to ~20min; a big 5s gap sits right after the 10-min mark.
    subs = []
    t = 0.0
    idx = 1
    while t < 1200:
        subs.append(_sub(idx, t, t + 1.0))
        # inject a 5s gap right after 600s (10 min), else 1s gaps
        gap = 5.0 if 600 <= t < 601 else 1.0
        t += 1.0 + gap
        idx += 1
    chunks = chunk_subs(subs, target_min=10, min_split_min=12, gap_min_s=2.0)
    assert len(chunks) == 2
    # split lands in the 5s gap after the sub ending ~601s; second chunk rebased
    assert chunks[1].offset_s > 600
    # every sub appears exactly once across chunks, order preserved
    all_idx = [s.index for c in chunks for s in c.subs]
    assert all_idx == [s.index for s in subs]


def test_no_qualifying_gap_allows_oversized_chunk():
    # continuous 1s-gap talk for 25 min, one 3s gap only at ~18min.
    subs = []
    t = 0.0
    idx = 1
    while t < 1500:
        subs.append(_sub(idx, t, t + 1.0))
        gap = 3.0 if 1080 <= t < 1081 else 1.0
        t += 1.0 + gap
        idx += 1
    chunks = chunk_subs(subs, target_min=10, min_split_min=12, gap_min_s=2.0)
    # first chunk runs past 12min to the only >2s gap near/after target (~18min)
    assert len(chunks) == 2
    assert chunks[0].subs[-1].end.total_seconds() <= 1081
    assert chunks[1].offset_s > 1080
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_song_detect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'song_detect'` (or ImportError for `Chunk`/`chunk_subs`).

- [ ] **Step 3: Write minimal implementation**

```python
# song_detect.py
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
    # split-after-left-index -> midpoint, for gaps that qualify
    qualifying = {i: mid for (g, mid, i) in gaps if g > gap_min_s}

    for i, s in enumerate(subs):
        cur.append(s)
        # once we're at/past the target mark, split at the first qualifying gap
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_song_detect.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add song_detect.py tests/test_song_detect.py
git commit -m "feat(song): deterministic subtitle-audio chunker"
```

---

## Task 2: Per-chunk script with rebased timestamps

Build the indexed script for one chunk with timestamps shifted to the chunk's 0-based audio slice, while keeping original IDs. Reuses `build_script`'s format by constructing a temporary re-timed sub list.

**Files:**
- Modify: `song_detect.py`
- Test: `tests/test_song_detect.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_song_detect.py
from song_detect import build_chunk_script


def test_build_chunk_script_rebases_times_keeps_ids():
    subs = [_sub(41, 605, 607, "hello"), _sub(42, 610, 612, "world")]
    chunk = Chunk(subs=subs, offset_s=600.0, end_s=612.0)
    script = build_chunk_script(chunk)
    # original IDs preserved
    assert "[41]" in script and "[42]" in script
    # times rebased by -600s: 605->5.00, 610->10.00
    assert "5.00-7.00" in script
    assert "10.00-12.00" in script
    # no absolute 605/610 leaked
    assert "605" not in script and "610" not in script
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_song_detect.py::test_build_chunk_script_rebases_times_keeps_ids -v`
Expected: FAIL with `ImportError: cannot import name 'build_chunk_script'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to song_detect.py
from datetime import timedelta


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_song_detect.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add song_detect.py tests/test_song_detect.py
git commit -m "feat(song): per-chunk script with rebased timestamps"
```

---

## Task 3: Drop helper — remove sung IDs from an SRT

Removes given IDs from an SRT file in place with `reindex=False` so surviving IDs stay stable (mirrors `_drop_line` in `qc_fixes.py:135`).

**Files:**
- Modify: `song_detect.py`
- Test: `tests/test_song_detect.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_song_detect.py
from song_detect import drop_sub_ids


def test_drop_sub_ids_removes_and_keeps_original_indices(tmp_path):
    subs = [_sub(1, 0, 1, "a"), _sub(2, 2, 3, "b"),
            _sub(3, 4, 5, "c"), _sub(4, 6, 7, "d")]
    p = tmp_path / "translated.srt"
    p.write_text(srt.compose(subs, reindex=False), encoding="utf-8")

    removed = drop_sub_ids(str(p), [2, 3])
    assert removed == [2, 3]

    kept = list(srt.parse(p.read_text(encoding="utf-8")))
    assert [s.index for s in kept] == [1, 4]  # originals preserved, not renumbered


def test_drop_sub_ids_empty_is_noop(tmp_path):
    subs = [_sub(1, 0, 1, "a"), _sub(2, 2, 3, "b")]
    p = tmp_path / "translated.srt"
    original = srt.compose(subs, reindex=False)
    p.write_text(original, encoding="utf-8")
    assert drop_sub_ids(str(p), []) == []
    assert list(s.index for s in srt.parse(p.read_text(encoding="utf-8"))) == [1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_song_detect.py -k drop_sub_ids -v`
Expected: FAIL with `ImportError: cannot import name 'drop_sub_ids'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to song_detect.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_song_detect.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add song_detect.py tests/test_song_detect.py
git commit -m "feat(song): drop-by-id SRT helper (reindex=False)"
```

---

## Task 4: Audio slicing and the per-chunk Gemini call

`slice_audio_opus` cuts a chunk's audio window to 0-based mono opus bytes; `detect_songs_in_chunk` sends those bytes + the rebased script and returns the sung IDs the model reports.

**Files:**
- Modify: `song_detect.py`
- Test: none (network + audio I/O — covered by the Task 6 integration probe, not a unit test)

- [ ] **Step 1: Add the slicing + call implementation**

```python
# add to song_detect.py
from google import genai

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
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `python -c "import song_detect; print('ok', song_detect.slice_audio_opus, song_detect.detect_songs_in_chunk)"`
Expected: prints `ok <function ...> <function ...>` with no ImportError.

- [ ] **Step 3: Commit**

```bash
git add song_detect.py
git commit -m "feat(song): audio slicing + per-chunk Gemini sung-id call"
```

---

## Task 5: Orchestrator `detect_songs`

Chunks the subs, runs each chunk's call in parallel, unions the sung IDs. Returns sorted unique original IDs.

**Files:**
- Modify: `song_detect.py`
- Test: `tests/test_song_detect.py` (union logic via a monkeypatched per-chunk call — no network)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_song_detect.py
import song_detect as sd


def test_detect_songs_unions_ids_across_chunks(tmp_path, monkeypatch):
    # 20 min of subs -> forces >1 chunk; stub the per-chunk call.
    subs = []
    t = 0.0
    idx = 1
    while t < 1200:
        subs.append(_sub(idx, t, t + 1.0))
        gap = 5.0 if 600 <= t < 601 else 1.0
        t += 1.0 + gap
        idx += 1
    p = tmp_path / "subs.srt"
    p.write_text(srt.compose(subs, reindex=False), encoding="utf-8")

    calls = {"n": 0}

    def fake_chunk_call(chunk, audio_path, client, model, thinking_level="MEDIUM"):
        calls["n"] += 1
        # each chunk "finds" its first sub's id as sung
        return [chunk.subs[0].index]

    monkeypatch.setattr(sd, "detect_songs_in_chunk", fake_chunk_call)

    ids = sd.detect_songs(audio_path="/nonexistent.wav", subtitles_path=str(p),
                          client=object(), model="m")
    assert calls["n"] >= 2               # multiple chunks were processed
    assert ids == sorted(set(ids))       # sorted unique
    assert len(ids) == calls["n"]        # one id unioned per chunk


def test_detect_songs_empty_subs_returns_empty(tmp_path):
    p = tmp_path / "subs.srt"
    p.write_text("", encoding="utf-8")
    assert sd.detect_songs(audio_path="/x.wav", subtitles_path=str(p),
                           client=object(), model="m") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_song_detect.py -k detect_songs -v`
Expected: FAIL with `AttributeError`/`ImportError` — `detect_songs` not defined.

- [ ] **Step 3: Write minimal implementation**

```python
# add to song_detect.py
from concurrent.futures import ThreadPoolExecutor


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_song_detect.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add song_detect.py tests/test_song_detect.py
git commit -m "feat(song): detect_songs orchestrator with per-chunk union"
```

---

## Task 6: Integration probe against known ground truth

A guarded script-style test that hits the real API only when `GEMINI_API_KEY` is set; otherwise it skips. Validates the known answer on `20260903_taxi_CUT_02` (find 101-107, exclude 24-25).

**Files:**
- Modify: `tests/test_song_detect.py`

- [ ] **Step 1: Add the guarded integration test**

```python
# append to tests/test_song_detect.py
import os
import pytest


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"),
                    reason="needs GEMINI_API_KEY + network")
def test_detect_songs_taxi_ground_truth():
    from google import genai
    run = "output/20260903_taxi_CUT_02"
    audio = os.path.join(run, "audio/audio.wav")
    subs = os.path.join(run, "data/subtitles.srt")
    if not (os.path.exists(audio) and os.path.exists(subs)):
        pytest.skip("taxi run not present")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    ids = set(sd.detect_songs(audio, subs, client, model))
    assert {101, 102, 103, 104, 105, 106, 107} <= ids, f"missed rap: {sorted(ids)}"
    assert not ({24, 25} & ids), f"false positive on dialogue-over-music: {sorted(ids)}"
```

- [ ] **Step 2: Run the unit suite (integration auto-skips without a key in sandbox)**

Run: `python -m pytest tests/test_song_detect.py -v`
Expected: 8 passed, 1 skipped (integration skipped — no key/network in sandbox).

- [ ] **Step 3: Run the integration test in a real shell (user action)**

The sandbox blocks the Gemini host, so this must run in an unproxied terminal with the key set:
```bash
export GEMINI_API_KEY=...   # or: set -a; source .env; set +a
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy GRPC_PROXY grpc_proxy
python -m pytest tests/test_song_detect.py::test_detect_songs_taxi_ground_truth -v
```
Expected: PASS (finds 101-107, excludes 24-25).

- [ ] **Step 4: Commit**

```bash
git add tests/test_song_detect.py
git commit -m "test(song): guarded integration probe on taxi ground truth"
```

---

## Task 7: Wire `t_detect_songs` into the pipeline

Add the Prefect task and submit it in the EMOTION fan-out; drop its result from the translated SRT after TRANSLATE completes, before GENERATE. Always on, wrapped so failure never kills the dub.

**Files:**
- Modify: `main_prefect_dag.py` (import near the other `from ... import` lines ~`:47`; new task near the other `@task` defs; submit at the EMOTION fan-out `:689`; drop after `translated_fut.result()` `:731`)
- Test: none (Prefect flow wiring — validated by the integration probe + a real run)

- [ ] **Step 1: Add the import**

Near the QC imports (`main_prefect_dag.py:47`), add:
```python
from song_detect import detect_songs, drop_sub_ids
```

- [ ] **Step 2: Add the Prefect task**

Add alongside the other `@task` definitions (e.g. near `t_gemini_extract_emotions`). Match the file's existing `@task`/`timer` style:
```python
@task
def t_detect_songs(config, audio_file, subtitles_file):
    """Return subtitle IDs that are sung lyrics (to be dropped before GENERATE).
    Runs in the EMOTION-stage parallel fan-out. Never raises: on error returns []
    so the dub proceeds with songs dubbed (pre-existing behavior)."""
    with timer("Detect songs"):
        try:
            from google import genai
            client = genai.Client(api_key=config.gemini_api_key)
            ids = detect_songs(audio_file, subtitles_file, client,
                               config.gemini_model)
            print(f"🎵 song detection: {len(ids)} sung line(s) -> drop {ids}")
            return ids
        except Exception as e:  # noqa: BLE001 — must never fail the dub
            print(f"⚠️  song detection failed ({e}); keeping all lines")
            return []
```

- [ ] **Step 3: Submit in the EMOTION fan-out**

In the `if stage <= STAGES.EMOTION:` block (`:689`), alongside `gemini_emotions_fut`/`detect_gender_fut`/`mouth_windows_fut`, add:
```python
        detect_songs_fut = t_detect_songs.submit(config, config.audio_file,
                                                  config.subtitles)
```
And initialize it to `None` with the other futures above the block (near `:683`):
```python
    detect_songs_fut = None
```

- [ ] **Step 4: Drop the IDs after TRANSLATE, before GENERATE**

Immediately after `translate_stats = translated_fut.result()` and `translated_file = config.subtitles_translated_file` (`:731-732`), add:
```python
    if detect_songs_fut is not None:
        song_ids = detect_songs_fut.result()
        if song_ids:
            dropped = drop_sub_ids(config.subtitles_translated_file, song_ids)
            print(f"🎵 dropped {len(dropped)} sung line(s) from translated SRT: {dropped}")
```

- [ ] **Step 5: Verify the flow imports and the task is registered**

Run: `python -c "import main_prefect_dag as m; print('ok', m.t_detect_songs.name)"`
Expected: prints `ok t_detect_songs` (or the Prefect-assigned task name) with no ImportError.

- [ ] **Step 6: Commit**

```bash
git add main_prefect_dag.py
git commit -m "feat(song): wire t_detect_songs into EMOTION fan-out + drop before GENERATE"
```

---

## Task 8: Share the chunker with the QA script

Replace the single full-audio `compress_audio` call in `test_dub_qc.py`'s `main()` with the shared chunker so QA handles long audio, running one QC pass-set per chunk and unioning issues via the existing `dedup_issues`.

**Files:**
- Modify: `test_dub_qc.py` (`main()`, `:213-230`)
- Test: none (CLI script; exercised manually)

- [ ] **Step 1: Replace the single-call block with chunked calls**

In `test_dub_qc.py`'s `main()`, replace the block that builds one `script`/`audio_bytes` and calls `qc_check` once (`:213-230`) with:
```python
    import srt as _srt
    from song_detect import chunk_subs, build_chunk_script, slice_audio_opus

    subs = list(_srt.parse(open(subs_path, encoding="utf-8").read()))
    chunks = chunk_subs(subs)
    print(f"audio: {len(chunks)} chunk(s)")

    api_key = os.getenv("GEMINI_API_KEY")
    model = args.model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    print(f"model: {model}")
    client = genai.Client(api_key=api_key)

    all_issues: list[Issue] = []
    for ci, chunk in enumerate(chunks):
        chunk_script = build_chunk_script(chunk)
        audio_bytes = slice_audio_opus(audio_path, chunk.offset_s, chunk.end_s)
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
```

- [ ] **Step 2: Verify the QA script still imports and parses args**

Run: `python test_dub_qc.py --help`
Expected: prints the argparse usage with no ImportError.

- [ ] **Step 3: Run the unit suite to confirm nothing regressed**

Run: `python -m pytest tests/test_song_detect.py -v`
Expected: 8 passed, 1 skipped.

- [ ] **Step 4: Commit**

```bash
git add test_dub_qc.py
git commit -m "refactor(qc): QA script uses shared chunker for long audio"
```

---

## Self-Review Notes

- **Spec coverage:** detection via full original audio + IDs (Tasks 4–5), chunker with 12min floor / 10min target / >2s gap midpoint / oversized fallback (Task 1), per-chunk timestamp rebasing (Task 2), delete-by-ID with `reindex=False` (Task 3), EMOTION-stage parallel placement + drop-after-TRANSLATE, always-on, error-safe (Task 7), shared chunker with QA (Task 8), MEDIUM thinking (Task 4 default), integration probe on taxi ground truth (Task 6). All spec sections map to a task.
- **Type consistency:** `Chunk(subs, offset_s, end_s)` used identically across Tasks 1–8; `detect_songs_in_chunk`/`detect_songs`/`drop_sub_ids`/`build_chunk_script`/`slice_audio_opus`/`chunk_subs` signatures match every call site.
- **No placeholders:** every code step shows full code; every run step shows the command + expected output.
