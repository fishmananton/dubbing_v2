# Song Detection & Drop — Design

**Date:** 2026-09-13
**Status:** Approved (pending spec review)

## Problem

When a film contains a song, its lyrics should NOT be translated or dubbed —
the original singing should play. But our STT transcribes the sung lyrics like
any dialogue, TRANSLATE translates them, and TTS voices them, so the song comes
out as spoken dubbed lines over the instrumental. The dub-quality QC
(`test_dub_qc.py`) cannot catch this: it only sees the dubbed audio + the script
it was built from, and by that point the lyrics *are* the script — content
matches, no defect. It has no access to the original audio.

Two cases:
1. **Pure musical number** — song plays, nobody talks over it. Every sub in the
   span is lyrics. Dropping them is correct.
2. **Song + simultaneous foreground dialogue** — decided RARE for our content;
   handled best-effort (minor original-speaker bleed acceptable). Not the
   optimization target.

## Desired Output

For a detected song span the viewer should hear the **original singing over the
instrumental** — the song "left untranslated." This requires NO new mixing work:
dropping a sub creates a subtitle-free span, and the existing
`split_vocal` → `original_speech_layer` path
(`main_prefect_dag.py:105`, `final_audio.py:205`) already restores the original
vocal for spans not covered by a subtitle (same mechanism used for laughs, etc.).
So "drop the song sub" automatically yields "original singing plays."

## Detection Approach

Send the **full original audio** (full mix, not vocal-vs-background stems) + the
indexed subtitle script to Gemini in one call, and ask which subtitle **IDs** are
sung lyrics rather than spoken dialogue. No energy pre-filter, no pitch analysis.

Rationale (both explored and rejected during brainstorming):
- **Energy `m/glob` block-detection** cleanly flags loud-music spans but cannot
  distinguish song lyrics from loud dialogue-over-music. Rejected as the decision
  maker.
- **Pitch/F0 analysis** (sustained notes, semitone-lock, vibrato) fails on rap —
  rap is rhythmic speech, pitch-indistinguishable from dialogue. Rejected.
- Only a model that "hears" content can reliably tell sung from spoken. Cost is
  acceptable: QA already sends full final audio to Gemini in 3 parallel passes,
  so one more full-audio call is routine.

### Return IDs, not timestamps

The model returns subtitle **IDs**, not time ranges. IDs are strictly better:
- The script already gives the model ground-truth timestamps per line
  (`[107] 502.30-509.10  Speaker: text`), so it classifies over a fixed labeled
  set rather than regressing free-form time (which drifts in long audio).
- An ID maps 1:1 to a line to delete — zero matching logic.
- A song span with no covering sub is a non-issue: nothing to translate/dub, and
  the no-sub→original-vocal fallback already covers it.

### ID stability (load-bearing assumption — verified)

Subtitle IDs are assigned once at transcribe assembly
(`transcribe_common.py:419`) and never reindexed downstream: TRANSLATE and all QC
fixes compose with `reindex=False` (`qc_fixes.py:141,155,170`;
`translate_duration.py:437`). So the IDs the model sees == the IDs we delete ==
the IDs TTS would have built from. Delete-by-ID is safe.

## Chunking (shared with QA script)

The current QC inlines audio as raw bytes (works only for short tracks — the
`compress_audio` docstring says "3-min track"). Full-length parts must be
chunked. This chunker is a **shared utility used both by the song-detection
module and the QA script** (QA needs it too, for the same size reason).

Algorithm:
- **≤12min → single chunk** (no split). The 12min floor avoids splitting a
  10:30 part into 10 + 0:30.
- **>12min → split targeting ~10min chunks.** For each ~10min target, choose the
  split at an **inter-sub gap > 2s nearest the target**, and cut at that gap's
  **midpoint**. Inter-sub gaps are already silence (nobody speaking), so no audio
  VAD/silence-scan is needed.
- **No qualifying gap near target → allow oversized chunk:** let the chunk run
  until the next >2s gap appears, even past ~10-12min. Prioritizes clean pause
  cuts over chunk size. Only if a chunk would exceed the inline **size ceiling
  (~20MB; opus @48k mono ≈ 0.36MB/min, so ~40min ≈ 14MB is still safe)** does it
  fall back to the largest available gap near the target rather than failing.

Splitting never cuts mid-sub (cuts land in gaps between subs), so every ID has
its full audio in exactly one chunk. A song straddling a chunk boundary is
harmless: each chunk hears its fragment (a fragment of singing still sounds
sung), flags its half, and results are **unioned across chunks** to recover all
sung IDs.

Per chunk: send only the subs whose timestamps fall in that chunk (adjust the
sent script per chunk) + that chunk's audio slice; collect returned IDs; union
across chunks after receiving.

### Per-chunk timestamp rebasing (correctness)

A chunk's audio slice starts at **t=0**, but subs carry **absolute** timestamps.
If we send 0-based audio with absolute times, the model's time anchors are wrong
for every chunk after the first (it hears a line at ~22s but the script labels it
502s), which defeats the ID-attribution the whole approach relies on. So when
building each chunk's script we **subtract the chunk's start offset** from each
sub's start/end so the displayed times match the 0-based audio slice.

- IDs are **not** changed — we still delete by original ID; only the *displayed*
  times in the per-chunk script are rebased.
- This is NOT the pipeline's timing correction (`fix_timing_subs.py`); it's a
  local offset subtraction while building the chunk script. The on-disk SRT is
  untouched.
- Dropping the times entirely (send `[id] text` only) is rejected: it discards
  the per-line anchor that makes ID classification robust, forcing purely
  acoustic localization.

## Placement & Wiring

Runs as a standard EMOTION-stage parallel task — **always on, no feature flag.**

After TRANSCRIBE, EMOTION submits `gemini_emotions`, `detect_gender`, and
`mouth_windows` non-blocking (`main_prefect_dag.py:689-695`), then TRANSLATE is
submitted (`:709`) and runs concurrently with them; all are awaited afterward
(`:718-731`). `detect_songs` slots into this exact fan-out:

```
TRANSCRIBE (:676)
  ├─ submit gemini_emotions / detect_gender / mouth_windows   (:689)
  ├─ submit detect_songs                                       (NEW, alongside)
  ├─ submit translate                                          (:709)
  └─ await emotions, mouth, translate                          (:718-731)
        songs = detect_songs_fut.result()          (NEW)
        drop_ids(translated_file, songs)           (NEW — delete sung IDs
                                                     from the TRANSLATED SRT)
  → GENERATE (:734)
```

Because `detect_songs` and TRANSLATE are already awaited at the same point, the
drop happens right after `translated_fut.result()` — we have both the sung-ID
list and the finished translated SRT there. Delete the IDs from the **translated
file** in place (`reindex=False`), so GENERATE/TTS never sees them.

**Wall-clock cost ≈ zero:** the call runs in the existing parallel window beside
emotions/gender/mouth/translate. As long as it finishes before GENERATE it adds
no serial time. TRANSLATE wastefully translates the few song lines that then get
dropped — a few cents of GPT text, done in parallel regardless; no TTS is spent.

## Components / Interfaces

- **`chunk_audio_by_subs(audio_path, subs, target_min=10, min_split_min=12, gap_min_s=2.0) -> list[Chunk]`**
  Shared utility (song module + QA). Each `Chunk` carries its audio slice (or
  slice bounds) and the subset of subs whose timestamps fall inside it. Pure /
  deterministic; unit-testable without a model.

- **`detect_songs(original_audio_path, subtitles_path, client, model, thinking_level="MEDIUM") -> list[int]`**
  Chunks, sends each chunk (0-based audio slice + its indexed script with
  timestamps **rebased to the chunk offset** via reused `build_script`,
  `test_dub_qc.py:120`; audio compressed via reused `compress_audio`,
  `test_dub_qc.py:131`), one Gemini call per chunk with a structured `list[int]`
  response schema, unions returned IDs. Returns sorted unique sung IDs (original
  IDs, never rebased).

- **Drop helper** — removes given IDs from an SRT file with `reindex=False`
  (reuse/mirror `_drop_line`, `qc_fixes.py:135`; may drop several IDs at once).

- **`t_detect_songs`** — Prefect task wrapping `detect_songs`, submitted at the
  EMOTION fan-out.

## Model Config

- **Thinking level: MEDIUM.** Sung-vs-spoken is acoustic + text-alignment, not
  deep multi-step reasoning; matches existing QC (`test_dub_qc.py:146`). Bump to
  HIGH only if boundary misses appear on rap/spoken-word. Higher thinking also
  raises latency in the parallel window, so MEDIUM keeps wall-clock at zero.
- **Part length: recommend 10min parts** (user pre-splits films by scene pauses).
  30min works (opus ≈ 11MB, under cap) but 10min tightens attribution
  (less timestamp drift, fewer IDs to pick among). This is a
  quality-vs-convenience choice, not a hard limit.

## Error Handling

Wrapped so a failure never kills the dub (same philosophy as the QC tail,
`main_prefect_dag.py:1092`): on any exception, log and proceed with no drops
(song lines get dubbed — the pre-existing behavior, i.e. safe degradation).

## Out of Scope

- Clean separation of overlapping song + foreground dialogue (decided rare;
  best-effort only).
- A-cappella songs with no instrumental that also evade the model (not observed;
  deferred).
- Energy/pitch pre-filters (explored, rejected above).
- Any change to the final mix (the existing no-sub→original-vocal fallback
  already produces the desired output).

## Testing

- **Chunker (deterministic):** unit tests — ≤12min single chunk; >12min splits at
  nearest >2s gap midpoint; no-gap → oversized chunk; song straddling a boundary
  appears in both chunks' sub subsets; size-ceiling fallback; per-chunk scripts
  have times rebased to the chunk offset while IDs stay original.
- **detect_songs (integration):** run on `20260903_taxi_CUT_02` — must return the
  rap block (subs 101–107) and NOT the loud-dialogue-over-music lines (24–25).
- **End-to-end:** confirm dropped song IDs are absent from the translated SRT and
  that the final audio plays original singing over the instrumental for those
  spans.
