# Behavior-Preserving Pipeline Speedups (QWEN3TTS + INDEXTTS2)

**Date:** 2026-09-16
**Status:** Approved — implementing
**Scope:** Full pipeline, both Modal TTS engines. **Behavior-preserving only** — every
change must produce byte/waveform-identical output. No persistent idle Modal spend.

## Motivation

A full COMBINE→QC-fix run (`20260922_rapuntsel_CUT_02/03`, Qwen3) took ~7m12s. The
critical-path map (`main_prefect_dag.py` GENERATE→COMBINE) surfaced CPU-side serial
waste that costs wall-clock without affecting output:

- `measure_loudness(video_file)` runs a full-length ffmpeg loudnorm scan of the **source
  video** on every `build_audio` (2–4×/run), though the source loudness is invariant.
- The final `tts_build_final(testing=False)` applies ffmpeg `atempo` **serially, one
  subprocess per segment** (~160 segments ≈ 27s).
- `run_line_loudness_stage` meters/apply gains **serially per line**.
- `cut_speakers` and the IndexTTS2 prewarm-wait run as back-to-back serial gates.

**Explicitly out of scope** (output-affecting; deferred): Qwen3 in-run pre-warm; replacing
ffmpeg `atempo` with an in-process stretch (rubberband/sox); skipping the remeasure build;
collapsing `build_audio`'s two loudnorm passes; reducing QC passes.

## Design — four levers + one shared helper

### Shared: `parallel_map(fn, items, max_workers=None)`
Bounded `ThreadPoolExecutor` fan-out that **preserves input order** in its results and
propagates exceptions. Default `max_workers = min(32, (os.cpu_count() or 4))`, caller may
lower. Used by #2 and #4. New small util module.

### #2 — Parallelize final-build `atempo` (biggest win, both engines)
`tts_build_final(testing=False)` calls `adjust_speed()` (ffmpeg `atempo`) per segment,
serially (`tts_v2.py:400`). Each call is independent (own WAV in/out); only final
assembly order matters. Extract the per-segment speed application into a helper and run it
through `parallel_map`, then assemble in the original index order exactly as today.
- **Behavior-preserving:** identical ffmpeg command per segment, identical assembly order.
- **Verify:** parallel result == serial result (waveform-identical) on real segments.
- **Saving:** ~20–25s per final build; ×2 on a QC-fix run.

### #3 — Cache `measure_loudness(video_file)`
`measure_loudness` (`final_audio.py:183`, invoked `dag:443`) scans the whole source video
every `build_audio`. Memoize the (integrated LUFS, LRA) result keyed by `(path, mtime)` in
a sidecar JSON under the run's data dir; subsequent calls read the cache.
- **Behavior-preserving:** same numbers fed to loudnorm → identical output.
- **Verify:** second call with unchanged (path, mtime) does not invoke ffmpeg; returns the
  cached values; a changed mtime recomputes.
- **Saving:** ~25s (clean) to ~50s (QC-fix).

### #4 — Parallelize `loudness_adjust` (lower priority)
`run_line_loudness_stage` (`loudness_adjust.py:172`) is serial per line: measure → smooth
→ apply. Parallelize the **measurement** loop and the **apply/write** loop via
`parallel_map`; keep any cross-line gain **smoothing** as a serial pass in between.
- **Precondition:** confirm gain smoothing is a distinct pass over measured values (not
  interleaved with measurement). Only parallelize the independent loops.
- **Behavior-preserving:** per-line measurement and per-line application are independent;
  smoothing stays serial.
- **Saving:** small (~1–3s).

### #5 — Overlap the pre-GENERATE serial gates — **DROPPED (no-op on inspection)**
Intended to overlap `cut_speakers` (`dag:842`) with the IndexTTS2 prewarm-wait
(`dag:845`). On inspection this saves nothing: the prewarm containers are `.spawn()`'d at
EMOTION (`dag:795`) and boot in the background on Modal regardless of when `h.get()` is
called; `cut_speakers.result()` already runs *before* that barrier (giving boot extra
time); and `cut_speakers` cannot move earlier because it consumes `emotions_tags`
produced by EMOTION. No real serial waste to reclaim — not implemented.

## Estimated total
~45–75s off a full run (more on QC-fix runs), zero extra idle spend, identical output.
Both engines gain #2/#3; #4 applies to both. #5 dropped as a no-op.

## Implemented
- `parallel.py`: `parallel_map` (order-preserving bounded thread pool).
- #2 `tts_v2.py`: `_nonoverflow_speed_factor` + `_prerender_nonoverflow`; final build
  pre-renders placement-independent atempo in parallel, sequential loop unchanged
  (byte-identical; overflow lines still render inline).
- #3 `final_audio.py`: `measure_loudness` memoized by (path, mtime, size).
- #4 `loudness_adjust.py`: per-line measure and apply loops via `parallel_map`; gain
  smoothing stays serial between them.
Tests: `test_parallel.py`, `test_tts_speed_prerender.py` (incl. byte-identity),
`test_measure_loudness_cache.py`, `test_loudness_parallel.py` (characterization).

## Post-instrumentation finding (the real bottleneck)
Phase timers on the final `tts_build_final` (CUT_05, 158 segs) showed the ~26s build was
**85% preload**: preload=22.3s, atempo_prerender(parallel #2)=2.1s [126 segs],
loop+assemble=1.6s, export=0.2s. So #2 worked but atempo was never the bottleneck.

Root cause: the preload re-ran `strip_silence` (pydub `detect_nonsilent`, 1ms `seek_step`,
GIL-bound) on every segment. But **every engine already silence-trims at generation**
(Modal `_trim_silence`; Inworld/Fish/ElevenLabs/Cartesia `strip_silence`) and
`loudness_adjust` only applies gain/fades — so the build re-strip was redundant. Measured:
15.8s to change **2/158** segments by ≤87ms.

Fix (two parts):
- Removed the redundant `strip_silence` at the preload (`tts_v2.py`) — already trimmed at
  generation. Byte-identical for 156/158; 2 segments keep ≤87ms more (within MIN_GAP/atempo
  noise). This alone: preload 22.3s → 15.4s.
- Found the remaining 15.4s was pydub decoding **float32** `_loudness_out.wav` via an
  ffmpeg subprocess (68ms/file). Switched `loudness_adjust` to write **int32 PCM**
  (`write_wav_int32`): pydub reads int32 natively (0.1ms/file, measured 658× faster) and
  int32 is bit-exact for the signal (−999 dBFS round-trip error) AND matches the int32 the
  pipeline already converts these files to on load — so quality is unchanged. (int16 was
  rejected: also fast but −94 dBFS lossy for no benefit.)

Combined: preload 22.3s → ~0s; final build ~26s → ~5s. The atempo-prerender instrumentation
(`⏱️ build_final phases:` print) is left in place to confirm on the next run.

## Verification strategy
TDD per lever. `parallel_map`, #2 (serial==parallel), #3 (memo behavior) are unit-tested.
#4 is unit-tested after confirming the smoothing structure. #5 is an orchestration
reorder, verified by inspection. Full existing suite must stay green.
