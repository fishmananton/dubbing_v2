# QC-Fix Agent — Design

Date: 2026-09-13
Status: Approved for planning

## Purpose

After the dubbing pipeline renders final audio, a Gemini QC pass (`test_dub_qc.py`) listens
to the whole track and reports defects (STT hallucinations on shouts, text-normalization
errors, wrong emotion, timing spill, artifacts). Today a human reads that report and
manually fixes lines. This project adds a **QC-Fix Agent**: a live, end-of-pipeline stage
that reads the QC report, diagnoses each issue against the pipeline's own artifacts,
**auto-applies reversible high-confidence fixes**, regenerates only the affected lines, and
re-QCs — while emitting everything it did (and everything it couldn't safely fix) as a
structured log for later human review.

The agent does **not** hardcode a fixed taxonomy of bugs. It reasons from evidence to a
small set of primitives, guided (not constrained) by a human-curated playbook, and defaults
to *proposing* rather than *applying* whenever confidence is low or a fix is risky.

## Scope

In scope:
- A new agent module (`qc_agent.py`) containing all LLM reasoning.
- A `qc_fix` targeted-regeneration mode inside `dubbing_flow` (`main_prefect_dag.py`).
- Extraction of the GENERATE→timing→COMBINE sequence into a reusable helper so it can run
  a second time inline.
- A post-COMBINE tail in `dubbing_flow` that wires QC → agent → regen → re-QC.
- Playbook (input), per-run fix log (output), and a cross-run promotion queue (output).

Out of scope:
- Changes to `api/app.py` (web backend — handled separately later).
- Changing how QC itself detects issues (QC prompt/behavior stays as-is).
- Any human-interactive step inside the pipeline run (the stage is non-interactive).
- Self-modifying playbook (agent never writes the playbook; see Learning below).

## Key Decisions (locked during brainstorming)

1. **Authority = auto-apply + propose.** Reversible, high-confidence fixes are applied
   automatically; everything else is emitted as a proposal (data, not a chat prompt).
2. **Live stage** running after COMBINE, inside `dubbing_flow`.
3. **One fix cycle per run:** batch all auto-fixes → one targeted regen → re-QC once →
   remainder become proposals. No fix→regen→QC loop.
4. **Gate is risk, not novelty.** `auto_apply = (primitive is reversible) AND
   (confidence ≥ per-primitive threshold)`. Novel (non-playbook) issues still get a
   free-reasoning fix attempt under the same gate. A playbook match raises confidence.
5. **Evidence = full deterministic artifact bundle by default; audio on request.** The
   model always receives the QC report + a rich text/stats bundle (see Evidence Bundle).
   It requests audio only when it decides it must, and it chooses the window. No forced
   clips, no stats-only auto-apply blind spot (audio is a *fallback*, most fixes need none).
6. **Single-call, structured per-issue output** (loop shape C). One LLM context sees all
   issues together (handles collisions and systemic patterns); output is one complete
   object per issue. Re-QC is the empirical check that per-issue depth did not suffer.
7. **QC_FIX regen operates on `subtitles_retranslated` as frozen truth**, via a pre-sync
   `copyfile(retranslated → translated)`; skips base-atempo recompute and reroll;
   clamp+spill for correct-but-longer fixes (never reroll).
8. **One flow run.** The QC-fix cycle runs inline as a post-COMBINE tail by calling an
   extracted `_regen_and_combine` helper a second time — not by restarting the flow.

## Architecture

### Components

- **`qc_agent.py`** (new) — all LLM reasoning. Pure module: given the QC report + artifact
  bundle, returns structured decisions. Knows nothing about Prefect. Testable in isolation.
- **`main_prefect_dag.py`** (modified) — orchestration only. Extract `_regen_and_combine`;
  add `qc_fix` branch in the timing block; add the post-COMBINE tail.
- **`config/qc_playbook.json`** (new, human-curated) — known pattern→primitive entries with
  per-entry confidence boost. Read-only to the agent.
- **`output/{run_id}/data/qc_fix_log.json`** (new, per run) — complete record of every
  issue's diagnosis, chosen primitive, params, confidence, playbook-matched flag,
  auto-applied vs. proposed, and re-QC outcome.
- **`config/qc_promotion_queue.json`** (new, cross-run, append-only) — novel fixes confirmed
  good by re-QC, queued for a human to review and possibly promote into the playbook.

### Primitives (the agent's entire action surface)

All primitives are reversible and their effect is verifiable by re-QC.

| Primitive | Params | Effect (deterministic apply) |
|-----------|--------|------------------------------|
| `edit_text` | `idx, new_text` | Rewrite the line's text (normalization, wording). |
| `drop_line` | `idx` | Remove invented text (STT hallucination on non-speech). |
| `set_emotion` | `idx, tag, vector, category` | Overwrite the line's emotion tag/vector. |
| `propose` | `idx, diagnosis, suggested_fix` | No auto-apply; emit for human review. |

Text edits are written to `subtitles_retranslated` (then pre-synced into `translated`
before regen — see QC_FIX Mode). Emotion edits overwrite `emotions_tags.json`; they survive
regen because QC_FIX enters below the EMOTION stage, which loads tags from file rather than
recomputing them.

### Playbook and Learning

- **Playbook** (`config/qc_playbook.json`): JSON list of entries, each roughly
  `{ pattern (human description of the issue signature), primitive, guidance, confidence_boost }`.
  The agent reads it as context. A match nudges confidence up toward the auto-apply threshold.
- **Human-curated only.** The agent never writes to the playbook — critical in a live
  pipeline, or it would silently widen its own auto-apply surface run over run.
- **Promotion queue:** when the agent solves a *novel* (non-playbook) issue and re-QC
  confirms it, it appends the pattern→fix it used to `config/qc_promotion_queue.json`. A
  human periodically reviews this queue and hand-promotes recurring wins into the playbook.

## Data Flow

```
dubbing_flow(...):
    stages 0–7 (unchanged)  ──▶  final_audio.wav        [render pass 1]
    ── post-COMBINE tail (same flow run) ──
    issues = qc_check(final_audio)
    if not issues:            # Q13: zero-issue path
        write empty qc_fix_log; return
    bundle   = build_evidence_bundle(issues, run artifacts)
    decisions = qc_agent.decide(issues, bundle)          # Phase 1 (+Phase 2 if audio)
    auto      = [d for d in decisions if d.auto_apply]
    proposals = [d for d in decisions if not d.auto_apply]
    if auto:
        auto    = collision_resolve(auto)                # dedupe by idx FIRST
        apply_fixes(auto)                                # edit/drop/set_emotion
        changed = [d.idx for d in auto]
        _regen_and_combine(config, changed, qc_fix=True) # render pass 2, inline
        reqc = qc_check(final_audio)
        diff reqc vs issues:
            was-flagged-now-clean → fixed
            still/newly-flagged   → regression → downgrade to proposal
        append novel+confirmed fixes → qc_promotion_queue.json
    write qc_fix_log.json (decisions + outcomes + proposals)
```

### The agent's two-phase call (runtime C)

- **Phase 1 (one call):** full text bundle + all issues + playbook. For each issue the model
  emits *either* a final decision *or* a declared `needs_audio: {idx, window}` field.
- **Phase 2 (one call, only if any issue asked for audio):** re-invoke with the requested
  clips attached (mono-opus slices of `vocal.wav` / tts segment / `final_audio.wav`, windowed
  by the model's requested span — which may cover unsubtitled gaps, e.g. a scream that spills
  past the SRT box). Model finalizes those issues.
- Max LLM calls per run: **2** (Phase 1 + optional Phase 2). Plus one re-QC pass.

### Evidence Bundle (always provided, all text/numeric)

Per flagged issue, deterministically gathered — the SRT box is never trusted as the sole
window; diarization and original-audio structure are authoritative:

- All three text layers: original (`subtitles.srt`), translated, retranslated.
- **Diarization spans** overlapping the region (`speakers_segments_data.json`) — who really
  spoke, when; immune to STT corruption.
- **Original-vocal energy/VAD stats** over the region and a padded window (`vocal.wav`).
- **Precomputed mismatch signal:** "continuous-vocal-activity span vs. transcribed-text
  length" — the labeled red flag for the hallucination class (long loud original ↔ few words).
- Emotion tag + vector for the line (`emotions_tags.json`).
- Timing/fit row (`build_final_stats.json`): fit ratio, spill, applied speed, status.
- Per-line atempo and any sentinel outlier (`natural_timing.json`).
- Visibility/mouth window (`subtitles_visibility.json`).
- TTS segment metadata (path/duration for `tts_segments/<speaker>/<idx>.wav`).

## QC_FIX Mode (targeted regeneration)

A single `qc_fix: bool` flag on `dubbing_flow` / the extracted helper. When set, the
GENERATE→timing→COMBINE sequence reuses existing helpers but composes them differently:

1. **Pre-sync (top of timing block):** `copyfile(subtitles_retranslated → subtitles_translated)`
   so the file GENERATE reads holds the frozen, actually-spoken truth (all prior rerolls +
   agent edits). This makes the existing `copyfile(translated → retranslated)` at the top of
   TIMING_FIX a harmless no-op that propagates the same content forward.
2. **GENERATE** runs normally, scoped to `changed_list` (regenerates only fixed lines).
3. **Timing:** do **not** call `compute_speaker_base_atempo` (would re-measure whole track and
   shift other lines). Load the saved `speaker_base_atempo.json`. Run `classify_lines` on the
   changed line(s) only; merge their `per_line_atempo` into the saved `natural_timing.json`.
4. **Skip retranslation entirely** (no `retranslate_timing_fix`). A correct-but-longer fix
   clamps to `max_speed_factor` and spills into the gap (`warn_too_long`) — never rerolled.
   (Consistent with the established "Pass-2 reroll degrades quality" preference.)
5. **COMBINE** runs normally (reads `subtitles_retranslated`, reuses `build_cache` so
   unchanged segments are not reprocessed).

## Auto-Apply Gate

`auto_apply = (primitive ∈ {edit_text, drop_line, set_emotion}) AND (confidence ≥ threshold[primitive])`

The model emits an explicit `confidence` (0–1) per issue. Per-primitive thresholds reflect
blast radius (all reversible, but not equal):

- `edit_text`  ≥ 0.70
- `set_emotion` ≥ 0.80
- `drop_line`  ≥ 0.90

A playbook match adds that entry's `confidence_boost`. Anything below threshold, or a
`propose` primitive, is emitted as a proposal (not applied).

## Collision Resolution (deterministic)

Before regen, dedupe auto-fix decisions by `idx`. If two decisions target the same line,
keep the higher-risk-tier action (`drop_line` > `set_emotion` > `edit_text`); if they
genuinely conflict (e.g., two different `edit_text` on one line), downgrade both to a
proposal rather than guess. This is done in code, not left to the model.

## Error Handling

- **Zero QC issues:** tail is a no-op; write an empty `qc_fix_log.json`; flow ends after
  render pass 1.
- **No auto-fixes (all proposals):** write log with proposals; **no regen, no re-QC.**
- **Agent/LLM failure (API error, unparseable output):** log the failure, emit all issues as
  proposals, skip regen. The pipeline still ships render pass 1 — the agent never blocks a
  successful dub.
- **Regen failure:** log it; keep render pass 1 as the shipped output; mark the attempted
  fixes as proposals.
- **Re-QC regression:** the changed line is re-flagged → downgraded to a proposal in the log
  (render pass 2 still ships — re-QC is advisory review data, matching "one cycle" and
  "clamp+spill" decisions; we do not roll back to pass 1).

## Testing

- **`qc_agent.py` unit tests:** feed canned QC reports + synthetic bundles; assert the chosen
  primitive, confidence, and auto/propose split. Cover: the scream-hallucination bundle
  (long activity ↔ short text → `drop_line`, high confidence), the `8:45` normalization
  (→ `edit_text`), a mislabeled emotion (→ `set_emotion`), and a novel/ambiguous case
  (→ `propose`). No live API needed — mock the model call, test the parsing/gate/collision
  logic deterministically.
- **QC_FIX mode test:** on a real run dir, invoke `_regen_and_combine(changed_list=[k],
  qc_fix=True)`; assert only line k's segment changed, `speaker_base_atempo.json` unchanged,
  no OpenAI reroll call fired, retranslated/translated stay consistent, COMBINE produced a
  new final audio.
- **Playbook/threshold config test:** confidence boost applied on match; per-primitive
  thresholds enforced.
- **End-to-end (manual, live API):** run the full tail on `output/20260903_barkoni_10` (has a
  known scream hallucination on the line at 48.8–51.1s and an `8:45` normalization error);
  verify the hallucinated line is dropped and the time is normalized, and re-QC comes back
  clean on those two.

## Open Items For Implementation Planning

- Exact JSON schemas for `qc_playbook.json`, `qc_fix_log.json`, and the agent's structured
  output (`response_schema`).
- Signature of `_regen_and_combine` and the minimal diff to extract it from the current
  inline flow body without disturbing pass-1 behavior.
- Where `qc_check` lives relative to `test_dub_qc.py` (reuse its compress/prompt code; the
  agent stage should call a shared function, not the CLI).
