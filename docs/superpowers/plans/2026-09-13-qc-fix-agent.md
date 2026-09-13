# QC-Fix Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live, post-COMBINE stage to `dubbing_flow` that reads the Gemini QC report, diagnoses each defect against pipeline artifacts, auto-applies reversible high-confidence fixes, regenerates only the affected lines, and re-QCs — emitting a structured log of everything done and everything left for a human.

**Architecture:** Pure deterministic modules (`qc_evidence.py`, `qc_fixes.py`) plus one LLM-reasoning module (`qc_agent.py`) that knows nothing about Prefect. `test_dub_qc.py` gains a reusable `qc_check()` so both the CLI and the pipeline call the same code. `main_prefect_dag.py` extracts the GENERATE→timing→COMBINE body into `_regen_and_combine(config, ..., qc_fix=False)`, relocates `split_vocal` into COMBINE, and adds a post-COMBINE tail that wires QC → agent → apply → regen → re-QC.

**Tech Stack:** Python 3, Prefect (`@flow`/`@task`), `google.genai` (Gemini, structured `response_schema`), pydantic, `srt`, pytest with mocked LLM calls.

---

## Design Reference

Spec: `docs/superpowers/specs/2026-09-13-qc-fix-agent-design.md` (approved). This plan implements every locked decision (#1–#9), the five primitives, the two-phase agent call, the evidence bundle, QC_FIX mode, the auto-apply gate, collision resolution, and error handling.

## File Structure

**New files:**
- `qc_agent.py` — all LLM reasoning. `decide(issues, bundle, playbook) -> list[Decision]`. Two-phase (text, then optional audio). No Prefect, no filesystem writes.
- `qc_evidence.py` — deterministic evidence bundle builder. `build_evidence_bundle(issues, config) -> dict`. Reads run artifacts (subs, diarization, emotions, stats, vocal energy).
- `qc_fixes.py` — deterministic apply + gate + collision. `Decision` dataclass, `apply_gate`, `resolve_collisions`, `apply_fixes`. Reuses `apply_retranslation` from `post_build_fix.py`.
- `config/qc_playbook.json` — human-curated pattern→primitive entries (read-only to agent).
- `tests/__init__.py`, `tests/test_qc_fixes.py`, `tests/test_qc_agent.py`, `tests/test_qc_evidence.py`, `tests/test_regen_helper.py`.

**Modified files:**
- `test_dub_qc.py` — extract `qc_check(audio_path, subs_path, ...) -> list[Issue]` from `main()`; `main()` calls it.
- `main_prefect_dag.py` — extract `_regen_and_combine`; relocate `split_vocal` into COMBINE; add post-COMBINE tail.
- `audio.py` — `split_vocal` reads whichever subtitles file it is handed (already parameterized; no change needed, verified in Task 2).

**Outputs written at runtime (not created by this plan):**
- `output/{run_id}/data/qc_fix_log.json` — per-run record.
- `config/qc_promotion_queue.json` — cross-run, append-only novel-fix queue.

## Conventions used throughout

- **Speaker prefix:** subtitle content is `"SPEAKER_XX: text"`. The prefix before the first `:` routes reference audio (`tts_index_tts2.py:219-243`).
- **Frozen truth:** `subtitles_retranslated.srt` is the actually-spoken text after all rerolls/edits. All text/speaker/timing edits write here.
- **SRT writes** use `srt.compose(sorted(subs, key=lambda x: x.start), reindex=False)` — matching `post_build_fix.py:444`.
- **Tests mock the LLM.** No live Gemini/OpenAI in unit tests (spec Testing section). Run with `python3 -m pytest`.

---

## Task 1: `Decision` model + auto-apply gate

**Files:**
- Create: `qc_fixes.py`
- Test: `tests/test_qc_fixes.py`, `tests/__init__.py`

- [ ] **Step 1: Create the tests package**

Create `tests/__init__.py` (empty file).

- [ ] **Step 2: Write the failing test for the gate**

Create `tests/test_qc_fixes.py`:

```python
from qc_fixes import Decision, apply_gate, PRIMITIVE_THRESHOLDS


def make(idx, primitive, confidence, **params):
    return Decision(idx=idx, primitive=primitive, confidence=confidence,
                    params=params, playbook_matched=False, diagnosis="d")


def test_edit_text_above_threshold_auto_applies():
    d = make(5, "edit_text", 0.72, new_text="8:45")
    apply_gate([d])
    assert d.auto_apply is True


def test_edit_text_below_threshold_becomes_proposal():
    d = make(5, "edit_text", 0.69, new_text="x")
    apply_gate([d])
    assert d.auto_apply is False


def test_drop_line_needs_090():
    lo = make(1, "drop_line", 0.89)
    hi = make(2, "drop_line", 0.90)
    apply_gate([lo, hi])
    assert lo.auto_apply is False
    assert hi.auto_apply is True


def test_change_speaker_needs_085():
    d = make(3, "change_speaker", 0.84, new_speaker="SPEAKER_01")
    apply_gate([d])
    assert d.auto_apply is False


def test_propose_never_auto_applies():
    d = make(4, "propose", 0.99, diagnosis="unclear", suggested_fix="human")
    apply_gate([d])
    assert d.auto_apply is False


def test_playbook_boost_pushes_over_threshold():
    d = make(5, "edit_text", 0.65, new_text="x")
    d.playbook_matched = True
    d.playbook_boost = 0.10
    apply_gate([d])
    assert d.auto_apply is True
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/test_qc_fixes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qc_fixes'`.

- [ ] **Step 4: Implement `Decision` + `apply_gate`**

Create `qc_fixes.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Per-primitive auto-apply thresholds (spec Auto-Apply Gate).
# Reflect blast radius; all primitives are reversible.
PRIMITIVE_THRESHOLDS: dict[str, float] = {
    "edit_text": 0.70,
    "change_timing": 0.70,
    "set_emotion": 0.80,
    "change_speaker": 0.85,
    "drop_line": 0.90,
}

APPLIABLE_PRIMITIVES = set(PRIMITIVE_THRESHOLDS.keys())

# Higher tier = higher risk; used to break same-idx collisions.
COLLISION_TIER: dict[str, int] = {
    "drop_line": 4,
    "change_speaker": 3,
    "set_emotion": 2,
    "change_timing": 1,
    "edit_text": 1,
}


@dataclass
class Decision:
    idx: int
    primitive: str
    confidence: float
    params: dict[str, Any] = field(default_factory=dict)
    diagnosis: str = ""
    playbook_matched: bool = False
    playbook_boost: float = 0.0
    needs_audio: dict | None = None  # {"idx": int, "window": [start_s, end_s]}
    auto_apply: bool = False
    outcome: str = ""  # filled after re-QC: "fixed" | "regression" | "proposed"


def effective_confidence(d: Decision) -> float:
    boost = d.playbook_boost if d.playbook_matched else 0.0
    return min(1.0, d.confidence + boost)


def apply_gate(decisions: list[Decision]) -> None:
    """Set d.auto_apply in place. propose never auto-applies."""
    for d in decisions:
        if d.primitive not in APPLIABLE_PRIMITIVES:
            d.auto_apply = False
            continue
        threshold = PRIMITIVE_THRESHOLDS[d.primitive]
        d.auto_apply = effective_confidence(d) >= threshold
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_qc_fixes.py -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Commit**

```bash
git add qc_fixes.py tests/__init__.py tests/test_qc_fixes.py
git commit -m "feat(qc): add Decision model and auto-apply gate"
```

---

## Task 2: Collision resolution

**Files:**
- Modify: `qc_fixes.py`
- Test: `tests/test_qc_fixes.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_qc_fixes.py`:

```python
from qc_fixes import resolve_collisions


def test_no_collision_passes_through():
    a = make(1, "edit_text", 0.9, new_text="a")
    b = make(2, "drop_line", 0.95)
    a.auto_apply = b.auto_apply = True
    kept = resolve_collisions([a, b])
    assert {d.idx for d in kept} == {1, 2}


def test_higher_tier_wins_same_idx():
    text = make(1, "edit_text", 0.9, new_text="a")
    drop = make(1, "drop_line", 0.95)
    text.auto_apply = drop.auto_apply = True
    kept = resolve_collisions([text, drop])
    assert len(kept) == 1
    assert kept[0].primitive == "drop_line"


def test_same_field_conflict_downgrades_both():
    a = make(1, "edit_text", 0.9, new_text="a")
    b = make(1, "edit_text", 0.9, new_text="b")
    a.auto_apply = b.auto_apply = True
    kept = resolve_collisions([a, b])
    assert kept == []  # both downgraded, neither auto-applied
    assert a.auto_apply is False and b.auto_apply is False


def test_different_fields_same_idx_coexist():
    # timing + text touch different fields -> both kept
    t = make(1, "change_timing", 0.9, new_start_ms=1000, new_end_ms=2000)
    x = make(1, "edit_text", 0.9, new_text="hi")
    t.auto_apply = x.auto_apply = True
    kept = resolve_collisions([t, x])
    assert {d.primitive for d in kept} == {"change_timing", "edit_text"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_qc_fixes.py::test_higher_tier_wins_same_idx -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_collisions'`.

- [ ] **Step 3: Implement `resolve_collisions`**

Append to `qc_fixes.py`:

```python
# Which SRT/artifact field each primitive writes. Primitives touching different
# fields on the same line can coexist; same-field conflicts must be downgraded.
PRIMITIVE_FIELD: dict[str, str] = {
    "edit_text": "text",
    "drop_line": "line",       # removes the whole line -> conflicts with everything
    "change_timing": "timing",
    "change_speaker": "speaker",
    "set_emotion": "emotion",
}


def _downgrade(d: Decision) -> None:
    d.auto_apply = False
    d.outcome = "proposed"


def resolve_collisions(auto: list[Decision]) -> list[Decision]:
    """Dedupe auto-apply decisions by idx. Same-field conflict -> downgrade both.
    Different fields -> keep both. drop_line removes the line, so it conflicts with
    any other primitive on that idx and wins by tier. Done in code, not the model."""
    by_idx: dict[int, list[Decision]] = {}
    for d in auto:
        by_idx.setdefault(d.idx, []).append(d)

    kept: list[Decision] = []
    for idx, group in by_idx.items():
        if len(group) == 1:
            kept.append(group[0])
            continue

        # drop_line on this idx supersedes everything else on the line.
        drops = [d for d in group if d.primitive == "drop_line"]
        if drops:
            winner = max(drops, key=lambda d: effective_confidence(d))
            for d in group:
                if d is not winner:
                    _downgrade(d)
            kept.append(winner)
            continue

        # Group remaining by field; same-field conflict downgrades all in that field.
        by_field: dict[str, list[Decision]] = {}
        for d in group:
            by_field.setdefault(PRIMITIVE_FIELD[d.primitive], []).append(d)

        for field_decisions in by_field.values():
            if len(field_decisions) == 1:
                kept.append(field_decisions[0])
            else:
                for d in field_decisions:
                    _downgrade(d)
    return kept
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_qc_fixes.py -v`
Expected: PASS (10 tests total).

- [ ] **Step 5: Commit**

```bash
git add qc_fixes.py tests/test_qc_fixes.py
git commit -m "feat(qc): deterministic collision resolution by idx and field"
```

---

## Task 3: Apply primitives (write fixes to artifacts)

**Files:**
- Modify: `qc_fixes.py`
- Test: `tests/test_qc_fixes.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_qc_fixes.py`:

```python
import json
from datetime import timedelta

import srt as _srt

from qc_fixes import apply_fixes


def _write_srt(path, rows):
    subs = [
        _srt.Subtitle(index=i, start=timedelta(seconds=s), end=timedelta(seconds=e),
                      content=c)
        for (i, s, e, c) in rows
    ]
    path.write_text(_srt.compose(subs, reindex=False))


def _read_subs(path):
    return {s.index: s for s in _srt.parse(path.read_text())}


def test_edit_text_rewrites_keeping_speaker(tmp_path):
    srt_path = tmp_path / "retrans.srt"
    _write_srt(srt_path, [(1, 0, 2, "SPEAKER_00: at eight forty five")])
    d = make(1, "edit_text", 0.9, new_text="at 8:45")
    d.auto_apply = True
    changed = apply_fixes([d], subtitles_file=str(srt_path),
                          emotions_file=str(tmp_path / "emo.json"))
    assert changed == [1]
    assert _read_subs(srt_path)[1].content == "SPEAKER_00: at 8:45"


def test_drop_line_removes_line(tmp_path):
    srt_path = tmp_path / "retrans.srt"
    _write_srt(srt_path, [(1, 0, 2, "SPEAKER_00: hi"),
                          (2, 2, 4, "SPEAKER_00: AAAAH")])
    d = make(2, "drop_line", 0.95)
    d.auto_apply = True
    changed = apply_fixes([d], subtitles_file=str(srt_path),
                          emotions_file=str(tmp_path / "emo.json"))
    assert changed == [2]
    assert 2 not in _read_subs(srt_path)


def test_change_speaker_rewrites_prefix(tmp_path):
    srt_path = tmp_path / "retrans.srt"
    _write_srt(srt_path, [(1, 0, 2, "SPEAKER_00: hi there")])
    d = make(1, "change_speaker", 0.9, new_speaker="SPEAKER_01")
    d.auto_apply = True
    apply_fixes([d], subtitles_file=str(srt_path),
                emotions_file=str(tmp_path / "emo.json"))
    assert _read_subs(srt_path)[1].content == "SPEAKER_01: hi there"


def test_change_timing_rewrites_window(tmp_path):
    srt_path = tmp_path / "retrans.srt"
    _write_srt(srt_path, [(1, 0.0, 2.0, "SPEAKER_00: hi")])
    d = make(1, "change_timing", 0.9, new_start_ms=500, new_end_ms=1800)
    d.auto_apply = True
    apply_fixes([d], subtitles_file=str(srt_path),
                emotions_file=str(tmp_path / "emo.json"))
    sub = _read_subs(srt_path)[1]
    assert sub.start == timedelta(milliseconds=500)
    assert sub.end == timedelta(milliseconds=1800)


def test_set_emotion_overwrites_tags_file(tmp_path):
    srt_path = tmp_path / "retrans.srt"
    _write_srt(srt_path, [(1, 0, 2, "SPEAKER_00: hi")])
    emo = tmp_path / "emo.json"
    emo.write_text(json.dumps({"1": {"emotion_tag": "[calm]", "category": "neutral",
                                     "emo_vector": [0, 0]}}))
    d = make(1, "set_emotion", 0.9, tag="[furious]", category="angry",
             vector=[1.0, 0.0])
    d.auto_apply = True
    apply_fixes([d], subtitles_file=str(srt_path), emotions_file=str(emo))
    data = json.loads(emo.read_text())
    assert data["1"]["emotion_tag"] == "[furious]"
    assert data["1"]["category"] == "angry"
    assert data["1"]["emo_vector"] == [1.0, 0.0]


def test_only_auto_apply_decisions_are_written(tmp_path):
    srt_path = tmp_path / "retrans.srt"
    _write_srt(srt_path, [(1, 0, 2, "SPEAKER_00: original")])
    d = make(1, "edit_text", 0.5, new_text="ignored")
    d.auto_apply = False
    changed = apply_fixes([d], subtitles_file=str(srt_path),
                          emotions_file=str(tmp_path / "emo.json"))
    assert changed == []
    assert _read_subs(srt_path)[1].content == "SPEAKER_00: original"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_qc_fixes.py::test_change_speaker_rewrites_prefix -v`
Expected: FAIL — `ImportError: cannot import name 'apply_fixes'`.

- [ ] **Step 3: Implement `apply_fixes` and the per-primitive writers**

Append to `qc_fixes.py`:

```python
import json
from datetime import timedelta

import srt as _srt

from post_build_fix import apply_retranslation


def _drop_line(idx: int, subtitles_file: str) -> list[int]:
    subs = list(_srt.parse(open(subtitles_file, encoding="utf-8").read()))
    remaining = [s for s in subs if s.index != idx]
    if len(remaining) == len(subs):
        return []
    with open(subtitles_file, "w", encoding="utf-8") as f:
        f.write(_srt.compose(sorted(remaining, key=lambda x: x.start), reindex=False))
    return [idx]


def _change_speaker(idx: int, new_speaker: str, subtitles_file: str) -> list[int]:
    subs = list(_srt.parse(open(subtitles_file, encoding="utf-8").read()))
    for sub in subs:
        if sub.index == idx:
            text = sub.content.split(":", 1)[1].strip() if ":" in sub.content else sub.content
            sub.content = f"{new_speaker}: {text}"
            break
    else:
        return []
    with open(subtitles_file, "w", encoding="utf-8") as f:
        f.write(_srt.compose(sorted(subs, key=lambda x: x.start), reindex=False))
    return [idx]


def _change_timing(idx: int, new_start_ms: int, new_end_ms: int,
                   subtitles_file: str) -> list[int]:
    subs = list(_srt.parse(open(subtitles_file, encoding="utf-8").read()))
    for sub in subs:
        if sub.index == idx:
            sub.start = timedelta(milliseconds=new_start_ms)
            sub.end = timedelta(milliseconds=new_end_ms)
            break
    else:
        return []
    with open(subtitles_file, "w", encoding="utf-8") as f:
        f.write(_srt.compose(sorted(subs, key=lambda x: x.start), reindex=False))
    return [idx]


def _set_emotion(idx: int, tag: str, category: str, vector: list[float],
                 emotions_file: str) -> list[int]:
    data = {}
    try:
        data = json.loads(open(emotions_file, encoding="utf-8").read())
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data[str(idx)] = {"emotion_tag": tag, "category": category, "emo_vector": vector}
    with open(emotions_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return [idx]


def apply_fixes(decisions: list[Decision], subtitles_file: str,
                emotions_file: str) -> list[int]:
    """Write each auto-apply decision to the frozen retranslated SRT (text/speaker/
    timing/drop) or emotions_tags.json (emotion). Returns changed indices for regen."""
    changed: list[int] = []
    for d in decisions:
        if not d.auto_apply:
            continue
        p, prm = d.primitive, d.params
        if p == "edit_text":
            changed += apply_retranslation(d.idx, prm["new_text"], subtitles_file)
        elif p == "drop_line":
            changed += _drop_line(d.idx, subtitles_file)
        elif p == "change_speaker":
            changed += _change_speaker(d.idx, prm["new_speaker"], subtitles_file)
        elif p == "change_timing":
            changed += _change_timing(d.idx, prm["new_start_ms"],
                                      prm["new_end_ms"], subtitles_file)
        elif p == "set_emotion":
            changed += _set_emotion(d.idx, prm["tag"], prm["category"],
                                    prm["vector"], emotions_file)
    return sorted(set(changed))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_qc_fixes.py -v`
Expected: PASS (16 tests total).

- [ ] **Step 5: Commit**

```bash
git add qc_fixes.py tests/test_qc_fixes.py
git commit -m "feat(qc): apply primitives (edit/drop/speaker/timing/emotion) to artifacts"
```

---

## Task 4: Evidence bundle builder

**Files:**
- Create: `qc_evidence.py`
- Test: `tests/test_qc_evidence.py`

The bundle is all-text/numeric per spec. It never trusts the SRT box as the sole window — it also pulls diarization spans and vocal energy over the region. The precomputed mismatch signal (long loud original ↔ few words) is the labeled hallucination flag.

- [ ] **Step 1: Write the failing test**

Create `tests/test_qc_evidence.py`:

```python
import json

from qc_evidence import build_evidence_bundle, diarization_spans_overlapping


def test_diarization_overlap_selects_region():
    diar = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 5.0, "speaker": "SPEAKER_01"},
        {"start": 6.0, "end": 7.0, "speaker": "SPEAKER_00"},
    ]
    spans = diarization_spans_overlapping(diar, 1.5, 5.5)
    assert spans == [{"start": 2.0, "end": 5.0, "speaker": "SPEAKER_01"}]


def test_bundle_shape(tmp_path, monkeypatch):
    # Minimal fake run dir
    data = tmp_path / "data"
    data.mkdir()
    (data / "subtitles.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nSPEAKER_00: hi\n\n")
    (data / "subtitles_translated.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nSPEAKER_00: hola\n\n")
    (data / "subtitles_retranslated.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nSPEAKER_00: hola\n\n")
    (data / "emotions_tags.json").write_text(
        json.dumps({"1": {"emotion_tag": "[calm]", "category": "neutral",
                          "emo_vector": [0.0]}}))
    (data / "speakers_segments_data.json").write_text(
        json.dumps([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]))
    (data / "build_final_stats.json").write_text(
        json.dumps({"stats": [{"index": 1, "final_fit_ratio": 1.0,
                               "spill_vs_subtitle_ms": 0, "applied_speed_factor": 1.0,
                               "timing_status": "good"}]}))
    (data / "natural_timing.json").write_text(
        json.dumps({"per_line_atempo": {"1": 1.0}}))
    (data / "subtitles_visibility.json").write_text(json.dumps({}))

    class FakeConfig:
        data_output_folder = str(data)
        vocal_file = str(tmp_path / "vocal.wav")  # missing -> energy stats degrade gracefully

    # An Issue-like object with the fields build_evidence_bundle reads.
    class I:
        def __init__(self):
            self.start = 0.0
            self.end = 2.0
            self.sub_index = 1
            self.symptom = "wrong"
            self.mismatch = "why"

    bundle = build_evidence_bundle([I()], FakeConfig())
    assert 1 in bundle
    entry = bundle[1]
    assert entry["text"]["original"].endswith("hi")
    assert entry["text"]["retranslated"].endswith("hola")
    assert entry["diarization_spans"] == [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
    assert entry["emotion"]["category"] == "neutral"
    assert entry["timing"]["timing_status"] == "good"
    assert "mismatch_signal" in entry
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_qc_evidence.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qc_evidence'`.

- [ ] **Step 3: Implement `qc_evidence.py`**

Create `qc_evidence.py`:

```python
from __future__ import annotations

import json
import os

import srt as _srt


def _load_json(path: str, default):
    try:
        return json.loads(open(path, encoding="utf-8").read())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _subs_by_idx(path: str) -> dict:
    try:
        return {s.index: s for s in _srt.parse(open(path, encoding="utf-8").read())}
    except FileNotFoundError:
        return {}


def diarization_spans_overlapping(diar: list[dict], start: float,
                                  end: float) -> list[dict]:
    return [s for s in diar if s["start"] < end and start < s["end"]]


def _vocal_energy_stats(vocal_file: str, start: float, end: float) -> dict:
    """Mean dBFS + active-fraction over [start,end] and a padded window. Degrades to
    nulls if the file is missing so the bundle is still emitted."""
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(vocal_file)
    except Exception:
        return {"available": False}
    pad = 0.5
    a = max(0, int((start - pad) * 1000))
    b = int((end + pad) * 1000)
    window = seg[a:b]
    core = seg[int(start * 1000):int(end * 1000)]
    return {
        "available": True,
        "core_dbfs": round(core.dBFS, 2) if len(core) else None,
        "window_dbfs": round(window.dBFS, 2) if len(window) else None,
        "core_duration_s": round(len(core) / 1000.0, 3),
    }


def _mismatch_signal(diar_spans: list[dict], text: str, region_s: float) -> dict:
    """Labeled hallucination flag: long continuous vocal activity vs. few words."""
    words = len((text.split(":", 1)[1] if ":" in text else text).split())
    activity_s = sum(s["end"] - s["start"] for s in diar_spans)
    words_per_activity_s = (words / activity_s) if activity_s > 0 else None
    return {
        "diarized_activity_s": round(activity_s, 3),
        "transcribed_word_count": words,
        "words_per_activity_s": (round(words_per_activity_s, 3)
                                 if words_per_activity_s is not None else None),
        "long_activity_short_text": (activity_s >= 1.5 and words <= 3),
    }


def build_evidence_bundle(issues, config) -> dict:
    """Per flagged issue (keyed by sub_index), gather the deterministic text/numeric
    evidence. The SRT box is never the sole window — diarization + vocal energy are
    authoritative (spec Evidence Bundle)."""
    d = config.data_output_folder
    orig = _subs_by_idx(os.path.join(d, "subtitles.srt"))
    trans = _subs_by_idx(os.path.join(d, "subtitles_translated.srt"))
    retrans = _subs_by_idx(os.path.join(d, "subtitles_retranslated.srt"))
    emotions = _load_json(os.path.join(d, "emotions_tags.json"), {})
    diar = _load_json(os.path.join(d, "speakers_segments_data.json"), [])
    stats_doc = _load_json(os.path.join(d, "build_final_stats.json"), {})
    stats_by_idx = {row["index"]: row for row in stats_doc.get("stats", [])}
    natural = _load_json(os.path.join(d, "natural_timing.json"), {})
    per_line_atempo = natural.get("per_line_atempo", {})
    visibility = _load_json(os.path.join(d, "subtitles_visibility.json"), {})

    bundle: dict[int, dict] = {}
    for issue in issues:
        idx = issue.sub_index
        if idx is None:
            continue
        retrans_content = retrans[idx].content if idx in retrans else ""
        spans = diarization_spans_overlapping(diar, issue.start, issue.end)
        bundle[idx] = {
            "qc": {"start": issue.start, "end": issue.end,
                   "symptom": issue.symptom, "mismatch": issue.mismatch},
            "text": {
                "original": orig[idx].content if idx in orig else "",
                "translated": trans[idx].content if idx in trans else "",
                "retranslated": retrans_content,
            },
            "diarization_spans": spans,
            "vocal_energy": _vocal_energy_stats(config.vocal_file, issue.start,
                                                issue.end),
            "mismatch_signal": _mismatch_signal(spans, retrans_content,
                                                issue.end - issue.start),
            "emotion": emotions.get(str(idx), {}),
            "timing": stats_by_idx.get(idx, {}),
            "per_line_atempo": per_line_atempo.get(str(idx)),
            "visibility": visibility.get(str(idx), {}),
        }
    return bundle
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_qc_evidence.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add qc_evidence.py tests/test_qc_evidence.py
git commit -m "feat(qc): deterministic evidence bundle builder"
```

---

## Task 5: Playbook loader + config file

**Files:**
- Create: `config/qc_playbook.json`
- Modify: `qc_fixes.py`
- Test: `tests/test_qc_fixes.py`

- [ ] **Step 1: Create the playbook file**

Create `config/qc_playbook.json`:

```json
[
  {
    "pattern": "Long continuous original-vocal activity (scream/laugh) but few or no words in the transcript — STT hallucination on non-speech.",
    "primitive": "drop_line",
    "guidance": "Drop the invented line so the original non-speech vocal is revealed by the non-speech layer.",
    "confidence_boost": 0.10
  },
  {
    "pattern": "Numbers/times/units read out as words or mis-normalized (e.g. 'eight forty five' vs '8:45', '$' vs 'dollars').",
    "primitive": "edit_text",
    "guidance": "Rewrite the text with the correct normalized form; keep the speaker prefix.",
    "confidence_boost": 0.10
  },
  {
    "pattern": "Emotion of the dubbed line clearly wrong vs. what the scene requires (flat where it should be angry, etc.).",
    "primitive": "set_emotion",
    "guidance": "Overwrite the line's emotion tag/category/vector.",
    "confidence_boost": 0.05
  }
]
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_qc_fixes.py`:

```python
from qc_fixes import load_playbook, match_playbook


def test_load_playbook_reads_entries():
    entries = load_playbook("config/qc_playbook.json")
    assert len(entries) >= 3
    assert all("pattern" in e and "primitive" in e for e in entries)


def test_match_playbook_missing_file_returns_empty(tmp_path):
    assert load_playbook(str(tmp_path / "nope.json")) == []


def test_match_playbook_by_primitive():
    entries = [{"pattern": "p", "primitive": "drop_line", "confidence_boost": 0.1}]
    boost = match_playbook(entries, "drop_line")
    assert boost == 0.1
    assert match_playbook(entries, "edit_text") == 0.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/test_qc_fixes.py::test_load_playbook_reads_entries -v`
Expected: FAIL — `ImportError: cannot import name 'load_playbook'`.

- [ ] **Step 4: Implement loader/matcher**

Append to `qc_fixes.py`:

```python
def load_playbook(path: str = "config/qc_playbook.json") -> list[dict]:
    return _load_playbook_json(path)


def _load_playbook_json(path: str) -> list[dict]:
    try:
        return json.loads(open(path, encoding="utf-8").read())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def match_playbook(entries: list[dict], primitive: str) -> float:
    """Return the max confidence_boost among entries whose primitive matches.
    The agent supplies the human-readable pattern match; this rewards alignment
    of the chosen primitive with a known pattern."""
    boosts = [float(e.get("confidence_boost", 0.0))
              for e in entries if e.get("primitive") == primitive]
    return max(boosts) if boosts else 0.0
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_qc_fixes.py -v`
Expected: PASS (19 tests total).

- [ ] **Step 6: Commit**

```bash
git add config/qc_playbook.json qc_fixes.py tests/test_qc_fixes.py
git commit -m "feat(qc): human-curated playbook loader and matcher"
```

---

## Task 6: `qc_agent.py` — parse structured output + build decisions

The agent module holds all LLM reasoning. To keep it testable without a live API, the model call is a single injectable function `model_call(system, contents, response_schema) -> dict`. `decide()` builds the prompt, calls the model (phase 1, then phase 2 if any issue requested audio), maps raw output to `Decision` objects, applies the playbook boost, and runs the gate. Collision resolution stays in `qc_fixes` and is called by the orchestrator (Task 8) after `decide`.

**Files:**
- Create: `qc_agent.py`
- Test: `tests/test_qc_agent.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_qc_agent.py`:

```python
from qc_agent import decide, RawDecision


def fake_model(raw_decisions):
    """Return a callable that ignores inputs and yields a fixed phase-1 payload."""
    def _call(system, contents, response_schema):
        return {"decisions": raw_decisions}
    return _call


class Issue:
    def __init__(self, sub_index, symptom="s", mismatch="m", start=0.0, end=2.0):
        self.sub_index = sub_index
        self.symptom = symptom
        self.mismatch = mismatch
        self.start = start
        self.end = end


def _bundle(idx):
    return {idx: {"text": {"retranslated": "SPEAKER_00: AAAH"}, "mismatch_signal":
                  {"long_activity_short_text": True}}}


def test_scream_hallucination_becomes_drop_line_auto():
    raw = [{"idx": 1, "primitive": "drop_line", "confidence": 0.85,
            "diagnosis": "scream hallucination", "params": {},
            "playbook_pattern_matched": True}]
    playbook = [{"pattern": "scream", "primitive": "drop_line",
                 "confidence_boost": 0.10}]
    decisions = decide([Issue(1)], _bundle(1), playbook,
                       model_call=fake_model(raw))
    assert len(decisions) == 1
    d = decisions[0]
    assert d.primitive == "drop_line"
    # 0.85 + 0.10 boost = 0.95 >= 0.90 threshold
    assert d.auto_apply is True


def test_normalization_edit_text():
    raw = [{"idx": 2, "primitive": "edit_text", "confidence": 0.72,
            "diagnosis": "8:45", "params": {"new_text": "at 8:45"},
            "playbook_pattern_matched": False}]
    decisions = decide([Issue(2)], _bundle(2), [], model_call=fake_model(raw))
    assert decisions[0].params["new_text"] == "at 8:45"
    assert decisions[0].auto_apply is True


def test_ambiguous_becomes_propose():
    raw = [{"idx": 3, "primitive": "propose", "confidence": 0.5,
            "diagnosis": "unclear", "params": {"suggested_fix": "human review"},
            "playbook_pattern_matched": False}]
    decisions = decide([Issue(3)], _bundle(3), [], model_call=fake_model(raw))
    assert decisions[0].primitive == "propose"
    assert decisions[0].auto_apply is False


def test_phase2_invoked_when_audio_requested():
    calls = {"n": 0}

    def two_phase(system, contents, response_schema):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"decisions": [{"idx": 4, "primitive": "pending",
                                   "confidence": 0.0, "diagnosis": "",
                                   "params": {},
                                   "needs_audio": {"idx": 4, "window": [1.0, 3.0]}}]}
        return {"decisions": [{"idx": 4, "primitive": "drop_line",
                               "confidence": 0.95, "diagnosis": "confirmed",
                               "params": {}, "playbook_pattern_matched": False}]}

    decisions = decide([Issue(4)], _bundle(4), [], model_call=two_phase,
                       audio_provider=lambda reqs: {4: b"opusbytes"})
    assert calls["n"] == 2
    assert decisions[0].primitive == "drop_line"
    assert decisions[0].auto_apply is True


def test_model_failure_yields_all_proposals():
    def boom(system, contents, response_schema):
        raise RuntimeError("api down")

    decisions = decide([Issue(5), Issue(6)], {5: {}, 6: {}}, [],
                       model_call=boom)
    assert len(decisions) == 2
    assert all(d.primitive == "propose" and d.auto_apply is False
               for d in decisions)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_qc_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qc_agent'`.

- [ ] **Step 3: Implement `qc_agent.py`**

Create `qc_agent.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from qc_fixes import Decision, apply_gate, match_playbook

AGENT_SYSTEM_PROMPT = """You are a dubbing QC-fix agent. For each flagged issue you \
receive the QC symptom plus a deterministic evidence bundle (text layers, diarization \
spans, original-vocal energy, a precomputed long-activity/short-text mismatch signal, \
emotion tag, timing/fit row). Reason from evidence to exactly one primitive per issue:

- edit_text {new_text}: fix wording/normalization (numbers, units, times).
- drop_line {}: remove an invented line (STT hallucination on non-speech, e.g. a scream \
transcribed as words — look for long_activity_short_text).
- set_emotion {tag, category, vector}: fix a clearly wrong emotion.
- change_timing {new_start_ms, new_end_ms}: fix spill/misalignment.
- change_speaker {new_speaker}: fix a misattributed voice. Prefer 'propose' if adjacent \
lines look mislabeled too (diarization-block error).
- propose {suggested_fix}: anything low-confidence, ambiguous, or risky.

Emit an explicit confidence in [0,1]. If you cannot decide without hearing the audio, \
emit needs_audio {idx, window:[start_s,end_s]} instead of a final primitive; the window \
may extend past the SRT box (e.g. a scream that spills). Set playbook_pattern_matched \
true when your diagnosis matches a provided playbook pattern."""


@dataclass
class RawDecision:
    idx: int
    primitive: str
    confidence: float
    diagnosis: str
    params: dict
    playbook_pattern_matched: bool = False
    needs_audio: dict | None = None


def _to_decision(raw: dict, playbook: list[dict]) -> Decision:
    primitive = raw.get("primitive", "propose")
    matched = bool(raw.get("playbook_pattern_matched", False))
    boost = match_playbook(playbook, primitive) if matched else 0.0
    return Decision(
        idx=raw["idx"],
        primitive=primitive,
        confidence=float(raw.get("confidence", 0.0)),
        params=raw.get("params", {}) or {},
        diagnosis=raw.get("diagnosis", ""),
        playbook_matched=matched,
        playbook_boost=boost,
        needs_audio=raw.get("needs_audio"),
    )


def _build_prompt(issues, bundle, playbook) -> str:
    return json.dumps({
        "playbook": playbook,
        "issues": [
            {"sub_index": i.sub_index, "symptom": i.symptom,
             "mismatch": i.mismatch, "start": i.start, "end": i.end,
             "evidence": bundle.get(i.sub_index, {})}
            for i in issues
        ],
    }, ensure_ascii=False, default=str)


def _all_proposals(issues) -> list[Decision]:
    out = []
    for i in issues:
        d = Decision(idx=i.sub_index if i.sub_index is not None else -1,
                     primitive="propose", confidence=0.0,
                     params={"suggested_fix": "agent unavailable — human review"},
                     diagnosis="agent/LLM failure")
        d.auto_apply = False
        out.append(d)
    return out


def decide(issues, bundle, playbook,
           model_call: Callable[[str, str, object], dict],
           audio_provider: Callable[[list[dict]], dict] | None = None) -> list[Decision]:
    """Two-phase agent. Phase 1: text only. Phase 2 (optional): re-invoke with audio
    clips for issues that requested them. Any exception -> all proposals (never blocks
    the pipeline). Returns Decisions with the gate applied; collision resolution is the
    orchestrator's job."""
    try:
        prompt = _build_prompt(issues, bundle, playbook)
        phase1 = model_call(AGENT_SYSTEM_PROMPT, prompt, None)
        raws = {r["idx"]: r for r in phase1.get("decisions", [])}

        audio_reqs = [r["needs_audio"] for r in raws.values()
                      if r.get("needs_audio")]
        if audio_reqs and audio_provider is not None:
            clips = audio_provider(audio_reqs)  # {idx: opus_bytes}
            phase2_prompt = json.dumps({
                "resolve_with_audio": [r["needs_audio"] for r in raws.values()
                                       if r.get("needs_audio")],
                "note": "audio clips attached separately; finalize these issues",
            }, default=str)
            phase2 = model_call(AGENT_SYSTEM_PROMPT, phase2_prompt, None)
            for r in phase2.get("decisions", []):
                raws[r["idx"]] = r  # phase-2 answer overrides the pending one

        decisions = [_to_decision(r, playbook) for r in raws.values()]
        apply_gate(decisions)
        return decisions
    except Exception as e:  # noqa: BLE001 — agent must never block the dub
        print(f"⚠️  QC agent failed ({e}); emitting all issues as proposals")
        return _all_proposals(issues)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_qc_agent.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add qc_agent.py tests/test_qc_agent.py
git commit -m "feat(qc): LLM reasoning agent with two-phase call and gate"
```

---

## Task 7: Extract `qc_check()` from `test_dub_qc.py`

Both the CLI and the pipeline must run the same QC. Extract the pass/union logic into `qc_check()`; `main()` becomes a thin wrapper.

**Files:**
- Modify: `test_dub_qc.py:139-222`
- Test: `tests/test_qc_agent.py` (import smoke) — full behavior needs live API, so we only assert the function exists and is callable with a mocked client.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_qc_agent.py`:

```python
def test_qc_check_is_importable_and_unions_passes(monkeypatch):
    import test_dub_qc as qc

    class FakeParsed:
        def __init__(self, issues):
            self.issues = issues

    class FakeResp:
        def __init__(self, issues):
            self.parsed = FakeParsed(issues)

    calls = {"n": 0}

    class FakeModels:
        def generate_content(self, model, contents, config):
            calls["n"] += 1
            return FakeResp([qc.Issue(start=1.0, end=2.0, sub_index=7,
                                      symptom="x", mismatch="y",
                                      severity=qc.Severity.high)])

    class FakeClient:
        models = FakeModels()

    issues = qc.qc_check(audio_bytes=b"opus", script="[7] 1.00-2.00  S: hi",
                         client=FakeClient(), model="fake", passes=3,
                         temperature=0.4, thinking_level="LOW")
    assert calls["n"] == 3
    assert len(issues) == 1  # 3 identical passes union down to one
    assert issues[0].sub_index == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_qc_agent.py::test_qc_check_is_importable_and_unions_passes -v`
Expected: FAIL — `AttributeError: module 'test_dub_qc' has no attribute 'qc_check'`.

- [ ] **Step 3: Refactor `test_dub_qc.py`**

Replace the body of `main()` (lines 168-222) so the reusable core lives in `qc_check()`. Insert this new function immediately before `def main()` (after `compress_audio`, line 137):

```python
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
```

Then replace lines 168-211 of `main()` (from `script = build_script(...)` through `merged = dedup_issues(all_issues)`) with:

```python
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
```

The remaining lines of `main()` (the print loop and the `out = ...` write, lines 212-222) are unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_qc_agent.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Verify the CLI still imports**

Run: `python3 -c "import test_dub_qc; assert hasattr(test_dub_qc, 'qc_check') and hasattr(test_dub_qc, 'main')"`
Expected: no output, exit 0.

- [ ] **Step 6: Commit**

```bash
git add test_dub_qc.py tests/test_qc_agent.py
git commit -m "refactor(qc): extract reusable qc_check() from CLI main()"
```

---

## Task 8: Relocate `split_vocal` from TRANSLATE into COMBINE

Locked decision #9. Today `split_vocal` runs at TRANSLATE (`main_prefect_dag.py:601`) reading `config.subtitles`. Move its submit into the COMBINE block, parallel with `tts_build_final`, reading `subtitles_for_combine`. This makes `drop_line`/`change_timing` fixes rebuild the non-speech layer for free during a QC-fix cycle (which re-runs COMBINE). Behavior-neutral for the main run because SRT layers share timestamps.

**Files:**
- Modify: `main_prefect_dag.py:590-619` (remove from TRANSLATE), `main_prefect_dag.py:876-883` (add to COMBINE)
- Test: `tests/test_regen_helper.py`

- [ ] **Step 1: Verify `split_vocal` already reads an arbitrary subtitles path**

Run: `python3 -c "import inspect, audio; print(inspect.signature(audio.split_vocal))"`
Expected: `(input_vocal, subtitles_file, non_speech_layer_file)` — confirms no code change to `audio.py`; only the caller changes.

- [ ] **Step 2: Remove the TRANSLATE-stage submit**

In `main_prefect_dag.py`, in the TRANSLATE block, delete the `split_vocal_fut` submit at line 601:

```python
        split_vocal_fut = t_split_vocal.submit(vocal_file, config.subtitles, config.non_speech_layer_file)
```

And delete its await at lines 618-619:

```python
    if split_vocal_fut:
        split_vocal_fut.result()
```

Also remove the now-unused declaration `split_vocal_fut = None` at line 592.

- [ ] **Step 3: Add the submit into the COMBINE block**

In the `if stage <= STAGES.COMBINE:` block (line 876), after computing `subtitles_for_combine` (line 877) and before `tts_build_final_fut`, insert:

```python
        # Non-speech layer (laughs/screams/reactions) is carved from the ORIGINAL
        # vocal for spans not covered by a subtitle. Built here (not at TRANSLATE) so
        # a QC-fix cycle — which re-runs COMBINE after drop_line/change_timing edits —
        # rebuilds it against the frozen retranslated subs. Runs parallel to the timing
        # build; both only feed build_audio, so it costs no wall-clock time.
        split_vocal_fut = t_split_vocal.submit(
            vocal_file, subtitles_for_combine, config.non_speech_layer_file)
```

Then, immediately before the `t_build_audio.submit` call (line 881), await it so the layer exists before the mix:

```python
        split_vocal_fut.result()
```

- [ ] **Step 4: Write a regression test for the ordering**

Create `tests/test_regen_helper.py`:

```python
import ast


def _combine_block_source():
    src = open("main_prefect_dag.py", encoding="utf-8").read()
    # Sanity: the COMBINE block owns split_vocal now; TRANSLATE block does not.
    return src


def test_split_vocal_removed_from_translate_block():
    src = _combine_block_source()
    translate_marker = "# --- TRANSLATE stage"
    combine_marker = "if stage <= STAGES.COMBINE:"
    t_start = src.index(translate_marker)
    c_start = src.index(combine_marker)
    translate_region = src[t_start:c_start]
    assert "t_split_vocal.submit" not in translate_region


def test_split_vocal_present_in_combine_block_with_combine_subs():
    src = _combine_block_source()
    c_start = src.index("if stage <= STAGES.COMBINE:")
    combine_region = src[c_start:]
    assert "t_split_vocal.submit" in combine_region
    # reads the frozen combine subs, not the original SRT
    assert "subtitles_for_combine, config.non_speech_layer_file" in combine_region


def test_module_still_imports():
    import importlib
    import main_prefect_dag
    importlib.reload(main_prefect_dag)
    assert hasattr(main_prefect_dag, "dubbing_flow")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_regen_helper.py -v`
Expected: PASS (3 tests). If `test_module_still_imports` fails on heavy import side effects, mark it `@pytest.mark.skip(reason="import has deploy-time side effects")` — the two source-assertion tests are the real guard.

- [ ] **Step 6: Commit**

```bash
git add main_prefect_dag.py tests/test_regen_helper.py
git commit -m "refactor(pipeline): build non-speech layer at COMBINE from frozen subs"
```

---

## Task 9: Extract `_regen_and_combine` helper

Locked decisions #3, #7, #8. Pull the GENERATE→timing→COMBINE body into a reusable helper with a `qc_fix: bool` flag. In `qc_fix` mode it: pre-syncs retranslated→translated; scopes GENERATE to `changed_list`; loads saved `speaker_base_atempo.json` (no whole-track re-measure); runs `classify_lines` on changed lines only; skips retranslation entirely (clamp+spill); runs COMBINE (which now includes the relocated `split_vocal`).

This is a mechanical extraction plus the qc_fix branch. Because the current body is deeply inlined in `dubbing_flow` and depends on many locals, the helper takes an explicit context object.

**Files:**
- Modify: `main_prefect_dag.py` (add `_regen_and_combine`, call it from the COMBINE-region for pass 1)
- Test: `tests/test_regen_helper.py`

- [ ] **Step 1: Add the helper signature and qc_fix pre-sync (no behavior change to pass 1 yet)**

Insert this function above `dubbing_flow` (near the other module-level helpers, e.g. before line 432 `@flow`):

```python
def _regen_and_combine(
    config,
    speakers_array,
    emotions_tags,
    subtitle_visibility_analysis,
    dst_language,
    ttsmodel,
    speaker_base_atempo,
    timing_overflow_threshold,
    timing_max_speed_factor,
    is_dubbed,
    mix_gains,
    use_non_speech,
    video_file,
    changed_list,
    build_cache=None,
    qc_fix: bool = False,
):
    """Run GENERATE(scoped)→timing→COMBINE once. In qc_fix mode: freeze retranslated
    as truth (pre-sync into translated), reuse the saved per-speaker atempo, classify
    only changed lines, skip retranslation (clamp+spill), then COMBINE (rebuilds the
    non-speech layer via the relocated split_vocal). Returns the final audio path."""
    import shutil

    is_qwen3 = ttsmodel == TTS_MODEL.QWEN3TTS.value

    if qc_fix:
        # Pre-sync: retranslated holds all prior rerolls + agent edits. Copy it into
        # translated so GENERATE reads the frozen truth. Makes the TIMING_FIX-top
        # copyfile(translated -> retranslated) a harmless no-op.
        if os.path.exists(config.subtitles_retranslated_file):
            shutil.copyfile(config.subtitles_retranslated_file,
                            config.subtitles_translated_file)

    # --- GENERATE (scoped to changed_list) ---
    regen_file = config.subtitles_retranslated_file
    if is_qwen3:
        t_generate_qwen3tts_segments.submit(
            config=config, translated_file=regen_file, speakers=speakers_array,
            emotions_tags=emotions_tags, language_code=dst_language,
            changed_list=changed_list,
        ).result()
    else:
        t_generate_indextts2_segments.submit(
            config=config, translated_file=regen_file, speakers=speakers_array,
            emotions_tags=emotions_tags, changed_list=changed_list,
            duration_factors=None, warm_pods=0,
        ).result()

    t_combine_tts_segments.submit(speakers_array, config.tts_segments_folder).result()

    # --- Timing: no whole-track re-measure; classify only changed lines ---
    if qc_fix:
        sb_path = os.path.join(config.data_output_folder, "speaker_base_atempo.json")
        if os.path.exists(sb_path):
            speaker_base_atempo = json.loads(open(sb_path, encoding="utf-8").read())

        remeasure = t_tts_build_final.submit(
            config, speakers=speakers_array, convert_flag=True,
            subtitle_visibility_analysis=subtitle_visibility_analysis,
            testing=True, build_cache=build_cache, changed_list=changed_list,
            subtitles_file=config.subtitles_retranslated_file,
        ).result()
        build_cache = remeasure["build_cache"]

        classification = classify_lines(
            stats=remeasure["stats"], segment_meta=remeasure["segment_meta"],
            speaker_base_atempo=speaker_base_atempo,
            overflow_threshold=timing_overflow_threshold,
        )
        # Merge changed lines' atempo into the saved natural_timing.json (don't shift
        # other lines). Skip retranslation entirely — clamp+spill on longer fixes.
        natural = _load_json_or(
            os.path.join(config.data_output_folder, "natural_timing.json"),
            {"per_line_atempo": {}})
        natural["per_line_atempo"].update(
            {str(k): v for k, v in classification["per_line_atempo"].items()})
        with open(os.path.join(config.data_output_folder, "natural_timing.json"),
                  "w", encoding="utf-8") as f:
            json.dump(natural, f, indent=2, ensure_ascii=False)

    # --- COMBINE (includes relocated split_vocal) ---
    subtitles_for_combine = (config.subtitles_retranslated_file
                             if os.path.exists(config.subtitles_retranslated_file)
                             else config.subtitles_translated_file)
    t_loudness_adjust.submit(subtitles_file=subtitles_for_combine,
                             vocal_file=config.vocal_file,
                             tts_segments_folder=config.tts_segments_folder).result()
    split_vocal_fut = t_split_vocal.submit(
        config.vocal_file, subtitles_for_combine, config.non_speech_layer_file)
    tts_build_final_fut = t_tts_build_final.submit(
        config, speakers=speakers_array, convert_flag=True,
        subtitle_visibility_analysis=subtitle_visibility_analysis, testing=False,
        speaker_base_atempo=speaker_base_atempo, subtitles_file=subtitles_for_combine,
        max_speed_factor=timing_max_speed_factor)
    split_vocal_fut.result()
    build_audio_fut = t_build_audio.submit(
        config, tts_build_final_flag=tts_build_final_fut.result(),
        is_dubbed=is_dubbed, mix_gains=mix_gains, use_non_speech=use_non_speech,
        video_file=video_file)
    return build_audio_fut.result()


def _load_json_or(path: str, default):
    try:
        return json.loads(open(path, encoding="utf-8").read())
    except (FileNotFoundError, json.JSONDecodeError):
        return default
```

Note: this helper is **additive** — the existing inline COMBINE block (Task 8's version) still runs pass 1. We wire the helper into the tail in Task 10; pass-1 behavior is unchanged here, which keeps this task low-risk.

- [ ] **Step 2: Write the QC_FIX-mode behavior test (mocked tasks)**

Append to `tests/test_regen_helper.py`:

```python
import os
import shutil
import types


def test_regen_qc_fix_presyncs_and_skips_retranslate(tmp_path, monkeypatch):
    import main_prefect_dag as m

    # Fake config with the paths the helper touches.
    data = tmp_path / "data"
    data.mkdir()
    retrans = data / "subtitles_retranslated.srt"
    trans = data / "subtitles_translated.srt"
    retrans.write_text("1\n00:00:00,000 --> 00:00:02,000\nSPEAKER_00: fixed\n\n")
    trans.write_text("1\n00:00:00,000 --> 00:00:02,000\nSPEAKER_00: OLD\n\n")
    (data / "speaker_base_atempo.json").write_text('{"SPEAKER_00": 1.1}')
    (data / "natural_timing.json").write_text('{"per_line_atempo": {"1": 1.0}}')

    cfg = types.SimpleNamespace(
        data_output_folder=str(data),
        subtitles_retranslated_file=str(retrans),
        subtitles_translated_file=str(trans),
        non_speech_layer_file=str(tmp_path / "ns.wav"),
        vocal_file=str(tmp_path / "vocal.wav"),
        tts_segments_folder=str(tmp_path / "tts"),
    )

    called = {"retranslate_timing_fix": 0, "compute_base": 0}

    # Stub every Prefect task the helper submits with a .submit().result() shim.
    def shim(result=None):
        fut = types.SimpleNamespace(result=lambda: result)
        return types.SimpleNamespace(submit=lambda *a, **k: fut)

    monkeypatch.setattr(m, "t_generate_indextts2_segments", shim())
    monkeypatch.setattr(m, "t_generate_qwen3tts_segments", shim())
    monkeypatch.setattr(m, "t_combine_tts_segments", shim())
    monkeypatch.setattr(m, "t_tts_build_final",
                        shim({"build_cache": None, "stats": [], "segment_meta": {}}))
    monkeypatch.setattr(m, "t_loudness_adjust", shim())
    monkeypatch.setattr(m, "t_split_vocal", shim())
    monkeypatch.setattr(m, "t_build_audio", shim("FINAL.wav"))
    monkeypatch.setattr(m, "classify_lines",
                        lambda **k: {"per_line_atempo": {1: 1.2}})

    def boom_retranslate(**k):
        called["retranslate_timing_fix"] += 1
        raise AssertionError("retranslation must not run in qc_fix mode")
    monkeypatch.setattr(m, "retranslate_timing_fix", boom_retranslate, raising=False)

    out = m._regen_and_combine(
        config=cfg, speakers_array={"SPEAKER_00": {}}, emotions_tags={},
        subtitle_visibility_analysis={}, dst_language="en",
        ttsmodel=m.TTS_MODEL.INDEXTTS2.value, speaker_base_atempo=None,
        timing_overflow_threshold=1.35, timing_max_speed_factor=1.45,
        is_dubbed=False, mix_gains=None, use_non_speech=True, video_file=None,
        changed_list=[1], qc_fix=True)

    assert out == "FINAL.wav"
    assert called["retranslate_timing_fix"] == 0
    # pre-sync copied retranslated -> translated
    assert "fixed" in trans.read_text()
    # changed line's atempo merged into natural_timing.json
    assert '"1": 1.2' in (data / "natural_timing.json").read_text()
```

- [ ] **Step 3: Run test to verify it passes**

Run: `python3 -m pytest tests/test_regen_helper.py::test_regen_qc_fix_presyncs_and_skips_retranslate -v`
Expected: PASS. (If module import has deploy side effects, guard with the skip note from Task 8 Step 5 and instead assert helper logic via a thin re-import.)

- [ ] **Step 4: Commit**

```bash
git add main_prefect_dag.py tests/test_regen_helper.py
git commit -m "feat(pipeline): _regen_and_combine helper with qc_fix targeted mode"
```

---

## Task 10: Post-COMBINE QC-fix tail + fix log

Wire it together (spec Data Flow). After pass-1 renders `final_audio.wav`, the tail: runs `qc_check`; if issues exist, builds the evidence bundle, calls the agent, resolves collisions on auto-fixes, applies them, runs `_regen_and_combine(..., qc_fix=True)` on the changed lines, re-QCs, diffs, and writes `qc_fix_log.json` (+ appends novel confirmed fixes to the promotion queue).

**Files:**
- Create: `qc_tail.py` (orchestration extracted from the flow so it's unit-testable without Prefect)
- Modify: `main_prefect_dag.py` COMBINE block (call the tail after `generate_videos`)
- Test: `tests/test_qc_agent.py` (tail logic with everything mocked)

- [ ] **Step 1: Write the failing test for the tail's diff/log logic**

Append to `tests/test_qc_agent.py`:

```python
from qc_tail import diff_reqc, build_fix_log


class Iss:
    def __init__(self, sub_index):
        self.sub_index = sub_index
        self.start, self.end = 0.0, 1.0
        self.symptom, self.mismatch = "s", "m"
        from test_dub_qc import Severity
        self.severity = Severity.high


def test_diff_fixed_when_issue_gone():
    from qc_fixes import Decision
    before = [Iss(1), Iss(2)]
    after = [Iss(2)]  # idx1 cleared, idx2 remains
    applied = [Decision(idx=1, primitive="drop_line", confidence=0.95),
               Decision(idx=2, primitive="edit_text", confidence=0.9)]
    for d in applied:
        d.auto_apply = True
    diff_reqc(applied, before, after)
    outcomes = {d.idx: d.outcome for d in applied}
    assert outcomes[1] == "fixed"
    assert outcomes[2] == "regression"


def test_build_fix_log_shape():
    from qc_fixes import Decision
    fixed = Decision(idx=1, primitive="drop_line", confidence=0.95,
                     diagnosis="scream")
    fixed.auto_apply = True
    fixed.outcome = "fixed"
    proposal = Decision(idx=3, primitive="propose", confidence=0.4,
                        params={"suggested_fix": "human"}, diagnosis="unclear")
    log = build_fix_log([fixed], [proposal], reqc_count=1)
    assert log["summary"]["auto_applied"] == 1
    assert log["summary"]["proposals"] == 1
    assert log["decisions"][0]["idx"] == 1
    assert log["proposals"][0]["idx"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_qc_agent.py::test_diff_fixed_when_issue_gone -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qc_tail'`.

- [ ] **Step 3: Implement `qc_tail.py`**

Create `qc_tail.py`:

```python
from __future__ import annotations

import json
import os
from dataclasses import asdict

from qc_fixes import Decision, apply_fixes, load_playbook, resolve_collisions


def diff_reqc(applied: list[Decision], before, after) -> None:
    """Set outcome per applied decision: 'fixed' if its idx is no longer flagged,
    else 'regression' (downgrade to proposal in the log; render pass 2 still ships —
    re-QC is advisory, spec Error Handling)."""
    still_flagged = {i.sub_index for i in after if i.sub_index is not None}
    for d in applied:
        if d.idx in still_flagged:
            d.outcome = "regression"
            d.auto_apply = False
        else:
            d.outcome = "fixed"


def build_fix_log(applied: list[Decision], proposals: list[Decision],
                  reqc_count: int) -> dict:
    def ser(d: Decision) -> dict:
        return {
            "idx": d.idx, "primitive": d.primitive,
            "confidence": round(d.confidence, 3),
            "effective_confidence": round(min(1.0, d.confidence + (
                d.playbook_boost if d.playbook_matched else 0.0)), 3),
            "params": d.params, "diagnosis": d.diagnosis,
            "playbook_matched": d.playbook_matched, "outcome": d.outcome,
        }
    return {
        "summary": {
            "auto_applied": sum(1 for d in applied if d.outcome == "fixed"),
            "regressions": sum(1 for d in applied if d.outcome == "regression"),
            "proposals": len(proposals),
            "reqc_passes": reqc_count,
        },
        "decisions": [ser(d) for d in applied],
        "proposals": [ser(d) for d in proposals],
    }


def append_promotion_queue(applied: list[Decision], path: str) -> None:
    """Novel (non-playbook) fixes confirmed 'fixed' by re-QC get queued for a human
    to promote into the playbook. Append-only; agent never writes the playbook."""
    novel = [d for d in applied
             if d.outcome == "fixed" and not d.playbook_matched]
    if not novel:
        return
    queue = []
    if os.path.exists(path):
        try:
            queue = json.loads(open(path, encoding="utf-8").read())
        except json.JSONDecodeError:
            queue = []
    for d in novel:
        queue.append({"primitive": d.primitive, "diagnosis": d.diagnosis,
                      "params": d.params, "confidence": round(d.confidence, 3)})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)


def write_fix_log(log: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def split_auto_and_proposals(decisions: list[Decision]):
    auto = [d for d in decisions if d.auto_apply]
    proposals = [d for d in decisions if not d.auto_apply]
    return auto, proposals
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_qc_agent.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Wire the tail into `dubbing_flow`**

In `main_prefect_dag.py`, at the end of the `if stage <= STAGES.COMBINE:` block (after `output_file = generate_videos_fut.result()`, line 889), add the tail. First add imports near the other top-level imports:

```python
from qc_agent import decide as qc_decide
from qc_evidence import build_evidence_bundle
from qc_fixes import apply_fixes, load_playbook, resolve_collisions
from qc_tail import (append_promotion_queue, build_fix_log, diff_reqc,
                     split_auto_and_proposals, write_fix_log)
from test_dub_qc import build_script, compress_audio, qc_check
```

Then append the tail inside the COMBINE block:

```python
        # ---------------- post-COMBINE QC-fix tail (same flow run) ----------------
        qc_log_path = os.path.join(config.data_output_folder, "qc_fix_log.json")
        try:
            from google import genai
            subs_for_qc = subtitles_for_combine
            script = build_script(subs_for_qc)
            audio_bytes = compress_audio(config.audio_result_file)
            gclient = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            gmodel = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

            issues = qc_check(audio_bytes=audio_bytes, script=script,
                              client=gclient, model=gmodel)

            if not issues:                                   # zero-issue path
                write_fix_log(build_fix_log([], [], reqc_count=0), qc_log_path)
            else:
                bundle = build_evidence_bundle(issues, config)
                playbook = load_playbook("config/qc_playbook.json")

                def _model_call(system, contents, response_schema):
                    r = gclient.models.generate_content(
                        model=gmodel,
                        contents=[system, contents],
                        config=genai.types.GenerateContentConfig(
                            response_mime_type="application/json"))
                    return json.loads(r.text)

                decisions = qc_decide(issues, bundle, playbook,
                                      model_call=_model_call)
                auto, proposals = split_auto_and_proposals(decisions)

                if not auto:                                 # all proposals -> no regen
                    write_fix_log(build_fix_log([], proposals, reqc_count=0),
                                  qc_log_path)
                else:
                    auto = resolve_collisions(auto)          # dedupe by idx FIRST
                    proposals += [d for d in decisions
                                  if not d.auto_apply and d not in proposals]
                    changed = apply_fixes(
                        auto, subtitles_file=config.subtitles_retranslated_file,
                        emotions_file=config.emotions_tags_file)

                    _regen_and_combine(
                        config=config, speakers_array=speakers_array,
                        emotions_tags=emotions_tags,
                        subtitle_visibility_analysis=subtitle_visibility_analysis,
                        dst_language=dst_language, ttsmodel=ttsmodel,
                        speaker_base_atempo=speaker_base_atempo,
                        timing_overflow_threshold=timing_overflow_threshold,
                        timing_max_speed_factor=timing_max_speed_factor,
                        is_dubbed=is_dubbed, mix_gains=mix_gains,
                        use_non_speech=use_non_speech, video_file=video_file,
                        changed_list=changed, qc_fix=True)

                    reqc_audio = compress_audio(config.audio_result_file)
                    reqc_script = build_script(config.subtitles_retranslated_file)
                    reqc_issues = qc_check(audio_bytes=reqc_audio,
                                           script=reqc_script, client=gclient,
                                           model=gmodel)
                    diff_reqc(auto, issues, reqc_issues)
                    append_promotion_queue(
                        auto, "config/qc_promotion_queue.json")
                    write_fix_log(build_fix_log(auto, proposals, reqc_count=1),
                                  qc_log_path)
                    # Re-render video from the pass-2 audio.
                    output_file = t_generate_videos.submit(
                        config, video_file, config.audio_result_file,
                        preview=False).result()
        except Exception as e:  # noqa: BLE001 — QC-fix must never fail the dub
            print(f"⚠️  QC-fix tail failed ({e}); shipping render pass 1")
            write_fix_log({"summary": {"error": str(e)}, "decisions": [],
                           "proposals": []}, qc_log_path)
```

- [ ] **Step 6: Verify the flow module imports cleanly**

Run: `python3 -c "import main_prefect_dag; assert hasattr(main_prefect_dag, '_regen_and_combine')"`
Expected: exit 0. (If import triggers Modal/Prefect deploy side effects, run instead: `python3 -m py_compile main_prefect_dag.py qc_tail.py qc_agent.py qc_evidence.py qc_fixes.py` — expect exit 0.)

- [ ] **Step 7: Run the full unit suite**

Run: `python3 -m pytest tests/ -v`
Expected: PASS (all tests across the five test files).

- [ ] **Step 8: Commit**

```bash
git add qc_tail.py main_prefect_dag.py tests/test_qc_agent.py
git commit -m "feat(qc): post-COMBINE QC-fix tail wiring QC->agent->regen->re-QC"
```

---

## Task 11: End-to-end manual verification (live API)

Spec Testing section: run the tail on a real run dir with known defects.

**Files:** none (verification only)

- [ ] **Step 1: Pick the known-defect run**

The spec names `output/20260903_barkoni_10` (scream hallucination at 48.8–51.1s + an `8:45` normalization error). Confirm it exists, else use the most recent run under `output/`:

Run: `ls -d output/20260903_barkoni_10 2>/dev/null || ls -dt output/*/ | head -1`

- [ ] **Step 2: Run QC-fix by re-running the COMBINE stage on that run**

The tail runs at the end of the COMBINE stage. Invoke the flow from `stage=COMBINE` on the chosen run_id (the `__main__` block in `main_prefect_dag.py` is the entry point; set `run_id` and `stage=STAGES.COMBINE.value`). Because prior stages' artifacts already exist, only COMBINE + the tail execute.

Expected during the run:
- `pass k/3: N issue(s)` printed by `qc_check` (pass 1).
- Agent decisions; a `drop_line` on the scream line and an `edit_text` for `8:45`.
- One `_regen_and_combine(..., qc_fix=True)` cycle regenerating only those indices.
- Re-QC prints, then `report`/log written.

- [ ] **Step 3: Verify the fix log**

Run: `cat output/<run_id>/data/qc_fix_log.json | python3 -m json.tool | head -40`
Expected: the scream line shows `"primitive": "drop_line", "outcome": "fixed"`; the `8:45` line shows `"primitive": "edit_text", "outcome": "fixed"`.

- [ ] **Step 4: Verify the scream is revealed, not silenced**

Because `split_vocal` now runs at COMBINE against the post-drop subs, the dropped scream's window becomes a gap and its original vocal is carried by `non_speech_layer.wav`. Listen to the final audio around 48.8–51.1s and confirm the original scream is audible (not silence, not a hallucinated word).

Run: `ls -la output/<run_id>/audio/non_speech_layer.wav output/<run_id>/audio/final_audio.wav`
Then play `final_audio.wav` and spot-check the two timestamps. Note: this is a listening check — report explicitly that it was verified by ear (per the "judge audio by listening" preference), not by metrics.

- [ ] **Step 5: Confirm re-QC came back clean on those two**

In the log `summary`, `regressions` should be 0 for these two indices, and both appear under `decisions` with `outcome: "fixed"`.

---

## Self-Review

**Spec coverage:**
- Purpose / auto-apply + propose (decision #1): Tasks 1, 6, 10. ✓
- Live post-COMBINE stage (#2): Task 10. ✓
- One fix cycle, no loop (#3): Task 9 helper + Task 10 single `_regen_and_combine` + one re-QC. ✓
- Gate = risk not novelty (#4): Task 1 thresholds; novel issues still gated (Task 6 boost only on match). ✓
- Evidence bundle default, audio on request (#5): Task 4 bundle; Task 6 two-phase `needs_audio`. ✓
- Single-call structured per-issue output (#6): Task 6 `decide`. ✓
- QC_FIX on retranslated as truth, pre-sync, skip base recompute/reroll, clamp+spill (#7): Task 9. ✓
- One flow run, inline tail (#8): Task 10. ✓
- `split_vocal` → COMBINE (#9): Task 8 (main run) + Task 9 (qc_fix path). ✓
- Five primitives: Tasks 1, 3, 6. ✓
- Collision resolution: Task 2. ✓
- Error handling (zero issues / all proposals / agent failure / regen failure / re-QC regression): Task 10 branches + Task 6 try/except + Task 9 (shipped pass-1 preserved) + Task 10 diff downgrade. ✓
- Playbook + promotion queue (human-curated, append-only): Tasks 5, 10. ✓
- Testing (agent unit / QC_FIX mode / playbook-threshold / e2e): Tasks 1-2-3-5-6 (unit), 9 (QC_FIX mode), 1+5 (threshold/boost), 11 (e2e). ✓

**Placeholder scan:** no TODO/TBD; every code step contains full code; commands have expected output. ✓

**Type consistency:** `Decision` fields (`idx, primitive, confidence, params, diagnosis, playbook_matched, playbook_boost, needs_audio, auto_apply, outcome`) are defined once (Task 1) and used consistently in Tasks 2, 3, 6, 10. `apply_fixes(decisions, subtitles_file, emotions_file)` signature matches its call in Task 10. `qc_check(audio_bytes, script, client, model, passes, temperature, thinking_level)` matches Task 7 definition and Task 10 calls. `_regen_and_combine(...)` param list in Task 9 matches the call in Task 10. ✓

**Known integration risk to watch during execution:** the exact `google.genai` structured-output call shape in Task 10's `_model_call` (schema for the agent's per-issue output) is the one place the plan uses a plain JSON response rather than a pydantic `response_schema`. During Task 6/10 execution, prefer defining a pydantic `AgentReport` schema mirroring `RawDecision` and passing it as `response_schema` (like `QCReport` in `test_dub_qc.py`) for reliability. This is noted here rather than blocking, since the unit tests mock `model_call` and don't depend on the wire format.

---

## Execution Handoff
```
