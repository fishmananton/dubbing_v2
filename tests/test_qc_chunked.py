"""Chunked QC: split long audio into chunks (like the CLI), run all (chunk x pass)
requests in parallel, rebase each issue by its chunk offset, and union. These pin the
rebase + union contract with a stubbed per-pass call (no real Gemini/audio)."""
from datetime import timedelta

import srt

import test_dub_qc as q
from test_dub_qc import Issue, Severity, qc_check_chunked


def _sub(i, s, e):
    return srt.Subtitle(index=i, start=timedelta(seconds=s), end=timedelta(seconds=e),
                        content=f"SPK: line{i}")


def _long_subs():
    # >12 min with a wide gap near 600s so chunk_subs splits into 2 chunks
    subs, t, i = [], 0.0, 1
    while t < 1200:
        subs.append(_sub(i, t, t + 1.0))
        t += 1.0 + (5.0 if 600 <= t < 601 else 1.0)
        i += 1
    return subs


def test_rebases_by_chunk_offset_and_unions_passes(monkeypatch):
    monkeypatch.setattr(q, "slice_audio_opus", lambda *a, **k: b"opus")

    # every pass "hears" one issue at LOCAL 1-2s of its chunk
    def fake_pass(audio_bytes, script, client, model, temperature, thinking_level,
                 max_retries):
        return [Issue(start=1.0, end=2.0, sub_index=None, symptom="x", mismatch="y",
                      severity=Severity.high)]

    monkeypatch.setattr(q, "_qc_single_pass", fake_pass)

    out = qc_check_chunked(seg=object(), subs=_long_subs(), client=object(),
                           model="m", passes=2, thinking_level="LOW")

    # 2 chunks -> 2 distinct issues after union; passes within a chunk collapse to one
    assert len(out) == 2
    starts = sorted(round(o.start) for o in out)
    assert starts[0] == 1            # chunk 1 issue stays at ~1s
    assert starts[1] > 600           # chunk 2 issue rebased by its offset


def test_single_chunk_short_input(monkeypatch):
    monkeypatch.setattr(q, "slice_audio_opus", lambda *a, **k: b"opus")
    calls = {"n": 0}

    def fake_pass(*a, **k):
        calls["n"] += 1
        return [Issue(start=0.5, end=1.0, sub_index=1, symptom="s", mismatch="m",
                      severity=Severity.medium)]

    monkeypatch.setattr(q, "_qc_single_pass", fake_pass)
    subs = [_sub(1, 0, 1), _sub(2, 2, 3)]
    out = qc_check_chunked(seg=object(), subs=subs, client=object(), model="m",
                           passes=3, thinking_level="MEDIUM")
    assert calls["n"] == 3           # 1 chunk x 3 passes
    assert len(out) == 1             # unioned
    assert out[0].sub_index == 1
