from datetime import timedelta

import srt

from song_detect import Chunk, chunk_subs


def _sub(idx, start_s, end_s, text="x"):
    return srt.Subtitle(index=idx,
                        start=timedelta(seconds=start_s),
                        end=timedelta(seconds=end_s),
                        content=f"Speaker: {text}")


def test_short_input_is_single_chunk():
    subs = [_sub(1, 0, 2), _sub(2, 100, 102), _sub(3, 470, 480)]
    chunks = chunk_subs(subs, target_min=10, min_split_min=12, gap_min_s=2.0)
    assert len(chunks) == 1
    assert chunks[0].offset_s == 0.0
    assert [s.index for s in chunks[0].subs] == [1, 2, 3]


def test_splits_long_input_at_nearest_gap_over_threshold():
    subs = []
    t = 0.0
    idx = 1
    while t < 1200:
        subs.append(_sub(idx, t, t + 1.0))
        gap = 5.0 if 600 <= t < 601 else 1.0
        t += 1.0 + gap
        idx += 1
    chunks = chunk_subs(subs, target_min=10, min_split_min=12, gap_min_s=2.0)
    assert len(chunks) == 2
    assert chunks[1].offset_s > 600
    all_idx = [s.index for c in chunks for s in c.subs]
    assert all_idx == [s.index for s in subs]


def test_no_qualifying_gap_allows_oversized_chunk():
    subs = []
    t = 0.0
    idx = 1
    while t < 1500:
        subs.append(_sub(idx, t, t + 1.0))
        gap = 3.0 if 1080 <= t < 1081 else 1.0
        t += 1.0 + gap
        idx += 1
    chunks = chunk_subs(subs, target_min=10, min_split_min=12, gap_min_s=2.0)
    assert len(chunks) == 2
    assert chunks[0].subs[-1].end.total_seconds() <= 1081
    assert chunks[1].offset_s > 1080


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


import json


def test_drop_sub_ids_stashes_dropped_blocks_to_sidecar(tmp_path):
    # Song detection deletes lines that were ALREADY translated. Instead of destroying
    # that English text, soft-drop stashes each dropped block {idx,start,end,content}
    # into a sidecar so QC's restore_line can bring it back with no re-translation.
    subs = [_sub(1, 0, 1, "a"), _sub(3, 4, 5, "locked him in the closet"),
            _sub(4, 6, 7, "d")]
    p = tmp_path / "subtitles_translated.srt"
    p.write_text(srt.compose(subs, reindex=False), encoding="utf-8")
    sidecar = tmp_path / "dropped_song_lines.json"

    removed = drop_sub_ids(str(p), [3], sidecar_path=str(sidecar))
    assert removed == [3]
    # line 3 gone from the translated SRT
    assert [s.index for s in srt.parse(p.read_text(encoding="utf-8"))] == [1, 4]
    # but preserved verbatim in the sidecar, keyed by idx
    stash = json.loads(sidecar.read_text(encoding="utf-8"))
    row = stash["3"]
    assert row["start_ms"] == 4000 and row["end_ms"] == 5000
    # full block content preserved verbatim, incl. the "Speaker:" prefix
    assert row["content"] == "Speaker: locked him in the closet"


def test_drop_sub_ids_sidecar_accumulates_across_calls(tmp_path):
    # Two drop passes must both persist; the second must not clobber the first.
    subs = [_sub(1, 0, 1, "a"), _sub(2, 2, 3, "b"), _sub(3, 4, 5, "c")]
    p = tmp_path / "subtitles_translated.srt"
    p.write_text(srt.compose(subs, reindex=False), encoding="utf-8")
    sidecar = tmp_path / "dropped_song_lines.json"

    drop_sub_ids(str(p), [2], sidecar_path=str(sidecar))
    drop_sub_ids(str(p), [3], sidecar_path=str(sidecar))
    stash = json.loads(sidecar.read_text(encoding="utf-8"))
    assert set(stash.keys()) == {"2", "3"}


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
    # decode is stubbed: this test exercises union logic only (no real audio)
    class _FakeSeg:
        def set_channels(self, n):
            return self
    monkeypatch.setattr(sd.AudioSegment, "from_file",
                        staticmethod(lambda *a, **k: _FakeSeg()))

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
