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


def test_edit_text_strips_duplicate_speaker_prefix(tmp_path):
    # The model sometimes returns new_text WITH the speaker prefix already on it
    # (e.g. "Paulo: ..."). The writer must not double it into "Paulo: Paulo: ...",
    # which regen would then speak aloud. Real taxi_CUT_03 idx-102 regression.
    srt_path = tmp_path / "retrans.srt"
    _write_srt(srt_path, [(1, 0, 2, "Paulo: старый текст")])
    d = make(1, "edit_text", 0.9, new_text="Paulo: новый текст")
    d.auto_apply = True
    apply_fixes([d], subtitles_file=str(srt_path),
                emotions_file=str(tmp_path / "emo.json"))
    assert _read_subs(srt_path)[1].content == "Paulo: новый текст"


def test_edit_text_strips_prefix_only_when_it_matches_speaker(tmp_path):
    # A colon inside genuine dialogue (different from the speaker label) must survive.
    srt_path = tmp_path / "retrans.srt"
    _write_srt(srt_path, [(1, 0, 2, "SPEAKER_00: old")])
    d = make(1, "edit_text", 0.9, new_text="he said: run")
    d.auto_apply = True
    apply_fixes([d], subtitles_file=str(srt_path),
                emotions_file=str(tmp_path / "emo.json"))
    assert _read_subs(srt_path)[1].content == "SPEAKER_00: he said: run"


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


def test_restore_line_auto_applies_at_threshold():
    d = make(-1, "restore_line", 0.70, window=[100.0, 113.0])
    apply_gate([d])
    assert d.auto_apply is True


def test_two_window_restores_both_survive_collision():
    # Null-idx restores both carry idx=-1 but target different time spans; they must
    # NOT downgrade each other (idx<0 means "not a specific line", never a conflict).
    a = make(-1, "restore_line", 0.9, window=[100.0, 113.0])
    b = make(-1, "restore_line", 0.9, window=[702.0, 748.0])
    a.auto_apply = b.auto_apply = True
    kept = resolve_collisions([a, b])
    assert len(kept) == 2
    assert all(d.auto_apply for d in kept)


def _write_stash(path, rows):
    # rows: list of (idx, start_ms, end_ms, content)
    stash = {str(i): {"start_ms": s, "end_ms": e, "content": c}
             for (i, s, e, c) in rows}
    path.write_text(json.dumps(stash, ensure_ascii=False))


def test_restore_line_window_reinserts_overlapping_stashed_lines(tmp_path):
    # A mis-dropped dialogue block (idx 3,4,5) sits in the sidecar. A window restore
    # re-inserts every stashed line overlapping the window into the retranslated SRT.
    srt_path = tmp_path / "retrans.srt"
    _write_srt(srt_path, [(2, 66.0, 66.7, "Rapunzel: Ah."),
                          (6, 113.7, 117.2, "Rapunzel: So I can't...")])
    stash = tmp_path / "dropped_song_lines.json"
    _write_stash(stash, [
        (3, 100800, 103100, "Rapunzel: So I locked him in the closet."),
        (4, 103700, 105800, "Rapunzel: I locked him in the closet."),
        (5, 107500, 111100, "Rapunzel: I went and locked him in the closet."),
    ])
    d = make(-1, "restore_line", 0.9, window=[100.0, 113.0])
    d.auto_apply = True
    changed = apply_fixes([d], subtitles_file=str(srt_path),
                          emotions_file=str(tmp_path / "emo.json"),
                          dropped_lines_file=str(stash))
    assert changed == [3, 4, 5]
    subs = _read_subs(srt_path)
    assert subs[3].content == "Rapunzel: So I locked him in the closet."
    assert subs[3].start == timedelta(milliseconds=100800)
    assert set(subs.keys()) == {2, 3, 4, 5, 6}


def test_restore_line_by_explicit_idx(tmp_path):
    srt_path = tmp_path / "retrans.srt"
    _write_srt(srt_path, [(1, 0, 1, "S: a")])
    stash = tmp_path / "dropped_song_lines.json"
    _write_stash(stash, [(1, 0, 1, "S: a"),
                         (7, 5000, 6000, "S: restored")])
    d = make(7, "restore_line", 0.9)
    d.auto_apply = True
    changed = apply_fixes([d], subtitles_file=str(srt_path),
                          emotions_file=str(tmp_path / "emo.json"),
                          dropped_lines_file=str(stash))
    assert changed == [7]
    assert _read_subs(srt_path)[7].content == "S: restored"


def test_restore_line_no_match_is_safe_noop(tmp_path):
    srt_path = tmp_path / "retrans.srt"
    _write_srt(srt_path, [(1, 0, 1, "S: a")])
    stash = tmp_path / "dropped_song_lines.json"
    _write_stash(stash, [(3, 100000, 101000, "S: dropped")])
    d = make(-1, "restore_line", 0.9, window=[500.0, 600.0])  # no overlap
    d.auto_apply = True
    changed = apply_fixes([d], subtitles_file=str(srt_path),
                          emotions_file=str(tmp_path / "emo.json"),
                          dropped_lines_file=str(stash))
    assert changed == []
    assert set(_read_subs(srt_path).keys()) == {1}


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
