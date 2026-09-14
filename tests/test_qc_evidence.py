import json

from qc_evidence import (build_evidence_bundle, build_shared_context,
                         diarization_spans_overlapping)


def _write_min_run(tmp_path):
    """A minimal fake run dir; returns (FakeConfig, data_dir)."""
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
    (data / "speakers_data.json").write_text(
        json.dumps({"SPEAKER_00": {"gender": "male", "reference_audio": "x.wav"}}))
    (data / "build_final_stats.json").write_text(
        json.dumps([{"index": 1, "subtitle_start_ms": 0, "subtitle_end_ms": 2000,
                     "actual_start_ms": 0, "actual_end_ms": 1900,
                     "applied_speed_factor": 1.0, "timing_status": "good"}]))

    class FakeConfig:
        data_output_folder = str(data)
        vocal_file = str(tmp_path / "vocal.wav")  # missing -> energy degrades gracefully

    return FakeConfig(), data


class _Issue:
    def __init__(self, sub_index=1, start=0.0, end=2.0):
        self.start, self.end = start, end
        self.sub_index = sub_index
        self.symptom, self.mismatch = "wrong", "why"


def test_diarization_overlap_selects_region():
    diar = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 5.0, "speaker": "SPEAKER_01"},
        {"start": 6.0, "end": 7.0, "speaker": "SPEAKER_00"},
    ]
    spans = diarization_spans_overlapping(diar, 1.5, 5.5)
    assert spans == [{"start": 2.0, "end": 5.0, "speaker": "SPEAKER_01"}]


def test_per_issue_bundle_is_slim(tmp_path):
    # The per-issue slice carries ONLY the line-specific numeric signals; the
    # whole-clip timeline (subs/diarization/timing) lives in build_shared_context.
    config, _ = _write_min_run(tmp_path)
    bundle = build_evidence_bundle([_Issue(1)], config)
    assert 1 in bundle
    entry = bundle[1]
    assert set(entry) == {"vocal_energy", "mismatch_signal", "emotion"}
    assert entry["emotion"]["category"] == "neutral"
    assert "long_activity_short_text" in entry["mismatch_signal"]
    # dropped from the per-issue slice: text layers, diarization, timing, visibility
    assert "text" not in entry and "diarization_spans" not in entry
    assert "timing" not in entry and "visibility" not in entry


def test_null_sub_index_has_no_slice(tmp_path):
    # Gap-audio issues (no sub_index) get no per-issue key; the agent locates them
    # on the shared timeline via their start/end.
    config, _ = _write_min_run(tmp_path)
    bundle = build_evidence_bundle([_Issue(sub_index=None)], config)
    assert bundle == {}


def test_shared_context_has_whole_clip_timeline(tmp_path):
    config, _ = _write_min_run(tmp_path)
    ctx = build_shared_context(config)
    assert ctx["subtitles_retranslated"][0]["text"].endswith("hola")
    assert ctx["subtitles_original"][0]["text"].endswith("hi")
    assert ctx["diarization"] == [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
    assert ctx["timing"][0]["box_start_ms"] == 0
    assert ctx["timing"][0]["actual_end_ms"] == 1900
    assert ctx["timing"][0]["status"] == "good"
    assert ctx["speakers"]["SPEAKER_00"]["gender"] == "male"
    # translated is intentionally NOT in the shared context (retranslated is truth)
    assert "subtitles_translated" not in ctx
