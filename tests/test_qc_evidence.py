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
