import ast

import pytest


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
