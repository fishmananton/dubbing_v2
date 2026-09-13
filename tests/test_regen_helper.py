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


@pytest.mark.skip(reason="import has deploy-time side effects")
def test_module_still_imports():
    import importlib
    import main_prefect_dag
    importlib.reload(main_prefect_dag)
    assert hasattr(main_prefect_dag, "dubbing_flow")
