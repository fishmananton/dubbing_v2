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
