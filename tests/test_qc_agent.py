import json

from qc_agent import decide, RawDecision


def test_omitted_confidence_defaults_to_apply_not_bounce():
    # The model sometimes commits to a concrete primitive with valid params but drops
    # the `confidence` field entirely (observed on CUT_11 change_timing). A committed
    # primitive IS the certainty signal, so a missing confidence must default to
    # auto-apply, not bounce the fix to a proposal. The model can still self-downgrade
    # by emitting an explicit LOW confidence.
    raw = [{"idx": 23, "primitive": "change_timing",
            "diagnosis": "extend box over leaked span",
            "params": {"new_start_ms": 68145, "new_end_ms": 71145}}]
    bundle = {23: {"mismatch_signal": {"long_activity_short_text": False}}}
    decisions = decide([Issue(23)], bundle, [], model_call=fake_model(raw))
    assert decisions[0].primitive == "change_timing"
    assert decisions[0].confidence == 1.0
    assert decisions[0].auto_apply is True


def test_explicit_low_confidence_still_bounces():
    # Omission -> apply, but an EXPLICIT low number still means self-doubt -> bounce.
    raw = [{"idx": 23, "primitive": "change_timing", "confidence": 0.4,
            "diagnosis": "unsure", "params": {"new_start_ms": 68145,
                                              "new_end_ms": 71145}}]
    bundle = {23: {"mismatch_signal": {"long_activity_short_text": False}}}
    decisions = decide([Issue(23)], bundle, [], model_call=fake_model(raw))
    assert decisions[0].auto_apply is False


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


def test_edit_text_missing_param_not_auto_applied():
    # High confidence edit_text but no new_text param. The writer would KeyError
    # mid-loop, so the gate must refuse to auto-apply (C2).
    raw = [{"idx": 2, "primitive": "edit_text", "confidence": 0.99,
            "diagnosis": "reword", "params": {},
            "playbook_pattern_matched": False}]
    bundle = {2: {"mismatch_signal": {"long_activity_short_text": False}}}
    decisions = decide([Issue(2)], bundle, [], model_call=fake_model(raw))
    assert decisions[0].primitive == "edit_text"
    assert decisions[0].auto_apply is False


def test_ambiguous_becomes_propose():
    raw = [{"idx": 3, "primitive": "propose", "confidence": 0.5,
            "diagnosis": "unclear", "params": {"suggested_fix": "human review"},
            "playbook_pattern_matched": False}]
    # Genuinely ambiguous: no concrete fix param and no hallucination signal.
    bundle = {3: {"mismatch_signal": {"long_activity_short_text": False}}}
    decisions = decide([Issue(3)], bundle, [], model_call=fake_model(raw))
    assert decisions[0].primitive == "propose"
    assert decisions[0].auto_apply is False


def test_propose_with_concrete_text_promoted_to_edit_text():
    # The model hedged to 'propose' but supplied a concrete new_text at high
    # confidence (the real idx-29 8:45 case). Promote to edit_text so the gate applies.
    raw = [{"sub_index": 29, "primitive": "propose", "confidence": 0.95,
            "params": {"new_text": "DevOps: at eight forty-five"},
            "playbook_pattern_matched": True}]
    decisions = decide([Issue(29)], _bundle(29), [], model_call=fake_model(raw))
    assert decisions[0].primitive == "edit_text"
    assert decisions[0].auto_apply is True


def test_propose_without_concrete_fix_stays_propose():
    # A bare propose with no actionable param AND no hallucination signal stays propose.
    raw = [{"idx": 4, "primitive": "propose", "confidence": 0.99,
            "params": {"suggested_fix": "human should relisten"}}]
    bundle = {4: {"mismatch_signal": {"long_activity_short_text": False}}}
    decisions = decide([Issue(4)], bundle, [], model_call=fake_model(raw))
    assert decisions[0].primitive == "propose"
    assert decisions[0].auto_apply is False


def test_propose_on_hallucination_evidence_promoted_to_drop_line():
    # Model hedged to propose with empty params, but the evidence shows the non-speech
    # hallucination signature (the real scream idx-20 case). Promote to drop_line; with
    # the playbook boost it clears the 0.90 gate and auto-applies.
    raw = [{"sub_index": 20, "primitive": "propose", "confidence": 0.85,
            "params": {}, "playbook_pattern_matched": True}]
    bundle = {20: {"mismatch_signal": {"long_activity_short_text": True}}}
    playbook = [{"pattern": "scream", "primitive": "drop_line",
                 "confidence_boost": 0.10}]
    decisions = decide([Issue(20)], bundle, playbook, model_call=fake_model(raw))
    assert decisions[0].primitive == "drop_line"
    assert decisions[0].auto_apply is True  # 0.85 + 0.10 = 0.95 >= 0.90


def test_multiple_actions_same_line_survive():
    # French bleed on one line needs BOTH a text fix (adieu->see you) AND a box
    # extension over the leaked audio. Different fields, so decide() must return both
    # (today the idx-keyed collapse drops all but the last raw). Collision resolution
    # keeps different-field edits — that's the orchestrator's job, not decide()'s.
    raw = [
        {"idx": 47, "primitive": "edit_text", "confidence": 0.9,
         "diagnosis": "adieu -> see you", "params": {"new_text": "See you."}},
        {"idx": 47, "primitive": "change_timing", "confidence": 0.9,
         "diagnosis": "extend box over bleed",
         "params": {"new_start_ms": 282225, "new_end_ms": 285000}},
    ]
    bundle = {47: {"mismatch_signal": {"long_activity_short_text": False}}}
    decisions = decide([Issue(47)], bundle, [], model_call=fake_model(raw))
    prims = sorted(d.primitive for d in decisions if d.idx == 47)
    assert prims == ["change_timing", "edit_text"]
    assert all(d.auto_apply for d in decisions)


def test_multiple_lines_from_one_issue_all_survive():
    # A block-level speaker misattribution: the flagged issue is on line 101, but the
    # fix spans the mislabeled block — reassign 101 & 102 and correct the gender-agreement
    # text on 103. One issue fans out to three lines; all must reach the gate.
    raw = [
        {"idx": 101, "primitive": "change_speaker", "confidence": 0.9,
         "diagnosis": "female voice on male block", "params": {"new_speaker": "SPEAKER_02"}},
        {"idx": 102, "primitive": "change_speaker", "confidence": 0.9,
         "diagnosis": "same block", "params": {"new_speaker": "SPEAKER_02"}},
        {"idx": 103, "primitive": "edit_text", "confidence": 0.9,
         "diagnosis": "fix masculine verb agreement", "params": {"new_text": "он сказал"}},
    ]
    decisions = decide([Issue(101)], {101: {}}, [], model_call=fake_model(raw))
    got = {(d.idx, d.primitive) for d in decisions}
    assert got == {(101, "change_speaker"), (102, "change_speaker"), (103, "edit_text")}


def test_edit_text_correction_lifted_from_diagnosis_when_param_missing():
    # Model chose edit_text and quoted the corrected line in the diagnosis but forgot
    # to fill new_text. Lift the cue-anchored correction so it can auto-apply instead
    # of degrading to an empty proposal (real taxi_CUT_03 idx-102 symptom).
    raw = [{"idx": 102, "primitive": "edit_text", "confidence": 0.95, "params": {},
            "diagnosis": "TTS synthesized 'Шесть ли'; correct to 'Шествие особенно'."}]
    bundle = {102: {"mismatch_signal": {"long_activity_short_text": False}}}
    decisions = decide([Issue(102)], bundle, [], model_call=fake_model(raw))
    assert decisions[0].params.get("new_text") == "Шествие особенно"
    assert decisions[0].auto_apply is True


def test_edit_text_without_liftable_correction_stays_proposal():
    # edit_text, no new_text, and no cue-anchored quote in the diagnosis: we must NOT
    # guess a random quote (wrong text gets spoken aloud). Stays a safe proposal.
    raw = [{"idx": 5, "primitive": "edit_text", "confidence": 0.95, "params": {},
            "diagnosis": "The wording sounds off and needs a human relisten."}]
    bundle = {5: {"mismatch_signal": {"long_activity_short_text": False}}}
    decisions = decide([Issue(5)], bundle, [], model_call=fake_model(raw))
    assert decisions[0].auto_apply is False
    assert "new_text" not in decisions[0].params


def test_shared_context_forwarded_to_prompt():
    # The whole-clip timeline must reach the model in phase 1 so it can reason across
    # neighboring lines/spans (e.g. diarization onset vs. subtitle box for bleed).
    seen = {}

    def capture(system, contents, response_schema):
        seen["prompt"] = contents
        return {"decisions": [{"idx": 1, "primitive": "propose", "confidence": 0.3,
                               "diagnosis": "d", "params": {"suggested_fix": "x"}}]}

    ctx = {"diarization": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
           "subtitles_retranslated": [{"idx": 1, "start": 0.5, "end": 2.0,
                                       "text": "S: hi"}]}
    bundle = {1: {"mismatch_signal": {"long_activity_short_text": False}}}
    decide([Issue(1)], bundle, [], model_call=capture, context=ctx)
    payload = json.loads(seen["prompt"])
    assert payload["timeline"]["diarization"][0]["speaker"] == "SPEAKER_00"
    assert payload["timeline"]["subtitles_retranslated"][0]["idx"] == 1


def test_phase2_invoked_when_audio_requested():
    calls = {"n": 0}
    seen = {}

    def two_phase(system, contents, response_schema):
        calls["n"] += 1
        seen[calls["n"]] = contents
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
    # phase-1 contents is a plain string prompt; phase-2 must forward the audio bytes
    assert isinstance(seen[1], str)
    assert isinstance(seen[2], list)
    assert b"opusbytes" in seen[2]
    assert decisions[0].primitive == "drop_line"
    assert decisions[0].auto_apply is True


def test_phase2_skipped_when_no_provider_leaves_pending_as_propose():
    # needs_audio but no provider wired -> phase 2 can't run -> the pending decision
    # (no primitive) degrades to a propose with empty params. This is the taxi_CUT_03
    # symptom: high-confidence-but-empty proposals that never auto-applied.
    def one_phase(system, contents, response_schema):
        return {"decisions": [{"idx": 4, "confidence": 0.92, "diagnosis": "",
                               "params": {},
                               "needs_audio": {"idx": 4, "window": [1.0, 3.0]}}]}

    bundle = {4: {"mismatch_signal": {"long_activity_short_text": False}}}
    decisions = decide([Issue(4)], bundle, [], model_call=one_phase)
    assert decisions[0].primitive == "propose"
    assert decisions[0].auto_apply is False


def test_model_failure_yields_all_proposals():
    def boom(system, contents, response_schema):
        raise RuntimeError("api down")

    decisions = decide([Issue(5), Issue(6)], {5: {}, 6: {}}, [],
                       model_call=boom)
    assert len(decisions) == 2
    assert all(d.primitive == "propose" and d.auto_apply is False
               for d in decisions)


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


def test_write_qc_issues_persists_symptoms(tmp_path):
    from qc_tail import write_qc_issues
    from test_dub_qc import Issue, Severity
    issues = [
        Issue(start=1.0, end=2.5, sub_index=57, symptom="flat delivery",
              mismatch="line calls for feeling", severity=Severity.medium),
        Issue(start=3.0, end=4.0, sub_index=None, symptom="click",
              mismatch="artifact", severity=Severity.low),
    ]
    p = tmp_path / "qc_issues.json"
    write_qc_issues(issues, str(p))
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["count"] == 2
    assert data["issues"][0]["sub_index"] == 57
    assert data["issues"][0]["symptom"] == "flat delivery"
    assert data["issues"][0]["severity"] == "medium"
    assert data["issues"][1]["sub_index"] is None


def test_write_qc_issues_empty(tmp_path):
    from qc_tail import write_qc_issues
    p = tmp_path / "qc_issues.json"
    write_qc_issues([], str(p))
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["count"] == 0
    assert data["issues"] == []


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
