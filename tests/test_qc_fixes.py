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
