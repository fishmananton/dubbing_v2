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
