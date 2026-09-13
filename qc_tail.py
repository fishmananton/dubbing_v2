from __future__ import annotations

import json
import os
from dataclasses import asdict

from qc_fixes import Decision, apply_fixes, load_playbook, resolve_collisions


def diff_reqc(applied: list[Decision], before, after) -> None:
    """Set outcome per applied decision: 'fixed' if its idx is no longer flagged,
    else 'regression' (downgrade to proposal in the log; render pass 2 still ships —
    re-QC is advisory, spec Error Handling)."""
    still_flagged = {i.sub_index for i in after if i.sub_index is not None}
    for d in applied:
        if d.idx in still_flagged:
            d.outcome = "regression"
            d.auto_apply = False
        else:
            d.outcome = "fixed"


def build_fix_log(applied: list[Decision], proposals: list[Decision],
                  reqc_count: int) -> dict:
    def ser(d: Decision) -> dict:
        return {
            "idx": d.idx, "primitive": d.primitive,
            "confidence": round(d.confidence, 3),
            "effective_confidence": round(min(1.0, d.confidence + (
                d.playbook_boost if d.playbook_matched else 0.0)), 3),
            "params": d.params, "diagnosis": d.diagnosis,
            "playbook_matched": d.playbook_matched, "outcome": d.outcome,
        }
    return {
        "summary": {
            "auto_applied": sum(1 for d in applied if d.outcome == "fixed"),
            "regressions": sum(1 for d in applied if d.outcome == "regression"),
            "proposals": len(proposals),
            "reqc_passes": reqc_count,
        },
        "decisions": [ser(d) for d in applied],
        "proposals": [ser(d) for d in proposals],
    }


def append_promotion_queue(applied: list[Decision], path: str) -> None:
    """Novel (non-playbook) fixes confirmed 'fixed' by re-QC get queued for a human
    to promote into the playbook. Append-only; agent never writes the playbook."""
    novel = [d for d in applied
             if d.outcome == "fixed" and not d.playbook_matched]
    if not novel:
        return
    queue = []
    if os.path.exists(path):
        try:
            queue = json.loads(open(path, encoding="utf-8").read())
        except json.JSONDecodeError:
            queue = []
    for d in novel:
        queue.append({"primitive": d.primitive, "diagnosis": d.diagnosis,
                      "params": d.params, "confidence": round(d.confidence, 3)})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)


def write_fix_log(log: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def write_qc_issues(issues, path: str) -> None:
    """Persist the raw issues QC reported (symptom/mismatch/severity/window), so a
    human can see exactly what the listen pass heard — independent of what the agent
    later did with them. `issues` are test_dub_qc.Issue objects (severity is an enum)."""
    def ser(i):
        sev = getattr(i, "severity", None)
        return {
            "sub_index": i.sub_index,
            "start": i.start,
            "end": i.end,
            "symptom": i.symptom,
            "mismatch": i.mismatch,
            "severity": getattr(sev, "value", sev),
        }
    out = {"count": len(issues), "issues": [ser(i) for i in issues]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def split_auto_and_proposals(decisions: list[Decision]):
    auto = [d for d in decisions if d.auto_apply]
    proposals = [d for d in decisions if not d.auto_apply]
    return auto, proposals
