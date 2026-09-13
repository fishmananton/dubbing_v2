from __future__ import annotations

import json
import os

import srt as _srt


def _load_json(path: str, default):
    try:
        return json.loads(open(path, encoding="utf-8").read())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _subs_by_idx(path: str) -> dict:
    try:
        return {s.index: s for s in _srt.parse(open(path, encoding="utf-8").read())}
    except FileNotFoundError:
        return {}


def diarization_spans_overlapping(diar: list[dict], start: float,
                                  end: float) -> list[dict]:
    return [s for s in diar if s["start"] < end and start < s["end"]]


def _vocal_energy_stats(vocal_file: str, start: float, end: float) -> dict:
    """Mean dBFS + active-fraction over [start,end] and a padded window. Degrades to
    nulls if the file is missing so the bundle is still emitted."""
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(vocal_file)
    except Exception:
        return {"available": False}
    pad = 0.5
    a = max(0, int((start - pad) * 1000))
    b = int((end + pad) * 1000)
    window = seg[a:b]
    core = seg[int(start * 1000):int(end * 1000)]
    return {
        "available": True,
        "core_dbfs": round(core.dBFS, 2) if len(core) else None,
        "window_dbfs": round(window.dBFS, 2) if len(window) else None,
        "core_duration_s": round(len(core) / 1000.0, 3),
    }


def _mismatch_signal(diar_spans: list[dict], text: str, region_s: float) -> dict:
    """Labeled hallucination flag: long continuous vocal activity vs. few words."""
    words = len((text.split(":", 1)[1] if ":" in text else text).split())
    activity_s = sum(s["end"] - s["start"] for s in diar_spans)
    words_per_activity_s = (words / activity_s) if activity_s > 0 else None
    return {
        "diarized_activity_s": round(activity_s, 3),
        "transcribed_word_count": words,
        "words_per_activity_s": (round(words_per_activity_s, 3)
                                 if words_per_activity_s is not None else None),
        "long_activity_short_text": (activity_s >= 1.5 and words <= 3),
    }


def build_evidence_bundle(issues, config) -> dict:
    """Per flagged issue (keyed by sub_index), gather the deterministic text/numeric
    evidence. The SRT box is never the sole window — diarization + vocal energy are
    authoritative (spec Evidence Bundle)."""
    d = config.data_output_folder
    orig = _subs_by_idx(os.path.join(d, "subtitles.srt"))
    trans = _subs_by_idx(os.path.join(d, "subtitles_translated.srt"))
    retrans = _subs_by_idx(os.path.join(d, "subtitles_retranslated.srt"))
    emotions = _load_json(os.path.join(d, "emotions_tags.json"), {})
    diar = _load_json(os.path.join(d, "speakers_segments_data.json"), [])
    stats_doc = _load_json(os.path.join(d, "build_final_stats.json"), {})
    stats_by_idx = {row["index"]: row for row in stats_doc.get("stats", [])}
    natural = _load_json(os.path.join(d, "natural_timing.json"), {})
    per_line_atempo = natural.get("per_line_atempo", {})
    visibility = _load_json(os.path.join(d, "subtitles_visibility.json"), {})

    bundle: dict[int, dict] = {}
    for issue in issues:
        idx = issue.sub_index
        if idx is None:
            continue
        retrans_content = retrans[idx].content if idx in retrans else ""
        spans = diarization_spans_overlapping(diar, issue.start, issue.end)
        bundle[idx] = {
            "qc": {"start": issue.start, "end": issue.end,
                   "symptom": issue.symptom, "mismatch": issue.mismatch},
            "text": {
                "original": orig[idx].content if idx in orig else "",
                "translated": trans[idx].content if idx in trans else "",
                "retranslated": retrans_content,
            },
            "diarization_spans": spans,
            "vocal_energy": _vocal_energy_stats(config.vocal_file, issue.start,
                                                issue.end),
            "mismatch_signal": _mismatch_signal(spans, retrans_content,
                                                issue.end - issue.start),
            "emotion": emotions.get(str(idx), {}),
            "timing": stats_by_idx.get(idx, {}),
            "per_line_atempo": per_line_atempo.get(str(idx)),
            "visibility": visibility.get(str(idx), {}),
        }
    return bundle
