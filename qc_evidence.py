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
    """Mean dBFS over [start,end] and a padded window. Degrades to nulls if the file
    is missing so the bundle is still emitted."""
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


def _mismatch_signal(diar_spans: list[dict], text: str) -> dict:
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


def _subs_rows(path: str) -> list[dict]:
    """Compact subtitle rows (index, start_s, end_s, content) for the whole clip."""
    try:
        subs = list(_srt.parse(open(path, encoding="utf-8").read()))
    except FileNotFoundError:
        return []
    return [{"idx": s.index,
             "start": round(s.start.total_seconds(), 3),
             "end": round(s.end.total_seconds(), 3),
             "text": s.content} for s in subs]


def build_shared_context(config) -> dict:
    """The whole-clip timeline handed to the agent ONCE (not per issue). Lets the
    agent reason across neighboring lines/spans instead of a single flagged box —
    e.g. compare a diarization onset to a subtitle box to detect original-audio bleed.

    Trimmed to the alignment-relevant fields; the heavy per-line emotion/energy blobs
    stay in the per-issue slice (build_evidence_bundle)."""
    d = config.data_output_folder
    diar = _load_json(os.path.join(d, "speakers_segments_data.json"), [])
    stats_doc = _load_json(os.path.join(d, "build_final_stats.json"), [])
    stats_rows = stats_doc.get("stats", []) if isinstance(stats_doc, dict) else stats_doc
    speakers = _load_json(os.path.join(d, "speakers_data.json"), {})

    timing = [{
        "idx": r.get("index"),
        "box_start_ms": r.get("subtitle_start_ms"),
        "box_end_ms": r.get("subtitle_end_ms"),
        "actual_start_ms": r.get("actual_start_ms"),
        "actual_end_ms": r.get("actual_end_ms"),
        "status": r.get("timing_status"),
        "speed_factor": r.get("applied_speed_factor"),
    } for r in stats_rows]

    return {
        # what each line was TTS-spoken from (the ground truth the audio was built on)
        "subtitles_retranslated": _subs_rows(
            os.path.join(d, "subtitles_retranslated.srt")),
        # source-language transcript, for "was this foreign word actually said?" checks
        "subtitles_original": _subs_rows(os.path.join(d, "subtitles.srt")),
        # where real speech actually is in the original vocal (ground truth for timing)
        "diarization": [{"start": round(s["start"], 3), "end": round(s["end"], 3),
                         "speaker": s.get("speaker")} for s in diar],
        "timing": timing,
        # the cast: who exists and their gender (for change_speaker reasoning)
        "speakers": {k: {"gender": v.get("gender")} for k, v in speakers.items()}
        if isinstance(speakers, dict) else {},
    }


def build_evidence_bundle(issues, config) -> dict:
    """Slim per-issue evidence (keyed by sub_index): only the line-specific numeric
    signals — emotion, vocal energy around the window, and the hallucination mismatch
    signal. The whole-clip timeline (subs/diarization/timing/speakers) is shared once
    via build_shared_context, so it is NOT duplicated here. Issues with no sub_index
    carry no slice; the agent locates them on the shared timeline via their start/end."""
    d = config.data_output_folder
    retrans = _subs_by_idx(os.path.join(d, "subtitles_retranslated.srt"))
    emotions = _load_json(os.path.join(d, "emotions_tags.json"), {})
    diar = _load_json(os.path.join(d, "speakers_segments_data.json"), [])

    bundle: dict[int, dict] = {}
    for issue in issues:
        idx = issue.sub_index
        if idx is None:
            continue
        retrans_content = retrans[idx].content if idx in retrans else ""
        spans = diarization_spans_overlapping(diar, issue.start, issue.end)
        bundle[idx] = {
            "vocal_energy": _vocal_energy_stats(config.vocal_file, issue.start,
                                                issue.end),
            "mismatch_signal": _mismatch_signal(spans, retrans_content),
            "emotion": emotions.get(str(idx), {}),
        }
    return bundle
