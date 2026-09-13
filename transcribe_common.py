"""Shared transcription assembly: one aligner + AI-fix + SRT writer for all STT
engines. Each engine front-end returns words_data = [{text, start(ms), end(ms)}]
and this module turns that into diarized, AI-corrected SRT subtitles.

Language-aware: for CJK languages (zh, ja) tokens are joined without spaces and
sentences split on CJK punctuation; Latin languages use space-joins and ASCII
sentence enders."""
import json
import re

from openai import OpenAI

SYSTEM_PROMPT = """You are a subtitle quality-control editor. Your task is to fix logical errors in speaker diarization, assign descriptive role names, and flag fragmented subtitles.

    TASKS & RULES:
    1. DIARIZATION: Identify and correct logical speaker assignment errors. Do not invent a new conversation flow from scratch, but actively fix obvious flaws where the original diarization failed (including splitting a single original label if it mistakenly groups a back-and-forth conversation).
    2. ROLES: Replace generic speaker labels with consistent, descriptive English role names based on context. Target exactly {num_speakers} unique roles unless your corrections change the actual speaker count.
    3. SEGMENTATION: If a grammatical phrase or sentence is split across 2 or MORE consecutive subtitles by the SAME speaker, Set "merge_into_next": true on EVERY subtitle that must attach to the following one.
    4. Maintain exactly consistent role names throughout the entire array. Do not use synonyms for the same character.

    OUTPUT FORMAT:
    Return a STRICT JSON object containing only the `subtitles` array. Do NOT include "text" — only index, speaker, and merge_into_next.

    {
      "subtitles": [
        {
          "index": 1,
          "speaker": "Assigned Role",
          "merge_into_next": false
        }
      ]
    }
    """

CJK_RE = re.compile(r'[一-鿿぀-ヿ]')  # CJK ideographs + hiragana/katakana
CJK_LANGS = {"zh", "ja"}
_CJK_SENTENCE_ENDERS = "。！？"  # 。！？


def is_cjk_lang(lang: str) -> bool:
    return (lang or "")[:2].lower() in CJK_LANGS


def has_cjk(token: str) -> bool:
    return bool(CJK_RE.search(token))


def srt_timestamp(seconds):
    h, m = divmod(seconds, 3600)
    m, s = divmod(m, 60)
    return f"{int(h):02}:{int(m):02}:{s:06.3f}".replace('.', ',')


def find_speaker(start_time, end_time, speaker_segments):
    """Find the speaker by maximum overlap duration."""
    max_overlap = 0
    best_speaker = None
    for seg in speaker_segments:
        overlap_start = max(start_time, seg["start"])
        overlap_end = min(end_time, seg["end"])
        overlap_duration = overlap_end - overlap_start
        if overlap_duration > max_overlap:
            max_overlap = overlap_duration
            best_speaker = seg["speaker"]
    return best_speaker


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def join_words(tokens: list[str], lang: str) -> str:
    """Join word tokens. CJK: no space between two CJK tokens; a space otherwise.
    Latin: plain space-join."""
    if not tokens:
        return ""
    if not is_cjk_lang(lang):
        return " ".join(tokens)
    result = [tokens[0]]
    for prev, cur in zip(tokens, tokens[1:]):
        if has_cjk(prev) and has_cjk(cur):
            result.append(cur)
        else:
            result.append(" " + cur)
    return "".join(result)


def _is_sentence_end(token: str, lang: str) -> bool:
    """True if the token ends a sentence. For CJK also treats 。！？ as enders.
    Latin path guards initials (J.), decimals (3.5), initialisms (U.S.)."""
    t = (token or "").strip().rstrip('"\')]}»”’')
    if not t:
        return False
    if is_cjk_lang(lang) and t[-1] in _CJK_SENTENCE_ENDERS:
        return True
    if t.endswith(("?", "!")):
        return True
    if not t.endswith("."):
        return False
    core = t[:-1]
    parts = core.split()
    last = parts[-1] if parts else core
    stripped = last.strip('"\'([{«“‘')
    if len(stripped) == 1 and stripped.isalpha():
        return False
    if any(ch.isdigit() for ch in stripped):
        return False
    if "." in stripped:
        return False
    return True


def apply_ai_segmentation_merges(segments: list[dict], pause_split_threshold: float = 0.5) -> list[dict]:
    if not segments:
        return []
    result = []
    i = 0
    while i < len(segments):
        cur = segments[i].copy()
        while cur.get("merge_into_next") and i + 1 < len(segments):
            nxt = segments[i + 1]
            time_gap = nxt["start"] - cur["end"]
            if cur.get("speaker") == nxt.get("speaker") and time_gap < pause_split_threshold:
                cur["end"] = nxt["end"]
                cur["text"] = normalize_space(f"{cur.get('text', '')} {nxt.get('text', '')}")
                cur["merge_into_next"] = nxt.get("merge_into_next", False)
                i += 1
            else:
                break
        result.append(cur)
        i += 1
    return result


def fix_sub_diarization_with_ai(client: OpenAI, model, srt_res: list, num_speakers: int):
    text = f"Subtitles:\n{json.dumps(srt_res, ensure_ascii=False, indent=2)}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "developer", "content": SYSTEM_PROMPT.replace("{num_speakers}", f"{num_speakers}")},
            {"role": "user", "content": text},
        ],
        reasoning_effort="medium",
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    data = json.loads(raw)

    fix_map = {}
    for item in data['subtitles']:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if idx is None:
            continue
        try:
            idx = int(idx)
        except Exception:
            continue
        fix_map[idx] = {
            "speaker": str(item.get("speaker", "")).strip(),
            "merge_into_next": bool(item.get("merge_into_next", False)),
        }

    corrected = []
    for row in srt_res:
        idx = row.get("index")
        fix = fix_map.get(idx)
        new_row = row.copy()
        if fix:
            if fix["speaker"]:
                new_row["speaker"] = fix["speaker"]
            new_row["merge_into_next"] = fix["merge_into_next"]
        else:
            new_row["merge_into_next"] = False
        corrected.append(new_row)

    corrected = apply_ai_segmentation_merges(corrected)
    for i, row in enumerate(corrected, 1):
        row["index"] = i
        row.pop("merge_into_next", None)
    return corrected


def filter_speakable_subs(subs: list[dict]) -> list[dict]:
    result = []
    for sub in subs:
        text = sub.get("text", "")
        if text and any(ch.isalnum() for ch in text):
            result.append(sub)
    return result


_NUM2WORDS_LANG_MAP: dict[str, str] = {
    "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it",
    "pt": "pt", "nl": "nl", "pl": "pl", "ru": "ru", "uk": "uk",
    "sv": "sv", "cs": "cz", "tr": "tr", "ar": "ar", "he": "he",
    "ko": "ko", "ja": "ja", "vi": "vi", "th": "th",
}

_DIGIT_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _expand_numbers(text: str, lang: str) -> str:
    if not _DIGIT_RE.search(text):
        return text
    code = lang[:2].lower()
    n2w_lang = _NUM2WORDS_LANG_MAP.get(code)
    if n2w_lang is None:
        return text
    try:
        import num2words
    except ImportError:
        return text

    def _replace(match: re.Match) -> str:
        raw = match.group(0).replace(",", ".")
        try:
            val = float(raw)
            if val == int(val) and "." not in match.group(0) and "," not in match.group(0):
                return num2words.num2words(int(val), lang=n2w_lang)
            return num2words.num2words(val, lang=n2w_lang)
        except (ValueError, NotImplementedError, OverflowError):
            return match.group(0)

    return _DIGIT_RE.sub(_replace, text)


def _merge_short_adjacent_subs(segments: list[dict], max_dur: float = 0.4, max_gap: float = 1.0,
                               lang: str = "en") -> list[dict]:
    """Merge ultra-short same-speaker adjacent subs that are undubbable individually."""
    if not segments:
        return segments
    merged = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        dur = seg["end"] - seg["start"]
        if dur < max_dur and i + 1 < len(segments):
            nxt = segments[i + 1]
            nxt_dur = nxt["end"] - nxt["start"]
            gap = nxt["start"] - seg["end"]
            if seg["speaker"] == nxt["speaker"] and gap < max_gap and nxt_dur < max_dur:
                merged.append({
                    **seg,
                    "end": nxt["end"],
                    "text": join_words([seg["text"], nxt["text"]], lang),
                })
                i += 2
                continue
        merged.append(seg)
        i += 1
    return merged


def _merge_same_speaker_below_threshold(segments: list[dict], pause_split_threshold: float,
                                        max_dur: float = 7.0) -> list[dict]:
    """Re-join adjacent same-speaker subs whose gap is below the split threshold,
    undoing sentence-splits the AI kept on one speaker. Capped at max_dur."""
    if not segments:
        return segments
    merged = []
    for seg in segments:
        if merged:
            prev = merged[-1]
            gap = seg["start"] - prev["end"]
            if (prev["speaker"] == seg["speaker"] and gap < pause_split_threshold
                    and (seg["end"] - prev["start"]) <= max_dur):
                prev["end"] = seg["end"]
                prev["text"] = normalize_space(f"{prev['text']} {seg['text']}")
                continue
        merged.append(seg.copy())
    return merged


def assemble_transcription(words_data: list, trans_language: str, speaker_segments: list,
                           subtitles_file: str, openai_client: OpenAI, openai_model: str,
                           num_speakers: int | None, pause_split_threshold: float = 0.4):
    """Turn engine word output (ms timings) into diarized, AI-corrected SRT.
    Shared across all STT engines; language-aware for CJK joining/splitting."""
    segments = []
    max_subtitle_duration = 7
    last_known_speaker = ""

    current_text = []
    segment_start = None
    prev_end = None

    for w in words_data:
        w_text = w["text"]
        w_start = w["start"] / 1000.0
        w_end = w["end"] / 1000.0

        expanded_text_for_math = _expand_numbers(w_text, trans_language)

        start_in_speech = any(seg["start"] <= w_start <= seg["end"] for seg in speaker_segments)
        end_in_speech = any(seg["start"] <= w_end <= seg["end"] for seg in speaker_segments)

        if not start_in_speech or not end_in_speech:
            if not start_in_speech and not end_in_speech:
                min_dist = min(
                    (min(abs(w_start - seg["end"]), abs(w_start - seg["start"])) for seg in speaker_segments),
                    default=999
                )
                if min_dist > 0.5:
                    continue

            w_idx = words_data.index(w)

            fwd_anchor_start = None
            fwd_seg = None
            for fw in words_data[w_idx + 1:]:
                fw_start = fw["start"] / 1000.0
                for seg in speaker_segments:
                    if seg["start"] <= fw_start <= seg["end"]:
                        fwd_anchor_start = fw_start
                        fwd_seg = seg
                        break
                if fwd_anchor_start is not None:
                    break

            bwd_anchor_end = None
            bwd_seg = None
            for bw in reversed(words_data[:w_idx]):
                bw_end = bw["end"] / 1000.0
                for seg in speaker_segments:
                    if seg["start"] <= bw_end <= seg["end"]:
                        bwd_anchor_end = bw_end
                        bwd_seg = seg
                        break
                if bwd_anchor_end is not None:
                    break

            fwd_gap = (fwd_anchor_start - fwd_seg["start"]) if fwd_seg and fwd_anchor_start else 0
            bwd_gap = (bwd_seg["end"] - bwd_anchor_end) if bwd_seg and bwd_anchor_end else 0

            if not start_in_speech and not end_in_speech:
                if fwd_gap > 0.05 and fwd_seg:
                    w_start = fwd_seg["start"]
                    w_end = fwd_anchor_start
                elif bwd_gap > 0.05 and bwd_seg:
                    w_start = bwd_anchor_end
                    w_end = bwd_seg["end"]
                elif fwd_seg:
                    w_start = fwd_seg["start"]
                    w_end = fwd_anchor_start if fwd_anchor_start else fwd_seg["end"]
            elif not start_in_speech and end_in_speech:
                for seg in speaker_segments:
                    if seg["start"] <= w_end <= seg["end"]:
                        w_start = seg["start"]
                        break
            elif start_in_speech and not end_in_speech:
                for seg in speaker_segments:
                    if seg["start"] <= w_start <= seg["end"]:
                        w_end = seg["end"]
                        break

        actual_w_duration = w_end - w_start
        max_allowed_w_duration = max(0.6, len(expanded_text_for_math) * 0.25)
        if actual_w_duration > max_allowed_w_duration:
            w_end = w_start + max_allowed_w_duration

        if not current_text:
            segment_start = w_start

        time_since_last_word = (w_start - prev_end) if prev_end else 0
        current_chunk_duration = prev_end - segment_start if prev_end else 0
        word_speaker = find_speaker(w_start, w_end, speaker_segments)
        speaker_changed = (word_speaker and last_known_speaker and word_speaker != last_known_speaker) if current_text else False

        if prev_end and (
                speaker_changed or time_since_last_word >= pause_split_threshold or current_chunk_duration >= max_subtitle_duration):
            speaker = find_speaker(segment_start, prev_end, speaker_segments)
            if speaker:
                last_known_speaker = speaker
            segments.append({
                "speaker": last_known_speaker,
                "start": segment_start,
                "end": prev_end,
                "text": join_words(current_text, trans_language),
            })
            if word_speaker:
                last_known_speaker = word_speaker
            current_text = [w_text]
            segment_start = w_start
        else:
            current_text.append(w_text)

        prev_end = w_end

        if current_text and _is_sentence_end(w_text, trans_language):
            speaker = find_speaker(segment_start, w_end, speaker_segments)
            if speaker:
                last_known_speaker = speaker
            segments.append({
                "speaker": last_known_speaker,
                "start": segment_start,
                "end": w_end,
                "text": join_words(current_text, trans_language),
            })
            current_text = []
            segment_start = None

    if current_text:
        speaker = find_speaker(segment_start, prev_end, speaker_segments)
        if speaker:
            last_known_speaker = speaker
        segments.append({
            "speaker": last_known_speaker,
            "start": segment_start,
            "end": prev_end,
            "text": join_words(current_text, trans_language),
        })

    segments = filter_speakable_subs(segments)
    segments = _merge_short_adjacent_subs(segments, lang=trans_language)

    segments.sort(key=lambda x: x["start"])
    for i, seg in enumerate(segments, 1):
        seg["index"] = i

    unique_speakers = len(set(seg["speaker"] for seg in segments))
    actual_num_speakers = num_speakers if num_speakers is not None else unique_speakers

    segments = fix_sub_diarization_with_ai(openai_client, openai_model, segments, actual_num_speakers)
    segments = _merge_same_speaker_below_threshold(segments, pause_split_threshold)

    with open(subtitles_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{srt_timestamp(seg['start'])} --> {srt_timestamp(seg['end'])}\n")
            f.write(f"{seg['speaker']}: {seg['text']}\n\n")

    return trans_language
