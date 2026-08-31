from pydub import AudioSegment
from pydub.generators import WhiteNoise
import requests
import tempfile
import os
import json
import re

from openai import OpenAI

SYSTEM_PROMPT = """You are a subtitle quality-control editor. Your task is to fix logical errors in speaker diarization, assign descriptive role names, and flag fragmented subtitles.

    TASKS & RULES:
    1. DIARIZATION: Identify and correct logical speaker assignment errors. Do not invent a new conversation flow from scratch, but actively fix obvious flaws where the original diarization failed (including splitting a single original label if it mistakenly groups a back-and-forth conversation).
    2. ROLES: Replace generic speaker labels with consistent, descriptive English role names based on context. Target exactly {num_speakers} unique roles unless your corrections change the actual speaker count.
    3. SEGMENTATION: If a grammatical phrase or sentence is split across 2 or MORE consecutive subtitles by the SAME speaker, Set "merge_into_next": true on EVERY subtitle that must attach to the following one.
    4. TEXT PRESERVATION: Keep the `"text"` exactly as provided. Do NOT translate or paraphrase. Do NOT combine the text yourself when flagging a merge.
    5. Maintain exactly consistent role names throughout the entire array. Do not use synonyms for the same character.

    OUTPUT FORMAT:
    Return a STRICT JSON object containing only the `subtitles` array.

    {
      "subtitles": [
        {
          "index": 1,
          "speaker": "Assigned Role",
          "text": "original text strictly preserved",
          "merge_into_next": false
        }
      ]
    }
    """


def srt_timestamp(seconds):
    h, m = divmod(seconds, 3600)
    m, s = divmod(m, 60)
    return f"{int(h):02}:{int(m):02}:{s:06.3f}".replace('.', ',')


def find_speaker(start_time, end_time, speaker_segments):
    """Find the speaker by calculating maximum overlap duration."""
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


def fix_sub_diarization_with_ai(
        client: OpenAI,
        model,
        srt_res: list,
        num_speakers: int,
):
    text = f"Subtitles:\n{json.dumps(srt_res, ensure_ascii=False, indent=2)}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "developer",
                "content": SYSTEM_PROMPT.replace("{num_speakers}", f"{num_speakers}"),
            },
            {"role": "user", "content": text},
        ],
        reasoning_effort="medium",
        response_format={"type": "json_object"}
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
            "text": str(item.get("text", "")).strip(),
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

            if fix["text"]:
                new_row["text"] = fix["text"]

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


def _merge_short_adjacent_subs(segments: list[dict], max_dur: float = 0.4, max_gap: float = 1.0) -> list[dict]:
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
                    "text": seg["text"] + " " + nxt["text"],
                })
                i += 2
                continue
        merged.append(seg)
        i += 1
    return merged


def deepgram_transcribe(audio_file_raw: str, subtitles_file: str, speaker_segments: list, deepgram_api_key: str,
                        openai_client: OpenAI,
                        openai_model: str,
                        num_speakers: int | None,
                        language: str = 'auto', pause_split_threshold: float = 0.4):
    # Fill only silent regions with -50dB white noise to prevent ASR timing bugs on digital silence
    from pydub.silence import detect_silence
    audio = AudioSegment.from_file(audio_file_raw)
    silent_ranges = detect_silence(audio, min_silence_len=200, silence_thresh=-50)
    if silent_ranges:
        if silent_ranges[0][0] == 0:
            silent_ranges = silent_ranges[1:]
        if silent_ranges and silent_ranges[-1][1] >= len(audio) - 50:
            silent_ranges = silent_ranges[:-1]
    if silent_ranges:
        noise_ref = WhiteNoise().to_audio_segment(duration=1)
        target_dbfs = -50
        gain_adjust = target_dbfs - noise_ref.dBFS
        for start_ms, end_ms in silent_ranges:
            chunk_noise = WhiteNoise().to_audio_segment(duration=end_ms - start_ms).apply_gain(gain_adjust)
            chunk_noise = chunk_noise.set_channels(audio.channels).set_frame_rate(audio.frame_rate).set_sample_width(audio.sample_width)
            audio = audio.overlay(chunk_noise, position=start_ms)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio.export(tmp.name, format="wav")
        tmp.close()
        upload_file = tmp.name
    else:
        upload_file = audio_file_raw

    base_url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {deepgram_api_key}",
        "Content-Type": "audio/wav",
    }

    params = {
        "model": "nova-3",
        "smart_format": "false",
        "punctuate": "true",
    }
    if language != "auto":
        params["language"] = language
    else:
        params["detect_language"] = "true"

    try:
        with open(upload_file, "rb") as f:
            response = requests.post(
                base_url,
                headers=headers,
                params=params,
                data=f,
                timeout=(10, 600),
            )
        response.raise_for_status()
    finally:
        if upload_file != audio_file_raw:
            os.unlink(upload_file)

    result = response.json()
    channel = result["results"]["channels"][0]
    alternative = channel["alternatives"][0]

    detected_language = channel.get("detected_language", language)
    trans_language = detected_language if language == "auto" else language

    # Use word-level data for precise segmentation (same approach as assemblyai)
    words_data = alternative.get("words", [])
    if not words_data:
        raise RuntimeError("Deepgram transcription returned no words.")

    segments = []
    max_subtitle_duration = 7
    last_known_speaker = ""

    current_text = []
    segment_start = None
    prev_end = None

    for w in words_data:
        w_text = w["word"]
        w_start = float(w["start"])
        w_end = float(w["end"])

        # SANITIZE STRETCHED WORDS:
        # 1. Expand numbers purely for the math check
        expanded_text_for_math = _expand_numbers(w_text, trans_language)

        # 2. Fix words floating in silence (start/end outside speech segments)
        start_in_speech = any(seg["start"] <= w_start <= seg["end"] for seg in speaker_segments)
        end_in_speech = any(seg["start"] <= w_end <= seg["end"] for seg in speaker_segments)

        if not start_in_speech or not end_in_speech:
            w_idx = words_data.index(w)

            # Find forward anchor: next word that's in speech
            fwd_anchor_start = None
            fwd_seg = None
            for fw in words_data[w_idx + 1:]:
                fw_start = float(fw["start"])
                for seg in speaker_segments:
                    if seg["start"] <= fw_start <= seg["end"]:
                        fwd_anchor_start = fw_start
                        fwd_seg = seg
                        break
                if fwd_anchor_start is not None:
                    break

            # Find backward anchor: prev word that's in speech
            bwd_anchor_end = None
            bwd_seg = None
            for bw in reversed(words_data[:w_idx]):
                bw_end = float(bw["end"])
                for seg in speaker_segments:
                    if seg["start"] <= bw_end <= seg["end"]:
                        bwd_anchor_end = bw_end
                        bwd_seg = seg
                        break
                if bwd_anchor_end is not None:
                    break

            # Check available gap in each direction
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

        # 3. Check duration against the expanded mathematical length
        actual_w_duration = w_end - w_start
        max_allowed_w_duration = max(0.6, len(expanded_text_for_math) * 0.25)

        if actual_w_duration > max_allowed_w_duration:
            w_end = w_start + max_allowed_w_duration

        if not current_text:
            segment_start = w_start

        # Check split conditions
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
                "text": " ".join(current_text)
            })
            if word_speaker:
                last_known_speaker = word_speaker
            current_text = [w_text]
            segment_start = w_start
        else:
            current_text.append(w_text)

        prev_end = w_end

    # Append the final chunk
    if current_text:
        speaker = find_speaker(segment_start, prev_end, speaker_segments)
        if speaker:
            last_known_speaker = speaker
        segments.append({
            "speaker": last_known_speaker,
            "start": segment_start,
            "end": prev_end,
            "text": " ".join(current_text)
        })

    segments = filter_speakable_subs(segments)

    segments = _merge_short_adjacent_subs(segments)

    segments.sort(key=lambda x: x["start"])
    for i, seg in enumerate(segments, 1):
        seg["index"] = i

    unique_speakers = len(set(seg["speaker"] for seg in segments))
    actual_num_speakers = num_speakers if num_speakers is not None else unique_speakers

    segments = fix_sub_diarization_with_ai(
        openai_client,
        openai_model,
        segments,
        actual_num_speakers,
    )

    with open(subtitles_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{srt_timestamp(seg['start'])} --> {srt_timestamp(seg['end'])}\n")
            f.write(f"{seg['speaker']}: {seg['text']}\n\n")

    return trans_language
