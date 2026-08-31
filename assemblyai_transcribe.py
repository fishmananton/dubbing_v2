import requests
import time
import subprocess
import tempfile
import os
from openai import OpenAI
from pydub import AudioSegment
import json
import re

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


def srt_timestamp(seconds):
    h, m = divmod(seconds, 3600)
    m, s = divmod(m, 60)
    return f"{int(h):02}:{int(m):02}:{s:06.3f}".replace('.', ',')


def find_speaker(start_time, end_time, speaker_segments):
    """Find the speaker by calculating maximum overlap duration."""
    max_overlap = 0
    best_speaker = None

    for seg in speaker_segments:
        # Calculate how much the audio segment overlaps with the diarization segment
        overlap_start = max(start_time, seg["start"])
        overlap_end = min(end_time, seg["end"])
        overlap_duration = overlap_end - overlap_start

        if overlap_duration > max_overlap:
            max_overlap = overlap_duration
            best_speaker = seg["speaker"]

    return best_speaker


def get_split_points(diar_segments, max_chunk_sec=600, pause_threshold=10):
    if not diar_segments:
        return []

    # --- Step 1: collect all pauses ---
    pauses = []
    for i in range(len(diar_segments) - 1):
        gap = diar_segments[i + 1]["start"] - diar_segments[i]["end"]
        if gap >= pause_threshold:
            pauses.append(diar_segments[i]["end"])  # cut point at the end of the previous segment

    if not pauses:
        return []
    total_duration = diar_segments[-1]["end"]
    split_points = []

    # --- Step 2: generate ideal targets ---
    targets = [t for t in range(max_chunk_sec, int(total_duration), max_chunk_sec)]

    # --- Step 3: for each target, pick the nearest valid pause ---
    for t in targets:
        nearest = min(pauses, key=lambda p: abs(p - t))
        split_points.append(nearest)

    # --- Step 4: deduplicate and sort ---
    split_points = sorted(set(split_points))

    # --- Step 5: remove last split if final segment too short ---
    if split_points:
        last_split = split_points[-1]
        if total_duration - last_split < max_chunk_sec / 3:
            split_points.pop()

        return split_points


def split_audio_by_points(audio_file, split_points, out_dir):
    audio = AudioSegment.from_file(audio_file)
    split_points = [0] + split_points + [len(audio) / 1000]

    parts = []
    for i in range(len(split_points) - 1):
        start_ms = int(split_points[i] * 1000)
        end_ms = int(split_points[i + 1] * 1000)
        part_file = f"{out_dir}/part_{i + 1:02d}.mp3"
        audio[start_ms:end_ms].export(part_file, format="mp3")
        parts.append({'file': part_file, 'start': start_ms, 'end': end_ms})
    return parts


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def apply_ai_segmentation_merges(segments: list[dict], pause_split_threshold: float = 0.5) -> list[dict]:
    if not segments:
        return []

    result = []
    i = 0

    while i < len(segments):
        cur = segments[i].copy()

        # Continuously merge as long as the chain continues
        while cur.get("merge_into_next") and i + 1 < len(segments):
            nxt = segments[i + 1]
            time_gap = nxt["start"] - cur["end"]

            # Verify same speaker and small gap
            if cur.get("speaker") == nxt.get("speaker") and time_gap < pause_split_threshold:
                cur["end"] = nxt["end"]
                cur["text"] = normalize_space(f"{cur.get('text', '')} {nxt.get('text', '')}")
                # Inherit the next item's merge flag to see if the chain continues
                cur["merge_into_next"] = nxt.get("merge_into_next", False)
                i += 1  # Move pointer past the absorbed segment
            else:
                break  # Stop chaining if speakers don't match or gap is too wide

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


# num2words language code mapping
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


def assemblyai_transcribe_raw(audio_file_raw: str, assemblyai_api_key: str):
    t0 = time.time()

    mp3_file = tempfile.mktemp(suffix=".mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "quiet", "-i", audio_file_raw, "-q:a", "2", mp3_file],
        check=True,
    )
    file_size_mb = os.path.getsize(mp3_file) / 1024 / 1024
    print(f"  [transcribe] mp3 encode: {time.time() - t0:.1f}s (file: {file_size_mb:.1f}MB)")

    base_url = "https://api.assemblyai.com"
    headers = {
        "authorization": f"{assemblyai_api_key}",
        "content-type": "application/octet-stream"
    }
    t1 = time.time()
    try:
        with open(mp3_file, "rb") as f:
            response = requests.post(base_url + "/v2/upload", headers=headers, data=f, timeout=(10, 300))
        response.raise_for_status()
    finally:
        os.unlink(mp3_file)

    upload_response = response.json()
    audio_url = upload_response["upload_url"]
    print(f"  [transcribe] upload to AssemblyAI: {time.time() - t1:.1f}s")

    data = {
        "audio_url": audio_url,
        "speech_models": ["universal-3-5-pro", "universal-2"],
        "language_detection": True,
        "disfluencies": False,
        "format_text": False,
    }

    t2 = time.time()
    response = requests.post(base_url + "/v2/transcript", json=data, headers=headers, timeout=(10, 300))
    response.raise_for_status()
    transcript_id = response.json()['id']

    poll_count = 0
    while True:
        response = requests.get(base_url + "/v2/transcript/" + transcript_id, headers=headers, timeout=(10, 300))
        response.raise_for_status()
        transcription = response.json()
        if transcription['status'] == 'completed':
            break
        elif transcription['status'] == 'error':
            raise RuntimeError(f"Transcription failed: {transcription['error']}")
        else:
            poll_count += 1
            time.sleep(3)
    print(f"  [transcribe] AssemblyAI processing: {time.time() - t2:.1f}s ({poll_count} polls)")

    trans_language = transcription.get("language_code", "ko").split('_')[0]
    words_data = transcription.get("words", [])
    print(f"  [transcribe] got {len(words_data)} words, language: {trans_language}")
    print(f"  [transcribe] raw transcription total: {time.time() - t0:.1f}s")

    return words_data, trans_language


def assemble_transcription(words_data: list, trans_language: str, speaker_segments: list,
                           subtitles_file: str, openai_client: OpenAI, openai_model: str,
                           num_speakers: int | None, pause_split_threshold: float = 0.4):
    t0 = time.time()

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

        # SANITIZE STRETCHED WORDS:
        # 1. Expand numbers purely for the math check
        expanded_text_for_math = _expand_numbers(w_text, trans_language)

        # 2. Fix words floating in silence (start/end outside speech segments)
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

            # Find forward anchor: next word that's in speech
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

            # Find backward anchor: prev word that's in speech
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

            # Check available gap in each direction
            fwd_gap = (fwd_anchor_start - fwd_seg["start"]) if fwd_seg and fwd_anchor_start else 0
            bwd_gap = (bwd_seg["end"] - bwd_anchor_end) if bwd_seg and bwd_anchor_end else 0

            if not start_in_speech and not end_in_speech:
                # Both in silence — prefer forward if gap exists, else backward
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
                # Start in silence, end in speech — find the segment containing end
                for seg in speaker_segments:
                    if seg["start"] <= w_end <= seg["end"]:
                        w_start = seg["start"]
                        break
            elif start_in_speech and not end_in_speech:
                # Start in speech, end in silence — clamp end to segment boundary
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

        # Check split conditions:
        # - Is there a natural pause?
        # - Have we exceeded the max duration for this chunk?
        # - Did the speaker change?
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

    # ---- 4.5 Merge ultra-short same-speaker adjacent subs (dubbing-aware)
    segments = _merge_short_adjacent_subs(segments)

    # ---- 5. Sort and save to SRT
    segments.sort(key=lambda x: x["start"])
    for i, seg in enumerate(segments, 1):
        seg["index"] = i

    unique_speakers = len(set(seg["speaker"] for seg in segments))
    actual_num_speakers = num_speakers if num_speakers is not None else unique_speakers

    t4 = time.time()
    segments = fix_sub_diarization_with_ai(
        openai_client,
        openai_model,
        segments,
        actual_num_speakers,
    )
    print(f"  [transcribe] AI diarization fix: {time.time() - t4:.1f}s ({len(segments)} final segments)")

    with open(subtitles_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{srt_timestamp(seg['start'])} --> {srt_timestamp(seg['end'])}\n")
            f.write(f"{seg['speaker']}: {seg['text']}\n\n")

    print(f"  [transcribe] assembly total: {time.time() - t0:.1f}s")
    return trans_language


def assemblyai_transcribe(audio_file_raw: str, subtitles_file: str, speaker_segments: list, assemblyai_api_key: str,
                          openai_client: OpenAI,
                          openai_model: str,
                          num_speakers: int | None,
                          language: str = 'auto', pause_split_threshold: float = 0.4):
    words_data, trans_language = assemblyai_transcribe_raw(audio_file_raw, assemblyai_api_key)
    return assemble_transcription(words_data, trans_language, speaker_segments, subtitles_file,
                                  openai_client, openai_model, num_speakers, pause_split_threshold)