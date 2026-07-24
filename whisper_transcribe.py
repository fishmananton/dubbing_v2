from pydub import AudioSegment
import requests
import time
from openai import OpenAI
import json
import os
from pydub.silence import detect_nonsilent
import langcodes

SYSTEM_PROMPT = """You are given subtitles with imperfect speaker diarization.

Your task:
1. Correct speaker assignments based on conversation context and flow.
2. Replace generic speaker labels (e.g., SPEAKER_01, Person_1) with consistent, meaningful role names in English.
3. Ensure the same character always uses the same role name throughout the entire dialogue.
4. Minimize the number of unique roles. Do NOT create unnecessary new roles.
5. Number of speakers is known to be {num_speakers}.

Correction rules:
- If a speaker appears for less than 5 seconds total, assume it is a misclassification and reassign those lines to the most contextually appropriate existing role.
- Use dialogue logic (who responds to whom, tone, relationships, etc.) to infer correct speakers.
- Preserve narrative consistency (e.g., family roles, hierarchy, tone).

Output requirements:
- Return STRICT JSON ONLY.
- Do not include explanations or extra text.
- Output format:

[
  {
    "index": <subtitle index>,
    "speaker": "<role_name>"
  }
]
"""

def srt_timestamp(seconds):
    h, m = divmod(seconds, 3600)
    m, s = divmod(m, 60)
    return f"{int(h):02}:{int(m):02}:{s:06.3f}".replace('.', ',')


def find_speaker(time, speaker_segments):
    """Find which speaker segment covers a given timestamp."""
    for seg in speaker_segments:
        if seg["start"] <= time <= seg["end"]:
            return seg["speaker"]
    return "Unknown"


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


def fix_sub_diarization_with_ai(client:OpenAI, model,  srt_res:list,num_speakers:int):
    text = f"Subtitles:\n{json.dumps(srt_res, ensure_ascii=False, indent=2)}"
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.replace("{num_speakers}", f"{num_speakers}")},
            {"role": "user", "content": text},
        ],
    )
    fix_map = {item["index"]: item["speaker"] for item in json.loads(response.choices[0].message.content.strip())}
    corrected = []
    for row in srt_res:
        new_row = row.copy()
        if row.get("index") in fix_map:
            new_row["speaker"] = fix_map[row["index"]]
        corrected.append(new_row)

    return corrected


def filter_speakable_subs(subs: list[dict]) -> list[dict]:
    result = []

    for sub in subs:
        text = sub.get("text", "")
        if text and any(ch.isalpha() for ch in text):
            result.append(sub)

    return result

def merge_segments_by_speaker(segments, pause_split_threshold=0.8):
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        prev = merged[-1]

        same_speaker = seg["speaker"] == prev["speaker"]
        short_pause = (seg["start"] - prev["end"]) < pause_split_threshold

        if same_speaker and short_pause:
            # Extend previous segment
            prev["end"] = seg["end"]
            prev["text"] = f'{prev["text"].rstrip()} {seg["text"].lstrip()}'.strip()
        else:
            merged.append(seg.copy())

    # Optional: reindex after merging
    for i, seg in enumerate(merged, start=1):
        seg["index"] = i

    return merged

def get_leading_silence_seconds(
    wav_path,
    silence_thresh=-40,
    min_silence_len=100
):
    """
    Returns leading silence duration in seconds.

    silence_thresh: dBFS threshold (lower = stricter silence)
    min_silence_len: ms of silence to consider
    """
    audio = AudioSegment.from_file(wav_path, format="wav")

    nonsilent = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )

    if not nonsilent:
        # whole file is silence
        return len(audio) / 1000.0

    first_sound_start_ms = nonsilent[0][0]

    return first_sound_start_ms / 1000.0

def groq_whisper_large_v3_transcribe(
    audio_file_raw: str,
    subtitles_file: str,
    speaker_segments: list,
    groq_api_key: str,
    openai_client,
    openai_model: str,
    num_speakers: int | None,
    language: str = "auto",
    pause_split_threshold: float = 0.25,
):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
    }

    data = {
        "model": "whisper-large-v3",
        "response_format": "verbose_json",
        "temperature": "0",
        # Groq supports "word" and "segment" timestamp granularities
        # when response_format is verbose_json.
        "timestamp_granularities[]": ["word"],
    }

    # Groq accepts ISO-639-1 language codes like "en", "ko", etc.
    # If omitted, it auto-detects.
    if language != "auto":
        data["language"] = language

    with open(audio_file_raw, "rb") as f:
        files = {
            "file": (os.path.basename(audio_file_raw), f, "application/octet-stream")
        }

        response = requests.post(
            url,
            headers=headers,
            data=data,
            files=files,
            timeout=(10, 300),
        )

    response.raise_for_status()
    transcription = response.json()

    # Groq verbose_json returns word timestamps in seconds.
    words = transcription.get("words", [])

    raw_lang = transcription.get("language")

    if raw_lang:
        trans_language = langcodes.find(raw_lang).language
    else:
        trans_language = language if language != "auto" else "auto"

    # ---- 4. Build merged segments based on time + speaker
    segments = []
    current_text = []
    current_speaker = None
    segment_start = None
    prev_end = None

    #fix whisper bug with 1st word

    first_offset = get_leading_silence_seconds(audio_file_raw)
    if words:
        first_word = words[0]

        if first_word["start"] < 0.05 and first_offset > 0.2 and first_offset < first_word["end"]:
            first_word["start"] = first_offset

    for w in words:
        word = w["word"]
        start = float(w["start"])
        end = float(w["end"])
        speaker = find_speaker((start + end) / 2, speaker_segments)  # midpoint lookup
        if speaker == "Unknown":
            speaker = current_speaker if current_speaker is not None else "Speaker_01"

        # Start new segment if time gap or speaker change
        if (prev_end is not None and start - prev_end >= pause_split_threshold) or (speaker != current_speaker):
            if current_text:
                segments.append({
                    "speaker": current_speaker,
                    "start": segment_start,
                    "end": prev_end,
                    "text": " ".join(current_text)
                })
            current_text = [word]
            segment_start = start
            current_speaker = speaker
        else:
            if not current_text:
                segment_start = start
                current_speaker = speaker
            current_text.append(word)

        prev_end = end

    # Add final segment
    if current_text:
        segments.append({
            "speaker": current_speaker,
            "start": segment_start,
            "end": prev_end,
            "text": " ".join(current_text)
        })

    segments = filter_speakable_subs(segments)

    # ---- 5. Sort and save to SRT
    segments.sort(key=lambda x: x["start"])
    for i, seg in enumerate(segments, 1):
        seg["index"] = i

    unique_speakers = len(set(seg["speaker"] for seg in segments))
    actual_num_speakers = num_speakers if num_speakers is not None else unique_speakers

    if actual_num_speakers > 1:
        segments = fix_sub_diarization_with_ai(openai_client, openai_model, segments, num_speakers)

    segments = merge_segments_by_speaker(segments, pause_split_threshold=pause_split_threshold)


    with open(subtitles_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{srt_timestamp(seg['start'])} --> {srt_timestamp(seg['end'])}\n")
            f.write(f"{seg['speaker']}: {seg['text']}\n\n")

    return trans_language



