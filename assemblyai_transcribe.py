from pydub import AudioSegment
import requests
import time
from openai import OpenAI
import json

SYSTEM_PROMPT = """You are a subtitle quality-control editor. Your task is to fix logical errors in speaker diarization, assign descriptive role names, and flag fragmented subtitles.

TASKS & RULES:
1. DIARIZATION: Identify and correct logical speaker assignment errors. Do not invent a new conversation flow from scratch, but actively fix obvious flaws where the original diarization failed (including splitting a single original label if it mistakenly groups a back-and-forth conversation).
2. ROLES: Replace generic speaker labels with consistent, descriptive English role names based on context. Target exactly {num_speakers} unique roles unless your corrections change the actual speaker count.
3. SEGMENTATION: If a grammatical phrase is unnaturally split across adjacent subtitles by the SAME corrected speaker, set `"merge_into_next": true` on the FIRST subtitle of that split.
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


def apply_ai_segmentation_merges(segments: list[dict], pause_split_threshold: float = 0.4) -> list[dict]:
    if not segments:
        return []

    result = []
    i = 0

    while i < len(segments):
        cur = segments[i].copy()

        if cur.get("merge_into_next") and i + 1 < len(segments):
            nxt = segments[i + 1].copy()

            # Calculate the actual silence gap between the two subtitles
            time_gap = nxt["start"] - cur["end"]

            # If the gap is smaller than our intentional pause threshold, allow the heal
            if cur.get("speaker") == nxt.get("speaker") and time_gap < pause_split_threshold:
                nxt["start"] = cur["start"]
                nxt["text"] = normalize_space(
                    f"{cur.get('text', '')} {nxt.get('text', '')}"
                )
                result.append(nxt)
                i += 2
                continue

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
        # temperature=0,
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
        if text and any(ch.isalpha() for ch in text):
            result.append(sub)

    return result

def assemblyai_transcribe(audio_file_raw: str,  subtitles_file: str, speaker_segments: list, assemblyai_api_key: str,
                          openai_client: OpenAI,
                          openai_model: str,
                          num_speakers: int | None,
                          language: str = 'auto', pause_split_threshold:float = 0.4):


    base_url = "https://api.assemblyai.com"
    headers = {
        "authorization": f"{assemblyai_api_key}",
        "content-type": "application/octet-stream"
    }
    with open(audio_file_raw, "rb") as f:
        response = requests.post(base_url+"/v2/upload", headers=headers, data=f, timeout=(10, 300))
    response.raise_for_status()  # Raise error if upload failed

    upload_response = response.json()
    audio_url = upload_response["upload_url"]


    data = {
        "audio_url": audio_url,
        "speech_models" : ["universal-3-5-pro", "universal-2"],
        "language_detection": True,
        "disfluencies": False
    }
    if language != "auto":
        data["language_detection"] = False
        data["language_code"] = language

    response = requests.post(base_url+"/v2/transcript", json=data, headers=headers, timeout=(10, 300))
    response.raise_for_status()
    transcript_id = response.json()['id']
    while True:
        response = requests.get(base_url + "/v2/transcript/" + transcript_id, headers=headers, timeout=(10, 300))
        response.raise_for_status()  # Raise error if upload failed
        transcription = response.json()
        if transcription['status'] == 'completed':
            break
        elif transcription['status'] == 'error':
            raise RuntimeError(f"Transcription failed: {transcription['error']}")
        else:
            time.sleep(3)

    trans_language = transcription.get("language_code", "ko").split('_')[0] if language == "auto" else language

    # Fetch AssemblyAI Native Sentences (which include internal word arrays)
    sentences_res = requests.get(f"{base_url}/v2/transcript/{transcript_id}/sentences", headers=headers,
                                 timeout=(10, 300))
    sentences_res.raise_for_status()
    sentences_data = sentences_res.json().get("sentences", [])

    segments = []
    max_subtitle_duration = 7  # Max seconds a subtitle should stay on screen
    last_known_speaker = ""

    for sentence in sentences_data:
        text = sentence.get("text", "").strip()
        if not text:
            continue

        s_start = sentence["start"] / 1000.0
        s_end = sentence["end"] / 1000.0
        duration = s_end - s_start

        # # 2. SENTENCE IS SHORT ENOUGH -> Keep it whole
        # if duration <= max_subtitle_duration:
        #     speaker = find_speaker(s_start, s_end, speaker_segments)
        #     if speaker:
        #         last_known_speaker = speaker
        #     segments.append({
        #         "speaker": last_known_speaker,
        #         "start": s_start,
        #         "end": s_end,
        #         "text": text
        #     })
        #     continue

        # 3. SENTENCE IS TOO LONG -> Split it using its internal words
        current_text = []
        segment_start = None
        prev_end = None

        for w in sentence.get("words", []):
            w_text = w["text"]
            w_start = w["start"] / 1000.0
            w_end = w["end"] / 1000.0

            # SANITIZE STRETCHED WORDS:
            # If AssemblyAI stretches a 1-character word to 3 seconds, clamp it.
            actual_w_duration = w_end - w_start
            max_allowed_w_duration = max(0.6, len(w_text) * 0.25)
            if actual_w_duration > max_allowed_w_duration:
                w_end = w_start + max_allowed_w_duration

            if not current_text:
                segment_start = w_start

            # Check split conditions inside the long sentence:
            # - Is there a natural pause?
            # - Have we exceeded the max duration for this chunk?
            time_since_last_word = (w_start - prev_end) if prev_end else 0
            current_chunk_duration = prev_end - segment_start if prev_end else 0

            if prev_end and (
                    time_since_last_word >= pause_split_threshold or current_chunk_duration >= max_subtitle_duration):
                speaker = find_speaker(segment_start, prev_end, speaker_segments)
                if speaker:
                    last_known_speaker = speaker
                segments.append({
                    "speaker": last_known_speaker,
                    "start": segment_start,
                    "end": prev_end,
                    "text": " ".join(current_text)
                })
                # Reset for next chunk
                current_text = [w_text]
                segment_start = w_start
            else:
                current_text.append(w_text)

            prev_end = w_end

        # Append the final chunk of the split sentence
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

    # ---- 5. Sort and save to SRT
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














