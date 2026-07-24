from pydub import AudioSegment
import requests
import time
from openai import OpenAI
import json

SYSTEM_PROMPT = """You are given subtitles with imperfect speaker diarization and imperfect subtitle segmentation.

Tasks:
1. Correct speaker labels.
2. Replace generic labels with consistent English role names.
3. Use exactly {num_speakers} speakers unless the input clearly proves fewer are present.
4. Fix only obvious subtitle-boundary problems that break grammar, meaning, or natural speech.

Rules:
- Keep the same character under the same speaker name.
- Minimize speaker names; do not invent unnecessary roles.
- If a speaker appears for less than 5 seconds total, treat it as likely diarization noise and reassign it when context supports that.
- Preserve the original text as much as possible.
- Do not translate, summarize, or freely paraphrase.
- Only adjust adjacent subtitles when needed for grammar or natural flow.
- Only merge adjacent subtitles when they have the same corrected speaker.
- Do not create new subtitle indices.

merge_into_next:
- Set true only when this subtitle should be removed and its own text prepended to the next subtitle.
- If merge_into_next is true, keep this subtitle's own corrected text only.
- Do NOT repeat this subtitle's text inside the next subtitle.
- The next subtitle should contain only its own corrected text.

Return STRICT JSON ONLY.
Return one object per input subtitle index:

[
  {
    "index": <subtitle index>,
    "speaker": "<speaker_name>",
    "text": "<corrected subtitle text without speaker label>",
    "merge_into_next": false
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


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def apply_ai_segmentation_merges(segments: list[dict]) -> list[dict]:


    if not segments:
        return []

    result = []
    i = 0

    while i < len(segments):
        cur = segments[i].copy()

        if cur.get("merge_into_next") and i + 1 < len(segments):
            nxt = segments[i + 1].copy()

            # Safety: only merge same speaker after AI correction.
            if cur.get("speaker") == nxt.get("speaker"):
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
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.replace("{num_speakers}", f"{num_speakers}"),
            },
            {"role": "user", "content": text},
        ],
    )

    raw = response.choices[0].message.content.strip()
    data = json.loads(raw)

    fix_map = {}
    for item in data:
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
        "speech_models" : ["universal-3-pro", "universal-2"],
        "language_detection": True
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

    words = transcription["words"]
    trans_language  =transcription["language_code"].split('_')[0] if language == "auto" else language

    # ---- 4. Build merged segments based on time + speaker
    segments = []
    current_text = []
    current_speaker = None
    segment_start = None
    prev_end = None

    for w in words:
            word = w["text"]
            start = float(w["start"]/1000)
            end = float(w["end"]/1000)
            speaker = find_speaker((start + end) / 2, speaker_segments)  # midpoint lookup
            if speaker == "Unknown":
                speaker = current_speaker if current_speaker is not None else "Speaker_01"
            # Start new segment if time gap or speaker change
            if (prev_end and start - prev_end >= pause_split_threshold) or (speaker != current_speaker):
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














