import json
import requests
import time

from openai import OpenAI


SYSTEM_PROMPT = """You are a subtitle quality-control editor. Your task is to fix logical errors in speaker diarization, assign descriptive role names, and flag fragmented subtitles.

TASKS & RULES:
1. DIARIZATION: Identify and correct logical speaker assignment errors. Do not invent a new conversation flow from scratch, but actively fix obvious flaws where the original diarization failed (including splitting a single original label if it mistakenly groups a back-and-forth conversation).
2. ROLES: Replace generic speaker labels with consistent, descriptive English role names based on context. Target exactly {num_speakers} unique roles unless your corrections change the actual speaker count.
3. SEGMENTATION: If a grammatical phrase is unnaturally split across adjacent subtitles by the SAME corrected speaker, set `"merge_into_next": true` on the FIRST subtitle of that split.
4. TEXT PRESERVATION: Keep the `"text"` exactly as provided. Do NOT translate or paraphrase. Do NOT combine the text yourself when flagging a merge.

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


def find_speaker(time_sec, speaker_segments):
    for seg in speaker_segments:
        if seg["start"] <= time_sec <= seg["end"]:
            return seg["speaker"]
    return "Unknown"


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
        messages=[
            {
                "role": "developer",
                "content": SYSTEM_PROMPT.replace("{num_speakers}", f"{num_speakers}"),
            },
            {"role": "user", "content": text},
        ],
        reasoning_effort="low",
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


def deepgram_transcribe(
    audio_file_raw: str,
    subtitles_file: str,
    speaker_segments: list,
    deepgram_api_key: str,
    openai_client: OpenAI,
    openai_model: str,
    num_speakers: int | None,
    language: str = "auto",
):
    base_url = "https://api.deepgram.com/v1/listen"

    headers = {
        "Authorization": f"Token {deepgram_api_key}",
        "Content-Type": "audio/wav",
    }

    params = {
        "model": "nova-3",
        "smart_format": "true",
        "punctuate": "true",
        "paragraphs": "true",
    }
    if language != "auto":
        params["language"] = language
    else:
        params["detect_language"] = "true"

    with open(audio_file_raw, "rb") as f:
        response = requests.post(
            base_url,
            headers=headers,
            params=params,
            data=f,
            timeout=(10, 600),
        )
    response.raise_for_status()

    result = response.json()
    channel = result["results"]["channels"][0]
    alternative = channel["alternatives"][0]

    paragraphs_data = alternative.get("paragraphs", {}).get("paragraphs", [])
    if not paragraphs_data:
        raise RuntimeError("Deepgram transcription returned no paragraphs.")

    detected_language = channel.get("detected_language", language)
    trans_language = detected_language if language == "auto" else language

    # Build segments from sentences, assign speakers via diarization
    segments = []
    for paragraph in paragraphs_data:
        for sentence in paragraph.get("sentences", []):
            start = float(sentence["start"])
            end = float(sentence["end"])
            text = sentence["text"].strip()
            if not text:
                continue

            speaker = find_speaker((start + end) / 2, speaker_segments)
            if speaker == "Unknown":
                speaker = "Speaker_01"

            segments.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
            })

    segments = filter_speakable_subs(segments)

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
