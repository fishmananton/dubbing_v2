import requests
import time
from openai import OpenAI

from assemblyai_transcribe import (
    srt_timestamp,
    find_speaker,
    fix_sub_diarization_with_ai,
    filter_speakable_subs,
)

BASE_URL = "https://asr.api.speechmatics.com/v2"

LANGUAGE_MAP = {
    "en": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "nl": "nl",
    "sv": "sv",
    "da": "da",
    "no": "no",
    "fi": "fi",
    "pl": "pl",
    "ru": "ru",
    "zh": "cmn",
    "ja": "ja",
    "ko": "ko",
    "ar": "ar",
    "hi": "hi",
    "tr": "tr",
    "uk": "uk",
    "cs": "cs",
    "ro": "ro",
    "hu": "hu",
    "el": "el",
    "he": "he",
    "th": "th",
    "vi": "vi",
    "id": "id",
    "ms": "ms",
    "bg": "bg",
    "hr": "hr",
    "sk": "sk",
    "sl": "sl",
    "lt": "lt",
    "lv": "lv",
    "et": "et",
    "ca": "ca",
    "gl": "gl",
    "eu": "eu",
}


def speechmatics_transcribe(
    audio_file_raw: str,
    subtitles_file: str,
    speaker_segments: list,
    speechmatics_api_key: str,
    openai_client: OpenAI,
    openai_model: str,
    num_speakers: int | None,
    language: str = "auto",
    pause_split_threshold: float = 0.4,
):
    headers = {"Authorization": f"Bearer {speechmatics_api_key}"}

    transcription_config = {"language": "auto", "model": "enhanced"}
    if language != "auto":
        sm_lang = LANGUAGE_MAP.get(language, language)
        transcription_config["language"] = sm_lang

    job_config = {
        "type": "transcription",
        "transcription_config": transcription_config,
    }

    with open(audio_file_raw, "rb") as f:
        files = {
            "data_file": (audio_file_raw.split("/")[-1], f, "application/octet-stream"),
            "config": (None, __import__("json").dumps(job_config), "application/json"),
        }
        response = requests.post(
            f"{BASE_URL}/jobs", headers=headers, files=files, timeout=(30, 300)
        )
    response.raise_for_status()
    job_id = response.json()["id"]

    while True:
        response = requests.get(
            f"{BASE_URL}/jobs/{job_id}", headers=headers, timeout=(10, 60)
        )
        response.raise_for_status()
        job_status = response.json()["job"]
        if job_status["status"] == "done":
            break
        elif job_status["status"] == "rejected":
            raise RuntimeError(
                f"Speechmatics transcription rejected: {job_status.get('errors', '')}"
            )
        time.sleep(3)

    response = requests.get(
        f"{BASE_URL}/jobs/{job_id}/transcript",
        headers=headers,
        params={"format": "json-v2"},
        timeout=(10, 120),
    )
    response.raise_for_status()
    transcript = response.json()

    try:
        trans_language = transcript["metadata"]["language_identification"]["predicted_language"]
    except (KeyError, TypeError):
        trans_language = language
    if trans_language == "auto":
        trans_language = "en"
    # Speechmatics may return full locale like "en" or mapped code
    # Normalize back to our 2-letter codes
    reverse_map = {v: k for k, v in LANGUAGE_MAP.items()}
    trans_language = reverse_map.get(trans_language, trans_language.split("-")[0])

    tokens = []
    for result in transcript.get("results", []):
        rtype = result.get("type")
        if rtype not in ("word", "punctuation"):
            continue
        tokens.append(
            {
                "text": result["alternatives"][0]["content"],
                "start": result["start_time"],
                "end": result["end_time"],
                "type": rtype,
                "attaches_to": result.get("attaches_to"),
                "is_eos": result.get("is_eos", False),
            }
        )

    segments = []
    current_text = []
    current_speaker = None
    segment_start = None
    prev_end = None

    def flush_segment():
        nonlocal current_text, current_speaker, segment_start, prev_end
        if current_text:
            segments.append(
                {
                    "speaker": current_speaker,
                    "start": segment_start,
                    "end": prev_end,
                    "text": "".join(current_text),
                }
            )
        current_text = []

    for tok in tokens:
        text = tok["text"]
        start = float(tok["start"])
        end = float(tok["end"])

        if tok["type"] == "punctuation":
            if tok["attaches_to"] == "previous" and current_text:
                current_text.append(text)
            elif current_text:
                current_text.append(text)
            if end > (prev_end or 0):
                prev_end = end
            if tok["is_eos"]:
                flush_segment()
            continue

        speaker = find_speaker((start + end) / 2, speaker_segments)
        if speaker == "Unknown":
            speaker = current_speaker if current_speaker is not None else "Speaker_01"

        if (prev_end and start - prev_end >= pause_split_threshold) or (
            speaker != current_speaker
        ):
            flush_segment()
            current_text = [text]
            segment_start = start
            current_speaker = speaker
        else:
            if not current_text:
                segment_start = start
                current_speaker = speaker
                current_text = [text]
            else:
                current_text.append(" " + text)

        prev_end = end

    flush_segment()

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
