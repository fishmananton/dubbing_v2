"""Speechmatics STT front-end. Returns raw word data (ms timings) for the shared
assembler in transcribe_common. Default engine for all non-en/zh/ja languages."""
import json
import time

import requests

BASE_URL = "https://asr.api.speechmatics.com/v2"

LANGUAGE_MAP = {
    "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it", "pt": "pt",
    "nl": "nl", "sv": "sv", "da": "da", "no": "no", "fi": "fi", "pl": "pl",
    "ru": "ru", "zh": "cmn", "ja": "ja", "ko": "ko", "ar": "ar", "hi": "hi",
    "tr": "tr", "uk": "uk", "cs": "cs", "ro": "ro", "hu": "hu", "el": "el",
    "he": "he", "th": "th", "vi": "vi", "id": "id", "ms": "ms", "bg": "bg",
    "hr": "hr", "sk": "sk", "sl": "sl", "lt": "lt", "lv": "lv", "et": "et",
    "ca": "ca", "gl": "gl", "eu": "eu",
}


def speechmatics_transcribe_raw(audio_file_raw: str, speechmatics_api_key: str, language: str = "auto"):
    """Returns (words_data, trans_language) with ms start/end. Punctuation tokens
    are appended to the preceding word so the shared assembler can split on
    sentence ends."""
    t0 = time.time()
    headers = {"Authorization": f"Bearer {speechmatics_api_key}"}

    transcription_config = {"language": "auto", "model": "enhanced", "enable_entities": True}
    if language != "auto":
        transcription_config["language"] = LANGUAGE_MAP.get(language, language)

    job_config = {"type": "transcription", "transcription_config": transcription_config}

    with open(audio_file_raw, "rb") as f:
        files = {
            "data_file": (audio_file_raw.split("/")[-1], f, "application/octet-stream"),
            "config": (None, json.dumps(job_config), "application/json"),
        }
        response = requests.post(f"{BASE_URL}/jobs", headers=headers, files=files)
    response.raise_for_status()
    job_id = response.json()["id"]

    poll_count = 0
    while True:
        response = requests.get(f"{BASE_URL}/jobs/{job_id}", headers=headers, timeout=(10, 60))
        response.raise_for_status()
        job_status = response.json()["job"]
        if job_status["status"] == "done":
            break
        elif job_status["status"] == "rejected":
            raise RuntimeError(f"Speechmatics rejected: {job_status.get('errors', '')}")
        poll_count += 1
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
    reverse_map = {v: k for k, v in LANGUAGE_MAP.items()}
    trans_language = reverse_map.get(trans_language, trans_language.split("-")[0])

    words_data = []
    for result in transcript.get("results", []):
        rtype = result.get("type")
        alt = result["alternatives"][0]
        content = alt["content"]
        if rtype == "word":
            words_data.append({
                "text": content,
                "start": int(round(float(result["start_time"]) * 1000)),
                "end": int(round(float(result["end_time"]) * 1000)),
                "speaker": None,
            })
        elif rtype == "entity":
            # enable_entities emits numbers/times/dates as a single "entity" whose
            # alternatives[0].content is the written form ("8:45"). We want the
            # spoken form the TTS can read ("eight forty five"), matching how
            # AssemblyAI behaves with format_text=False. spoken_form is a list of
            # word tokens with their own timings.
            spoken = result.get("spoken_form")
            if spoken:
                for tok in spoken:
                    words_data.append({
                        "text": tok["alternatives"][0]["content"],
                        "start": int(round(float(tok["start_time"]) * 1000)),
                        "end": int(round(float(tok["end_time"]) * 1000)),
                        "speaker": None,
                    })
            else:
                words_data.append({
                    "text": content,
                    "start": int(round(float(result["start_time"]) * 1000)),
                    "end": int(round(float(result["end_time"]) * 1000)),
                    "speaker": None,
                })
        elif rtype == "punctuation" and words_data:
            words_data[-1]["text"] += content
            words_data[-1]["end"] = int(round(float(result["end_time"]) * 1000))

    print(f"  [speechmatics] got {len(words_data)} words, language: {trans_language} "
          f"({poll_count} polls, {time.time() - t0:.1f}s)")
    return words_data, trans_language
