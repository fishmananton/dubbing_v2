from http import HTTPStatus
from dashscope.audio.asr import Transcription
import dashscope
import requests
import time
import re
import boto3
from openai import OpenAI
import json

SYSTEM_PROMPT = """You are given subtitles with imperfect speaker diarization.

Your task:
1. Correct speaker assignments based on conversation context and flow.
2. Replace generic speaker labels (e.g., SPEAKER_01, Person_1) with consistent, meaningful role names in English.
3. Ensure the same character always uses the same role name throughout the entire dialogue.
4. Minimize the number of unique roles. Do NOT create unnecessary new roles.
5. Number of speakers is known to be {num_speakers} or less.

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


CJK_RE = re.compile(r'[\u4e00-\u9fff]')

def srt_timestamp(seconds):
    h, m = divmod(seconds, 3600)
    m, s = divmod(m, 60)
    return f"{int(h):02}:{int(m):02}:{s:06.3f}".replace('.', ',')


def find_speaker(time_sec, speaker_segments):
    """Find which speaker segment covers a given timestamp."""
    for seg in speaker_segments:
        if seg["start"] <= time_sec <= seg["end"]:
            return seg["speaker"]
    return "Unknown"



def has_cjk(token: str) -> bool:
    return bool(CJK_RE.search(token))


def _join_tokens(tokens):
    if not tokens:
        return ""

    result = [tokens[0]]

    for prev, cur in zip(tokens, tokens[1:]):
        if has_cjk(prev) and has_cjk(cur):
            result.append(cur)
        else:
            result.append(" " + cur)

    return "".join(result)


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

def alibabacloud_transcribe(
    audio_file_raw: str,
    subtitles_file: str,
    speaker_segments: list,
    alibaba_api_key: str,
    s3_bucket_name: str,
    boto_session: boto3.Session,
    openai_client: OpenAI,
    openai_model: str,
    num_speakers: int | None,
    language: str = "auto",
    run_id: str = "",
    model: str = "fun-asr",
    poll_interval: int = 3,
    dashscope_base_url: str = "https://dashscope-intl.aliyuncs.com/api/v1", #https://modelstudio.console.alibabacloud.com/
):
    """
    Transcribe audio with Alibaba DashScope and generate SRT subtitles,
    using external diarization speaker segments exactly like assemblyai_transcribe.

    Returns:
        trans_language (str)
    """

    s3 = boto_session.client("s3")

    s3_object_key = f"{run_id}/transcribe/input/{audio_file_raw}"

    s3.upload_file(audio_file_raw, s3_bucket_name, s3_object_key)

    input_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": s3_bucket_name, "Key": s3_object_key},
        ExpiresIn=3600
    )

    # ---- 2. Configure DashScope
    dashscope.base_http_api_url = dashscope_base_url
    dashscope.api_key = alibaba_api_key

    # ---- 3. Start transcription
    transcribe_response = Transcription.async_call(
        model=model,
        file_urls=[input_url]
    )

    # ---- 4. Poll until completion
    while True:
        task_status = transcribe_response.output.task_status

        if task_status in ("SUCCEEDED", "FAILED"):
            break

        time.sleep(poll_interval)
        transcribe_response = Transcription.fetch(
            task=transcribe_response.output.task_id
        )
    s3.delete_object(Bucket=s3_bucket_name, Key=s3_object_key)
    if transcribe_response.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f"Alibaba transcription request failed: "
            f"status_code={transcribe_response.status_code}, "
            f"response={transcribe_response}"
        )

    if transcribe_response.output.task_status == "FAILED":
        raise RuntimeError(f"Alibaba transcription failed: {transcribe_response.output}")

    # ---- 5. Download transcription result JSON
    result_url = transcribe_response.output.results[0]["transcription_url"]

    response = requests.get(result_url, timeout=(10, 120))
    response.raise_for_status()
    result_data = response.json()


    words = []

    for transcript in result_data.get("transcripts", []):

        for sentence in transcript.get("sentences", []):
            for w in sentence.get("words", []):
                token = w.get("text", "")
                punctuation = w.get("punctuation", "") or ""
                token_text = f"{token}{punctuation}"

                words.append({
                    "text": token_text,
                    "start": float(w["begin_time"]) / 1000.0,
                    "end": float(w["end_time"]) / 1000.0,
                })

    if not words:
        raise RuntimeError("Alibaba transcription succeeded but returned no words.")

    # ---- 8. Build merged segments based on time + speaker
    segments = []
    current_text = []
    current_speaker = None
    segment_start = None
    prev_end = None

    for w in words:
        word = w["text"]
        start = w["start"]
        end = w["end"]

        speaker = find_speaker((start + end) / 2, speaker_segments)

        if speaker == "Unknown":
            speaker = current_speaker if current_speaker is not None else "Speaker_01"

        # Start a new segment if:
        # - pause >= 250 ms
        # - speaker changed
        if (prev_end is not None and start - prev_end >= 0.25) or (speaker != current_speaker):
            if current_text:
                segments.append({
                    "speaker": current_speaker,
                    "start": segment_start,
                    "end": prev_end,
                    "text": _join_tokens(current_text)
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
            "text": _join_tokens(current_text)
        })
    segments = filter_speakable_subs(segments)
    # ---- 9. Sort and save to SRT
    segments.sort(key=lambda x: x["start"])
    for i, seg in enumerate(segments, 1):
        seg["index"] = i

    unique_speakers = len(set(seg["speaker"] for seg in segments))
    actual_num_speakers = num_speakers if num_speakers is not None else unique_speakers

    if actual_num_speakers > 1:
        segments = fix_sub_diarization_with_ai(openai_client, openai_model,segments,num_speakers)

    with open(subtitles_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{srt_timestamp(seg['start'])} --> {srt_timestamp(seg['end'])}\n")
            f.write(f"{seg['speaker']}: {seg['text']}\n\n")

    return language
