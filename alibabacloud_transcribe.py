"""Alibaba DashScope STT front-end. Returns raw word data (ms timings) for the
shared assembler in transcribe_common. Used for Chinese (zh)."""
from http import HTTPStatus

import boto3
import dashscope
import requests
import time
from dashscope.audio.asr import Transcription


def alibaba_transcribe_raw(
    audio_file_raw: str,
    alibaba_api_key: str,
    s3_bucket_name: str,
    boto_session: boto3.Session,
    language: str = "auto",
    run_id: str = "",
    model: str = "fun-asr",
    poll_interval: int = 3,
    dashscope_base_url: str = "https://dashscope-intl.aliyuncs.com/api/v1",
):
    """Returns (words_data, trans_language) with ms start/end. DashScope needs a
    public URL, so the file is uploaded to S3, transcribed, then removed."""
    t0 = time.time()
    s3 = boto_session.client("s3")
    s3_object_key = f"{run_id}/transcribe/input/{audio_file_raw}"
    s3.upload_file(audio_file_raw, s3_bucket_name, s3_object_key)
    input_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": s3_bucket_name, "Key": s3_object_key},
        ExpiresIn=3600,
    )

    dashscope.base_http_api_url = dashscope_base_url
    dashscope.api_key = alibaba_api_key

    transcribe_response = Transcription.async_call(model=model, file_urls=[input_url])

    while True:
        task_status = transcribe_response.output.task_status
        if task_status in ("SUCCEEDED", "FAILED"):
            break
        time.sleep(poll_interval)
        transcribe_response = Transcription.fetch(task=transcribe_response.output.task_id)

    s3.delete_object(Bucket=s3_bucket_name, Key=s3_object_key)

    if transcribe_response.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f"Alibaba transcription request failed: status_code={transcribe_response.status_code}, "
            f"response={transcribe_response}"
        )
    if transcribe_response.output.task_status == "FAILED":
        raise RuntimeError(f"Alibaba transcription failed: {transcribe_response.output}")

    result_url = transcribe_response.output.results[0]["transcription_url"]
    response = requests.get(result_url, timeout=(10, 120))
    response.raise_for_status()
    result_data = response.json()

    words_data = []
    for transcript in result_data.get("transcripts", []):
        for sentence in transcript.get("sentences", []):
            for w in sentence.get("words", []):
                token = w.get("text", "")
                punctuation = w.get("punctuation", "") or ""
                words_data.append({
                    "text": f"{token}{punctuation}",
                    "start": int(round(float(w["begin_time"]))),
                    "end": int(round(float(w["end_time"]))),
                    "speaker": None,
                })

    if not words_data:
        raise RuntimeError("Alibaba transcription succeeded but returned no words.")

    trans_language = "zh" if language == "auto" else language
    print(f"  [alibaba] got {len(words_data)} words, language: {trans_language} "
          f"({time.time() - t0:.1f}s)")
    return words_data, trans_language
