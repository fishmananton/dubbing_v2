import boto3
import srt
import json
import tempfile
import pickle
import io
from pathlib import Path
from modal_utils import run_modal_job


def extract_emotions(
    audio_file: str,
    subtitles_file: str,
    boto_session: boto3.Session,
    bucket_name: str,
    run_id: str = '',
):
    with open(subtitles_file, encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))

    segments = []
    for sub in subs:
        segments.append({
            "sub_id": str(sub.index),
            "start": sub.start.total_seconds(),
            "end": sub.end.total_seconds(),
        })

    s3 = boto_session.client("s3")
    s3_audio_path = f"{run_id}/emotion_extract/input/vocal_16k.wav"
    s3.upload_file(audio_file, bucket_name, s3_audio_path)
    audio_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket_name, "Key": s3_audio_path},
        ExpiresIn=3600,
    )

    result = run_modal_job(
        app_name="audio-dubbing-emotion-batch",
        function_name="extract_batch_emotions",
        timeout_minutes=30,
        poll_delay_sec=5,
        full_audio_url=audio_url,
        segments=segments,
    )

    s3.delete_object(Bucket=bucket_name, Key=s3_audio_path)

    if result["status"] != "COMPLETED":
        raise Exception(f"Emotion extraction failed: {result}")

    embeddings_dict = result["output"]["embeddings"]
    return embeddings_dict
