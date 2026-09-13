"""AssemblyAI STT front-end. Returns raw word data (ms timings) for the shared
assembler in transcribe_common. Used for English."""
import requests
import time
import subprocess
import tempfile
import os


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
