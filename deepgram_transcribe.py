"""Deepgram STT front-end. Returns raw word data (ms timings) for the shared
assembler in transcribe_common. Used for Japanese (ja)."""
import os
import tempfile
import time

import requests
from pydub import AudioSegment
from pydub.generators import WhiteNoise
from pydub.silence import detect_silence


def deepgram_transcribe_raw(audio_file_raw: str, deepgram_api_key: str, language: str = "auto"):
    """Returns (words_data, trans_language) with ms start/end. Fills interior
    silent regions with -50dB white noise first to avoid ASR timing bugs on
    digital silence."""
    t0 = time.time()

    audio = AudioSegment.from_file(audio_file_raw)
    silent_ranges = detect_silence(audio, min_silence_len=200, silence_thresh=-50)
    if silent_ranges:
        if silent_ranges[0][0] == 0:
            silent_ranges = silent_ranges[1:]
        if silent_ranges and silent_ranges[-1][1] >= len(audio) - 50:
            silent_ranges = silent_ranges[:-1]
    if silent_ranges:
        noise_ref = WhiteNoise().to_audio_segment(duration=1)
        gain_adjust = -50 - noise_ref.dBFS
        for start_ms, end_ms in silent_ranges:
            chunk_noise = WhiteNoise().to_audio_segment(duration=end_ms - start_ms).apply_gain(gain_adjust)
            chunk_noise = chunk_noise.set_channels(audio.channels).set_frame_rate(audio.frame_rate).set_sample_width(audio.sample_width)
            audio = audio.overlay(chunk_noise, position=start_ms)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio.export(tmp.name, format="wav")
        tmp.close()
        upload_file = tmp.name
    else:
        upload_file = audio_file_raw

    base_url = "https://api.deepgram.com/v1/listen"
    headers = {"Authorization": f"Token {deepgram_api_key}", "Content-Type": "audio/wav"}
    params = {"model": "nova-3", "smart_format": "false", "punctuate": "true"}
    if language != "auto":
        params["language"] = language
    else:
        params["detect_language"] = "true"

    try:
        with open(upload_file, "rb") as f:
            response = requests.post(base_url, headers=headers, params=params, data=f, timeout=(10, 600))
        response.raise_for_status()
    finally:
        if upload_file != audio_file_raw:
            os.unlink(upload_file)

    result = response.json()
    channel = result["results"]["channels"][0]
    alternative = channel["alternatives"][0]

    detected_language = channel.get("detected_language", language)
    trans_language = detected_language if language == "auto" else language

    raw_words = alternative.get("words", [])
    if not raw_words:
        raise RuntimeError("Deepgram transcription returned no words.")

    # Prefer punctuated_word so the shared assembler can split on sentence ends.
    words_data = [{
        "text": w.get("punctuated_word") or w["word"],
        "start": int(round(float(w["start"]) * 1000)),
        "end": int(round(float(w["end"]) * 1000)),
        "speaker": None,
    } for w in raw_words]

    print(f"  [deepgram] got {len(words_data)} words, language: {trans_language} "
          f"({time.time() - t0:.1f}s)")
    return words_data, trans_language
