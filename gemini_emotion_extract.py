import json
import io
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import srt
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from google import genai



GEMINI_EMOTION_PROMPT = """You are an expert ADR (Automated Dialogue Replacement) film director analyzing an audio segment and its subtitles.

For each subtitle line in this batch, carefully analyze the actor's vocal delivery in the audio:
- Pitch variation (monotone, rising pitch, voice cracking)
- Pacing & Cadence (rushed, drawn out, hesitant)
- Physical vocal properties (breathiness, raspy, trembling, whispers, shouting)
- Micro-expressions (sighs, gasps, chuckles, crying, swallowing)

For each subtitle, output a JSON object with:
1. "idx": <subtitle index number>
2. "category": One broad classification: "angry", "fearful", "sad", "neutral", "happy", "surprised", "disgusted".
3. "emotion_tag": A rich, descriptive stage direction in brackets [ ]. Focus on HOW the line is delivered physically and emotionally. Include inline markers like <breath>, <sigh>, or <gasp> if present in the audio.
4. "emo_vector": An array of EXACTLY 8 float values between 0.0 and 1.0 corresponding to IndexTTS-2.5 emotion dimensions:
   [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]

   Guidance for "emo_vector":
   - Dominant emotions should be between 0.6 and 1.0.
   - Minor/subtle emotional undertones should be between 0.1 and 0.4.
   - Calm/neutral speech should set the 8th index (calm) close to 0.8-1.0 and others near 0.0.

Example ideal JSON output for one subtitle:
{
  "idx": 1,
  "category": "angry",
  "emotion_tag": "[shouting hysterically, voice cracking with desperation]",
  "emo_vector": [0.0, 0.9, 0.2, 0.4, 0.0, 0.0, 0.3, 0.0]
}

Return a JSON array of these objects for ALL subtitles in this batch.
Only output valid JSON, no markdown outside the JSON."""


def _load_audio(audio_path: str):
    audio_data, sr = sf.read(audio_path, dtype="int16")
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1).astype(np.int16)
    return audio_data, sr


def _find_pause_boundaries(audio_data: np.ndarray, sr: int, min_batch_sec: float = 180.0, max_batch_sec: float = 300.0, min_pause_sec: float = 0.8):
    total_duration = len(audio_data) / sr

    frame_size = int(0.03 * sr)
    num_frames = len(audio_data) // frame_size
    trimmed = audio_data[:num_frames * frame_size].reshape(num_frames, frame_size).astype(np.float32)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1))

    threshold = np.percentile(rms[rms > 0], 10) if np.any(rms > 0) else 1e-6

    is_silence = rms < threshold
    min_pause_frames = int(min_pause_sec / 0.03)

    pauses = []
    in_pause = False
    pause_start = 0
    for i, silent in enumerate(is_silence):
        if silent and not in_pause:
            in_pause = True
            pause_start = i
        elif not silent and in_pause:
            in_pause = False
            if i - pause_start >= min_pause_frames:
                center_sec = ((pause_start + i) / 2) * 0.03
                pauses.append(center_sec)

    boundaries = [0.0]
    for pause_time in pauses:
        last_boundary = boundaries[-1]
        elapsed = pause_time - last_boundary

        if elapsed >= min_batch_sec:
            boundaries.append(pause_time)

    if total_duration - boundaries[-1] < min_batch_sec and len(boundaries) > 1:
        boundaries.pop()

    boundaries.append(total_duration)
    return boundaries


def _subs_for_batch(subs, batch_start: float, batch_end: float):
    batch = []
    for sub in subs:
        mid = (sub.start.total_seconds() + sub.end.total_seconds()) / 2
        if batch_start <= mid < batch_end:
            batch.append(sub)
    return batch


def _encode_ogg(audio_data: np.ndarray, sr: int, start_sec: float, end_sec: float) -> bytes:
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    batch_data = audio_data[start_sample:end_sample]

    segment = AudioSegment(
        data=batch_data.tobytes(),
        sample_width=2,
        frame_rate=sr,
        channels=1,
    )
    ogg_buf = io.BytesIO()
    segment.export(ogg_buf, format="ogg", codec="libopus", bitrate="64k")
    return ogg_buf.getvalue()


def _build_context(batch_subs):
    entries = []
    for sub in batch_subs:
        entries.append({
            "idx": sub.index,
            "start": round(sub.start.total_seconds(), 2),
            "end": round(sub.end.total_seconds(), 2),
            "text": sub.content,
        })
    return json.dumps(entries, ensure_ascii=False)


def _call_gemini(client: genai.Client, model_name: str, audio_bytes: bytes, context: str):
    response = client.models.generate_content(
        model=model_name,
        contents=[
            genai.types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"),
            f"Here is the subtitle array with timestamps (in seconds) for this audio segment:\n\n{context}",
        ],
        config=genai.types.GenerateContentConfig(
            system_instruction=GEMINI_EMOTION_PROMPT,  # Moved to system_instruction
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    return json.loads(response.text)


def extract_emotions_gemini(
    audio_file: str,
    subtitles_file: str,
    gemini_api_key: str,
    gemini_model_name: str,
    output_file: str,
    min_batch_sec: float = 45.0,
    max_batch_sec: float = 90.0,
    emo_scale: float = 0.5,
):
    client = genai.Client(api_key=gemini_api_key)

    audio_data, sr = _load_audio(audio_file)

    with open(subtitles_file, encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))

    boundaries = _find_pause_boundaries(audio_data, sr, min_batch_sec, max_batch_sec)

    batches = []
    for i in range(len(boundaries) - 1):
        batch_start = boundaries[i]
        batch_end = boundaries[i + 1]

        batch_subs = _subs_for_batch(subs, batch_start, batch_end)
        if not batch_subs:
            continue

        context = _build_context(batch_subs)
        batches.append((batch_start, batch_end, context))

    all_results = []

    def _process_batch(start, end, context):
        audio_bytes = _encode_ogg(audio_data, sr, start, end)
        return _call_gemini(client, gemini_model_name, audio_bytes, context)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_process_batch, s, e, ctx) for s, e, ctx in batches]
        for future in as_completed(futures):
            all_results.extend(future.result())

    def _normalize_emo_vector(vec: list[float]) -> list[float]:
        s = sum(vec)
        if s > 0:
            return [v / s * emo_scale for v in vec]
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, emo_scale]

    emotions = {
        item["idx"]: {
            "emotion_tag": item.get("emotion_tag", ""),
            "category": item.get("category", "neutral"),
            "emo_vector": _normalize_emo_vector(
                item.get("emo_vector", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            )
        }
        for item in all_results
    }

    Path(output_file).write_text(json.dumps(emotions, indent=2, ensure_ascii=False))
    return emotions
