import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import srt
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from google import genai


MAX_LINES = 10        # lines per speaker to concatenate into the clip
MIN_LINE_MS = 150     # skip ultra-short spans that carry no usable voice

GENDER_PROMPT = (
    "You are given a short audio clip containing ONLY one person's voice "
    "(concatenated from several of their lines) plus the transcript of what they say. "
    "Determine the speaker's gender for dubbing purposes. "
    "Primarily use vocal characteristics (pitch, timbre, resonance), "
    "but when the voice is ambiguous — especially a CHILD, whose high pitch is not "
    "informative — use the transcript and narrative context to infer the person's actual "
    "sex (e.g. a boy addressed as 'son', a girl called 'daughter'). "
    "Report male for boys and female for girls. "
    'Respond with a single JSON object: {"gender": "male"} or {"gender": "female"}. '
    "No other text."
)


def _load_vocal(audio_path: str):
    data, sr = sf.read(audio_path, dtype="int16")
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.int16)
    return data, sr


def _speaker_lines(srt_path: str):
    """Group subtitle spans + texts per speaker, dropping ultra-short spans."""
    with open(srt_path, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))
    by_spk: dict[str, dict] = {}
    for sub in subs:
        if ":" not in sub.content:
            continue
        spk, text = sub.content.split(":", 1)
        spk = spk.strip()
        start, end = sub.start.total_seconds(), sub.end.total_seconds()
        by_spk.setdefault(spk, {"spans": [], "texts": []})
        if (end - start) * 1000 >= MIN_LINE_MS:
            by_spk[spk]["spans"].append((start, end))
            by_spk[spk]["texts"].append(text.strip())
    return by_spk


def _build_clip(data: np.ndarray, sr: int, spans) -> bytes:
    """Concatenate the speaker's line spans into one OGG/opus clip."""
    seg = AudioSegment.empty()
    for start, end in spans:
        chunk = data[int(start * sr):int(end * sr)]
        if len(chunk) == 0:
            continue
        seg += AudioSegment(chunk.tobytes(), sample_width=2, frame_rate=sr, channels=1)
    buf = io.BytesIO()
    seg.export(buf, format="ogg", codec="libopus", bitrate="64k")
    return buf.getvalue()


def _call_gemini(client, model, audio_bytes, speaker, texts):
    transcript = "\n".join(f"- {t}" for t in texts)
    prompt = (
        f'Speaker label: "{speaker}"\n'
        f"Transcript of their lines:\n{transcript}\n\n"
        "Detect the gender of this single speaker."
    )
    r = client.models.generate_content(
        model=model,
        contents=[
            genai.types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"),
            prompt,
        ],
        config=genai.types.GenerateContentConfig(
            system_instruction=GENDER_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    gender = json.loads(r.text).get("gender", "").strip().lower()
    return gender if gender in ("male", "female") else "male"


def detect_gender(audio_path: str, srt_path: str, gemini_api_key: str, gemini_model: str):
    """Per-speaker gender detection: one Gemini call per speaker over a small
    concatenated voice clip + that speaker's transcript. Returns
    {speaker: {"gender": "male"|"female"}}. Falls back to "male" on any failure
    so a single bad call never blocks translation."""
    data, sr = _load_vocal(audio_path)
    by_spk = _speaker_lines(srt_path)
    client = genai.Client(api_key=gemini_api_key)

    clips = {}
    for spk, d in by_spk.items():
        spans, texts = d["spans"][:MAX_LINES], d["texts"][:MAX_LINES]
        if not spans:
            clips[spk] = None
            continue
        clips[spk] = (_build_clip(data, sr, spans), texts)

    result = {spk: {"gender": "male"} for spk in by_spk}

    def work(spk):
        clip = clips[spk]
        if clip is None:
            return spk, "male"
        try:
            return spk, _call_gemini(client, gemini_model, clip[0], spk, clip[1])
        except Exception as e:
            print(f"⚠️ gender detection failed for {spk}: {e}")
            return spk, "male"

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(work, spk) for spk in clips]
        for fut in as_completed(futs):
            spk, gender = fut.result()
            result[spk]["gender"] = gender

    return result
