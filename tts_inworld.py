import srt
import re
from pydub.silence import detect_nonsilent
from pydub import AudioSegment
import io
import subprocess
import os
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import base64
import time


def strip_silence(
    audio: AudioSegment,
    silence_thresh_db: float | None = None,
    min_silence_len: int = 180,
    margin_ms: int = 120,
    max_relative_trim_db: float = 24,
) -> AudioSegment:

    if len(audio) == 0:
        return audio

    # Fallback if dBFS is invalid for very quiet clips
    if silence_thresh_db is None:
        if audio.dBFS == float("-inf"):
            return audio
        silence_thresh_db = audio.dBFS - max_relative_trim_db

    nonsilent = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh_db,
    )

    if not nonsilent:
        return audio

    start = max(0, nonsilent[0][0] - margin_ms)
    end = min(len(audio), nonsilent[-1][1] + margin_ms)

    return audio[start:end]


def tts_generate_multivoice_inworld_segments(
    subtitles_file: str,
    api_key: str,
    speakers: dict,
    out_dir: str,
    language_code: str = "en",
    model_id: str = "inworld-tts-1.5-max",
    max_workers: int = 5,
    changed_list: list | None = None,
    voices: dict | None = None,
):
    # Your emotion labels -> Inworld-supported emotion markups
    # Note: these bracket-style emotion markups are currently English-only.
    EMOTION_MAP = {
        "angry": "[angry]",
        "fearful": "[fearful]",
        "sad": "[sad]",
        "neutral": None,
        "happy": "[happy]",
        "surprised": "[surprised]",
        "excited": "[happy]",     # closest built-in Inworld emotion tag
        "content": None,        # no direct built-in tag
        "scared": "[fearful]",
        "disgusted": "[disgusted]",
    }

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Basic {api_key}",
        "Content-Type": "application/json",
    })


    # --- Find candidate voices ---
    def get_voices(session, language_code):
        resp = session.get(
            "https://api.inworld.ai/voices/v1/voices",
            params={"languages": language_code},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        voice_list = data.get("voices", [])
        voice_list = [
            v for v in voice_list
            if str(v.get("source", "")).upper() == "SYSTEM"
        ]
        if not voice_list:
            raise RuntimeError(f"No Inworld voices found for language={language_code}")

        male_voice = pick_voice_by_gender(voice_list, "male")
        female_voice = pick_voice_by_gender(voice_list, "female")

        if not male_voice or not female_voice:
            raise RuntimeError(
                f"Could not find both male and female Inworld voices for language={language_code}."
            )

        return male_voice["name"], female_voice["name"]

    def score_voice(voice: dict) -> int:
        score = 0

        tags = [str(t).lower() for t in (voice.get("tags") or [])]
        categories = [str(c).lower() for c in (voice.get("categories") or [])]
        desc = (voice.get("description") or "").lower()
        name = (voice.get("displayName") or voice.get("name") or "").lower()

        text_parts = [name, desc] + tags + categories
        text_blob = " ".join(text_parts)

        # 1) Source preference
        if str(voice.get("source", "")).upper() == "SYSTEM":
            score += 30

        # 2) Strong positives: stable / neutral / broadly usable
        strong_positive = {
            "clear": 16,
            "neutral": 18,
            "steady": 16,
            "calm": 14,
            "warm": 12,
            "friendly": 12,
            "articulate": 14,
            "composed": 12,
            "conversational": 14,
            "supportive": 10,
            "helpful": 10,
            "professional": 10,
            "believable": 14,
            "approachable": 10,
            "general purpose assistant": 16,
            "general-purpose assistant": 16,
            "companion": 8,
            "companions": 8,
        }

        # 3) Mild positives: can support emotion without becoming theatrical
        mild_positive = {
            "expressive": 6,
            "engaging": 6,
            "empathetic": 8,
            "encouraging": 6,
            "soothing": 6,
            "gentle": 5,
            "narrative": 4,
            "insightful": 4,
            "lucid": 4,
            "captivating": 3,
        }

        # 4) Strong negatives: over-stylized / character / genre-heavy
        strong_negative = {
            "character": -18,
            "characters": -18,
            "villain": -30,
            "villains": -30,
            "sinister": -28,
            "ominous": -25,
            "menacing": -28,
            "manipulative": -25,
            "intimidating": -24,
            "gritty": -18,
            "gravelly": -20,
            "gruff": -18,
            "hoarse": -20,
            "raspy": -15,
            "robotic": -35,
            "monotone": -25,
            "shouty": -18,
            "performative": -16,
            "announcer": -18,
            "trailer": -22,
            "epic": -18,
            "sports": -18,
            "ads": -12,
            "motivational": -12,
            "fitness coach": -14,
            "call center": -10,
            "customer support": -8,
            "patient support": -8,
            "banking support": -10,
            "helpdesk": -8,
            "news": -12,
            "crime": -15,
            "gothic": -20,
            "fantasy": -18,
            "dark": -12,
            "romantic": -10,
            "spiritual": -10,
            "meditation": -8,
        }

        # 6) Delivery-quality penalties
        delivery_negative = {
            "breathy": -10,
            "high-pitched": -10,
            "nasal": -12,
            "rapid-fire": -10,
            "intense": -10,
            "commanding": -12,
            "authoritative": -6,  # not always bad, just less flexible
            "profound": -6,
            "resonant": -4,
            "controlled": 4,  # controlled is actually useful
            "deliberate": 4,
            "measured": 5,
            "precise": 6,
        }

        # Apply scores
        for word, value in strong_positive.items():
            if word in text_blob:
                score += value

        for word, value in mild_positive.items():
            if word in text_blob:
                score += value

        for word, value in strong_negative.items():
            if word in text_blob:
                score += value


        for word, value in delivery_negative.items():
            if word in text_blob:
                score += value

        # 7) Category shaping
        # Prefer broad/general content, mildly prefer education/companions, avoid interactive_media-heavy
        if "education_training" in categories:
            score += 8
        if "companions" in categories:
            score += 8
        if "enterprise" in categories:
            score += 4
        if "interactive_media" in categories:
            score -= 8

        # 8) Age shaping
        age_group = str(voice.get("ageGroup", "")).lower()
        if age_group == "middle_aged":
            score += 30
        elif age_group == "young":
            score -= 2
        elif age_group == "elderly":
            score -= 8

        return score

    def pick_voice_by_gender(voices: list[dict], gender: str) -> dict | None:
        gender_voices = [v for v in voices if v['gender'] == gender.lower()]

        # fallback: if metadata is weak, choose best overall system voice
        if not gender_voices:
            gender_voices = voices

        if not gender_voices:
            return None

        return max(gender_voices, key=score_voice)

    if voices:
        male_voice = voices["male"]
        female_voice = voices["female"]
    else:
        male_voice, female_voice = get_voices(session, language_code)

    # Build subtitle-index -> emotion map per speaker
    speaker_idx_to_emotion = {}
    for spk, data in speakers.items():
        groups = (data or {}).get("groups", {})
        idx_map = {}
        for emo, g in groups.items():
            inworld_emo = EMOTION_MAP.get(emo, None)
            for idx in g.get("idxs", []):
                idx_map[idx] = inworld_emo
        speaker_idx_to_emotion[spk] = idx_map

    with open(subtitles_file, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))

    if not subs:
        raise ValueError("Subtitle file is empty.")

    def synthesize_and_save(sub):
        idx = sub.index
        content = sub.content.strip()

        if ":" not in content:
            raise ValueError(f"Subtitle {idx} does not contain 'SPEAKER: text' format")

        speaker, text = content.split(":", 1)
        speaker = speaker.strip()
        text = text.strip().replace(":", " - ")

        output_dir = os.path.join(out_dir, speaker)
        os.makedirs(output_dir, exist_ok=True)
        if len(text.strip()) < 1:
            duration_ms = max(
                1,
                int(round((sub.end - sub.start).total_seconds() * 1000))
            )
            silence = AudioSegment.silent(duration=duration_ms, frame_rate=48000)
            silence.export(os.path.join(output_dir, f"{idx}.wav"), format="wav")
            return idx, speaker, True

        speaker_data = speakers.get(speaker, {})
        gender = speaker_data.get("gender", "male").lower()
        voice_id = female_voice if gender == "female" else male_voice

        emotion = speaker_idx_to_emotion.get(speaker, {}).get(idx, None)
        final_text = emotion + " " + text if emotion else text
        final_text = final_text.replace(",", "")
        payload = {
            "text": final_text,
            "voiceId": voice_id,
            "modelId": model_id,
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": 48000,
            },
            "temperature": 1.0,
            "speakingRate": 1.0,
            "applyTextNormalization": "ON",
        }

        audio_bytes = None
        last_error = None
        for attempt in range(3):
            try:
                resp = session.post(
                    "https://api.inworld.ai/tts/v1/voice",
                    json=payload,
                    timeout=180,
                )
                resp.raise_for_status()
                audio_b64 = resp.json().get("audioContent")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    break
                last_error = f"no audioContent in response"
            except Exception as e:
                last_error = str(e)
            time.sleep(2 ** attempt)

        output_path = os.path.join(output_dir, f"{idx}.wav")
        if audio_bytes is None:

            print(f"⚠️ Inworld TTS failed for subtitle {idx} after 3 attempts: {last_error} — writing silence")
            duration_ms = max(
                1,
                int(round((sub.end - sub.start).total_seconds() * 1000))
            )
            silence = AudioSegment.silent(duration=duration_ms, frame_rate=48000)
            silence.export(output_path, format="wav")
            return idx, speaker, False

        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
        audio = strip_silence(audio)
        audio.export(output_path, format="wav")

        return idx, speaker, True

    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if changed_list:
            changed_set = set(changed_list)
            subs_to_generate = [sub for sub in subs if sub.index in changed_set]
        else:
            subs_to_generate = subs

        futures = [executor.submit(synthesize_and_save, sub) for sub in subs_to_generate]
        for future in as_completed(futures):
            idx, speaker, success = future.result()
            if success:
                done += 1
            else:
                failed.append(idx)

    if failed:
        print(f"⚠️ Inworld TTS: {len(failed)} segment(s) failed and got silence: {sorted(failed)}")
    print(f"✅ Generated {done} Inworld TTS segments using {max_workers} threads.")
    return {"male": male_voice, "female": female_voice}