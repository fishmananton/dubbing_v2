from __future__ import annotations
import whisperx
import numpy as np

import io
import re

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from pydub.silence import detect_nonsilent

import srt
from fishaudio import FishAudio
from fishaudio.types.tts import TTSConfig, Prosody, FlushEvent
import queue
import threading

import io
from pathlib import Path
from pydub import AudioSegment
from fishaudio.types import FlushEvent
import tempfile

import asyncio

from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=2)

_align_future = None

def preload_alignment_model(language_code: str, device: str = "cpu"):
    global _align_future

    if _align_future is not None:
        return  # already loading or loaded

    def _load():
        import whisperx
        model, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=device,
        )
        return model, metadata

    _align_future = _executor.submit(_load)

def get_alignment_model():
    global _align_future

    if _align_future is None:
        raise RuntimeError("Alignment model was not preloaded")

    return _align_future.result()  # waits here if not ready


EMOTION_PROMPT_MAP = {
    "happy": "[happy]",
    "neutral": "[neutral]",
    "angry": "[angry]",
    "sad": "[sad]",
    "fearful": "[scared]",
    "surprised": "[surprised]",
    "excited": "[excited]",
    "content": "[calm]",
    "scared": "[scared]",
}

ACCENT_PROMPT_MAP = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ko": "Korean",
    "zh": "Mandarin Chinese",
    "ja": "Japanese",
    "ar": "Arabic",
    "hi": "Hindi",
    "he": "Hebrew",
    "pt": "Brazilian Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
}

GOOD_LANGS = {"en", "es", "fr", "de", "it", "nl", "pt", "pl", "ru"}
# MID_LANGS = {"zh", "ja", "ko"}
# BAD_LANGS = {"ar", "hi", "he","zh", "ja", "ko"}


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



def _pick_fish_public_voice(
    client: FishAudio,
    gender: str,
    language_code: str,
    page_size: int = 50,
) -> Optional[str]:
    """
    Fish supports listing voices and filtering by tags/language; use a simple
    best-effort selector here. Public-voice discovery exists, but Fish does not
    expose embedding-based similarity search in the SDK docs.  [oai_citation:2‡Fish Audio](https://docs.fish.audio/developer-guide/sdk-guide/python/voice-cloning)
    """
    gender = (gender or "male").lower()
    wanted_tag = "female" if gender == "female" else "male"

    resp = client.voices.list(
        language=language_code,
        tags=[wanted_tag],
        page_size=page_size,
    )

    candidates = list(resp.items or [])
    if not candidates:
        return None

    def score(v) -> int:
        s = 0
        tags = {str(t).lower() for t in (v.tags or [])}
        langs = {str(x).lower() for x in (v.languages or [])}

        if wanted_tag in tags:
            s += 20
        if str(language_code).lower() in langs:
            s += 20
        if "english" in tags and language_code.lower().startswith("en"):
            s += 8
        if "young" in tags:
            s -= 1  # tiny penalty for very stylized voices
        # prefer widely used / healthy public voices if available
        s += int(getattr(v, "task_count", 0) or 0) // 1000
        s += int(getattr(v, "like_count", 0) or 0) // 100
        return s

    best = max(candidates, key=score)
    return best.id


def _wait_until_voice_ready(
    client: FishAudio,
    voice_id: str,
    timeout_sec: int = 600,
    poll_sec: float = 3.0,
) -> None:
    """
    voices.get() returns a Voice object with a state field. Wait until ready or fail.  [oai_citation:3‡Fish Audio](https://docs.fish.audio/api-reference/sdk/python/types)
    """
    deadline = time.time() + timeout_sec
    last_state = None

    while time.time() < deadline:
        voice = client.voices.get(voice_id)
        last_state = str(getattr(voice, "state", "") or "").lower()

        if last_state == "trained":
            return
        if last_state == "failed":
            raise RuntimeError(f"Fish voice training failed for voice_id={voice_id}")

        time.sleep(poll_sec)

    raise TimeoutError(f"Voice {voice_id} did not become ready, last_state={last_state!r}")




def build_original_samples(
    translated_subs: list,
    original_subs_by_idxs: dict,
    audio_file: str):
    audio = AudioSegment.from_file(audio_file)

    result={}
    for sub in translated_subs:
        start_ms = int(sub.start.total_seconds() * 1000)
        end_ms = int(sub.end.total_seconds() * 1000)
        idx = sub.index
        chunk = audio[start_ms:end_ms]
        chunk = strip_silence(chunk)
        duration_ms = len(chunk)
        text =  original_subs_by_idxs[idx].content.strip().split(":", 1)[1]
        sample_item = {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": duration_ms,
            "audio": chunk,
            "text": text
        }
        result[idx] = sample_item


    return result


def build_fallback_decisions(
    speakers: dict,
    speaker_samples: dict,
    min_required_sec: float = 5.0,
    max_total_sec: float = 70.0,
    gap_ms: int = 150,
    chunk_target_sec: float = 15.0,
    changed_list: list = None,):

    decisions = []
    gap = AudioSegment.silent(duration=gap_ms)

    for speaker, data in speakers.items():
        groups = (data or {}).get("groups", {}) or {}
        if changed_list and not any(idx in changed_list for g in groups.values() for idx in g.get("idxs", [])):
            continue
        ordered_emotions = []
        if "neutral" in groups:
            ordered_emotions.append("neutral")
        ordered_emotions.extend(
            emo for emo in groups.keys() if emo != "neutral"
        )
        selected_idxs = []
        selected_sec = 0.0

        current_chunk_sec=  0.0
        global_chunks=[]
        current_chunk = {"text":[], "audio":[]}
        for emotion in ordered_emotions:
            group_data = groups.get(emotion, {}) or {}
            idxs = group_data.get("idxs", []) or []
            for idx in idxs:
                sample = speaker_samples.get(idx, {})
                duration_sec = float(sample.get("duration_ms", 0) or 0) / 1000.0
                audio = sample.get("audio")
                text = (sample.get("text") or "").strip()
                if duration_sec <= 0.0 or audio is None or not text:
                    continue
                selected_idxs.append(idx)
                selected_sec += duration_sec
                current_chunk_sec += duration_sec
                current_chunk["text"].append(text)
                current_chunk["audio"].append(audio)
                if current_chunk_sec >= chunk_target_sec:
                    global_chunks.append(current_chunk)
                    current_chunk = {"text":[], "audio":[]}
                    current_chunk_sec=0

                if selected_sec >= max_total_sec:
                    break
            if selected_sec >= max_total_sec:
                break
        if current_chunk["audio"]:
            global_chunks.append(current_chunk)
        fallback_required = selected_sec < float(min_required_sec)
        audio_bytes=[]
        merged_text=[]
        for chunk in global_chunks:
            merged_text.append(" ".join(chunk["text"]).strip())

            if chunk["audio"]:
                merged_audio = chunk["audio"][0]
                for audio in chunk["audio"][1:]:
                    merged_audio += gap + audio

                buf = io.BytesIO()
                merged_audio.export(buf, format="wav")
                audio_bytes.append(buf.getvalue() )
        decisions.append({
            "speaker": speaker,
            "gender": data.get("gender"),
            "fallback": fallback_required,
            "idxs": selected_idxs,
            "total_sec": selected_sec,
            "text": merged_text,
            "audio_bytes": audio_bytes,
        })
    return decisions

def prepare_one_voice(item, client, run_id):
    if item["fallback"]:
        return None

    voice = client.voices.create(
        title=f"tmp_{item['speaker']}_{run_id}",
        voices=item["audio_bytes"],
        texts=item["text"],
        visibility="private",
        enhance_audio_quality=True
    )

    _wait_until_voice_ready(client, voice.id)
    return {"speaker":item['speaker'], "voice_id":voice.id}

def synthesize_one(sub_text:str, idx:int, voice_id:str, emotion_text:str, output_dir : str, client : FishAudio, accent_str:str, config:TTSConfig):

    reference_id = voice_id
    final_text = f"{accent_str} {emotion_text} {sub_text}".strip()
    # final_text = sub_text.strip()
    audio_bytes = client.tts.convert(
        text=final_text,
        reference_id=reference_id,
        format="wav",
        latency="normal",
        config=config,
    )

    # post-trim using your existing silence stripper
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
    audio = strip_silence(audio)
    audio = audio.set_frame_rate(48000)
    out_path = os.path.join(output_dir, f"{idx}.wav")
    audio.export(out_path, format="wav")
    return idx

def split_words(text: str) -> list[str]:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split() if text else []
    return [w.strip("-'") for w in words if w.strip("-'")]


def assign_timings_from_words(
    subtitle_items,
    aligned,
    end_padding_sec: float = 0.01,
    start_padding_sec: float = 0.01,
    min_gap_sec: float = 0.02,
):
    words = aligned["word_segments"]
    pos = 0
    result = []

    # First pass: raw timings
    for item in subtitle_items:
        expected = split_words(item["text"])
        n = len(expected)

        chunk = words[pos:pos + n]
        actual = [token for w in chunk for token in split_words(w["word"])]

        ok = (len(actual) == len(expected) and actual == expected)
        if not ok:
            print ("Alignment error in assign_timings_from_words")

        if chunk:
            start = chunk[0]["start"]
            end = chunk[-1]["end"]
        else:
            start = None
            end = None

        result.append({
            "sub_id": item["sub_id"],
            "text": item["text"],
            "start": start,
            "end": end,
            "ok": ok,
            "expected_words": expected,
            "actual_words": actual,
        })

        pos += n

    # Second pass: safe padding
    for i, seg in enumerate(result):
        if seg["start"] is None or seg["end"] is None:
            continue

        new_start = max(0.0, seg["start"] - start_padding_sec)
        new_end = seg["end"] + end_padding_sec

        # clamp to next segment to avoid overlap
        if i < len(result) - 1:
            next_seg = result[i + 1]
            if next_seg["start"] is not None:
                new_end = min(new_end, next_seg["start"] - min_gap_sec)

        # never let end go before start
        if new_end < new_start:
            new_end = seg["end"]

        seg["start"] = new_start
        seg["end"] = new_end

    return result



def synthesize_group(lines:list, subtitle_items:list, voice_id:str,output_dir : str, client : FishAudio, config:TTSConfig):
    reference_id = voice_id
    final_text = "\n".join(lines)

    # final_text = sub_text.strip()

    audio_bytes = client.tts.convert(
        text=final_text,
        reference_id=reference_id,
        format="wav",
        latency="normal",
        config=config,
    )

    align_model, metadata = get_alignment_model()
    # post-trim using your existing silence stripper
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
    batch_duration = len(audio)/ 1000.0
    batch_audio = audio.set_frame_rate(16000).set_channels(1)
    raw = np.array(batch_audio.get_array_of_samples())

    if np.issubdtype(raw.dtype, np.integer):
        max_val = max(abs(np.iinfo(raw.dtype).min), np.iinfo(raw.dtype).max)
        samples = raw.astype(np.float32) / float(max_val)
    else:
        samples = raw.astype(np.float32)

    # Send normalized text to WhisperX
    full_text = " ".join(
        " ".join(split_words(item["text"]))
        for item in subtitle_items
    )
    segments=[{
            "start": 0.0,
            "end": batch_duration,  # scale to batch duration
            "text": full_text
        }]




    aligned = whisperx.align(
        transcript=segments,
        model=align_model,
        align_model_metadata=metadata,
        audio=samples,
        device='cpu',
        return_char_alignments=False,
    )
    final_segments = assign_timings_from_words(subtitle_items, aligned)
    for seg in final_segments:
        seg_audio = audio[seg["start"]*1000:seg["end"]*1000]
        if len(seg_audio) > 10:
            seg_audio = seg_audio.fade_in(5).fade_out(5)
        seg_audio = seg_audio.set_frame_rate(48000)
        out_path = os.path.join(output_dir, f"{seg['sub_id']}.wav")
        seg_audio.export(out_path, format="wav")
    return len(final_segments)



def build_emotion_idx( speakers:dict):
    speaker_idx_to_emotion = {}
    for spk, data in speakers.items():
        groups = (data or {}).get("groups", {})
        m = {}
        for emo, g in groups.items():
            for idx in g.get("idxs", []):
                m[idx] =EMOTION_PROMPT_MAP.get(emo, '[neutral]')
        speaker_idx_to_emotion[spk] = m
    return speaker_idx_to_emotion






def text_stream(q: queue.Queue):
    while True:
        item = q.get()
        if item is None:
            break
        yield item



def collect_until_idle(audio_queue, idle_timeout: float = 1.5) -> bytes:
    chunks = []

    while True:
        try:
            kind, payload = audio_queue.get(timeout=idle_timeout)
        except queue.Empty:
            # no new audio for a while -> treat as end of this flushed segment
            break

        if kind == "chunk":
            chunks.append(payload)
        elif kind == "error":
            raise payload
        elif kind == "end":
            break

    return b"".join(chunks)

def start_audio_reader(audio_stream):
    q = queue.Queue()

    def _reader():
        try:
            for chunk in audio_stream:
                q.put(("chunk", chunk))
        except Exception as e:
            q.put(("error", e))
        finally:
            q.put(("end", None))

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return q, t


def generate_sync(client, lines, idxs, voice_id, config, output_dir):
    q = queue.Queue()

    audio_stream = client.tts.stream_websocket(
        text_stream(q),
        latency="normal",
        reference_id=voice_id,
        format="pcm",
        config=config,
    )

    # only ONE reader thread for this stream
    audio_queue, reader_thread = start_audio_reader(audio_stream)

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    saved = []

    for sub_idx, line in zip(idxs, lines):
        q.put(line)
        q.put(FlushEvent())

        audio_bytes = collect_until_idle(audio_queue, idle_timeout=3)

        if not audio_bytes:
            print(f"{sub_idx}: no audio")
            continue

        audio = AudioSegment.from_raw(
            io.BytesIO(audio_bytes),
            sample_width=2,
            frame_rate=44100,
            channels=1,
        )
        audio = strip_silence(audio)
        audio = audio.set_frame_rate(48000)

        out_path = out_dir / f"{sub_idx}.wav"
        audio.export(out_path, format="wav")
        print(f"saved {out_path}")
        saved.append(sub_idx)

    q.put(None)
    return saved


def write_subtitle_silence(output_path: str, duration_ms: int, sample_rate: int = 48000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    duration_ms = max(1, int(duration_ms))

    silence = (
        AudioSegment
        .silent(duration=duration_ms, frame_rate=sample_rate)
        .set_channels(1)
        .set_sample_width(2)
    )

    silence.export(output_path, format="wav")

def tts_generate_multivoice_fish_segments(
    client: FishAudio,
    translated_subtitles_file: str,
    subtitles_file: str,
    original_voice_audio_file: str,
    speakers: dict,
    out_dir: str,
    language_code: str = "en",
    voices: dict | None = None,
    max_workers: int = 5,
    changed_list: Optional[list[int]] = None,
    min_clone_sec: float = 5.5,
    run_id: str = '',
    force_delete:bool = False,
    force_no_batch:bool=False
) -> dict:

    use_batch_alignment = True  if language_code in GOOD_LANGS and not force_no_batch else False


    if use_batch_alignment:
        print(f"Preloading alignment model for language_code={language_code}...")
        preload_alignment_model(language_code)


    with open(translated_subtitles_file, "r", encoding="utf-8") as f:
        target_subs =  list(srt.parse(f.read()))
    with open(subtitles_file, "r", encoding="utf-8") as f:
        original_subs =  list(srt.parse(f.read()))
    if not target_subs:
        raise ValueError("subtitles_file is empty")
    if not original_subs:
        raise ValueError("original_subtitles_file is empty")

    original_subs_by_idx = {sub.index: sub for sub in original_subs}
    translated_subs_by_idx = {sub.index: sub for sub in target_subs}


    original_samples = build_original_samples(target_subs,original_subs_by_idx, original_voice_audio_file)
    fallback_decisions = build_fallback_decisions(speakers, original_samples, min_required_sec=min_clone_sec, changed_list=changed_list if not use_batch_alignment else None)



    # Build fallback voices per gender once
    if any(
            item["fallback"]
            for item in fallback_decisions
            if str(item.get("gender", "")).lower() == "male"
    ):
        male_fallback = voices["male"] if voices and voices["male"] else _pick_fish_public_voice(client, "male", language_code)
    else:
        male_fallback = None
    if any(
            item["fallback"]
            for item in fallback_decisions
            if str(item.get("gender", "")).lower() == "female"
    ):
        female_fallback = voices["female"] if voices and voices["female"] else _pick_fish_public_voice(client, "female", language_code)
    else:
        female_fallback = None
    speaker_voices = {}
    for item in fallback_decisions:
        if not item.get("fallback"):
            continue
        speaker = item["speaker"]
        gender = str(item.get("gender", "")).lower()
        if gender == "male":
            speaker_voices[speaker] = male_fallback
        elif gender == "female":
            speaker_voices[speaker] = female_fallback

    process_speakers = [
        item for item in fallback_decisions
        if not item.get("fallback", False)
    ]

    groups = [
        process_speakers[i:i + 10]
        for i in range(0, len(process_speakers), 10)
    ]
    avaliable_voices=client.voices.list(self_only=True)
    for group in groups:
        prepare_jobs = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for speaker_dict in group:
                for voice in avaliable_voices.items or []:
                    if voice.title==f"tmp_{speaker_dict['speaker']}_{run_id}":
                        speaker_voices[speaker_dict['speaker']] = voice.id
                        break
                if speaker_voices.get(speaker_dict['speaker']) is None:
                    prepare_jobs.append(
                        ex.submit(prepare_one_voice, speaker_dict,client, run_id)
                    )
            for fut in as_completed(prepare_jobs):
                pv = fut.result()
                speaker_voices[pv['speaker']] = pv['voice_id']
    results = []
    speaker_idx_to_emotion = build_emotion_idx(speakers)
    config = TTSConfig(
        temperature=0.1,
        top_p=0.1,
        repetition_penalty=0.00,
        chunk_length=220,
        min_chunk_length=10,
        condition_on_previous_chunks=True,
        early_stop_threshold=1,
        latency="balanced",
        prosody=Prosody(speed=1.0, volume=0, normalize_loudness=True)
    )
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs=[]

            if use_batch_alignment:
                batch_size = 10

                for speaker, data in speakers.items():
                    voice_id = speaker_voices[speaker]
                    groups = (data or {}).get("groups", {}) or {}

                    accent_str = f"[natural {ACCENT_PROMPT_MAP.get(language_code, '')} accent]"

                    output_dir = os.path.join(out_dir, speaker)
                    os.makedirs(output_dir, exist_ok=True)

                    # Flatten all emotion groups into one ordered timeline for this speaker
                    merged_items = []
                    for emotion, group in groups.items():
                        idxs = group.get("idxs", []) or []
                        for sub_idx in idxs:
                            merged_items.append({
                                "sub_idx": sub_idx,
                                "emotion": emotion,
                            })

                    # Sort by subtitle order (or by subtitle start time if safer)
                    merged_items.sort(
                        key=lambda x: translated_subs_by_idx[x["sub_idx"]].start
                    )
                    # alternatively:
                    # merged_items.sort(key=lambda x: x["sub_idx"])

                    for start in range(0, len(merged_items), batch_size):
                        batch_items = merged_items[start:start + batch_size]

                        lines = []
                        subtitle_items = []

                        prev_emotion = None

                        for i, item in enumerate(batch_items):
                            sub_idx = item["sub_idx"]
                            emotion = item["emotion"]

                            raw_text = translated_subs_by_idx[sub_idx].content.strip()
                            text = raw_text.split(":", 1)[1].strip() if ":" in raw_text else raw_text

                            duration = (
                                    translated_subs_by_idx[sub_idx].end
                                    - translated_subs_by_idx[sub_idx].start
                            ).total_seconds()

                            if len(text.strip()) < 1:
                                duration_ms = max(1, int(round(duration * 1000)))
                                write_subtitle_silence(os.path.join(output_dir, f"{sub_idx}.wav"), duration_ms)
                                continue


                            emotion_str = EMOTION_PROMPT_MAP.get(emotion, "[neutral]")

                            # Add accent once at the start of each batch.
                            # Add emotion tag on first line or when emotion changes.
                            if i == 0:
                                line = f"{accent_str} [neutral] {text}".strip()
                                emotion = 'neutral'
                            elif emotion != prev_emotion:
                                line = f"{emotion_str} {text}".strip()
                            else:
                                line = text

                            lines.append(line)

                            subtitle_items.append({
                                "sub_id": sub_idx,
                                "text": text,  # clean text for Whisper alignment
                                "duration": duration,
                                "emotion": emotion,
                            })

                            prev_emotion = emotion

                        if not lines:
                            continue

                        futs.append(
                            ex.submit(
                                synthesize_group,
                                lines,
                                subtitle_items,
                                voice_id,
                                output_dir,
                                client,
                                config
                            )
                        )
            else:
                for sub in target_subs:
                    idx = sub.index
                    if changed_list and idx not in changed_list:
                        continue
                    speaker, sub_text = [x.strip() for x in sub.content.split(":", 1)]
                    output_dir = os.path.join(out_dir, speaker)


                    if len(sub_text.strip()) < 1:
                        duration_ms = max(
                            1,
                            int(round((sub.end - sub.start).total_seconds() * 1000))
                        )
                        write_subtitle_silence(os.path.join(output_dir, f"{idx}.wav"), duration_ms)
                        continue

                    voice_id = speaker_voices[speaker]
                    emotion = speaker_idx_to_emotion.get(speaker, {}).get(idx, "[neutral]")
                    # emotion = "[neutral]"
                    accent_str = f"[natural {ACCENT_PROMPT_MAP.get(language_code, '')} accent]"
                    os.makedirs(output_dir, exist_ok=True)
                    futs.append(
                        ex.submit(
                            synthesize_one,
                            sub_text,
                            idx,
                            voice_id,
                            emotion,
                            output_dir,
                            client,
                            accent_str,
                            config
                        )
                    )
            #
            #
            # for speaker, data in speakers.items():
            #     groups = (data or {}).get("groups", {}) or {}
            #     for emotion, group in groups.items():
            #         accent_str = f"[natural {ACCENT_PROMPT_MAP.get(language_code, "")} accent]"
            #         emotion_str = EMOTION_PROMPT_MAP.get(emotion, '[neutral]')
            #         output_dir = os.path.join(out_dir, speaker)
            #         os.makedirs(output_dir, exist_ok=True)
            #         lines = []
            #         for i, sub_idx in  enumerate(group["idxs"]):
            #             if i == 0:
            #                 lines.append(f"{accent_str} {emotion_str} {translated_subs_by_idx[sub_idx].content.strip().split(":", 1)[1]}")
            #             else:
            #                 lines.append(f"{translated_subs_by_idx[sub_idx].content.strip().split(":", 1)[1]}")
            #         futs.append(
            #             ex.submit(
            #                 generate_sync,
            #                 client,
            #                 lines,
            #                 group["idxs"],
            #                 speaker_voices[speaker],
            #                 config,
            #                 output_dir,
            #             )
            #         )




            for fut in as_completed(futs):
                results.append(fut.result())
    finally:
        # -----------------------------
        # Step 3: delete temp voices
        # -----------------------------
        if force_delete:
            delete_jobs = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for voice_id in speaker_voices.values():
                    if voice_id not in (male_fallback, female_fallback):
                        delete_jobs.append(ex.submit(client.voices.delete, voice_id))
                for fut in as_completed(delete_jobs):
                    try:
                        fut.result()
                    except Exception:
                        pass

    print(f"✅ Generated {sum(results) if use_batch_alignment else len(results)} Fish Audio TTS segments using {max_workers} threads.")

    return {"male": male_fallback, "female": female_fallback}

#
# import json
# from pathlib import Path
#
# translated_subtitles_file = "output/20260404_a9rz/data/subtitles_translated.srt"
# subtitles_file = "output/20260404_a9rz/data/subtitles.srt"
# original_voice_audio_file = "output/20260404_a9rz/audio/vocal.wav"
# api_key="39b62dc4dfa7448084a005f75818b2e5"
# speakers = json.loads(Path("output/20260404_a9rz/data/speakers_data.json").read_text())
# out_dir = "output/20260404_a9rz/audio/speakers"
# language_code = "en"
# tts_generate_multivoice_fish_segments(
#     translated_subtitles_file,
#     subtitles_file,
#     original_voice_audio_file,
#     api_key,
#     speakers,
#     out_dir,
#     language_code
# )
#
# print ("asd")