from __future__ import annotations

import os
import torch
import srt
import modal
import torchaudio
from pydub import AudioSegment


PIPELINE_SR = 48000

SILENCE_THRESHOLD = 0.01  # ~-40dB
SILENCE_MARGIN = 0.02     # keep 20ms margin after trim


def _trim_silence(audio: torch.Tensor, sr: int) -> torch.Tensor:
    """Trim leading and trailing silence from audio tensor (1, N)."""
    mono = audio[0] if audio.dim() == 2 else audio
    abs_audio = mono.abs()
    above = (abs_audio > SILENCE_THRESHOLD).nonzero(as_tuple=True)[0]
    if len(above) == 0:
        return audio

    margin_samples = int(SILENCE_MARGIN * sr)
    start = max(0, above[0].item() - margin_samples)
    end = min(len(mono), above[-1].item() + 1 + margin_samples)

    if audio.dim() == 2:
        return audio[:, start:end]
    return audio[start:end]


def _write_silence(output_path: str, duration_ms: int, sample_rate: int = 48000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    silence = (
        AudioSegment
        .silent(duration=max(1, int(duration_ms)), frame_rate=sample_rate)
        .set_channels(1)
        .set_sample_width(2)
    )
    silence.export(output_path, format="wav")


def tts_generate_index_tts2_segments(
    translated_subtitles_file: str,
    speakers_folder: str,
    speakers: dict,
    emotions_tags: dict,
    out_dir: str,
    max_pods: int = 1,
    changed_list: list[int] | None = None,
    duration_factors: dict[int, float] | None = None,
) -> dict:
    with open(translated_subtitles_file, "r", encoding="utf-8") as f:
        target_subs = list(srt.parse(f.read()))
    if not target_subs:
        raise ValueError("translated_subtitles_file is empty")

    translated_subs_by_idx = {sub.index: sub for sub in target_subs}

    speaker_sub_idxs: dict[str, list[int]] = {}
    for sub in target_subs:
        if ":" not in sub.content:
            continue
        spk = sub.content.split(":", 1)[0].strip()
        speaker_sub_idxs.setdefault(spk, []).append(sub.index)

    # Write silence for empty text lines
    for sub in target_subs:
        if ":" not in sub.content:
            continue
        spk = sub.content.split(":", 1)[0].strip()
        text = sub.content.split(":", 1)[1].strip()
        if not text:
            duration_ms = max(1, int((sub.end - sub.start).total_seconds() * 1000))
            _write_silence(os.path.join(out_dir, spk, f"{sub.index}.wav"), duration_ms)

    # Load reference audios keyed by speaker name
    speaker_ref_audio: dict[str, bytes] = {}
    for speaker in speakers.keys():
        ref_audio_path = os.path.join(speakers_folder, f"{speaker}.wav")
        if not os.path.exists(ref_audio_path):
            print(f"Warning: no reference audio for {speaker}, skipping")
            continue
        with open(ref_audio_path, "rb") as f:
            speaker_ref_audio[speaker] = f.read()

    # Build flat subtitle list ordered by speaker
    all_items: list[dict] = []
    for speaker in speakers.keys():
        if speaker not in speaker_ref_audio:
            continue
        sub_idxs = speaker_sub_idxs.get(speaker, [])
        if not sub_idxs:
            continue
        if changed_list:
            sub_idxs = [idx for idx in sub_idxs if idx in changed_list]
            if not sub_idxs:
                continue

        for idx in sub_idxs:
            sub = translated_subs_by_idx.get(idx)
            if sub is None:
                continue
            raw_text = sub.content.strip()
            text = raw_text.split(":", 1)[1].strip() if ":" in raw_text else raw_text
            if not text:
                continue

            duration_sec = (sub.end - sub.start).total_seconds()

            tag_data = emotions_tags.get(idx) or emotions_tags.get(str(idx))
            if tag_data and isinstance(tag_data, dict):
                emo_vector = tag_data.get("emo_vector", [0.0] * 8)
            else:
                emo_vector = [0.0] * 8

            if duration_factors and idx in duration_factors:
                duration_factor = duration_factors[idx]
            else:
                duration_factor = 1.0

            all_items.append({
                "speaker": speaker,
                "idx": idx,
                "text": text,
                "emo_vector": emo_vector,
                "duration_factor": duration_factor,
                "duration_sec": duration_sec,
            })

    if not all_items:
        print("IndexTTS2: no items to process")
        return {}

    # Split into max_pods even chunks
    if not all_items:
        chunks = []
    else:
        chunk_size = (len(all_items) + max_pods - 1) // max_pods
        chunks = [all_items[i:i + chunk_size] for i in range(0, len(all_items), chunk_size)]

    # Build map inputs: per chunk, collect only the ref audios used
    map_ref_audios = []
    map_subtitles = []
    for chunk in chunks:
        chunk_speakers = set(item["speaker"] for item in chunk)
        map_ref_audios.append({spk: speaker_ref_audio[spk] for spk in chunk_speakers})
        map_subtitles.append(chunk)

    IndexTTSGenerator = modal.Cls.from_name("index-tts-2-5-generator", "IndexTTSGenerator")
    tts_service = IndexTTSGenerator()

    # Build idx->speaker lookup for output routing
    idx_to_speaker = {item["idx"]: item["speaker"] for item in all_items}


    total_generated = 0
    for pod_results in tts_service.generate.map(map_ref_audios, map_subtitles):
        for item in pod_results:
            speaker = idx_to_speaker[item["idx"]]
            output_dir = os.path.join(out_dir, speaker)
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{item['idx']}.wav")
            with open(out_path, "wb") as f:
                f.write(item["audio_bytes"])
            audio, sr = torchaudio.load(out_path)
            if sr != PIPELINE_SR:
                audio = torchaudio.functional.resample(audio, sr, PIPELINE_SR)
            audio = _trim_silence(audio, PIPELINE_SR)
            torchaudio.save(out_path, audio, PIPELINE_SR)
            total_generated += 1

    print(f"IndexTTS2: {total_generated} segments, {len(chunks)} pods")

    return {}
