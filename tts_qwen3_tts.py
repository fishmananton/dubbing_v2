from __future__ import annotations

import json
import os
import re
import torch
import srt
import modal
import torchaudio
from pydub import AudioSegment


PIPELINE_SR = 48000

SILENCE_THRESHOLD = 0.01
SILENCE_MARGIN = 0.02
GLITCH_MAX_MS = 50    # a leading transient this short...
GLITCH_GAP_MS = 30    # ...followed by this much silence is a Seed-VC onset click, not speech


def _speech_onset(mono: torch.Tensor, sr: int) -> int:
    """First sample of real speech, skipping a short leading transient (Seed-VC
    onset click) that is separated from speech by a silent gap. The plain
    first-sample-above-threshold trim can't remove it: the click peaks at
    0.1-0.5, far above SILENCE_THRESHOLD, so it reads as the speech onset.
    Returns -1 if the whole clip is below threshold."""
    frame = max(1, int(sr * 0.010))
    n = mono.shape[-1]
    rms = torch.tensor([
        mono[i:i + frame].pow(2).mean().sqrt() for i in range(0, n, frame)
    ])
    above = rms > SILENCE_THRESHOLD
    if not bool(above.any()):
        return -1
    first = int(above.nonzero()[0].item())
    if first > 1:                       # leading silence, no transient at sample 0
        return first * frame
    run = first
    while run < len(above) and above[run]:
        run += 1
    gap = run
    while gap < len(above) and not above[gap]:
        gap += 1
    run_ms = (run - first) * 10
    gap_ms = (gap - run) * 10
    if run_ms <= GLITCH_MAX_MS and gap_ms >= GLITCH_GAP_MS and gap < len(above):
        return gap * frame             # real speech begins after the gap
    return first * frame               # no glitch signature — keep original onset


def _trim_silence(audio: torch.Tensor, sr: int) -> torch.Tensor:
    mono = audio[0] if audio.dim() == 2 else audio
    onset = _speech_onset(mono, sr)
    if onset < 0:
        return audio

    abs_audio = mono.abs()
    above = (abs_audio > SILENCE_THRESHOLD).nonzero(as_tuple=True)[0]
    margin_samples = int(SILENCE_MARGIN * sr)
    start = max(0, onset - margin_samples)
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


def _normalize_for_tts(text: str) -> str:
    text = re.sub(r'\bNo\.(?!\s*\d)', 'No', text)
    return text


def _strip_pause_punctuation(text: str) -> str:
    text = text.replace('...', '').replace('…', '')
    text = re.sub(r'[,;:—–]', '', text)
    text = re.sub(r'\s*-\s+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


CHARS_PER_SEC_PER_GPU = 8.5
TARGET_POD_SECONDS = 150

QWEN3_SUPPORTED_LANGUAGES = {"en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"}

DST_LANG_TO_QWEN3_CODE = {
    "en": "EN", "zh": "ZH", "ja": "JA", "ko": "KO",
    "de": "DE", "fr": "FR", "ru": "RU", "pt": "PT",
    "es": "ES", "it": "IT",
}


def _normalize_instruct(tag: str) -> str:
    """
    Emotion tags are authored for IndexTTS2, which uses bracketed stage directions
    and <breath>/<sigh> markers. Qwen3 expects plain prose instructions and has no
    tokens for either syntax, so strip them.
    """
    if not tag:
        return ""
    tag = re.sub(r'<[^>]*>', ' ', tag)
    tag = tag.replace('[', ' ').replace(']', ' ')
    tag = re.sub(r'\s{2,}', ' ', tag).strip().strip(',').strip()
    if tag and not tag.endswith(('.', '!', '?')):
        tag += '.'
    return tag[:1].upper() + tag[1:] if tag else ""


VOICE_MAP_FILE = os.path.join(os.path.dirname(__file__), "config", "qwen3_lora_map.json")


def _load_voice_map() -> dict:
    with open(VOICE_MAP_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_voice(voice_map: dict, lang_code: str, gender: str, speaker: str) -> str:
    """
    Pick the LoRA adapter for a (language, gender) pair.

    Speaker identity comes from Seed-VC downstream, so the adapter only supplies
    language and prosody quality and an unmapped pair can degrade to the fallback
    rather than failing the run.
    """
    key = f"{lang_code}_{gender}"
    voice = voice_map["voices"].get(key)
    if voice is None:
        voice = voice_map["fallback"]
        print(f"  Qwen3-TTS: no voice mapped for '{key}' (speaker '{speaker}'), "
              f"using fallback '{voice}'")
    return voice


def _normalize_for_dedup(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text).lower().strip()


def _dedup_candidates(candidate_texts: dict[int, list[str]]) -> dict[int, list[str]]:
    deduped = {}
    total_before = 0
    total_after = 0
    for idx, variants in candidate_texts.items():
        if not variants:
            continue
        total_before += len(variants)
        seen: set[str] = set()
        unique: list[str] = []
        for text in variants:
            normalized = _normalize_for_dedup(text)
            if normalized not in seen:
                seen.add(normalized)
                unique.append(text)
        deduped[idx] = unique
        total_after += len(unique)
    if total_before != total_after:
        print(f"  Dedup: {total_before} -> {total_after} variants ({total_before - total_after} duplicates removed)")
    return deduped


def tts_generate_qwen3_tts_segments(
    translated_subtitles_file: str,
    speakers_folder: str,
    speakers: dict,
    emotions_tags: dict,
    out_dir: str,
    language_code: str = "en",
    max_pods: int = 20,
    changed_list: list[int] | None = None,
    candidate_texts: dict[int, list[str]] | None = None,
    speaker_base_atempo: dict[str, float] | None = None,
) -> dict:
    """
    Generate TTS segments via Qwen3-TTS Modal service.

    Key differences from IndexTTS2:
      - Passes emotion as LLM text (instruct) not emo_vector
      - Passes gender and language per line
      - No warm/cold pod dispatch (no prewarming), so plain LPT bin-packing
    """
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

    for sub in target_subs:
        if ":" not in sub.content:
            continue
        spk = sub.content.split(":", 1)[0].strip()
        text = sub.content.split(":", 1)[1].strip()
        if not text:
            duration_ms = max(1, int((sub.end - sub.start).total_seconds() * 1000))
            _write_silence(os.path.join(out_dir, spk, f"{sub.index}.wav"), duration_ms)

    speaker_ref_audio: dict[str, bytes] = {}
    for speaker in speakers.keys():
        ref_audio_path = os.path.join(speakers_folder, f"{speaker}.wav")
        if not os.path.exists(ref_audio_path):
            print(f"Warning: no reference audio for {speaker}, skipping")
            continue
        with open(ref_audio_path, "rb") as f:
            speaker_ref_audio[speaker] = f.read()

    if candidate_texts is None:
        candidate_texts = {}
    else:
        candidate_texts = _dedup_candidates(candidate_texts)

    candidate_indices = set(candidate_texts.keys())

    if language_code.lower() not in DST_LANG_TO_QWEN3_CODE:
        raise ValueError(
            f"Qwen3-TTS does not support language '{language_code}'. "
            f"Supported: {sorted(DST_LANG_TO_QWEN3_CODE)}"
        )
    lang_code = DST_LANG_TO_QWEN3_CODE[language_code.lower()]
    voice_map = _load_voice_map()

    all_items: list[dict] = []
    for speaker, speaker_data in speakers.items():
        if speaker not in speaker_ref_audio:
            continue
        sub_idxs = speaker_sub_idxs.get(speaker, [])
        if not sub_idxs:
            continue
        if changed_list:
            sub_idxs = [idx for idx in sub_idxs if idx in changed_list]
            if not sub_idxs:
                continue

        gender = speaker_data.get("gender", "male")
        voice = _resolve_voice(voice_map, lang_code, gender, speaker)

        for idx in sub_idxs:
            sub = translated_subs_by_idx.get(idx)
            if sub is None:
                continue
            raw_text = sub.content.strip()
            text = raw_text.split(":", 1)[1].strip() if ":" in raw_text else raw_text
            # text = _strip_pause_punctuation(_normalize_for_tts(text))
            text = _normalize_for_tts(text)

            duration_sec = (sub.end - sub.start).total_seconds()

            tag_data = emotions_tags.get(idx) or emotions_tags.get(str(idx))
            instruct = ""
            if tag_data and isinstance(tag_data, dict):
                instruct = _normalize_instruct(tag_data.get("emotion_tag", ""))

            if idx in candidate_indices:
                for vi, variant_text in enumerate(candidate_texts[idx]):
                    if not variant_text:
                        continue
                    if vi == 0:
                        continue
                    all_items.append({
                        "speaker": speaker,
                        "idx": idx * 1000 + vi,
                        "text": _strip_pause_punctuation(_normalize_for_tts(variant_text)),
                        "instruct": instruct,
                        "lang": lang_code,
                        "gender": gender,
                        "voice_name": voice,
                        "duration_factor": 1.0,
                        "emo_vector": [],
                        "duration_sec": duration_sec,
                        "_real_idx": idx,
                        "_variant": vi,
                    })
            else:
                if not text:
                    continue
                all_items.append({
                    "speaker": speaker,
                    "idx": idx,
                    "text": text,
                    "instruct": instruct,
                    "lang": lang_code,
                    "gender": gender,
                    "voice_name": voice,
                    "duration_factor": 1.0,
                    "emo_vector": [],
                    "duration_sec": duration_sec,
                })

    if not all_items:
        print("Qwen3-TTS: no items to process")
        return {}

    total_chars = sum(len(item["text"]) for item in all_items)
    num_pods = max(1, min(max_pods, -(-total_chars // int(CHARS_PER_SEC_PER_GPU * TARGET_POD_SECONDS))))

    chunks: list[list[dict]] = [[] for _ in range(num_pods)]
    pod_loads = [0] * num_pods
    for item in sorted(all_items, key=lambda x: len(x["text"]), reverse=True):
        lightest = min(range(num_pods), key=lambda i: pod_loads[i])
        chunks[lightest].append(item)
        pod_loads[lightest] += len(item["text"])

    pod_loads = [load for chunk, load in zip(chunks, pod_loads) if chunk]
    chunks = [c for c in chunks if c]
    num_pods = len(chunks)

    def _build_map_inputs(chunk_list):
        refs, subs = [], []
        for chunk in chunk_list:
            chunk_speakers = set(item["speaker"] for item in chunk)
            refs.append({spk: speaker_ref_audio[spk] for spk in chunk_speakers})
            subs.append(chunk)
        return refs, subs

    Qwen3TTSGenerator = modal.Cls.from_name("qwen3-tts-generator", "Qwen3TTSGenerator")
    tts_service = Qwen3TTSGenerator()

    idx_to_speaker = {item["idx"]: item["speaker"] for item in all_items}

    virtual_to_real: dict[int, tuple[int, int]] = {}
    for item in all_items:
        if "_real_idx" in item:
            virtual_to_real[item["idx"]] = (item["_real_idx"], item["_variant"])

    candidate_results: dict[int, list[tuple[int, str, int]]] = {}

    total_generated = 0

    def _process_item(item):
        nonlocal total_generated
        item_idx = item["idx"]
        speaker = idx_to_speaker[item_idx]
        output_dir_path = os.path.join(out_dir, speaker)
        os.makedirs(output_dir_path, exist_ok=True)

        is_variant = item_idx in virtual_to_real
        if is_variant:
            real_idx, variant_num = virtual_to_real[item_idx]
            out_path = os.path.join(output_dir_path, f"{real_idx}_v{variant_num}.wav")
        else:
            out_path = os.path.join(output_dir_path, f"{item_idx}.wav")

        with open(out_path, "wb") as f:
            f.write(item["audio_bytes"])
        audio, sr = torchaudio.load(out_path)
        if sr != PIPELINE_SR:
            audio = torchaudio.functional.resample(audio, sr, PIPELINE_SR)
        audio = _trim_silence(audio, PIPELINE_SR)
        torchaudio.save(out_path, audio, PIPELINE_SR)
        total_generated += 1

        if is_variant:
            duration_ms = int(audio.shape[-1] * 1000 / PIPELINE_SR)
            candidate_results.setdefault(real_idx, []).append((variant_num, out_path, duration_ms))

    all_refs, all_subs = _build_map_inputs(chunks)
    for pod_results in tts_service.generate_lora_vc_multi.map(all_refs, all_subs):
        for item in pod_results:
            _process_item(item)

    for real_idx in candidate_texts:
        if real_idx not in candidate_results:
            candidate_results[real_idx] = []
        sub = translated_subs_by_idx.get(real_idx)
        if not sub:
            continue
        spk = sub.content.split(":", 1)[0].strip() if ":" in sub.content else ""
        existing_path = os.path.join(out_dir, spk, f"{real_idx}.wav")
        if os.path.exists(existing_path):
            audio, sr = torchaudio.load(existing_path)
            if sr != PIPELINE_SR:
                audio = torchaudio.functional.resample(audio, sr, PIPELINE_SR)
            duration_ms = int(audio.shape[-1] * 1000 / PIPELINE_SR)
            candidate_results[real_idx].insert(0, (0, existing_path, duration_ms))

    selected_candidates: dict[int, str] = {}
    for real_idx, variants in candidate_results.items():
        sub = translated_subs_by_idx.get(real_idx)
        if sub:
            available_ms = int((sub.end - sub.start).total_seconds() * 1000)
        else:
            available_ms = 999999

        if speaker_base_atempo:
            spk = sub.content.split(":", 1)[0].strip() if sub and ":" in sub.content else ""
            base = speaker_base_atempo.get(spk, 1.0)
            effective_available_ms = int(available_ms * base)
        else:
            effective_available_ms = available_ms

        fitting = [(v, path, dur) for v, path, dur in variants if dur <= effective_available_ms]
        if fitting:
            winner = max(fitting, key=lambda x: x[2])
        else:
            winner = min(variants, key=lambda x: x[2])

        winner_v, winner_path, winner_dur = winner
        spk_for_idx = ""
        if sub and ":" in sub.content:
            spk_for_idx = sub.content.split(":", 1)[0].strip()
        if not spk_for_idx:
            for vid, (rid, _vnum) in virtual_to_real.items():
                if rid == real_idx:
                    spk_for_idx = idx_to_speaker.get(vid, "")
                    break
        final_path = os.path.join(out_dir, spk_for_idx, f"{real_idx}.wav")
        if winner_path != final_path:
            os.replace(winner_path, final_path)
        selected_candidates[real_idx] = candidate_texts[real_idx][winner_v]

        for v, path, dur in variants:
            if path != final_path and path != winner_path and os.path.exists(path):
                os.unlink(path)

        print(f"  idx={real_idx}: picked variant {winner_v} ({winner_dur}ms / {available_ms}ms available)")

    print(f"Qwen3-TTS: {total_generated} segments, {num_pods} pods "
          f"(auto-scaled from {total_chars} chars, load: {pod_loads})")
    if selected_candidates:
        print(f"  Candidates resolved: {list(selected_candidates.keys())}")

    return selected_candidates
