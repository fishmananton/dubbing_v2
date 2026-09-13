from __future__ import annotations

import os
import re
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


def _normalize_for_tts(text: str) -> str:
    """Fix abbreviations that the TN normalizer misexpands."""
    # "No." meaning "no" (not "Number") — protect when NOT followed by a digit
    text = re.sub(r'\bNo\.(?!\s*\d)', 'No', text)
    return text


def _strip_pause_punctuation(text: str) -> str:
    """Remove punctuation that causes TTS pause tokens, keeping intonation markers."""
    text = text.replace('...', '').replace('…', '')
    text = re.sub(r'[,;:—–]', '', text)  # comma, semicolon, colon, em/en dash
    text = re.sub(r'\s*-\s+', ' ', text)  # spaced dashes (not in-word hyphens like "thirty-two")
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


COLD_BOOT_S = 75
CHARS_PER_SEC_PER_GPU = 8.5
TARGET_POD_SECONDS = 150

# Phonation styles the 8-dim emo_vector cannot represent. Detected from the
# free-text emotion_tag; when matched, the line is driven by a reference clip
# (emo_audio_prompt) instead of the vector. First match wins, so order matters.
PHONATION_PATTERNS = [
    ("whisper", re.compile(r"whisper|hushed", re.IGNORECASE)),
    ("shout", re.compile(r"shout|yell|scream|holler", re.IGNORECASE)),
    ("cry", re.compile(r"\bcry|crying|sobbing|sobs|weeping|tearful", re.IGNORECASE)),
]


def _detect_phonation_style(emotion_tag: str | None) -> str | None:
    """Return 'whisper' | 'shout' | 'cry' if the tag names that style, else None."""
    if not emotion_tag:
        return None
    for style, pat in PHONATION_PATTERNS:
        if pat.search(emotion_tag):
            return style
    return None


def _normalize_for_dedup(text: str) -> str:
    """Strip punctuation and lowercase for duplicate comparison."""
    return re.sub(r'[^\w\s]', '', text).lower().strip()


def _dedup_candidates(candidate_texts: dict[int, list[str]]) -> dict[int, list[str]]:
    """Remove duplicate variants per index, comparing without punctuation."""
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
        print(f"  🔍 Dedup: {total_before} -> {total_after} variants ({total_before - total_after} duplicates removed)")
    return deduped


def _compute_balanced_chunks(
    all_items: list[dict],
    num_pods: int,
    warm_pods: int,
) -> tuple[list[list[dict]], list[int]]:
    """
    Bin-pack items into chunks with balanced finish times for warm/cold pods.

    Warm pods get more chars (they start immediately), cold pods get fewer
    (they spend COLD_BOOT_S booting first). The goal:
      warm_gen_time == COLD_BOOT_S + cold_gen_time
    """
    total_chars = sum(len(item["text"]) for item in all_items)
    cold_pods = num_pods - warm_pods

    if cold_pods <= 0 or warm_pods <= 0:
        # All same type — equal distribution via standard LPT
        chunks: list[list[dict]] = [[] for _ in range(num_pods)]
        pod_loads = [0] * num_pods
        for item in sorted(all_items, key=lambda x: len(x["text"]), reverse=True):
            lightest = min(range(num_pods), key=lambda i: pod_loads[i])
            chunks[lightest].append(item)
            pod_loads[lightest] += len(item["text"])
        return chunks, pod_loads

    # Target chars per pod type so both finish at the same time
    c_target = (total_chars - warm_pods * COLD_BOOT_S * CHARS_PER_SEC_PER_GPU) / num_pods
    if c_target < 0:
        # Warm pods can absorb all work during cold boot — give cold pods nothing
        chunks = [[] for _ in range(warm_pods)]
        pod_loads = [0] * warm_pods
        for item in sorted(all_items, key=lambda x: len(x["text"]), reverse=True):
            lightest = min(range(warm_pods), key=lambda i: pod_loads[i])
            chunks[lightest].append(item)
            pod_loads[lightest] += len(item["text"])
        return chunks, pod_loads

    w_target = c_target + COLD_BOOT_S * CHARS_PER_SEC_PER_GPU

    # LPT bin-packing with per-bucket capacity targets
    warm_chunks: list[list[dict]] = [[] for _ in range(warm_pods)]
    cold_chunks: list[list[dict]] = [[] for _ in range(cold_pods)]
    warm_loads = [0.0] * warm_pods
    cold_loads = [0.0] * cold_pods

    for item in sorted(all_items, key=lambda x: len(x["text"]), reverse=True):
        item_chars = len(item["text"])
        # Find least-loaded bucket relative to its target
        best_idx = -1
        best_ratio = float('inf')

        for i in range(warm_pods):
            ratio = warm_loads[i] / w_target if w_target > 0 else float('inf')
            if ratio < best_ratio:
                best_ratio = ratio
                best_idx = i
                best_type = 'warm'

        for i in range(cold_pods):
            ratio = cold_loads[i] / c_target if c_target > 0 else float('inf')
            if ratio < best_ratio:
                best_ratio = ratio
                best_idx = i
                best_type = 'cold'

        if best_type == 'warm':
            warm_chunks[best_idx].append(item)
            warm_loads[best_idx] += item_chars
        else:
            cold_chunks[best_idx].append(item)
            cold_loads[best_idx] += item_chars

    chunks = warm_chunks + cold_chunks
    pod_loads = [int(l) for l in warm_loads] + [int(l) for l in cold_loads]
    return chunks, pod_loads


def tts_generate_index_tts2_segments(
    translated_subtitles_file: str,
    speakers_folder: str,
    speakers: dict,
    emotions_tags: dict,
    out_dir: str,
    max_pods: int = 20,
    changed_list: list[int] | None = None,
    duration_factors: dict[int, float] | None = None,
    candidate_texts: dict[int, list[str]] | None = None,
    speaker_base_atempo: dict[str, float] | None = None,
    warm_pods: int = 0,
) -> dict:
    """
    Generate TTS segments via IndexTTS2 Modal service.

    Args:
        candidate_texts: {idx: [text_v0, text_v1, ...]} — for these indices, generate
            ALL variants in one batch. After generation, measure durations and keep the
            longest variant that fits the subtitle window. Others are deleted.
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

    if candidate_texts is None:
        candidate_texts = {}
    else:
        candidate_texts = _dedup_candidates(candidate_texts)

    # Indices that have candidates — skip normal text extraction for these
    candidate_indices = set(candidate_texts.keys())

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
            text = _strip_pause_punctuation(_normalize_for_tts(text))

            duration_sec = (sub.end - sub.start).total_seconds()

            tag_data = emotions_tags.get(idx) or emotions_tags.get(str(idx))
            if tag_data and isinstance(tag_data, dict):
                emo_vector = tag_data.get("emo_vector", [0.0] * 8)
                phonation_style = _detect_phonation_style(tag_data.get("emotion_tag"))
            else:
                emo_vector = [0.0] * 8
                phonation_style = None

            if duration_factors and idx in duration_factors:
                duration_factor = duration_factors[idx]
            else:
                duration_factor = 1.0

            if idx in candidate_indices:
                # Variant 0 is the original text — reuse existing wav, only generate variants 1+
                for vi, variant_text in enumerate(candidate_texts[idx]):
                    if not variant_text:
                        continue
                    if vi == 0:
                        continue
                    all_items.append({
                        "speaker": speaker,
                        "idx": idx * 1000 + vi,
                        "text": _strip_pause_punctuation(_normalize_for_tts(variant_text)),
                        "emo_vector": emo_vector,
                        "phonation_style": phonation_style,
                        "duration_factor": duration_factor,
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
                    "emo_vector": emo_vector,
                    "phonation_style": phonation_style,
                    "duration_factor": duration_factor,
                    "duration_sec": duration_sec,
                })

    if not all_items:
        print("IndexTTS2: no items to process")
        return {}

    # Auto-scale pod count based on total text length
    total_chars = sum(len(item["text"]) for item in all_items)
    num_pods = max(1, min(max_pods, -(-total_chars // int(CHARS_PER_SEC_PER_GPU * TARGET_POD_SECONDS))))

    # Balanced bin-packing: warm pods get more work, cold pods get less
    warm_pods = min(warm_pods, num_pods)
    chunks, pod_loads = _compute_balanced_chunks(all_items, num_pods, warm_pods)
    # chunks may be fewer than num_pods if warm pods absorb everything
    num_pods = len(chunks)

    # Build map inputs: per chunk, collect only the ref audios used
    def _build_map_inputs(chunk_list):
        refs, subs = [], []
        for chunk in chunk_list:
            chunk_speakers = set(item["speaker"] for item in chunk)
            refs.append({spk: speaker_ref_audio[spk] for spk in chunk_speakers})
            subs.append(chunk)
        return refs, subs

    IndexTTSGenerator = modal.Cls.from_name("index-tts-2-5-generator", "IndexTTSGenerator")
    tts_service = IndexTTSGenerator()

    # Build idx->speaker lookup for output routing
    idx_to_speaker = {item["idx"]: item["speaker"] for item in all_items}

    # Track which virtual indices are candidate variants
    virtual_to_real: dict[int, tuple[int, int]] = {}  # virtual_idx -> (real_idx, variant_num)
    for item in all_items:
        if "_real_idx" in item:
            virtual_to_real[item["idx"]] = (item["_real_idx"], item["_variant"])

    # Track candidate variants: {real_idx: [(variant_num, out_path, duration_ms)]}
    candidate_results: dict[int, list[tuple[int, str, int]]] = {}

    # Two-phase dispatch: .map() warm chunks first (warm containers grab them),
    # then .spawn() cold chunks (warm containers are busy, Modal boots new ones).
    cold_count = max(0, num_pods - warm_pods) if warm_pods > 0 else 0
    warm_count = num_pods - cold_count
    warm_chunks = chunks[:warm_count]
    cold_chunks = chunks[warm_count:]

    warm_refs, warm_subs = _build_map_inputs(warm_chunks)
    cold_refs, cold_subs = _build_map_inputs(cold_chunks)

    # Phase 1: dispatch warm chunks via .map() — starts immediately on warm containers
    cold_handles = []
    if cold_chunks:
        # Phase 2: .spawn() cold chunks right after .map() starts
        # Warm containers are now occupied, so Modal must boot new ones for these
        for cr, cs in zip(cold_refs, cold_subs):
            cold_handles.append(tts_service.generate.spawn(cr, cs))

    if warm_chunks:
        print(f"IndexTTS2: dispatching {warm_count} warm + {cold_count} cold pods "
              f"(warm load: {pod_loads[:warm_count]}, cold load: {pod_loads[warm_count:]})")
    else:
        print(f"IndexTTS2: dispatching {cold_count} cold pods (no warm containers)")

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

    # Collect results: warm pods via .map(), cold pods via .spawn().get()
    if warm_chunks:
        for pod_results in tts_service.generate.map(warm_refs, warm_subs):
            for item in pod_results:
                _process_item(item)
    else:
        # No warm containers — single .map() for all (standard cold-start path)
        all_refs, all_subs = _build_map_inputs(chunks)
        for pod_results in tts_service.generate.map(all_refs, all_subs):
            for item in pod_results:
                _process_item(item)

    for handle in cold_handles:
        for item in handle.get():
            _process_item(item)

    # Inject existing wav as variant 0 for each candidate idx (original text, no regen needed)
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

    # Resolve candidates: pick longest that fits subtitle window, delete others
    selected_candidates: dict[int, str] = {}
    for real_idx, variants in candidate_results.items():
        sub = translated_subs_by_idx.get(real_idx)
        if sub:
            available_ms = int((sub.end - sub.start).total_seconds() * 1000)
        else:
            available_ms = 999999

        # In natural atempo mode, "fits" means audio_ms / base_atempo <= available_ms
        # So effective max audio = available_ms * base_atempo
        if speaker_base_atempo:
            spk = sub.content.split(":", 1)[0].strip() if sub and ":" in sub.content else ""
            base = speaker_base_atempo.get(spk, 1.0)
            effective_available_ms = int(available_ms * base)
        else:
            effective_available_ms = available_ms

        # Pick longest variant that fits (best meaning preservation)
        fitting = [(v, path, dur) for v, path, dur in variants if dur <= effective_available_ms]
        if fitting:
            winner = max(fitting, key=lambda x: x[2])
        else:
            # Nothing fits — pick shortest
            winner = min(variants, key=lambda x: x[2])

        # Move winner to the real output path
        winner_v, winner_path, winner_dur = winner
        # Find speaker for this real_idx
        spk_for_idx = ""
        if sub and ":" in sub.content:
            spk_for_idx = sub.content.split(":", 1)[0].strip()
        if not spk_for_idx:
            for vid, (rid, _) in virtual_to_real.items():
                if rid == real_idx:
                    spk_for_idx = idx_to_speaker.get(vid, "")
                    break
        final_path = os.path.join(out_dir, spk_for_idx, f"{real_idx}.wav")
        if winner_path != final_path:
            os.replace(winner_path, final_path)
        selected_candidates[real_idx] = candidate_texts[real_idx][winner_v]

        # Delete losers (skip final_path since winner is there now)
        for v, path, dur in variants:
            if path != final_path and path != winner_path and os.path.exists(path):
                os.unlink(path)

        print(f"  📝 idx={real_idx}: picked variant {winner_v} ({winner_dur}ms / {available_ms}ms available)")

    print(f"IndexTTS2: {total_generated} segments, {num_pods} pods ({warm_pods} warm + {num_pods - warm_pods} cold, "
          f"auto-scaled from {total_chars} chars, load: {pod_loads})")
    if selected_candidates:
        print(f"  🏆 Candidates resolved: {list(selected_candidates.keys())}")

    return selected_candidates
