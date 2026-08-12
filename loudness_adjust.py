from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import defaultdict
import os
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import srt

@dataclass
class LoudnessConfig:
    target_sr: int = 48000
    max_gain_db: float = 24.0
    min_gain_db: float = -24.0
    peak_target_dbfs: float = -2.0
    fallback_target_lufs: float = -19.0
    min_ref_duration_sec: float = 0.35
    silence_like_lufs_threshold: float = -45.0
    ref_pad_before_sec: float = 0.03
    ref_pad_after_sec: float = 0.05
    enable_gain_smoothing: bool = True
    max_adjacent_gain_jump_db: float = 3.0
    speaker_reset_gap_sec: float = 20.0
    fade_ms: int = 10


def read_wav_float(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    return audio, sr


def write_wav_float(path: str | Path, audio: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32, copy=False), sr, format="WAV", subtype="FLOAT")


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return np.mean(audio, axis=1, dtype=np.float32).astype(np.float32, copy=False)


def peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        return -120.0
    return 20.0 * np.log10(peak)


def apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    gain = 10.0 ** (gain_db / 20.0)
    return (audio * gain).astype(np.float32, copy=False)


def peak_limit_by_scaling(audio: np.ndarray, peak_target_dbfs: float) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        return audio.astype(np.float32, copy=False)

    peak_target = 10.0 ** (peak_target_dbfs / 20.0)
    if peak > peak_target:
        audio = audio * (peak_target / peak)
    return audio.astype(np.float32, copy=False)


def apply_fades(audio: np.ndarray, sr: int, fade_ms: int) -> np.ndarray:
    if fade_ms <= 0 or audio.size == 0:
        return audio.astype(np.float32, copy=False)

    fade_samples = int(sr * fade_ms / 1000.0)
    if fade_samples <= 0:
        return audio.astype(np.float32, copy=False)

    audio = audio.astype(np.float32, copy=True)

    if audio.ndim == 1:
        n = len(audio)
        fade_samples = min(fade_samples, n // 2)
        if fade_samples > 0:
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out
        return audio

    n = audio.shape[0]
    fade_samples = min(fade_samples, n // 2)
    if fade_samples > 0:
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)[:, None]
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)[:, None]
        audio[:fade_samples, :] *= fade_in
        audio[-fade_samples:, :] *= fade_out
    return audio


def try_measure_lufs(audio: np.ndarray, sr: int, min_duration_sec: float) -> Optional[float]:
    if len(audio) / float(sr) < min_duration_sec:
        return None
    try:
        meter = pyln.Meter(sr)
        value = meter.integrated_loudness(audio)
        return float(value) if np.isfinite(value) else None
    except Exception:
        return None


def extract_reference_segment(
    vocals_audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    cfg: LoudnessConfig,
) -> np.ndarray:
    start = max(0.0, start_sec - cfg.ref_pad_before_sec)
    end = max(start, end_sec + cfg.ref_pad_after_sec)
    start_sample = max(0, int(round(start * sr)))
    end_sample = max(start_sample, int(round(end * sr)))
    return vocals_audio[start_sample:end_sample].copy()


def compute_gain_db(
    tts_lufs: Optional[float],
    ref_lufs: Optional[float],
    cfg: LoudnessConfig,
) -> tuple[float, str]:
    if tts_lufs is None:
        return 0.0, "tts_lufs_invalid"

    target_lufs = ref_lufs if ref_lufs is not None else cfg.fallback_target_lufs
    gain_db = float(np.clip(target_lufs - tts_lufs, cfg.min_gain_db, cfg.max_gain_db))
    return gain_db, "matched" if ref_lufs is not None else "fallback"


def smooth_gains_by_speaker(
    entries: List[Dict[str, Any]],
    cfg: LoudnessConfig,
) -> None:
    if not cfg.enable_gain_smoothing or not entries:
        for e in entries:
            e["smoothed_gain_db"] = e["raw_gain_db"]
        return

    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["_speaker"]].append(entry)

    for speaker_entries in grouped.values():
        prev_gain = None
        prev_end = None

        for entry in speaker_entries:
            raw_gain = entry["raw_gain_db"]
            start = entry["_start"]
            end = entry["_end"]

            if prev_gain is None or prev_end is None or (start - prev_end) > cfg.speaker_reset_gap_sec:
                smoothed = raw_gain
            else:
                low = prev_gain - cfg.max_adjacent_gain_jump_db
                high = prev_gain + cfg.max_adjacent_gain_jump_db
                smoothed = float(np.clip(raw_gain, low, high))

            entry["smoothed_gain_db"] = smoothed
            prev_gain = smoothed
            prev_end = end


def run_line_loudness_stage(
    subtitles_file: str,
    tts_segments_dir: str,
    vocals_wav_path: str,
    cfg: LoudnessConfig | None = None,
) -> List[Dict[str, Any]]:

    cfg = cfg or LoudnessConfig()

    vocals_audio, vocals_sr = read_wav_float(vocals_wav_path)
    vocals_audio = to_mono(vocals_audio)

    if vocals_sr != cfg.target_sr:
        raise ValueError(f"Unexpected vocals sample rate: {vocals_sr}, expected {cfg.target_sr}")

    entries: List[Dict[str, Any]] = []
    with open(subtitles_file, 'r', encoding='utf-8') as f:
        subs = list(srt.parse(f.read()))
    for sub in subs:
        content = sub.content.strip()
        speaker = content.split(":", 1)[0].strip() if ":" in content else "__NO_SPEAKER__"

        idx = int(sub.index)
        start = sub.start.total_seconds()
        end = sub.end.total_seconds()
        tts_path = os.path.join(tts_segments_dir, speaker, f"{sub.index}.wav")
        out_path = os.path.join(tts_segments_dir, speaker, f"{sub.index}_loudness_out.wav")

        tts_audio, tts_sr = read_wav_float(tts_path)
        if tts_sr != cfg.target_sr:
            raise ValueError(f"Unexpected TTS sample rate for {tts_path}: {tts_sr}, expected {cfg.target_sr}")

        tts_mono = to_mono(tts_audio)
        ref_audio = extract_reference_segment(vocals_audio, vocals_sr, start, end, cfg)

        ref_lufs = try_measure_lufs(ref_audio, vocals_sr, cfg.min_ref_duration_sec)
        if ref_lufs is not None:
            ref_peak = float(np.max(np.abs(ref_audio))) if ref_audio.size else 0.0
            if ref_lufs < cfg.silence_like_lufs_threshold or ref_peak < 1e-4:
                ref_lufs = None

        tts_lufs = try_measure_lufs(tts_mono, tts_sr, cfg.min_ref_duration_sec)
        raw_gain_db, gain_mode = compute_gain_db(tts_lufs, ref_lufs, cfg)

        entries.append({
            "idx": idx,
            "_speaker": speaker,
            "_start": start,
            "_end": end,
            "tts_path": str(tts_path),
            "final_line_path": str(out_path),
            "ref_lufs": ref_lufs,
            "tts_lufs": tts_lufs,
            "raw_gain_db": raw_gain_db,
            "gain_mode": gain_mode,
            "peak_before_dbfs": peak_dbfs(tts_mono),
        })

    entries.sort(key=lambda x: (x["_start"], x["idx"]))
    smooth_gains_by_speaker(entries, cfg)

    final_entries: List[Dict[str, Any]] = []

    for entry in entries:
        audio, sr = read_wav_float(entry["tts_path"])
        gain_db = entry["smoothed_gain_db"]

        out_audio = apply_gain_db(audio, gain_db)
        out_audio = peak_limit_by_scaling(out_audio, cfg.peak_target_dbfs)
        out_audio = apply_fades(out_audio, sr, cfg.fade_ms)

        write_wav_float(entry["final_line_path"], out_audio, sr)

        final_entries.append({
            "idx": entry["idx"],
            "tts_path": entry["tts_path"],
            "final_line_path": entry["final_line_path"],
            "ref_lufs": entry["ref_lufs"],
            "tts_lufs": entry["tts_lufs"],
            "raw_gain_db": entry["raw_gain_db"],
            "smoothed_gain_db": entry["smoothed_gain_db"],
            "applied_gain_db": entry["smoothed_gain_db"],
            "gain_mode": entry["gain_mode"],
            "peak_before_dbfs": entry["peak_before_dbfs"],
            "peak_after_dbfs": peak_dbfs(to_mono(out_audio)),
        })

    return final_entries

