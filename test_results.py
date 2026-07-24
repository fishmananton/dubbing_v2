from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyloudnorm as pyln
import soundfile as sf


@dataclass
class QCConfig:
    target_sr: int = 48000
    target_lufs: float = -16.0
    target_true_peak_dbtp: float = -1.5

    # Hard checks
    max_sample_peak: float = 0.999
    recommended_sample_peak: float = 0.98
    min_mean_abs: float = 1e-4

    # Silence / dead sections
    silence_window_sec: float = 0.50
    silence_threshold_dbfs: float = -55.0
    long_silence_warn_sec: float = 2.0

    # DC offset
    max_dc_offset: float = 0.01

    # Stereo balance
    max_stereo_rms_diff_db: float = 3.0

    # Loudness tolerance
    lufs_tolerance: float = 1.0
    true_peak_tolerance_db: float = 0.5


def dbfs_from_linear(x: float) -> float:
    if x <= 0:
        return -120.0
    return 20.0 * math.log10(x)


def read_audio_float(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return audio, sr


def ensure_2d(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio[:, None]
    return audio


def to_mono(audio: np.ndarray) -> np.ndarray:
    audio2 = ensure_2d(audio)
    return np.mean(audio2, axis=1, dtype=np.float32)


def sample_peak(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


def rms_linear(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def mean_abs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.abs(audio), dtype=np.float64))


def dc_offset(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.mean(audio, dtype=np.float64))


def measure_lufs(audio: np.ndarray, sr: int) -> float | None:
    mono = to_mono(audio)
    duration_sec = len(mono) / float(sr)
    if duration_sec < 0.4:
        return None
    try:
        meter = pyln.Meter(sr)
        value = meter.integrated_loudness(mono)
        return float(value) if np.isfinite(value) else None
    except Exception:
        return None


def measure_channel_rms_db(audio: np.ndarray) -> list[float]:
    audio2 = ensure_2d(audio)
    values = []
    for ch in range(audio2.shape[1]):
        values.append(dbfs_from_linear(rms_linear(audio2[:, ch])))
    return values


def detect_silence_regions(
    audio: np.ndarray,
    sr: int,
    window_sec: float,
    silence_threshold_dbfs: float,
) -> list[dict[str, float]]:
    mono = to_mono(audio)
    win = max(1, int(round(window_sec * sr)))
    if len(mono) < win:
        level = dbfs_from_linear(rms_linear(mono))
        if level < silence_threshold_dbfs:
            return [{"start_sec": 0.0, "end_sec": len(mono) / sr, "duration_sec": len(mono) / sr}]
        return []

    silent = []
    in_silence = False
    start_idx = 0

    for i in range(0, len(mono), win):
        chunk = mono[i:i + win]
        level_db = dbfs_from_linear(rms_linear(chunk))
        is_silent = level_db < silence_threshold_dbfs

        if is_silent and not in_silence:
            in_silence = True
            start_idx = i
        elif not is_silent and in_silence:
            end_idx = i
            silent.append({
                "start_sec": start_idx / sr,
                "end_sec": end_idx / sr,
                "duration_sec": (end_idx - start_idx) / sr,
            })
            in_silence = False

    if in_silence:
        end_idx = len(mono)
        silent.append({
            "start_sec": start_idx / sr,
            "end_sec": end_idx / sr,
            "duration_sec": (end_idx - start_idx) / sr,
        })

    return silent


def ffmpeg_loudnorm_measure(path: str | Path, target_i: float = -16.0, target_lra: float = 7.0, target_tp: float = -1.5) -> dict[str, Any] | None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(path),
        "-af", f"loudnorm=I={target_i}:LRA={target_lra}:TP={target_tp}:print_format=json",
        "-f", "null",
        "-"
    ]
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception:
        return None

    match = re.search(r"\{\s*\"input_i\".*?\}", proc.stderr, flags=re.S)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def qc_check(audio_path: str | Path, cfg: QCConfig | None = None) -> dict[str, Any]:
    cfg = cfg or QCConfig()

    audio, sr = read_audio_float(audio_path)
    audio2 = ensure_2d(audio)
    mono = to_mono(audio)

    duration_sec = len(mono) / float(sr) if sr > 0 else 0.0
    peak = sample_peak(audio2)
    peak_dbfs = dbfs_from_linear(peak)
    rms_dbfs = dbfs_from_linear(rms_linear(mono))
    mean_abs_val = mean_abs(mono)
    dc = dc_offset(mono)
    lufs_py = measure_lufs(audio2, sr)

    channel_rms_db = measure_channel_rms_db(audio2)
    stereo_rms_diff_db = None
    if audio2.shape[1] == 2:
        stereo_rms_diff_db = abs(channel_rms_db[0] - channel_rms_db[1])

    silences = detect_silence_regions(
        audio2,
        sr,
        window_sec=cfg.silence_window_sec,
        silence_threshold_dbfs=cfg.silence_threshold_dbfs,
    )
    longest_silence_sec = max((x["duration_sec"] for x in silences), default=0.0)

    ffmpeg_stats = ffmpeg_loudnorm_measure(
        audio_path,
        target_i=cfg.target_lufs,
        target_lra=7.0,
        target_tp=cfg.target_true_peak_dbtp,
    )

    true_peak_dbtp = None
    measured_input_lufs = None
    measured_lra = None
    if ffmpeg_stats is not None:
        try:
            true_peak_dbtp = float(ffmpeg_stats["input_tp"])
        except Exception:
            true_peak_dbtp = None
        try:
            measured_input_lufs = float(ffmpeg_stats["input_i"])
        except Exception:
            measured_input_lufs = None
        try:
            measured_lra = float(ffmpeg_stats["input_lra"])
        except Exception:
            measured_lra = None

    issues: list[str] = []
    warnings: list[str] = []

    if sr != cfg.target_sr:
        warnings.append(f"sample rate is {sr}, expected {cfg.target_sr}")

    if peak >= cfg.max_sample_peak:
        issues.append(f"sample peak too high: {peak:.6f} ({peak_dbfs:.2f} dBFS)")
    elif peak >= cfg.recommended_sample_peak:
        warnings.append(f"sample peak is high: {peak:.6f} ({peak_dbfs:.2f} dBFS)")

    if mean_abs_val < cfg.min_mean_abs:
        issues.append(f"audio looks suspiciously quiet or empty (mean abs {mean_abs_val:.8f})")

    if abs(dc) > cfg.max_dc_offset:
        warnings.append(f"dc offset is high: {dc:.6f}")

    if stereo_rms_diff_db is not None and stereo_rms_diff_db > cfg.max_stereo_rms_diff_db:
        warnings.append(f"left/right RMS differs by {stereo_rms_diff_db:.2f} dB")

    if longest_silence_sec >= cfg.long_silence_warn_sec:
        warnings.append(f"long silence detected: {longest_silence_sec:.2f} sec")

    if measured_input_lufs is not None:
        if abs(measured_input_lufs - cfg.target_lufs) > cfg.lufs_tolerance:
            warnings.append(
                f"integrated loudness is {measured_input_lufs:.2f} LUFS, target is {cfg.target_lufs:.2f} LUFS"
            )
    elif lufs_py is not None:
        if abs(lufs_py - cfg.target_lufs) > cfg.lufs_tolerance:
            warnings.append(
                f"integrated loudness is {lufs_py:.2f} LUFS (pyloudnorm), target is {cfg.target_lufs:.2f} LUFS"
            )

    if true_peak_dbtp is not None:
        if true_peak_dbtp > cfg.target_true_peak_dbtp + cfg.true_peak_tolerance_db:
            issues.append(
                f"true peak too high: {true_peak_dbtp:.2f} dBTP, target <= {cfg.target_true_peak_dbtp:.2f} dBTP"
            )
        elif true_peak_dbtp > cfg.target_true_peak_dbtp:
            warnings.append(
                f"true peak slightly above target: {true_peak_dbtp:.2f} dBTP"
            )

    result = {
        "path": str(audio_path),
        "sample_rate": sr,
        "channels": int(audio2.shape[1]),
        "duration_sec": round(duration_sec, 3),

        "sample_peak_linear": peak,
        "sample_peak_dbfs": round(peak_dbfs, 2),
        "rms_dbfs_mono": round(rms_dbfs, 2),
        "mean_abs_mono": mean_abs_val,
        "dc_offset_mono": dc,

        "integrated_lufs_pyloudnorm": None if lufs_py is None else round(lufs_py, 2),
        "integrated_lufs_ffmpeg": None if measured_input_lufs is None else round(measured_input_lufs, 2),
        "true_peak_dbtp_ffmpeg": None if true_peak_dbtp is None else round(true_peak_dbtp, 2),
        "lra_ffmpeg": None if measured_lra is None else round(measured_lra, 2),

        "channel_rms_dbfs": [round(x, 2) for x in channel_rms_db],
        "stereo_rms_diff_db": None if stereo_rms_diff_db is None else round(stereo_rms_diff_db, 2),

        "longest_silence_sec": round(longest_silence_sec, 3),
        "silence_regions": silences,

        "issues": issues,
        "warnings": warnings,
        "ok": len(issues) == 0,
    }
    print_qc_report(result)
    return result


def print_qc_report(report: dict[str, Any], cfg: QCConfig | None = None) -> None:
    cfg = cfg or QCConfig()
    status = "✅ OK" if report["ok"] else "❌ FAIL"
    print(f"QC {status} | {report['duration_sec']}s | peak={report['sample_peak_dbfs']}dBFS | LUFS={report['integrated_lufs_ffmpeg']} | LRA={report['lra_ffmpeg']} | silence={report['longest_silence_sec']}s")

    if report["warnings"]:
        for w in report["warnings"]:
            print(f"  ⚠ {w}")

    if report["issues"]:
        for e in report["issues"]:
            print(f"  ✘ {e}")

    if report["longest_silence_sec"] >= cfg.long_silence_warn_sec:
        for s in report["silence_regions"]:
            if s["duration_sec"] >= 2.0:
                print(f"  silence: {s['start_sec']:.1f}s–{s['end_sec']:.1f}s ({s['duration_sec']:.1f}s)")

