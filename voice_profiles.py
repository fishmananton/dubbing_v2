import srt
from typing import Dict, List
import numpy as np
from pydub import AudioSegment


def extract_voice_profiles(audio_path: str, srt_path: str):
    audio = AudioSegment.from_file(audio_path)
    segments = load_srt_segments(srt_path)

    profiles = {}
    for seg in segments:
        profile = build_voice_profile(audio, seg)
        profiles[seg["idx"]] =profile
        # profiles.append({**seg, "voice_profile": profile})

    return profiles


def load_srt_segments(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))

    segments = []
    for sub in subs:
        segments.append({
            "start": sub.start.total_seconds(),
            "end": sub.end.total_seconds(),
            "text": sub.content.strip(),
            "idx": sub.index
        })
    return segments

def build_voice_profile(audio, segment: dict) -> Dict:
    start, end = segment["start"], segment["end"]
    duration = end - start

    pitch = extract_pitch(audio, start, end)
    energy = extract_energy_dbfs(audio, start, end)
    rate = extract_speech_rate(segment["text"], duration)
    bandwidth_profile = extract_bandwidth_profile(audio, start, end)

    return {
        "duration": duration,
        "pitch_median_hz": pitch["median"] if pitch else None,
        "pitch_range_hz": pitch["range"] if pitch else None,
        "speech_rate_wps": rate,
        "energy_dbfs": energy,
            "bandwidth_profile": bandwidth_profile
    }

def extract_pitch(audio: AudioSegment, start, end):
    import parselmouth
    seg = audio[start*1000:end*1000]
    if len(seg) < 300:
        return None

    seg = seg.set_channels(1).set_frame_rate(16000)
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)

    sound = parselmouth.Sound(samples, 16000)
    pitch = sound.to_pitch(time_step=0.01)

    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 50]  # убрать шум

    if len(f0) == 0:
        return None

    return {
        "median": float(np.median(f0)),
        "range": float(np.percentile(f0, 90) - np.percentile(f0, 10))
    }


def extract_energy_dbfs(audio: AudioSegment, start, end):
    seg = audio[start*1000:end*1000]
    if len(seg) < 300:
        return None
    x, _ = segment_to_float_mono(seg, sr=16000)
    rms = np.sqrt(np.mean(x**2) + 1e-12)
    dbfs = 20 * np.log10(rms + 1e-12)   # typically negative
    return float(dbfs)

def segment_to_float_mono(seg: AudioSegment, sr=16000):
    seg = seg.set_channels(1).set_frame_rate(sr)
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    # normalize based on sample width
    max_val = float(1 << (8 * seg.sample_width - 1))
    return samples / max_val, sr

def extract_speech_rate(text: str, duration: float):
    words = len(text.split())
    if duration <= 0:
        return None
    return words / duration

def extract_bandwidth_profile(audio: AudioSegment, start, end,
                              n_fft: int = 1024, hop: int = 256,
                              max_gain_db: float = 8.0,
                              ltas_bins: int = 64):
    seg = audio[start*1000:end*1000]
    sr=seg.frame_rate
    if len(seg) < 300:
        return None

    x, _ = segment_to_float_mono(seg, sr=sr)

    freqs, orig_log = ltas_logmag(x, sr, n_fft=n_fft, hop=hop)

    # ✅ compress for storage
    freqs_c, orig_log_c = compress_ltas(freqs, orig_log, n_bins=ltas_bins)

    return {
        "sr": sr,
        "n_fft": n_fft,
        "hop": hop,
        "freqs": freqs_c.astype(np.float32),
        "orig_logmag_db": orig_log_c.astype(np.float32),
        "max_gain_db": float(max_gain_db),
        "ltas_bins": int(ltas_bins),
    }


def compress_ltas(freqs, logmag_db, n_bins=64, fmin=80.0):

    freqs = np.asarray(freqs, dtype=np.float32)
    logmag_db = np.asarray(logmag_db, dtype=np.float32)

    fmax = float(freqs[-1])
    fmin = max(float(freqs[1]), float(fmin))  # avoid 0 Hz bin

    target_f = np.geomspace(fmin, fmax, n_bins).astype(np.float32)
    target_log = np.interp(target_f, freqs, logmag_db).astype(np.float32)
    return target_f, target_log

def ltas_logmag(y: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256):
    """
    Long-term average log-magnitude spectrum (LTAS) in dB.
    """
    y = y.astype(np.float32)
    y = y - np.mean(y)

    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))

    win = np.hanning(n_fft).astype(np.float32)
    mags = []

    for i in range(0, len(y) - n_fft + 1, hop):
        frame = y[i:i+n_fft] * win
        X = np.abs(np.fft.rfft(frame)) + 1e-9
        mags.append(X)

    M = np.mean(np.stack(mags, axis=0), axis=0)
    logM = 20.0 * np.log10(M + 1e-9)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    return freqs, logM


def apply_bandwidth_profile(y: np.ndarray, sr: int, bw_profile: dict):
    if not bw_profile:
        return y

    n_fft = int(bw_profile["n_fft"])
    hop = int(bw_profile["hop"])
    max_gain_db = float(bw_profile.get("max_gain_db", 8.0))

    freqs, tts_log = ltas_logmag(y, sr, n_fft=n_fft, hop=hop)

    orig_freqs = np.asarray(bw_profile["freqs"], dtype=np.float32)
    orig_log = np.asarray(bw_profile["orig_logmag_db"], dtype=np.float32)

    # ✅ avoid DC bin mismatch (0 Hz)
    mask = freqs >= orig_freqs[0]
    freqs2 = freqs[mask]
    tts_log2 = tts_log[mask]

    orig_log_interp = np.interp(freqs2, orig_freqs, orig_log).astype(np.float32)
    diff_db = (orig_log_interp - tts_log2).astype(np.float32)

    # smooth
    kernel = np.ones(9, dtype=np.float32) / 9.0
    diff_db = np.convolve(diff_db, kernel, mode="same")

    diff_db = np.clip(diff_db, -max_gain_db, max_gain_db)

    Y = np.fft.rfft(y)
    gain_full = np.ones(len(Y), dtype=np.float32)

    gain = (10.0 ** (diff_db / 20.0)).astype(np.float32)

    # map gain back into full FFT bins
    gain_full[mask[:len(gain_full)]] = gain[:np.sum(mask[:len(gain_full)])]

    y2 = np.fft.irfft(Y * gain_full, n=len(y)).astype(np.float32)
    return y2

