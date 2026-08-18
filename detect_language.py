from collections import defaultdict
import numpy as np
import whisper

MODEL_NAME = "small"  # 'small' or 'turbo' is fast and accurate on 30s pure speech


def detect_language_for_routing(audio_path: str, speaker_segments: list, zh_threshold: float = 0.40) -> str:
    """
    Stitches speech segments into a 30s pure speech buffer.
    Returns 'zh' if Chinese is detected (for Alibaba), otherwise 'auto' (for AssemblyAI).
    """
    if not speaker_segments:
        return "auto"

    # 1. Load raw audio tensor
    audio = whisper.load_audio(audio_path)
    SR = whisper.audio.SAMPLE_RATE
    target_samples = 30 * SR  # Exactly 30 seconds of speech

    # 2. Slice and collect speech-only arrays
    speech_chunks = []
    accumulated_samples = 0

    for seg in sorted(speaker_segments, key=lambda s: s["start"]):
        start_idx = int(seg["start"] * SR)
        end_idx = int(seg["end"] * SR)

        chunk = audio[start_idx:end_idx]
        if len(chunk) == 0:
            continue

        speech_chunks.append(chunk)
        accumulated_samples += len(chunk)

        if accumulated_samples >= target_samples:
            break

    if not speech_chunks:
        return "auto"

    # 3. Concatenate all speech chunks in RAM & cap at 30 seconds
    pure_speech = np.concatenate(speech_chunks)[:target_samples]
    pure_speech = whisper.pad_or_trim(pure_speech)

    # 4. Run single-pass Whisper language detection
    model = whisper.load_model(MODEL_NAME)
    mel = whisper.log_mel_spectrogram(pure_speech).to(model.device)
    _, probs = model.detect_language(mel)

    # 5. Routing logic
    if probs.get("zh", 0.0) >= zh_threshold:
        return "zh"
    initial_language = max(probs, key=probs.get)

    return initial_language