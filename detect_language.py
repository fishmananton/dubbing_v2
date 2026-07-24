from collections import defaultdict

MODEL_NAME = "base"
CONFIDENCE_THRESHOLD = 0.65

def first_speech_windows_by_duration(
    speaker_segments,
    target_speech_sec=30.0,
    min_seg_len=1.0,
    max_chunk_len=15.0,
    max_gap=0.75,   # allow small silence between speech pieces
):
    windows = []
    accumulated = 0.0

    segments = []
    for seg in sorted(speaker_segments, key=lambda s: s["start"]):
        start = float(seg["start"])
        end = float(seg["end"])
        if end <= start:
            continue
        if (end - start) < min_seg_len:
            continue
        segments.append((start, end))

    if not segments:
        return windows

    cur_start, cur_end = segments[0]

    for start, end in segments[1:]:
        gap = start - cur_end
        proposed_len = end - cur_start

        # merge if close enough and total merged chunk not too long
        if gap <= max_gap and proposed_len <= max_chunk_len:
            cur_end = end
        else:
            windows.append((cur_start, cur_end))
            accumulated += (cur_end - cur_start)
            if accumulated >= target_speech_sec:
                return windows
            cur_start, cur_end = start, end

    windows.append((cur_start, cur_end))
    accumulated += (cur_end - cur_start)

    return windows


def detect_language_from_chunks(audio_path, speaker_segments):
    import whisper
    SR = whisper.audio.SAMPLE_RATE
    model = whisper.load_model(MODEL_NAME)
    audio = whisper.load_audio(audio_path)

    windows = first_speech_windows_by_duration(
        speaker_segments,
        target_speech_sec=30.0,
        min_seg_len=1.0,
        max_chunk_len=15.0,
    )

    if not windows:
        return "auto"

    aggregated = defaultdict(float)
    used = 0

    for start, end in windows:
        chunk = audio[int(start * SR): int(end * SR)]

        # Skip extremely short chunks (they become mostly silence after padding)
        if len(chunk) < int(4.0 * SR):
            continue

        # Whisper requires fixed-length input for detect_language
        chunk = whisper.pad_or_trim(chunk)

        mel = whisper.log_mel_spectrogram(chunk).to(model.device)
        _, probs = model.detect_language(mel)

        for lang, p in probs.items():
            aggregated[lang] += p

        used += 1

    if used == 0:
        return "auto"

    # Average across used windows for a stable confidence number
    for lang in aggregated:
        aggregated[lang] /= used

    best_lang = max(aggregated, key=aggregated.get)
    confidence = aggregated[best_lang]

    if confidence < CONFIDENCE_THRESHOLD:
        return "auto"

    return best_lang
