import subprocess
from pydub import AudioSegment
import boto3
import srt
import copy
import tempfile
import soundfile as sf
import torchaudio
from modal_utils import run_modal_job
from parallel import parallel_map
import os
import threading

PIPELINE_SR = 48000

_silero_vad_model = None
_silero_get_speech_timestamps = None
_silero_vad_lock = threading.Lock()

def get_silero_vad():
    import torch
    import torchaudio
    global _silero_vad_model, _silero_get_speech_timestamps

    if _silero_vad_model is None:
        with _silero_vad_lock:
            if _silero_vad_model is None:
                model, utils = torch.hub.load(
                    "snakers4/silero-vad",
                    "silero_vad",
                    force_reload=False,
                )
                _silero_vad_model = model
                _silero_get_speech_timestamps = utils[0]

    return _silero_vad_model, _silero_get_speech_timestamps

def resample_wav(path: str, target_sr: int):
    audio, sr = torchaudio.load(path)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
        torchaudio.save(path, audio, target_sr)


def extract_audio(original_video: str, output_audio: str):
    subprocess.run([
        "ffmpeg", "-y",
        "-loglevel", "quiet",
        "-i", original_video,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "48000",
        "-ac", "2",
        output_audio
    ], check=True)




def split_vocal(
    input_vocal: str,
    subtitles_file: str,
    non_speech_layer_file: str
):
    audio = AudioSegment.from_file(input_vocal)

    with open(subtitles_file, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))

    audio = audio.set_frame_rate(48000).set_channels(1)
    audio_len = len(audio)

    PADDING = 150
    MIN_DURATION_MS = 350

    samples = [audio[i:i + 1000].dBFS for i in range(0, len(audio), 5000)]
    samples = [x for x in samples if x != float("-inf")]
    background = min(samples) if samples else -60
    LOUDNESS_THRESHOLD_DBFS = background + 8

    # 1. Build subtitle intervals
    sub_intervals = []
    for sub in subs:
        start = max(0, int(sub.start.total_seconds() * 1000) - PADDING)
        end = min(audio_len, int(sub.end.total_seconds() * 1000) + PADDING)
        sub_intervals.append((start, end))

    sub_intervals.sort()

    # 2. Merge overlapping subtitle intervals
    merged_subs = []
    for start, end in sub_intervals:
        if not merged_subs or start > merged_subs[-1][1]:
            merged_subs.append([start, end])
        else:
            merged_subs[-1][1] = max(merged_subs[-1][1], end)

    # 3. Find gaps not covered by subtitles
    gaps = []
    prev_end = 0

    for start, end in merged_subs:
        if start > prev_end:
            gaps.append((prev_end, start))
        prev_end = max(prev_end, end)

    if prev_end < audio_len:
        gaps.append((prev_end, audio_len))

    # 4. Start with full silence to preserve original timing
    result = AudioSegment.silent(duration=audio_len, frame_rate=audio.frame_rate)

    kept_segments = []

    # 5. Copy back only non-subtitle gaps that contain sound
    for start, end in gaps:
        seg = audio[start:end]

        if len(seg) < MIN_DURATION_MS:
            continue

        if seg.dBFS > LOUDNESS_THRESHOLD_DBFS:
            result = result[:start] + seg + result[end:]
            kept_segments.append((start, end, round(seg.dBFS, 2)))

    result.export(non_speech_layer_file, format="wav")
    return

# not sure if i need exactly 16Ghz
def prepare_vocal_asr(vocal_file: str, vocal_asr_file: str):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", vocal_file,
        "-ac", "1",
        "-ar", "16000",
        vocal_asr_file,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    return vocal_asr_file

def split_audio(
        run_id:str,
        boto_session: boto3.Session,
        bucket_name: str,
        input_audio: str,
        output_vocal: str,
        output_music: str,
        output_vocal_asr: str):
    import modal

    vol = modal.Volume.from_name("dubbing-transfer")
    vol_prefix = f"split_{run_id}"

    input_flac = tempfile.mktemp(suffix=".flac")
    try:
        print(f"[split_audio] Compressing WAV -> FLAC...")
        subprocess.run(["ffmpeg", "-y", "-i", input_audio, input_flac], check=True, capture_output=True)

        print(f"[split_audio] Uploading FLAC to Modal Volume...")
        with vol.batch_upload(force=True) as batch:
            batch.put_file(input_flac, f"{vol_prefix}/input.flac")
    finally:
        if os.path.exists(input_flac):
            os.remove(input_flac)

    result = run_modal_job(
        app_name="audio-dubbing-separator",
        function_name="split_audio_job",
        timeout_minutes=30,
        poll_delay_sec=5,
        run_id=vol_prefix,
    )

    if result["status"] != "COMPLETED":
        raise Exception(f"split_audio failed for {run_id}")

    print(f"[split_audio] Download + decompress + resample stems (parallel)...")

    def _fetch_stem(vol_name: str, out_wav: str, asr_out: str | None):
        # download one stem's FLAC, decompress to WAV, resample to PIPELINE_SR — identical
        # per-stem ops as before, just per stem so vocal & music go concurrently. The vocal
        # stem also builds its 16k ASR copy here, so ASR overlaps the music stem.
        flac = tempfile.mktemp(suffix=".flac")
        try:
            with open(flac, "wb") as f:
                for chunk in vol.read_file(f"{vol_prefix}/{vol_name}"):
                    f.write(chunk)
            subprocess.run(["ffmpeg", "-y", "-i", flac, out_wav],
                           check=True, capture_output=True)
            resample_wav(out_wav, PIPELINE_SR)
            if asr_out:
                prepare_vocal_asr(out_wav, asr_out)
        finally:
            if os.path.exists(flac):
                os.remove(flac)

    parallel_map(lambda t: _fetch_stem(*t),
                 [("vocal.flac", output_vocal, output_vocal_asr),
                  ("music.flac", output_music, None)])

    vol.remove_file(vol_prefix, recursive=True)






def get_voiced_duration_from_subs(
    audio: AudioSegment,
    subs_by_index: dict,
    collected_idxs: list[int],
    min_speech_duration: float = 0.1,
    min_silence_duration: float = 1.0,
    vad_method: str = "silero",
) -> float:
    if not collected_idxs:
        return 0.0

    ordered_idxs = sorted(set(collected_idxs))

    speaker_audio = AudioSegment.silent(duration=0, frame_rate=audio.frame_rate)

    for idx in ordered_idxs:
        sub = subs_by_index.get(idx)
        if not sub:
            continue

        start_ms = int(sub.start.total_seconds() * 1000)
        end_ms = int(sub.end.total_seconds() * 1000)

        if end_ms <= start_ms:
            continue

        speaker_audio += audio[start_ms:end_ms]

    if len(speaker_audio) == 0:
        return 0.0

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        speaker_audio.export(tmp.name, format="wav")

        from whisper_timestamped.transcribe import get_audio_tensor, get_vad_segments
        audio_vad = get_audio_tensor(tmp.name)
        segments = get_vad_segments(
            audio_vad,
            output_sample=True,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            method=vad_method,
        )

        vad_sample_rate = 16000
        voiced_duration = sum(
            (seg["end"] - seg["start"]) / vad_sample_rate
            for seg in segments
        )

    return float(voiced_duration)

def precompute_group_voiced_durations(audio: AudioSegment, speakers: dict, subs_by_index: dict):

    precomputed = {}

    for speaker_name, speaker_data in speakers.items():
        groups = speaker_data.get("groups", {})
        precomputed[speaker_name] = {}

        for emotion, group_data in groups.items():
            idxs = list(group_data.get("idxs", []))
            total_sec = get_voiced_duration_from_subs(audio, subs_by_index, idxs)

            precomputed[speaker_name][emotion] = {
                "idxs": idxs,
                "total_sec": total_sec,
            }

    return precomputed

def cut_speaker_audio(
    audio_path: str,
    subtitle_file: str,
    output_path: str,
    speakers: dict,
    emotions_tags: dict,
    min_sec: float = 3.0,
    max_sec: float = 8.0,
):
    os.makedirs(output_path, exist_ok=True)
    speakers_copy = copy.deepcopy(speakers)
    audio = AudioSegment.from_file(audio_path)

    with open(subtitle_file, 'r', encoding='utf-8') as f:
        subs = list(srt.parse(f.read()))

    subs_by_index = {sub.index: sub for sub in subs}

    # Build per-speaker index lists from subtitles (speaker label from content)
    speaker_idxs: dict[str, list[int]] = {}
    for sub in subs:
        if ":" in sub.content:
            spk = sub.content.split(":", 1)[0].strip()
        else:
            continue
        speaker_idxs.setdefault(spk, []).append(sub.index)

    # Category priority tiers
    PRIORITY_TIERS = [
        {"neutral"},
        {"happy", "sad"},
        {"angry", "fearful", "surprised", "disgusted"},
    ]

    for speaker_name in speakers.keys():
        all_idxs = speaker_idxs.get(speaker_name, [])
        if not all_idxs:
            continue

        # Group this speaker's indices by emotion category
        category_buckets: dict[str, list[int]] = {}
        for idx in all_idxs:
            tag_data = emotions_tags.get(idx) or emotions_tags.get(str(idx))
            if tag_data and isinstance(tag_data, dict):
                cat = tag_data.get("category", "neutral")
            else:
                cat = "neutral"
            category_buckets.setdefault(cat, []).append(idx)

        collected_idxs: list[int] = []
        total_sec = 0.0

        for tier in PRIORITY_TIERS:
            if total_sec >= min_sec:
                break
            tier_idxs = sorted(
                idx for cat in tier for idx in category_buckets.get(cat, [])
            )
            for idx in tier_idxs:
                sub = subs_by_index.get(idx)
                if not sub:
                    continue
                seg_dur = (sub.end - sub.start).total_seconds()
                if total_sec + seg_dur > max_sec:
                    continue
                collected_idxs.append(idx)
                total_sec += seg_dur
                if total_sec >= max_sec:
                    break

        collected_idxs.sort()

        # Build the audio file
        speaker_audio = AudioSegment.empty()
        used_idxs = []
        for idx in collected_idxs:
            sub = subs_by_index.get(idx)
            if not sub:
                continue
            start_ms = int(sub.start.total_seconds() * 1000)
            end_ms = int(sub.end.total_seconds() * 1000)
            if end_ms > start_ms:
                speaker_audio += audio[start_ms:end_ms]
                used_idxs.append(idx)

        if len(speaker_audio) == 0:
            continue

        speakers_copy[speaker_name]["reference_audio"] = f"{speaker_name}.wav"
        speakers_copy[speaker_name]["reference_duration_sec"] = len(speaker_audio) / 1000
        speakers_copy[speaker_name]["reference_idxs"] = used_idxs

        speaker_audio.export(
            os.path.join(output_path, f"{speaker_name}.wav"),
            format="wav",
        )

    return speakers_copy

def combine_audio_files(speakers: dict, tts_segments_folder: str,  output_path:str):

    for speaker_name, speaker_data in speakers.items():
        groups = speaker_data.get("groups", {})
        combined = AudioSegment.empty()
        for emotion, group_data in groups.items():
            collected_idxs = list(group_data.get("idxs", []))
            for idx in collected_idxs:
                if len(combined)/1000 >25:
                    break
                sound = AudioSegment.from_wav(f"{tts_segments_folder}/{speaker_name}/{idx}.wav")
                combined += sound
        combined.export(f"{output_path}/{speaker_name}_combined.wav", format="wav")
