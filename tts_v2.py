import srt
import re
from pydub.silence import detect_nonsilent
from pydub import AudioSegment
import io
import subprocess
import os
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import time


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


def adjust_speed(segment: AudioSegment, factor: float) -> AudioSegment:
    """Adjust playback speed of a pydub.AudioSegment using ffmpeg's atempo."""
    """Adjust playback speed of a pydub.AudioSegment using ffmpeg's atempo."""
    if factor <= 0:
        raise ValueError("Speed factor must be positive")

    # Export AudioSegment to WAV (in-memory)
    input_buffer = io.BytesIO()
    segment.export(input_buffer, format="wav")
    input_buffer.seek(0)

    # Run ffmpeg with atempo filter (up to 2.0x natively supported)
    with subprocess.Popen([
        "ffmpeg", "-y",
        "-f", "wav",
        "-i", "pipe:0",
        "-filter:a", f"atempo={factor}",
        "-f", "wav",
        "pipe:1"
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        out_data, _ = proc.communicate(input=input_buffer.read())

    # Load result into new AudioSegment
    return AudioSegment.from_file(io.BytesIO(out_data), format="wav")


def speed_len_ms(length_ms: int, factor: float) -> int:

    return max(1, int(round(length_ms / factor)))

def audiosegment_to_float32_mono(seg: AudioSegment) -> np.ndarray:
    samples = np.array(seg.get_array_of_samples())

    if seg.channels > 1:
        samples = samples.reshape((-1, seg.channels)).mean(axis=1)

    max_val = float(1 << (8 * seg.sample_width - 1))
    samples = samples.astype(np.float32) / max_val
    return samples


def trim_trailing_silence(audio: np.ndarray, sr: int, threshold: float = 1e-4, keep_tail_ms: int = 200) -> np.ndarray:
    """
    Trim only trailing silence from mono or stereo float audio.
    Keeps a small tail margin to avoid abrupt cutoff.
    """
    if audio.size == 0:
        return audio

    if audio.ndim == 1:
        mono = np.abs(audio)
    else:
        mono = np.max(np.abs(audio), axis=1)

    nz = np.where(mono > threshold)[0]
    if len(nz) == 0:
        return audio

    last_idx = nz[-1]
    keep_tail_samples = int(sr * keep_tail_ms / 1000.0)
    end_idx = min(len(audio), last_idx + 1 + keep_tail_samples)

    return audio[:end_idx]

def tts_build_final(
    speakers: dict,
    subtitles_file: str,
    tts_segments_dir: str,
    output_dir: str,
    subtitle_visibility_analysis_list: list | None = None,
    testing: bool = False,
    changed_list: list | None = None,
    build_cache: dict | None = None,
):


    if not subtitle_visibility_analysis_list:
        subtitle_visibility_analysis_list = []
    if changed_list is None:
        changed_list = []

    def to_ms(t):
        return int(t.total_seconds() * 1000)

    # ---------------- tuning ----------------
    MIN_GAP_MS = 120
    GOOD_ABS_DIFF = 300

    GOOD_RATIO_MIN = 0.75
    SHORT_WARN_RATIO = 0.50
    SHORT_WARN_DIFF = -600
    SLOWDOWN_RATIO_THRESHOLD = 0.80
    MAX_SLOWDOWN_FACTOR = 0.92

    SOFT_LONG_RATIO_MAX = 1.20
    HARD_LONG_DIFF_WARN = 600
    # ---------------------------------------

    with open(subtitles_file, "r", encoding="utf-8") as f:
        raw_subs = list(srt.parse(f.read()))

    if not raw_subs:
        raise ValueError("Subtitle file is empty.")

    subs = sorted(raw_subs, key=lambda s: s.start)
    changed_set = set(changed_list)
    subtitle_visibility_analysis = {x["index"]: x for x in subtitle_visibility_analysis_list}

    suffix = ".wav" if testing else "_loudness_out.wav"
    timeline_sr = 48000

    # generous global timeline because actual audio may shift right
    total_duration = to_ms(subs[-1].end) + 15000
    total_samples = int(round(total_duration * timeline_sr / 1000.0))


    if testing:
        if build_cache is None:
            build_cache = {
                "audio_cache": {},      # idx -> stripped AudioSegment
                "segment_meta": {},     # idx -> local per-segment metadata
                "timeline_meta": {},    # idx -> timeline-dependent metadata
                "sub_order": [s.index for s in subs],
            }

        audio_cache = build_cache["audio_cache"]
        segment_meta = build_cache["segment_meta"]
        timeline_meta = build_cache["timeline_meta"]

        cached_order = build_cache.get("sub_order")
        current_order = [s.index for s in subs]
        if cached_order != current_order:
            # safest: invalidate everything inside testing phase if subtitle order changed
            audio_cache.clear()
            segment_meta.clear()
            timeline_meta.clear()
            build_cache["sub_order"] = current_order
            changed_set = {s.index for s in subs}
    else:
        # Non-testing final pass: ignore previous cache entirely.
        build_cache = None
        audio_cache = {}
        segment_meta = {}
        timeline_meta = {}

    # ---------------- preload / refresh per-segment data ----------------
    for speaker in speakers:
        speaker_subs = [
            sub for sub in raw_subs
            if sub.content.strip().startswith(f"{speaker}:")
        ]

        for sub in speaker_subs:
            idx = sub.index
            seg_path = os.path.join(tts_segments_dir, speaker, f"{idx}{suffix}")

            needs_refresh = (
                not testing
                or idx not in audio_cache
                or idx not in segment_meta
                or idx in changed_set
            )

            if not needs_refresh:
                continue

            subtitle_start_ms = to_ms(sub.start)
            subtitle_end_ms = to_ms(sub.end)
            subtitle_duration = max(1, subtitle_end_ms - subtitle_start_ms)

            if os.path.exists(seg_path):
                seg_audio = strip_silence(AudioSegment.from_file(seg_path, format="wav"))
            else:
                print(f"⚠️ Missing segment {seg_path}, inserting silence instead")
                seg_audio = AudioSegment.silent(duration=subtitle_duration)

            audio_cache[idx] = seg_audio
            segment_meta[idx] = {
                "index": idx,
                "speaker": speaker,
                "seg_path": seg_path,
                "subtitle_start_ms": subtitle_start_ms,
                "subtitle_end_ms": subtitle_end_ms,
                "subtitle_duration_ms": subtitle_duration,
                "raw_len": len(seg_audio),
                "visibility": subtitle_visibility_analysis.get(
                    idx,
                    {
                        "index": idx,
                        "has_visible_speaking": False,
                        "silence_after_sub": False,
                        "speaking_starts_mid_sub": False,
                    },
                ),
            }

    # ---------------- determine where to start recomputing ----------------
    if testing and timeline_meta and changed_set:
        earliest_changed_pos = None
        for pos, sub in enumerate(subs):
            if sub.index in changed_set:
                earliest_changed_pos = pos
                break
    else:
        earliest_changed_pos = None


    if not testing:
        start_pos = 0
        next_free_start_ms = 0
        stats = []
        visibility_res = []
        final_audio = np.zeros(total_samples, dtype=np.float32)
    else:
        if not timeline_meta:
            start_pos = 0
            next_free_start_ms = 0
            stats = []
            visibility_res = []
            final_audio = None
        elif not changed_set:
            # full reuse in testing mode
            start_pos = len(subs)
            next_free_start_ms = 0
            stats = []
            visibility_res = []

            for sub in subs:
                idx = sub.index
                tm = timeline_meta[idx]
                stats.append(tm["stat"])
                visibility_res.extend(tm.get("visibility_entries", []))
                next_free_start_ms = tm["actual_end_ms"] + MIN_GAP_MS

            final_audio = None
        else:
            # reuse prefix, recompute suffix
            start_pos = earliest_changed_pos if earliest_changed_pos is not None else 0
            stats = []
            visibility_res = []
            next_free_start_ms = 0

            prefix_ok = True
            for sub in subs[:start_pos]:
                idx = sub.index
                if idx not in timeline_meta:
                    prefix_ok = False
                    break
                tm = timeline_meta[idx]
                stats.append(tm["stat"])
                visibility_res.extend(tm.get("visibility_entries", []))
                next_free_start_ms = tm["actual_end_ms"] + MIN_GAP_MS

            if not prefix_ok:
                start_pos = 0
                next_free_start_ms = 0
                stats = []
                visibility_res = []

            final_audio = None

    # ---------------- main timeline loop ----------------
    for i in range(start_pos, len(subs)):
        sub = subs[i]
        idx = sub.index
        meta = segment_meta[idx]

        speaker = meta["speaker"]
        subtitle_start_ms = meta["subtitle_start_ms"]
        subtitle_end_ms = meta["subtitle_end_ms"]
        subtitle_duration = meta["subtitle_duration_ms"]
        raw_len = meta["raw_len"]
        vis = meta["visibility"]
        seg_audio = audio_cache[idx]

        next_subtitle_start = to_ms(subs[i + 1].start) if i + 1 < len(subs) else None

        # 1) actual start first
        actual_start_ms = max(subtitle_start_ms, next_free_start_ms)

        if next_subtitle_start is not None:
            next_vis = segment_meta[subs[i + 1].index]["visibility"]
            if vis["has_visible_speaking"] and (
                not next_vis["has_visible_speaking"] or next_vis["speaking_starts_mid_sub"]
            ):
                actual_available_duration = max(1, subtitle_end_ms - actual_start_ms)
            else:
                actual_available_duration = max(1, next_subtitle_start - MIN_GAP_MS - actual_start_ms)
        else:
            actual_available_duration = max(1, subtitle_end_ms - actual_start_ms)
            if actual_available_duration <= 1:
                actual_available_duration = max(1, subtitle_duration)

        # Start from stripped base audio
        working_audio = seg_audio

        raw_subtitle_ratio = raw_len / subtitle_duration
        fit_ratio = raw_len / actual_available_duration

        # 3) if too long for actual available duration -> speed up
        applied_speed_factor = min(max(fit_ratio, 1.0), 1.08)

        if testing:
            current_len = raw_len
            if applied_speed_factor > 1.0:
                current_len = speed_len_ms(current_len, applied_speed_factor)
        else:
            working_audio = seg_audio
            if applied_speed_factor > 1.0:
                working_audio = adjust_speed(working_audio, applied_speed_factor)
            current_len = len(working_audio)

        current_subtitle_ratio = current_len / subtitle_duration
        current_subtitle_diff = current_len - subtitle_duration

        # 4) if too short -> optional gentle slowdown
        actual_end_ms = actual_start_ms + current_len
        diff = subtitle_end_ms - actual_end_ms

        if (
            current_subtitle_ratio < SLOWDOWN_RATIO_THRESHOLD
            or current_subtitle_diff < SHORT_WARN_DIFF
            or (diff > 200 and vis["has_visible_speaking"])
        ):
            desired_len = subtitle_duration
            slowdown_factor = current_len / desired_len
            slowdown_factor = max(MAX_SLOWDOWN_FACTOR, slowdown_factor)

            if slowdown_factor < 1.0:
                if testing:
                    slowed_len = speed_len_ms(current_len, slowdown_factor)
                    slowed_fit_ratio = slowed_len / actual_available_duration
                    if slowed_fit_ratio <= SOFT_LONG_RATIO_MAX:
                        current_len = slowed_len
                else:
                    slowed_audio = adjust_speed(working_audio, slowdown_factor)
                    slowed_len = len(slowed_audio)
                    slowed_fit_ratio = slowed_len / actual_available_duration
                    if slowed_fit_ratio <= SOFT_LONG_RATIO_MAX:
                        working_audio = slowed_audio
                        current_len = slowed_len

        # final metrics
        final_len = current_len
        final_subtitle_ratio = final_len / subtitle_duration
        final_subtitle_absolute_diff = final_len - subtitle_duration
        final_fit_ratio = final_len / actual_available_duration
        actual_end_ms = actual_start_ms + final_len

        if next_subtitle_start is None:
            spill_vs_subtitle_ms = 0
        else:
            spill_vs_subtitle_ms = max(0, actual_end_ms + MIN_GAP_MS - next_subtitle_start)

        # 5) classify
        if (
            final_subtitle_absolute_diff > HARD_LONG_DIFF_WARN
            or ((final_len - int(actual_available_duration) > GOOD_ABS_DIFF) and final_subtitle_absolute_diff > GOOD_ABS_DIFF)
        ):
            timing_status = "warn_too_long"
        elif final_subtitle_absolute_diff > GOOD_ABS_DIFF and final_fit_ratio > 1.0:
            timing_status = "allow_shift_next_audio"
        elif final_subtitle_absolute_diff < SHORT_WARN_DIFF:
            timing_status = "too_short"
        elif abs(final_subtitle_absolute_diff) <= GOOD_ABS_DIFF:
            timing_status = "good"
        elif final_subtitle_ratio >= GOOD_RATIO_MIN:
            timing_status = "good"
        elif final_subtitle_ratio >= SHORT_WARN_RATIO:
            timing_status = "acceptable_short"
        else:
            timing_status = "too_short"

        stat = {
            "index": idx,
            "speaker": speaker,
            "subtitle_start_ms": subtitle_start_ms,
            "subtitle_end_ms": subtitle_end_ms,
            "actual_start_ms": actual_start_ms,
            "actual_end_ms": actual_end_ms,
            "next_subtitle_start_ms": next_subtitle_start,
            "subtitle_duration_ms": subtitle_duration,
            "actual_available_duration_ms": int(actual_available_duration),
            "audio_duration_ms": final_len,
            "raw_subtitle_ratio": round(raw_subtitle_ratio, 3),
            "final_subtitle_ratio": round(final_subtitle_ratio, 3),
            "final_fit_ratio": round(final_fit_ratio, 3),
            "spill_vs_subtitle_ms": int(spill_vs_subtitle_ms),
            "timing_status": timing_status,
            "applied_speed_factor": min(max(fit_ratio, 1.0), 1.08),
        }
        stats.append(stat)

        local_visibility_entries = []

        if vis["has_visible_speaking"]:
            diff = subtitle_end_ms - actual_end_ms
            if diff > GOOD_ABS_DIFF and abs(final_subtitle_absolute_diff) > GOOD_ABS_DIFF:
                local_visibility_entries.append(
                    {"idx": idx, "len": final_len, "value": -1 * final_subtitle_absolute_diff, "change": "add"}
                )

        if vis["silence_after_sub"]:
            diff = actual_end_ms - subtitle_end_ms
            if diff > GOOD_ABS_DIFF and abs(final_subtitle_absolute_diff) > GOOD_ABS_DIFF:
                local_visibility_entries.append(
                    {"idx": idx, "len": final_len, "value": final_subtitle_absolute_diff, "change": "sub"}
                )

        visibility_res.extend(local_visibility_entries)

        # 6) place audio only in final non-testing pass
        if not testing:
            out_audio = working_audio.set_frame_rate(timeline_sr).set_channels(1)
            seg_np = audiosegment_to_float32_mono(out_audio)

            start_sample = int(round(actual_start_ms * timeline_sr / 1000.0))
            end_sample = start_sample + len(seg_np)

            if end_sample > len(final_audio):
                pad = end_sample - len(final_audio)
                final_audio = np.pad(final_audio, (0, pad))

            final_audio[start_sample:end_sample] += seg_np

        next_free_start_ms = actual_end_ms + MIN_GAP_MS

        if testing:
            timeline_meta[idx] = {
                "actual_start_ms": actual_start_ms,
                "actual_end_ms": actual_end_ms,
                "stat": stat,
                "visibility_entries": local_visibility_entries,
            }

    # ---------------- final write only in non-testing ----------------
    if not testing:
        os.makedirs(output_dir, exist_ok=True)
        final_path = f"{output_dir}/final.wav"

        peak = float(np.max(np.abs(final_audio))) if final_audio.size else 0.0
        if peak > 0.98:
            final_audio = final_audio * (0.98 / peak)

        final_audio = trim_trailing_silence(
            final_audio,
            timeline_sr,
            threshold=1e-4,
            keep_tail_ms=200,
        )

        sf.write(final_path, final_audio, timeline_sr, format="WAV", subtype="FLOAT")

    # ---------------- derive summary outputs ----------------
    too_long = [x for x in stats if x["timing_status"] == "warn_too_long"]
    too_short = [x for x in stats if x["timing_status"] == "too_short"]
    shifted = [x for x in stats if x["timing_status"] == "allow_shift_next_audio"]

    existing_visibility_res_indices = {x["idx"] for x in visibility_res}
    visibility_res.extend(
        {
            "idx": x["index"],
            "len": x["audio_duration_ms"],
            "value": x["subtitle_duration_ms"] - x["audio_duration_ms"],
            "change": "add",
        }
        for x in too_short if x["index"] not in existing_visibility_res_indices
    )
    visibility_res.extend(
        {
            "idx": x["index"],
            "len": x["audio_duration_ms"],
            "value": x["audio_duration_ms"] - x["subtitle_duration_ms"],
            "change": "sub",
        }
        for x in too_long if x["index"] not in existing_visibility_res_indices
    )

    vis_compact = [f"#{x['idx']}:{x['change']}({x['value']:+.0f}ms)" for x in visibility_res]
    print(
        f"ℹ️ segs={len(stats)} | too_long={len(too_long)} | too_short={len(too_short)} "
        f"| shifted={len(shifted)} | vis={len(visibility_res)}"
        f"{': ' + ' '.join(vis_compact) if vis_compact else ''}"
    )

    return {
        "stats": stats,
        "too_long": too_long,
        "too_short": too_short,
        "shifted": shifted,
        "visibility_res": visibility_res,
        "build_cache": build_cache if testing else None,
    }







def tts_generate_multivoice_elevenlabs_segments(subtitles_file: str, client, speakers:dict, model: str, out_dir: str, max_workers: int = 5, language_code: str ="en", changed_list:list|None = None,voices: dict| None = None, elevenlabs_emotions:int=1):
    """Generate one MP3 per subtitle line using parallel threads."""
    EMOTION_PROMPT_MAP = {
        "angry": "[angry, shouting]",
        "fearful": "[afraid, tense]",
        "sad": "[sad, quiet]",
        "neutral": "",
        "happy": "[cheerful, smiling]",
        "surprised": "[surprised, shocked]"
    }
    #
    # male_voice = None
    # female_voice = None

    def score_voice(voice, target_language: str):
        score = 0

        verified = voice.verified_languages or []

        langs = {
            item.language if hasattr(item, "language") else item.get("language")
            for item in verified
            if item is not None
        }

        # hard reject if target language is not supported
        if target_language not in langs:
            return -999

        score += 5

        # fewer verified languages often means more specialization
        if len(langs) == 1:
            score += 3
        elif len(langs) <= 3:
            score += 1
        elif len(langs) >= 8:
            score -= 2

        name = (getattr(voice, "name", "") or "").lower()
        desc = (getattr(voice, "description", "") or "").lower()

        # language-specific hints
        lang_keywords = {
            "es": ["spanish", "español", "latam", "castilian", "castellano"],
            "pt": ["portuguese", "português", "brazilian", "brasileiro"],
            "it": ["italian", "italiano"],
            "fr": ["french", "français"],
            "de": ["german", "deutsch"],
            "ru": ["russian", "русский"],
            "en": ["english"],
        }

        for kw in lang_keywords.get(target_language, []):
            if kw in name:
                score += 2
            if kw in desc:
                score += 1

        # style hints
        if "narration" in desc:
            score += 2
        if "natural" in desc:
            score += 2
        if "conversational" in desc:
            score += 1
        if "professional" in desc:
            score += 1

        # penalties
        if "character" in desc:
            score -= 2
        if "cartoon" in desc:
            score -= 3
        if "anime" in desc:
            score -= 3
        if "funny" in desc:
            score -= 2
        if "parody" in desc:
            score -= 3

        return score

    def get_all_voices(language_code):
        all_voices = []
        page = 1

        while True:
            response = client.voices.get_shared(
                category="professional",
                language=language_code,
                include_custom_rates=False,
                sort="cloned_by_count",
                page_size=100,
                page=page
            )

            voices = response.voices
            if not voices:
                break

            all_voices.extend(voices)

            # Stop if no more pages
            if not getattr(response, "has_more", False):
                break

            page += 1
        return all_voices

    def normalize_gender(g):
        g = (g or "").strip().lower()
        if g in {"male", "masculine", "man"}:
            return "male"
        if g in {"female", "feminine", "woman"}:
            return "female"
        return None

    def get_voices(language_code):
        voices = get_all_voices(language_code)

        scored_voices = []
        for voice in voices:
            gender = normalize_gender(getattr(voice, "gender", None))
            if gender not in {"male", "female"}:
                continue

            score = score_voice(voice, language_code)
            if score < 0:
                continue

            scored_voices.append({
                "voice": voice,
                "voice_id": voice.voice_id,
                "gender": gender,
                "score": score,
                "usage_7d": getattr(voice, "usage_character_count_7_d", 0) or 0,
                "cloned_by_count": getattr(voice, "cloned_by_count", 0) or 0,
                "usage_1y": getattr(voice, "usage_character_count_1_y", 0) or 0,
                "featured": 1 if getattr(voice, "featured", False) else 0,
                "name": getattr(voice, "name", "") or "",
            })

        male_candidates = [x for x in scored_voices if x["gender"] == "male"]
        female_candidates = [x for x in scored_voices if x["gender"] == "female"]

        male_candidates.sort(
            key=lambda x: (
                x["score"],
                x["usage_7d"],
                x["cloned_by_count"],
                x["usage_1y"],
                x["featured"],
                x["name"],
            ),
            reverse=True
        )

        female_candidates.sort(
            key=lambda x: (
                x["score"],
                x["usage_7d"],
                x["cloned_by_count"],
                x["usage_1y"],
                x["featured"],
                x["name"],
            ),
            reverse=True
        )

        male_voice = male_candidates[0]["voice_id"] if male_candidates else None
        female_voice = female_candidates[0]["voice_id"] if female_candidates else None

        if not male_voice or not female_voice:
            raise RuntimeError(
                f"Could not find suitable ElevenLabs male/female voices for language={language_code}"
            )

        return male_voice, female_voice
    if voices:
        male_voice = voices["male"]
        female_voice = voices["female"]
    else:
        male_voice, female_voice = get_voices(language_code)
    speaker_idx_to_emotion = {}
    for spk, data in speakers.items():
        groups = (data or {}).get("groups", {})
        m = {}
        for emo, g in groups.items():
            for idx in g.get("idxs", []):
                m[idx] =EMOTION_PROMPT_MAP.get(emo, '')
        speaker_idx_to_emotion[spk] = m
    with open(subtitles_file, 'r', encoding='utf-8') as f:
        subs = list(srt.parse(f.read()))
    if not subs:
        raise ValueError("Subtitle file is empty.")

    def synthesize_and_save(sub,elevenlabs_emotions):
        idx = sub.index
        speaker, text = sub.content.strip().split(":", 1)
        text = re.sub(r"(?<=\w)\.(?=\s+\w)", ",", text)
        output_dir = f"{out_dir}/{speaker}"
        os.makedirs(output_dir, exist_ok=True)

        voice = female_voice if speakers[speaker]["gender"] == "female" else male_voice
        emotion_prefix = speaker_idx_to_emotion.get(speaker, {}).get(idx, "")

        max_retries = 5
        retry_delay = 1.0

        if elevenlabs_emotions == 0:
            voice_settings = \
                {
                "stability": 0.55,
                "similarity_boost": 0.7,
                "style": 0.3,
                "use_speaker_boost": True
                }

        elif elevenlabs_emotions == 1:
            voice_settings =\
                {
                "stability": 0.45,
                "similarity_boost": 0.6,
                "style": 0.5,
                "use_speaker_boost": True
                }
        else:
            voice_settings = \
                {
                "stability": 0.3,
                "similarity_boost": 0.45,
                "style": 0.75,
                "use_speaker_boost": True
                }



        for attempt in range(1, max_retries + 1):
            try:
                with client.text_to_speech.with_raw_response.convert(
                        voice_id=voice,
                        output_format="wav_48000",
                        text=emotion_prefix + text,
                        model_id="eleven_v3",
                        apply_text_normalization="on",
                        voice_settings=voice_settings
                ) as response:
                    buffer = io.BytesIO()
                    for chunk in response.data:
                        if chunk:
                            buffer.write(chunk)
                    buffer.seek(0)

                    audio = AudioSegment.from_file(buffer, format="wav")
                    audio = strip_silence(audio)

                    output_path = os.path.join(output_dir, f"{idx}.wav")
                    audio.export(output_path, format="wav")
                    return idx, speaker, True

            except Exception as e:
                status_code = getattr(e, "status_code", None)
                body = getattr(e, "body", None)

                if status_code == 409:
                    print(
                        f"⚠️ ElevenLabs 409 for subtitle {idx} attempt {attempt}/{max_retries}). Retrying..."
                    )
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue

                raise

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if changed_list:
            changed_set = set(changed_list)
            subs_to_generate = [sub for sub in subs if sub.index in changed_set]
        else:
            subs_to_generate = subs
        futures = [executor.submit(synthesize_and_save, sub, elevenlabs_emotions) for sub in subs_to_generate]
        done = 0
        for future in as_completed(futures):
            idx, speaker, success = future.result()
            if success:
                done += 1

    print(f"✅ Generated {done} TTS segments using {max_workers} threads.")
    return {"male": male_voice, "female": female_voice}




def tts_generate_multivoice_cartesia_segments(
    subtitles_file: str,
    api_key: str,
    speakers: dict,
    out_dir: str,
    language_code: str = "en",
    model_id: str = "sonic-3",
    max_workers: int = 5,
    cartesia_version: str = "2026-03-01",
    changed_list: list| None = None,
    voices: dict| None = None
):
    """
    Generate one WAV per subtitle line using Cartesia TTS.

    speakers format example:
    {
        "SPEAKER_00": {
            "gender": "female",
            "groups": {
                "neutral": {"idxs": [1,2,3]},
                "angry": {"idxs": [10,11]}
            }
        },
        ...
    }
    """

    # Your emotion labels -> Cartesia-supported emotion labels
    EMOTION_MAP = {
        "angry": "angry",
        "fearful": "scared",
        "sad": "sad",
        "neutral": "neutral",
        "happy": "content",      # or "excited" depending on your preference
        "surprised": "excited",
        "excited": "excited",
        "content": "content",
        "scared": "scared",
    }

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Cartesia-Version": cartesia_version,
    })

    # --- Find candidate voices ---
    def get_voices(session, language_code):
        voices = []
        starting_after = None

        while True:
            params = {
                "limit": 100,
                "language": language_code,
                "expand[]": [
                    "is_starred",
                    "tags",
                    "preview_file_id",
                    "model_aliases",
                    "embedding",
                    "country",
                    "accent",
                    "is_featured",
                ],
            }
            if starting_after:
                params["starting_after"] = starting_after

            resp = session.get(
                "https://api.cartesia.ai/voices",
                params=params,
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()

            batch = data.get("data", [])
            voices.extend(batch)

            if not data.get("has_more") or not batch:
                break

            starting_after = batch[-1]["id"]

        male_voice = pick_voice_by_gender(voices, "masculine")
        female_voice = pick_voice_by_gender(voices, "feminine")
        if not male_voice or not female_voice:
            raise RuntimeError(
                f"Could not find both masculine and feminine Cartesia voices "
                f"for language={language_code}."
            )
        return male_voice["id"], female_voice["id"]

    def score_voice(voice: dict) -> int:
        score = 0

        # 1) strongest trust / support signals
        if voice.get("is_pro"):
            score += 100

        if voice.get("is_featured"):
            score += 50

        tags = [str(t.get("label", "")).lower() for t in (voice.get("tags") or [])]
        desc = (voice.get("description") or "").lower()
        name = (voice.get("name") or "").lower()
        text_blob = f"{name} {desc}"

        # 2) style preference: emotive > conversational
        if "emotive" in tags:
            score += 20

        if "conversational" in tags or "conversational" in text_blob:
            score += 10

        # 3) extra positive hints
        if "clear" in text_blob or "natural" in text_blob or "consistent" in text_blob:
            score += 8

        # 4) penalties for overly stylized voices
        if "entertainment" in tags:
            score -= 12

        if "expressive" in text_blob or "performer" in text_blob:
            score -= 8

        return score

    def pick_voice_by_gender(voices: list[dict], gender: str) -> dict | None:
        gender_voices = [
            v for v in voices
            if str(v.get("gender", "")).lower() == gender.lower()
        ]
        if not gender_voices:
            return None

        return max(gender_voices, key=score_voice)

    if voices:
        male_voice = voices["male"]
        female_voice = voices["female"]
    else:
        male_voice, female_voice = get_voices(session, language_code)




    # Build subtitle-index -> emotion map per speaker
    speaker_idx_to_emotion = {}
    for spk, data in speakers.items():
        groups = (data or {}).get("groups", {})
        idx_map = {}
        for emo, g in groups.items():
            cartesia_emo = EMOTION_MAP.get(emo, "neutral")
            for idx in g.get("idxs", []):
                idx_map[idx] = cartesia_emo
        speaker_idx_to_emotion[spk] = idx_map

    with open(subtitles_file, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))

    if not subs:
        raise ValueError("Subtitle file is empty.")

    def synthesize_and_save(sub):
        idx = sub.index
        content = sub.content.strip()

        if ":" not in content:
            raise ValueError(f"Subtitle {idx} does not contain 'SPEAKER: text' format")

        speaker, text = content.split(":", 1)
        speaker = speaker.strip()
        text = text.strip().replace(":", " - ")

        output_dir = os.path.join(out_dir, speaker)
        os.makedirs(output_dir, exist_ok=True)

        speaker_data = speakers.get(speaker, {})
        gender = speaker_data.get("gender", "male").lower()
        voice_id = female_voice if gender == "female" else male_voice

        emotion = speaker_idx_to_emotion.get(speaker, {}).get(idx, "neutral")

        payload = {
            "model_id": model_id,
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": voice_id,
            },
            "output_format": {
                "container": "wav",
                "encoding": "pcm_f32le",
                "sample_rate": 48000,
            },
            "language": language_code,
            "generation_config": {
                "emotion": emotion,
                "speed": 1.0,
                "volume": 1.0,
            },
            "save": False,
        }

        resp = session.post(
            "https://api.cartesia.ai/tts/bytes",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=180,
        )
        resp.raise_for_status()

        audio = AudioSegment.from_file(io.BytesIO(resp.content), format="wav")
        audio = strip_silence(audio)
        output_path = os.path.join(output_dir, f"{idx}.wav")
        audio.export(output_path, format="wav")

        return idx, speaker, True

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if changed_list:
            changed_set = set(changed_list)
            subs_to_generate = [sub for sub in subs if sub.index in changed_set]
        else:
            subs_to_generate = subs

        futures = [executor.submit(synthesize_and_save, sub) for sub in subs_to_generate]
        for future in as_completed(futures):
            idx, speaker, success = future.result()
            if success:
                done += 1

    print(f"✅ Generated {done} Cartesia TTS segments using {max_workers} threads.")
    return {"male": male_voice, "female": female_voice}

