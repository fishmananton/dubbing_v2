from __future__ import annotations
import os
#before 1 starting execute prefect server start
API_URL = "http://127.0.0.1:4200/api"
os.environ["PREFECT_API_URL"] = API_URL
os.environ["PREFECT_LOGGING_LEVEL"] = "INFO"
from enum import Enum
from pathlib import Path
from datetime import datetime
import random
import string

from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
from contextlib import contextmanager
import time
import json
from config import Configuration
from shutil import copyfile
import shutil


# === Import your existing functions ===
from audio import extract_audio, split_audio, split_vocal, cut_speaker_audio, combine_audio_files
from diarization import diarize
from translate_duration import translate_duration
from gender import detect_gender
from tts_v2 import tts_build_final, tts_generate_multivoice_elevenlabs_segments, tts_generate_multivoice_cartesia_segments
from video import generate_videos, cut_video, encode_preview_base
from subs_from_ocr import process_video_with_subs
from gemini_emotion_extract import extract_emotions_gemini
from detect_language import detect_language_for_routing
from voice_profiles import extract_voice_profiles
from assemblyai_transcribe import assemblyai_transcribe_raw
from speechmatics_transcribe import speechmatics_transcribe_raw, speechmatics_transcribe_raw_chunked
from alibabacloud_transcribe import alibaba_transcribe_raw
from deepgram_transcribe import deepgram_transcribe_raw
from transcribe_common import assemble_transcription
from loudness_adjust import run_line_loudness_stage
from detect_mouth_windows import detect_mouth_windows
from test_results import qc_check
from qc_agent import decide as qc_decide
from qc_evidence import build_evidence_bundle, build_shared_context
from qc_fixes import apply_fixes, load_playbook, resolve_collisions
from song_detect import detect_songs, drop_sub_ids
from qc_tail import (append_promotion_queue, build_fix_log, diff_reqc,
                     split_auto_and_proposals, write_fix_log, write_qc_issues)
from test_dub_qc import (build_script, compress_audio, _retry_call,
                         qc_check as qc_listen_check, qc_check_chunked)
from final_audio import build_audio, measure_loudness
from prefect.cache_policies import NO_CACHE
from tts_inworld import tts_generate_multivoice_inworld_segments
from tts_fish_audio import tts_generate_multivoice_fish_segments
from tts_index_tts2 import tts_generate_index_tts2_segments, CHARS_PER_SEC_PER_GPU, TARGET_POD_SECONDS
from tts_qwen3_tts import tts_generate_qwen3_tts_segments, QWEN3_SUPPORTED_LANGUAGES
from natural_tts_timing import compute_speaker_base_atempo, classify_lines, build_retranslation_request, build_underflow_retranslation_request, save_natural_timing_data


@contextmanager
def timer(label: str):
    """Context manager that logs execution time using Prefect's logger."""
    log = get_run_logger()
    start = time.time()
    log.info(f"⏳ Starting: {label}")
    try:
        yield
    finally:
        duration = time.time() - start
        log.warning(f"✅ {label} finished in {duration:.2f} seconds")

@task
def t_cut_video(video_file, output_file, duration_sec):
    with timer("Cut Video"):
        result = cut_video(video_file, output_file, duration_sec)
    return result


@task
def t_extract_audio(video_file, audio_file):
    with timer("MoviePy extraction"):
        extract_audio(video_file, audio_file)
    return audio_file


@task(cache_policy=NO_CACHE)
def t_split_audio(config, audio_file, run_id):
    with timer("audio split"):
        boto_session = config.get_boto_session()
        split_audio(run_id, boto_session, config.s3_bucket_name, audio_file, config.vocal_file, config.music_file, config.vocal_asr_file)
    return {"vocal_file": config.vocal_file, "music_file": config.music_file, "vocal_asr_file": config.vocal_asr_file}


@task(cache_policy=NO_CACHE)
def t_gemini_extract_emotions(config, audio_file, subtitles_file):
    with timer("Gemini emotion extraction"):
        speakers = extract_emotions_gemini(
            audio_file=audio_file,
            subtitles_file=subtitles_file,
            gemini_api_key=config.gemini_api_key,
            gemini_model_name=config.gemini_model,
            output_file=config.gemini_emotions_file,
        )
    return speakers


@task(cache_policy=NO_CACHE)
def t_detect_songs(config, audio_file, subtitles_file):
    """Return subtitle IDs that are sung lyrics (to be dropped before GENERATE).
    Runs in the EMOTION-stage parallel fan-out. Never raises: on error returns []
    so the dub proceeds with songs dubbed (pre-existing behavior)."""
    with timer("Detect songs"):
        try:
            from google import genai
            client = genai.Client(api_key=config.gemini_api_key)
            ids = detect_songs(audio_file, subtitles_file, client,
                               config.gemini_model)
            print(f"🎵 song detection: {len(ids)} sung line(s) -> drop {ids}")
            return ids
        except Exception as e:  # noqa: BLE001 — must never fail the dub
            print(f"⚠️  song detection failed ({e}); keeping all lines")
            return []


@task
def t_split_vocal(vocal_file,subtitles, non_speech_layer_file):
    with timer("Split Vocal"):
        split_vocal(vocal_file,subtitles, non_speech_layer_file)
    return non_speech_layer_file

@task(cache_policy=NO_CACHE)
def t_diarize(config, audio_file,num_speakers=None, run_id=''):
    with timer("diarize"):
        boto_session = config.get_boto_session()
        speaker_segments = diarize(boto_session, config.pyannote_key, config.s3_bucket_name, audio_file, num_speakers=num_speakers, run_id=run_id)

    return speaker_segments


@task(cache_policy=NO_CACHE)
def t_transcribe_raw(config, audio_file, language, run_id, speech_segments=None):
    """Language-routed raw STT. Returns (words_data, trans_language) with ms
    timings for the shared assembler. Engine per language:
      en -> AssemblyAI, zh -> Alibaba, ja -> Deepgram, else -> Speechmatics.
    speech_segments (whole-file VAD regions, seconds) let the Speechmatics path
    split at silence and transcribe chunks concurrently."""
    lang2 = (language or "auto")[:2].lower()
    with timer(f"transcribe_raw[{lang2}]"):
        if lang2 == "en":
            return assemblyai_transcribe_raw(
                audio_file_raw=audio_file,
                assemblyai_api_key=config.assemblyai_api_key,
            )
        if lang2 == "zh":
            return alibaba_transcribe_raw(
                audio_file_raw=audio_file,
                alibaba_api_key=config.alibaba_api_key,
                s3_bucket_name=config.s3_bucket_name,
                boto_session=config.get_boto_session(),
                language=language,
                run_id=run_id,
            )
        if lang2 == "ja":
            return deepgram_transcribe_raw(
                audio_file_raw=audio_file,
                deepgram_api_key=config.deepgram_api_key,
                language=language,
            )
        if speech_segments:
            return speechmatics_transcribe_raw_chunked(
                audio_file_raw=audio_file,
                speechmatics_api_key=config.speechmatics_api_key,
                speech_segments=speech_segments,
                language=language,
            )
        return speechmatics_transcribe_raw(
            audio_file_raw=audio_file,
            speechmatics_api_key=config.speechmatics_api_key,
            language=language,
        )


@task(cache_policy=NO_CACHE)
def t_assemble_transcription(config, words_data, trans_language, speaker_segments, num_speakers):
    with timer("assemble_transcription"):
        openai_client = config.get_openai_client()
        lang = assemble_transcription(
            words_data=words_data,
            trans_language=trans_language,
            speaker_segments=speaker_segments,
            subtitles_file=config.subtitles,
            openai_client=openai_client,
            openai_model=config.openai_diarization_model,
            num_speakers=num_speakers,
        )
    return lang

@task(cache_policy=NO_CACHE)
def t_process_video_with_subs(config, video_file, speaker_segments, language, num_speakers, run_id):
    with timer("process_video_with_subs"):
        boto_session = config.get_boto_session()
        openai_client = config.get_openai_client()
        lang = process_video_with_subs(video_file, boto_session, openai_client, config.openai_diarization_model, config.s3_bucket_name, speaker_segments, config.subtitles, language, num_speakers, run_id)
    return lang

@task
def t_detect_gender(audio_file, subtitle_file, gemini_api_key, gemini_model):
    with timer("detect gender"):
        result = detect_gender(audio_file, subtitle_file, gemini_api_key, gemini_model)
    return result

@task(cache_policy=NO_CACHE)
def t_translate(config, subtitles, src_lang, translate_to_language, subtitles_translated, punctuation, speakers_data=None):
    with timer("translate (duration-bounded)"):
        result = translate_duration(
            anthropic_api_key=config.anthropic_api_key,
            subtitles_file=subtitles,
            source_language=src_lang,
            target_language=translate_to_language,
            pass1_model=config.anthropic_translate_pass1_model,
            pass2_model=config.anthropic_translate_pass2_model,
            result_file=subtitles_translated,
            speakers_data=speakers_data,
        )
        from dataclasses import asdict
        translation_stats_path = os.path.join(os.path.dirname(subtitles_translated), "translation_stats.json")
        with open(translation_stats_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    return result


@task(cache_policy=NO_CACHE)
def t_retranslate_timing_fix(config, overflow_requests, underflow_requests, target_language):
    with timer("Timing retranslation"):
        from post_build_fix import retranslate_timing_fix
        result = retranslate_timing_fix(
            overflow_requests=overflow_requests,
            underflow_requests=underflow_requests,
            openai_client=config.get_openai_client(),
            target_language=target_language,
            model=config.timing_retranslate_model,
        )
    return result


@task(cache_policy=NO_CACHE)
def t_qc_listen_check(subtitles_file, audio_file, label="QC listen check"):
    with timer(label):
        import srt as _srt
        from google import genai
        from pydub import AudioSegment
        gclient = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        gmodel = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        # A/B knobs, env-tunable with no code edit: pass count + thinking level.
        passes = int(os.getenv("QC_PASSES", "3"))
        thinking = os.getenv("QC_THINKING_LEVEL", "HIGH")
        # Chunk long audio and run ALL (chunk x pass) reviews in parallel (each rebased
        # to absolute time, then unioned). Applies to both the first QC and the re-QC.
        subs = list(_srt.parse(open(subtitles_file, encoding="utf-8").read()))
        seg = AudioSegment.from_file(audio_file).set_channels(1)  # decode once
        return qc_check_chunked(seg=seg, subs=subs, client=gclient, model=gmodel,
                                passes=passes, thinking_level=thinking)


@task(cache_policy=NO_CACHE)
def t_qc_decide(issues, bundle, playbook, audio_file, context=None,
                dropped_lines=None):
    with timer("QC decide"):
        from google import genai
        from pydub import AudioSegment
        from song_detect import slice_audio_opus
        gclient = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        gmodel = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

        def _model_call(system, contents, response_schema):
            # contents is a plain prompt string (phase 1) or a list whose first item
            # is the prompt and whose remaining items are raw opus bytes (phase 2).
            if isinstance(contents, list):
                parts = []
                for c in contents:
                    if isinstance(c, (bytes, bytearray)):
                        parts.append(genai.types.Part.from_bytes(
                            data=bytes(c), mime_type="audio/ogg"))
                    else:
                        parts.append(c)
                wire = [system, *parts]
            else:
                wire = [system, contents]
            r = _retry_call(
                lambda: gclient.models.generate_content(
                    model=gmodel, contents=wire,
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        thinking_config=genai.types.ThinkingConfig(
                            thinking_level=genai.types.ThinkingLevel.MEDIUM))))
            return json.loads(r.text)

        # Decode the final audio once; slice the model-requested [start,end] windows.
        seg = AudioSegment.from_file(audio_file).set_channels(1)

        def _audio_provider(reqs):
            clips = {}
            for req in reqs:
                idx = req.get("idx")
                window = req.get("window") or []
                if idx is None or len(window) != 2:
                    continue
                start_s, end_s = float(window[0]), float(window[1])
                clips[idx] = slice_audio_opus(seg, start_s, end_s)
            return clips

        return qc_decide(issues, bundle, playbook, model_call=_model_call,
                         audio_provider=_audio_provider, context=context,
                         dropped_lines=dropped_lines)



@task
def t_cut_speakers(vocal_file, subtitles_file, speakers_array,emotions_tags, speakers_folder):
    with timer("Combine original audio"):
        result = cut_speaker_audio(vocal_file, subtitles_file, speakers_folder, speakers_array, emotions_tags)
    return result

@task
def t_detect_mouth_windows(video, subtitles_file):
    with timer("Detect mouth windows"):
        result = detect_mouth_windows(video, subtitles_file, sample_fps=4.0)
    return result


@task(cache_policy=NO_CACHE)
def t_generate_elevenlab_segments(config, speakers,translated_file, voices, language_code="ru", changed_list = None,  elevenlabs_emotions=1):
    with timer("generate segments"):
        elevenlabs_client = config.get_elevenlabs_client()
        result = tts_generate_multivoice_elevenlabs_segments(translated_file, elevenlabs_client, speakers, config.tts_model, config.tts_segments_folder, language_code=language_code, changed_list=changed_list, voices = voices, elevenlabs_emotions=elevenlabs_emotions)
    return result


@task(cache_policy=NO_CACHE)
def t_generate_cartesia_segments(config, speakers,translated_file, voices, language_code="ru", changed_list = None):
    with timer("generate cartesia segments"):
        result = tts_generate_multivoice_cartesia_segments(translated_file, config.cartesia_api_key, speakers, config.tts_segments_folder, language_code=language_code, changed_list=changed_list, voices = voices)
    return result


@task(cache_policy=NO_CACHE)
def t_generate_inworld_segments(config, speakers,translated_file, voices, language_code="ru", changed_list = None):
    with timer("generate inworld segments"):
        result = tts_generate_multivoice_inworld_segments(translated_file, config.inworld_key, speakers, config.tts_segments_folder, language_code=language_code, changed_list=changed_list, voices = voices)
    return result


@task(cache_policy=NO_CACHE)
def t_generate_fishaudio_segments(config, subtitles_file,translated_file, voice_audio,  speakers, voices, language_code="ru", changed_list = None, run_id="", force_delete=False, force_no_batch=False):
    with timer("generate segments"):
        fishaudio_client = config.get_fishaudio_client()
        result = tts_generate_multivoice_fish_segments(fishaudio_client, translated_file, subtitles_file, voice_audio, speakers, config.tts_segments_folder, language_code, voices, changed_list=changed_list, run_id=run_id, force_delete=force_delete, force_no_batch=force_no_batch)
    return result


def prewarm_indextts2(subtitles_file: str) -> tuple[list, int]:
    """Estimate pod count from source subtitles and fire dummy .spawn() calls to boot containers."""
    import modal
    import srt

    with open(subtitles_file, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))

    total_chars = 0
    for sub in subs:
        text = sub.content.split(":", 1)[1].strip() if ":" in sub.content else sub.content
        total_chars += len(text)

    threshold = int(CHARS_PER_SEC_PER_GPU * TARGET_POD_SECONDS)
    num_pods = max(1, min(20, -(-total_chars // threshold)))

    IndexTTSGenerator = modal.Cls.from_name("index-tts-2-5-generator", "IndexTTSGenerator")
    tts_service = IndexTTSGenerator()

    handles = []
    for _ in range(num_pods):
        handles.append(tts_service.generate.spawn({}, []))

    print(f"Pre-warm: fired {num_pods} dummy containers (from {total_chars} source chars)")
    return handles, num_pods


@task(cache_policy=NO_CACHE)
def t_generate_indextts2_segments(config, translated_file, speakers, emotions_tags, changed_list=None, max_pods=20, duration_factors=None, candidate_texts=None, speaker_base_atempo=None, warm_pods=0):
    with timer("generate index-tts2 segments"):
        result = tts_generate_index_tts2_segments(
            translated_subtitles_file=translated_file,
            speakers_folder=config.speakers_folder,
            speakers=speakers,
            emotions_tags=emotions_tags,
            out_dir=config.tts_segments_folder,
            max_pods=max_pods,
            changed_list=changed_list,
            duration_factors=duration_factors,
            candidate_texts=candidate_texts,
            speaker_base_atempo=speaker_base_atempo,
            warm_pods=warm_pods,
        )
    return result



@task(cache_policy=NO_CACHE)
def t_generate_qwen3tts_segments(config, translated_file, speakers, emotions_tags, language_code="en", changed_list=None, max_pods=20, candidate_texts=None, speaker_base_atempo=None):
    with timer("generate qwen3-tts segments"):
        result = tts_generate_qwen3_tts_segments(
            translated_subtitles_file=translated_file,
            speakers_folder=config.speakers_folder,
            speakers=speakers,
            emotions_tags=emotions_tags,
            out_dir=config.tts_segments_folder,
            language_code=language_code,
            max_pods=max_pods,
            changed_list=changed_list,
            candidate_texts=candidate_texts,
            speaker_base_atempo=speaker_base_atempo,
        )
    return result


@task
def t_combine_tts_segments(speakers, tts_segments_folder):
    combine_audio_files(speakers, tts_segments_folder, tts_segments_folder)
    return tts_segments_folder



@task(cache_policy=NO_CACHE)
def t_tts_build_final(config, speakers, convert_flag, subtitle_visibility_analysis, testing = False, build_cache=None, changed_list=None, per_line_atempo=None, subtitles_file=None, speaker_base_atempo=None, max_speed_factor=None):
    with timer("Build Final TTS"):
        subs_file = subtitles_file or config.subtitles_translated_file
        result = tts_build_final(speakers, subs_file, config.tts_segments_folder, config.tts_segments_folder, subtitle_visibility_analysis, testing, changed_list=changed_list, build_cache=build_cache, per_line_atempo=per_line_atempo, speaker_base_atempo=speaker_base_atempo, max_speed_factor=max_speed_factor)
    return result

@task
def t_get_voice_profiles(audio_file, subtitles_file):
    with timer("Get Voice Profiles"):
        segments = extract_voice_profiles( audio_path=audio_file, srt_path=subtitles_file)
    return segments


@task(cache_policy=NO_CACHE)
def t_build_audio(config, tts_build_final_flag, is_dubbed=False, mix_gains: list[float] | None = None, use_non_speech: bool = True, video_file: str | None = None):
    # mix_gains = [background_db, dialog_db, non_speech_db, original_underlay_db]
    gain_kwargs = {}
    if mix_gains and len(mix_gains) == 4:
        gain_kwargs = {
            "background_gain_db": mix_gains[0],
            "dialog_gain_db": mix_gains[1],
            "non_speech_gain_db": mix_gains[2],
            "original_underlay_gain_db": mix_gains[3],
        }

    # Match original video loudness and dynamic range
    if video_file:
        original_stats = measure_loudness(video_file)
        gain_kwargs.setdefault("target_loudness", original_stats["i"])
        gain_kwargs.setdefault("target_lra", original_stats["lra"])

    with timer("Bild Final Audio"):
        build_audio(
            tts_segments_folder=config.tts_segments_folder,
            background_file=config.music_file,
            output_audio_wav=config.audio_result_file,
            non_speech_layer=config.non_speech_layer_file if (not is_dubbed and use_non_speech) else None,
            original_speech_layer=config.vocal_file if is_dubbed else None,
            stem_background_out=config.stem_background,
            stem_dialog_out=config.stem_dialog,
            stem_original_out=config.stem_original if is_dubbed else None,
            **gain_kwargs,
        )
    return config.audio_result_file

@task(cache_policy=NO_CACHE)
def t_generate_videos(config, video_file, audio_result_file, preview=True):
    with timer("Bild Final Video"):
        output_file = config.final_video_preview_file if preview else config.final_video_file
        generate_videos(video_file, audio_result_file, config.subtitles_retranslated_file, output_file, preview=preview,
                        pre_encoded_video=config.preview_base_video_file if preview else None)
    return output_file

@task(cache_policy=NO_CACHE)
def t_detect_language(config, speaker_segments, audio = None):
    with timer("Detect Language"):
        audio = audio if audio else config.vocal_file
        lang, speech_segments = detect_language_for_routing(audio, speaker_segments)
        return lang, speech_segments

@task
def t_loudness_adjust(subtitles_file, vocal_file,tts_segments_folder):
    with timer("Loudness_adjust"):
        final_entries = run_line_loudness_stage(
            subtitles_file=subtitles_file,
            tts_segments_dir=tts_segments_folder,
            vocals_wav_path=vocal_file)
        return final_entries


@task
def t_test_results(final_audio_file):
    with timer("Test Results"):
        results = qc_check(final_audio_file)
        return results

class STAGES(int, Enum):
    SPLIT = 0
    DIARIZE = 1
    TRANSCRIBE = 2
    EMOTION = 3
    TRANSLATE = 4
    GENERATE = 5
    TIMING_FIX = 6
    COMBINE = 7

class ELEVENLABS_EMOTIONS(int, Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


class TTS_MODEL(int, Enum):
    ELEVENLABS = 0
    INWORLD = 1
    CARTESIA = 2
    FISHAUDIO = 3
    INDEXTTS2 = 4
    QWEN3TTS = 5


# Timing caps keyed by (ttsmodel, dst_language). Verbosity of the *target* language
# drives how much atempo headroom a line needs — Russian runs long, so Qwen3+ru gets
# a wider band. Lookup order: (model, lang) -> (model, None) -> DEFAULT_TIMING_CAPS.
TIMING_CAPS = {
    (TTS_MODEL.QWEN3TTS.value, None): {"base": 1.20, "overflow": 1.25, "max": 1.35},
    (TTS_MODEL.INDEXTTS2.value, None): {"base": 1.35, "overflow": 1.35, "max": 1.45},
}
DEFAULT_TIMING_CAPS = {"base": 1.35, "overflow": 1.35, "max": 1.45}


def resolve_timing_caps(ttsmodel: int, dst_language: str) -> dict:
    return (TIMING_CAPS.get((ttsmodel, dst_language))
            or TIMING_CAPS.get((ttsmodel, None))
            or DEFAULT_TIMING_CAPS)


def _regen_and_combine(
    config,
    speakers_array,
    emotions_tags,
    subtitle_visibility_analysis,
    dst_language,
    ttsmodel,
    speaker_base_atempo,
    timing_overflow_threshold,
    timing_max_speed_factor,
    is_dubbed,
    mix_gains,
    use_non_speech,
    video_file,
    changed_list,
    build_cache=None,
    qc_fix: bool = False,
):
    """Run GENERATE(scoped)→timing→COMBINE once. In qc_fix mode: freeze retranslated
    as truth (pre-sync into translated), reuse the saved per-speaker atempo, classify
    only changed lines, skip retranslation (clamp+spill), then COMBINE (rebuilds the
    non-speech layer via the relocated split_vocal). Returns the final audio path."""
    import shutil

    is_qwen3 = ttsmodel == TTS_MODEL.QWEN3TTS.value

    if qc_fix:
        # Pre-sync: retranslated holds all prior rerolls + agent edits. Copy it into
        # translated so GENERATE reads the frozen truth. Makes the TIMING_FIX-top
        # copyfile(translated -> retranslated) a harmless no-op.
        if os.path.exists(config.subtitles_retranslated_file):
            shutil.copyfile(config.subtitles_retranslated_file,
                            config.subtitles_translated_file)

    # --- GENERATE (scoped to changed_list) ---
    regen_file = config.subtitles_retranslated_file
    if is_qwen3:
        t_generate_qwen3tts_segments.submit(
            config=config, translated_file=regen_file, speakers=speakers_array,
            emotions_tags=emotions_tags, language_code=dst_language,
            changed_list=changed_list,
        ).result()
    else:
        t_generate_indextts2_segments.submit(
            config=config, translated_file=regen_file, speakers=speakers_array,
            emotions_tags=emotions_tags, changed_list=changed_list,
            duration_factors=None, warm_pods=0,
        ).result()

    t_combine_tts_segments.submit(speakers_array, config.tts_segments_folder).result()

    # --- Timing: no whole-track re-measure; classify only changed lines ---
    if qc_fix:
        sb_path = os.path.join(config.data_output_folder, "speaker_base_atempo.json")
        if os.path.exists(sb_path):
            speaker_base_atempo = json.loads(open(sb_path, encoding="utf-8").read())

        remeasure = t_tts_build_final.submit(
            config, speakers=speakers_array, convert_flag=True,
            subtitle_visibility_analysis=subtitle_visibility_analysis,
            testing=True, build_cache=build_cache, changed_list=changed_list,
            subtitles_file=config.subtitles_retranslated_file,
        ).result()
        build_cache = remeasure["build_cache"]

        classification = classify_lines(
            stats=remeasure["stats"], segment_meta=remeasure["segment_meta"],
            speaker_base_atempo=speaker_base_atempo,
            overflow_threshold=timing_overflow_threshold,
        )
        # Merge changed lines' atempo into the saved natural_timing.json (don't shift
        # other lines). Skip retranslation entirely — clamp+spill on longer fixes.
        natural = _load_json_or(
            os.path.join(config.data_output_folder, "natural_timing.json"),
            {"per_line_atempo": {}})
        natural["per_line_atempo"].update(
            {str(k): v for k, v in classification["per_line_atempo"].items()})
        with open(os.path.join(config.data_output_folder, "natural_timing.json"),
                  "w", encoding="utf-8") as f:
            json.dump(natural, f, indent=2, ensure_ascii=False)

    # --- COMBINE (includes relocated split_vocal) ---
    subtitles_for_combine = (config.subtitles_retranslated_file
                             if os.path.exists(config.subtitles_retranslated_file)
                             else config.subtitles_translated_file)
    t_loudness_adjust.submit(subtitles_file=subtitles_for_combine,
                             vocal_file=config.vocal_file,
                             tts_segments_folder=config.tts_segments_folder).result()
    split_vocal_fut = t_split_vocal.submit(
        config.vocal_file, subtitles_for_combine, config.non_speech_layer_file)
    tts_build_final_fut = t_tts_build_final.submit(
        config, speakers=speakers_array, convert_flag=True,
        subtitle_visibility_analysis=subtitle_visibility_analysis, testing=False,
        speaker_base_atempo=speaker_base_atempo, subtitles_file=subtitles_for_combine,
        max_speed_factor=timing_max_speed_factor)
    split_vocal_fut.result()
    build_audio_fut = t_build_audio.submit(
        config, tts_build_final_flag=tts_build_final_fut.result(),
        is_dubbed=is_dubbed, mix_gains=mix_gains, use_non_speech=use_non_speech,
        video_file=video_file)
    return build_audio_fut.result()


def _load_json_or(path: str, default):
    try:
        return json.loads(open(path, encoding="utf-8").read())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


# === Main Flow ===
@flow(name="video-dubbing-pipeline", task_runner=ConcurrentTaskRunner(max_workers=6))
def dubbing_flow(
        video_file:str,
        dst_language:str,
        trans_type:str = "default",
        ttsmodel:int = TTS_MODEL.INWORLD.value,
        num_speakers=None,
        emotions_flag:bool=True,
        elevenlabs_emotions:int = ELEVENLABS_EMOTIONS.MEDIUM.value,
        changed_list:list| None=None,
        is_dubbed: bool = False,
        test_mode: bool = False,
        test_duration_sec: float = 30.0,
        run_id:str = "",
        output_file:str| None = None,
        stage:int = STAGES.TRANSLATE.value,
        mix_gains: list[float] | None = None,
        use_non_speech: bool = True,
):
    if not run_id:
        run_id = generate_run_id()

    storagebox = Path("/mnt/storagebox") / "output" / run_id
    local_dir = Path("output") / run_id
    with timer("Move from storagebox to output"):
        if storagebox.exists():
            if local_dir.exists():
                shutil.rmtree(str(local_dir))
            shutil.move(str(storagebox), str(local_dir))

    config = Configuration(run_id)
    config.create_dirs()
    print(f"Starting dubbing flow with run_id: {run_id} on step {STAGES(stage).name}")

    # Save run parameters to general_config.json
    path = Path(config.general_config_file)
    data = json.loads(path.read_text()) if path.exists() else {}
    data["run_params"] = {
        "video_file": video_file,
        "dst_language": dst_language,
        "trans_type": trans_type,
        "ttsmodel": ttsmodel,
        "elevenlabs_emotions": elevenlabs_emotions,
        "num_speakers": num_speakers,
        "emotions_flag": emotions_flag,
        "test_mode": test_mode,
        "is_dubbed": is_dubbed,
        "test_duration_sec": test_duration_sec,
        "use_non_speech": use_non_speech,
        "output_file": None
    }
    path.write_text(json.dumps(data, indent=4))

    punctuation = False if dst_language != 'ru' else True # For Yandex it should go with punctuation
    voices = json.loads(Path(config.permanent_voices_file).read_text()).get(run_id, {}).get(TTS_MODEL(ttsmodel).name,{})

    preview_proc = None

    if stage == STAGES.SPLIT:
        if test_mode:
            with timer("Preparing test video"):
                cut_video_fut = t_cut_video.submit(video_file, config.test_video_file, test_duration_sec)
                video_file = cut_video_fut.result()
        audio_fut = t_extract_audio.submit(video_file, config.audio_file)
        audio_file = audio_fut.result()
        preview_proc = encode_preview_base(video_file, config.preview_base_video_file)
        split_fut  = t_split_audio.submit(config, audio_file, run_id)
        vocal_file = split_fut.result()["vocal_file"]
        vocal_asr_file = split_fut.result()["vocal_asr_file"]

    else:
        audio_file = config.audio_file
        vocal_file = config.vocal_file
        vocal_asr_file = config.vocal_asr_file
        if test_mode:
            video_file = config.test_video_file

    if stage <= STAGES.DIARIZE:
        # Detect language early using Silero VAD (no diarization needed)
        initial_language_fut = t_detect_language.submit(config, None, vocal_asr_file)
        # Start diarization in parallel
        diar_fut = t_diarize.submit(config, vocal_asr_file, num_speakers, run_id)

        initial_language, speech_segments = initial_language_fut.result()
        path = Path(config.general_config_file)
        data = json.loads(path.read_text()) if path.exists() else {}
        data["initial_src_language"] = initial_language
        path.write_text(json.dumps(data, indent=4))
    else:
        diar_fut = None
        speech_segments = None
        speakers_segments = json.loads(Path(config.speakers_segments_file).read_text())
        initial_language = json.loads(Path(config.general_config_file).read_text())["initial_src_language"]

    if stage <= STAGES.TRANSCRIBE:
        if trans_type == 'ocr':
            # OCR path needs diarization first
            if diar_fut is not None:
                speakers_segments = diar_fut.result()
                Path(config.speakers_segments_file).write_text(json.dumps(speakers_segments, indent=4))
            transcribe_fut = t_process_video_with_subs.submit(config, video_file, speakers_segments, initial_language, num_speakers, run_id)
            src_language = transcribe_fut.result()
        else:
            # Raw STT (engine chosen by language inside t_transcribe_raw) runs in
            # parallel with diarization, then the shared assembler aligns + AI-fixes.
            # Per-engine input: en=vocal_file, zh/ja=vocal_asr_file, else=audio_file.
            lang2 = (initial_language or "auto")[:2].lower()
            if lang2 == "en":
                asr_input = vocal_file
            elif lang2 in ("zh", "ja"):
                asr_input = vocal_asr_file
            else:
                asr_input = audio_file

            raw_fut = t_transcribe_raw.submit(config, asr_input, initial_language, run_id, speech_segments)

            if diar_fut is not None:
                speakers_segments = diar_fut.result()
                Path(config.speakers_segments_file).write_text(json.dumps(speakers_segments, indent=4))
            words_data, trans_language = raw_fut.result()

            src_language = t_assemble_transcription.submit(
                config, words_data, trans_language, speakers_segments, num_speakers
            ).result()

        path = Path(config.general_config_file)
        data = json.loads(path.read_text()) if path.exists() else {}
        data["src_language"] = src_language
        path.write_text(json.dumps(data, indent=4))
    else:
        if diar_fut is not None:
            speakers_segments = diar_fut.result()
            Path(config.speakers_segments_file).write_text(json.dumps(speakers_segments, indent=4))
        src_language = json.loads(Path(config.general_config_file).read_text())["src_language"]
    # --- EMOTION stage (must complete before TRANSLATE for duration estimation) ---
    gemini_emotions_fut = None
    detect_gender_fut = None
    mouth_windows_fut = None
    detect_songs_fut = None
    prewarm_handles = None
    prewarm_pods = 0

    if stage <= STAGES.EMOTION:
        gemini_emotions_fut = t_gemini_extract_emotions.submit(config, vocal_asr_file, config.subtitles)
        detect_gender_fut = t_detect_gender.submit(audio_file=config.vocal_file,
                                                   subtitle_file=config.subtitles,
                                                   gemini_api_key=config.gemini_api_key,
                                                   gemini_model=config.gemini_model)
        mouth_windows_fut = t_detect_mouth_windows.submit(video_file, config.subtitles)
        detect_songs_fut = t_detect_songs.submit(config, config.audio_file,
                                                 config.subtitles)
        #prewarm commented
        # if ttsmodel == TTS_MODEL.INDEXTTS2.value:
        #     prewarm_handles, prewarm_pods = prewarm_indextts2(config.subtitles)

    # Resolve detect_gender early so translate can start in parallel with Gemini
    if detect_gender_fut:
        speakers_array = detect_gender_fut.result()
        Path(config.speakers_file).write_text(json.dumps(speakers_array, indent=4))
    else:
        speakers_array = json.loads(Path(config.speakers_file).read_text())

    # --- TRANSLATE stage (runs in parallel with Gemini emotions + mouth detection) ---
    translated_fut = None

    if stage <= STAGES.TRANSLATE:
        translated_fut = t_translate.submit(config, subtitles=config.subtitles,
                                            src_lang=src_language,
                                            translate_to_language=dst_language,
                                            subtitles_translated=config.subtitles_translated_file,
                                            punctuation=punctuation,
                                            speakers_data=speakers_array)

    # Now wait for Gemini emotions (which has been running in parallel with translate)
    if gemini_emotions_fut:
        emotions_tags = gemini_emotions_fut.result()
        Path(config.emotions_tags_file).write_text(json.dumps(emotions_tags, indent=4))
    else:
        emotions_tags = json.loads(Path(config.emotions_tags_file).read_text())

    if mouth_windows_fut:
        subtitle_visibility_analysis = mouth_windows_fut.result()
        Path(config.subtitles_visibility_file).write_text(json.dumps(subtitle_visibility_analysis))
    else:
        subtitle_visibility_analysis = json.loads(Path(config.subtitles_visibility_file).read_text())

    if translated_fut:
        translate_stats = translated_fut.result()
    translated_file = config.subtitles_translated_file

    if detect_songs_fut is not None:
        song_ids = detect_songs_fut.result()
        if song_ids:
            dropped = drop_sub_ids(config.subtitles_translated_file, song_ids,
                                   sidecar_path=config.dropped_song_lines_file)
            print(f"🎵 dropped {len(dropped)} sung line(s) from translated SRT: {dropped}")

    if stage <= STAGES.GENERATE:
        cut_speakers_fut = t_cut_speakers.submit(vocal_file=vocal_asr_file, subtitles_file = config.subtitles, speakers_array=speakers_array, emotions_tags=emotions_tags, speakers_folder = config.speakers_folder)
        speakers_array = cut_speakers_fut.result()
        Path(config.speakers_file).write_text(json.dumps(speakers_array, indent=4))

        if prewarm_handles:
            with timer("Wait for pre-warm containers"):
                for h in prewarm_handles:
                    h.get()

        if ttsmodel == TTS_MODEL.ELEVENLABS.value:
            t_generate_segments_fut = t_generate_elevenlab_segments.submit(
                config,
                speakers=speakers_array,
                translated_file=translated_file,
                voices=voices,
                language_code=dst_language,
                changed_list=changed_list,
                elevenlabs_emotions=elevenlabs_emotions
            )
        elif ttsmodel == TTS_MODEL.INWORLD.value:
            t_generate_segments_fut = t_generate_inworld_segments.submit(
                config,
                speakers=speakers_array,
                translated_file=translated_file,
                voices=voices,
                language_code=dst_language,
                changed_list=changed_list
            )

        elif ttsmodel == TTS_MODEL.CARTESIA.value:
            t_generate_segments_fut = t_generate_cartesia_segments.submit(
                config,
                speakers=speakers_array,
                translated_file=translated_file,
                voices=voices,
                language_code=dst_language,
                changed_list=changed_list
            )
        elif ttsmodel == TTS_MODEL.FISHAUDIO.value:
            t_generate_segments_fut = t_generate_fishaudio_segments.submit(
                config=config,
                subtitles_file=config.subtitles,
                translated_file=translated_file,
                voice_audio=vocal_file,
                speakers=speakers_array,
                voices=voices,
                language_code=dst_language,
                changed_list=changed_list,
                run_id = run_id,
                force_delete=True,
                force_no_batch=True if changed_list else False,
                )
        elif ttsmodel == TTS_MODEL.INDEXTTS2.value:
            t_generate_segments_fut = t_generate_indextts2_segments.submit(
                config=config,
                translated_file=translated_file,
                speakers=speakers_array,
                emotions_tags=emotions_tags,
                changed_list=changed_list,
                duration_factors=None,
                warm_pods=prewarm_pods,
            )
        elif ttsmodel == TTS_MODEL.QWEN3TTS.value:
            t_generate_segments_fut = t_generate_qwen3tts_segments.submit(
                config=config,
                translated_file=translated_file,
                speakers=speakers_array,
                emotions_tags=emotions_tags,
                language_code=dst_language,
                changed_list=changed_list,
            )
        else:
            raise ValueError("Unknown ttsmodel {}".format(ttsmodel))

        voices = t_generate_segments_fut.result()
        path = Path(config.general_config_file)
        data = json.loads(path.read_text()) if path.exists() else {}
        data["voices"] = voices
        path.write_text(json.dumps(data, indent=4))

        all_voices = json.loads(Path(config.permanent_voices_file).read_text())
        all_voices.setdefault(run_id, {})[TTS_MODEL(ttsmodel).name] = voices

        Path(config.permanent_voices_file).write_text(json.dumps(all_voices, indent=4))
        combine_tts_segments_res = t_combine_tts_segments.submit(speakers_array, config.tts_segments_folder)
        combine_tts_segments_res.result()

        # Track how many pods the first TTS used — retranslation can reuse them as warm
        if ttsmodel == TTS_MODEL.INDEXTTS2.value:
            import srt as _srt
            with open(translated_file, "r", encoding="utf-8") as _f:
                _tgt_subs = list(_srt.parse(_f.read()))
            _tgt_chars = sum(
                len(s.content.split(":", 1)[1].strip()) if ":" in s.content else len(s.content)
                for s in _tgt_subs
            )
            _threshold = int(CHARS_PER_SEC_PER_GPU * TARGET_POD_SECONDS)
            first_tts_pods = max(1, min(20, -(-_tgt_chars // _threshold)))
        else:
            first_tts_pods = 0
    else:
        speakers_array=json.loads(Path(config.speakers_file).read_text())
        # voices = json.loads(Path(config.general_config_file).read_text())["voices"]
        first_tts_pods = 0

    # ---------------- TIMING_FIX stage: two-pass TTS timing ----------------
    # Caps depend on both engine and target-language verbosity (see TIMING_CAPS).
    is_qwen3 = ttsmodel == TTS_MODEL.QWEN3TTS.value
    _caps = resolve_timing_caps(ttsmodel, dst_language)
    timing_speaker_base_cap = _caps["base"]
    timing_overflow_threshold = _caps["overflow"]
    timing_max_speed_factor = _caps["max"]

    natural_per_line_atempo = None
    speaker_base_atempo = None
    build_cache = None

    if stage > STAGES.TIMING_FIX:
        # Load saved speaker base atempo from a previous TIMING_FIX run
        sb_path = os.path.join(config.data_output_folder, "speaker_base_atempo.json")
        if os.path.exists(sb_path):
            with open(sb_path, "r", encoding="utf-8") as f:
                speaker_base_atempo = json.load(f)

    if stage <= STAGES.TIMING_FIX:
        copyfile(config.subtitles_translated_file, config.subtitles_retranslated_file)

        # Step 1: Test build to measure actual durations (TTS was generated at factor=1.0)
        test_build = t_tts_build_final.submit(
            config, speakers=speakers_array, convert_flag=True,
            subtitle_visibility_analysis=subtitle_visibility_analysis,
            testing=True, build_cache=build_cache, changed_list=changed_list,
            subtitles_file=config.subtitles_retranslated_file,
        ).result()
        build_cache = test_build["build_cache"]

        # Step 2: Compute per-speaker base atempo
        speaker_base_atempo = compute_speaker_base_atempo(
            stats=test_build["stats"],
            segment_meta=test_build["segment_meta"],
            speaker_base_cap=timing_speaker_base_cap,
        )
        if speaker_base_atempo:
            print(f"🎯 Natural atempo — speaker bases: {speaker_base_atempo}")
            sf_path = os.path.join(config.data_output_folder, "speaker_base_atempo.json")
            with open(sf_path, "w", encoding="utf-8") as f:
                json.dump(speaker_base_atempo, f, indent=2, ensure_ascii=False)

        # Step 3: Classify lines (ok / overflow / underflow)
        classification = classify_lines(
            stats=test_build["stats"],
            segment_meta=test_build["segment_meta"],
            speaker_base_atempo=speaker_base_atempo,
            overflow_threshold=timing_overflow_threshold,
        )
        natural_per_line_atempo = classification["per_line_atempo"]
        overflow_indices = classification["overflow_indices"]
        underflow_indices = classification["underflow_indices"]

        save_natural_timing_data(
            output_dir=config.data_output_folder,
            speaker_base_atempo=speaker_base_atempo,
            per_line_atempo=natural_per_line_atempo,
            overflow_indices=overflow_indices,
            underflow_indices=underflow_indices,
        )

        print(f"📊 Classification: {len(overflow_indices)} overflow, {len(underflow_indices)} underflow, "
              f"{len(natural_per_line_atempo) - len(overflow_indices) - len(underflow_indices)} ok")

        # Step 4: Retranslate overflow lines (too long) and underflow lines (too short)
        if overflow_indices or underflow_indices:
            from post_build_fix import apply_retranslation

            overflow_requests = {}
            if overflow_indices:
                overflow_requests = build_retranslation_request(
                    overflow_indices=overflow_indices,
                    stats=test_build["stats"],
                    segment_meta=test_build["segment_meta"],
                    speaker_base_atempo=speaker_base_atempo,
                    subtitles_file=config.subtitles_retranslated_file,
                    source_subtitles_file=config.subtitles,
                    overflow_threshold=timing_overflow_threshold,
                )

            underflow_requests = {}
            if underflow_indices:
                underflow_requests = build_underflow_retranslation_request(
                    underflow_indices=underflow_indices,
                    stats=test_build["stats"],
                    segment_meta=test_build["segment_meta"],
                    speaker_base_atempo=speaker_base_atempo,
                    subtitles_file=config.subtitles_retranslated_file,
                    source_subtitles_file=config.subtitles,
                )

            candidate_texts = t_retranslate_timing_fix.submit(
                config=config,
                overflow_requests=overflow_requests,
                underflow_requests=underflow_requests,
                target_language=dst_language,
            ).result()

            if candidate_texts:
                if is_qwen3:
                    selected = t_generate_qwen3tts_segments.submit(
                        config=config,
                        translated_file=config.subtitles_retranslated_file,
                        speakers=speakers_array,
                        emotions_tags=emotions_tags,
                        language_code=dst_language,
                        changed_list=list(candidate_texts.keys()),
                        candidate_texts=candidate_texts,
                        speaker_base_atempo=speaker_base_atempo,
                    ).result()
                else:
                    selected = t_generate_indextts2_segments.submit(
                        config=config,
                        translated_file=config.subtitles_retranslated_file,
                        speakers=speakers_array,
                        emotions_tags=emotions_tags,
                        changed_list=list(candidate_texts.keys()),
                        duration_factors=None,
                        candidate_texts=candidate_texts,
                        speaker_base_atempo=speaker_base_atempo,
                        warm_pods=first_tts_pods,
                    ).result()

                if selected:
                    for idx, winning_text in selected.items():
                        apply_retranslation(idx, winning_text, config.subtitles_retranslated_file)

                # Re-measure after retranslation to update per_line_atempo
                remeasure = t_tts_build_final.submit(
                    config, speakers=speakers_array, convert_flag=True,
                    subtitle_visibility_analysis=subtitle_visibility_analysis,
                    testing=True, build_cache=build_cache,
                    changed_list=list(candidate_texts.keys()),
                    subtitles_file=config.subtitles_retranslated_file,
                ).result()
                build_cache = remeasure["build_cache"]

                # Recompute classification with updated measurements
                classification = classify_lines(
                    stats=remeasure["stats"],
                    segment_meta=remeasure["segment_meta"],
                    speaker_base_atempo=speaker_base_atempo,
                    overflow_threshold=timing_overflow_threshold,
                )
                natural_per_line_atempo = classification["per_line_atempo"]

    if stage <= STAGES.COMBINE:
        subtitles_for_combine = config.subtitles_retranslated_file if os.path.exists(config.subtitles_retranslated_file) else config.subtitles_translated_file
        # Non-speech layer (laughs/screams/reactions) is carved from the ORIGINAL
        # vocal for spans not covered by a subtitle. Built here (not at TRANSLATE) so
        # a QC-fix cycle — which re-runs COMBINE after drop_line/change_timing edits —
        # rebuilds it against the frozen retranslated subs. Runs parallel to the timing
        # build; both only feed build_audio, so it costs no wall-clock time.
        split_vocal_fut = t_split_vocal.submit(
            vocal_file, subtitles_for_combine, config.non_speech_layer_file)
        loudness_adjust_fut = t_loudness_adjust.submit(subtitles_file=subtitles_for_combine, vocal_file=config.vocal_file, tts_segments_folder = config.tts_segments_folder)
        loudness_adjust_fut.result()
        tts_build_final_fut = t_tts_build_final.submit(config, speakers=speakers_array, convert_flag=True, subtitle_visibility_analysis=subtitle_visibility_analysis, testing=False, speaker_base_atempo=speaker_base_atempo, subtitles_file=subtitles_for_combine, max_speed_factor=timing_max_speed_factor)
        split_vocal_fut.result()
        build_audio_fut = t_build_audio.submit(config, tts_build_final_flag=tts_build_final_fut.result(),
                                            is_dubbed=is_dubbed, mix_gains=mix_gains, use_non_speech=use_non_speech, video_file=video_file)
        audio_result_file = build_audio_fut.result()
        if preview_proc is not None:
            preview_proc.wait()
            if preview_proc.returncode != 0:
                Path(config.preview_base_video_file).unlink(missing_ok=True)

        # Render the pass-1 video concurrently with QC — QC only listens to the audio,
        # not the video. In the common branches (unsupported engine / no issues / all
        # proposals) we await this render, hidden behind QC's Gemini call. Only the
        # auto-fix branch supersedes it; there we await it first so the pass-1 write
        # can't race the pass-2 render on the same output file.
        video_fut = t_generate_videos.submit(config, video_file, audio_result_file, preview=False)

        def _finalize_video(output_file):
            path = Path(config.general_config_file)
            data = json.loads(path.read_text()) if path.exists() else {}
            data["output_file"] = output_file
            path.write_text(json.dumps(data, indent=4))
            return output_file

        # ---------------- post-COMBINE QC-fix tail (same flow run) ----------------
        qc_log_path = os.path.join(config.data_output_folder, "qc_fix_log.json")
        # _regen_and_combine only regenerates on the Modal engines (IndexTTS2/Qwen3).
        # For API engines a fix would regenerate the line on the wrong engine (different
        # voice), so the QC-fix cycle is skipped there until those paths are wired.
        if ttsmodel not in (TTS_MODEL.INDEXTTS2.value, TTS_MODEL.QWEN3TTS.value):
            print(f"ℹ️  QC-fix skipped: ttsmodel={TTS_MODEL(ttsmodel).name} not "
                  f"supported by targeted regen; shipping render pass 1")
            write_fix_log({"summary": {"skipped": "unsupported_ttsmodel"},
                           "decisions": [], "proposals": []}, qc_log_path)
            output_file = _finalize_video(video_fut.result())
        else:
            qc_issues_path = os.path.join(config.data_output_folder, "qc_issues.json")
            try:
                issues = t_qc_listen_check.submit(
                    subtitles_for_combine, config.audio_result_file).result()
                write_qc_issues(issues, qc_issues_path)

                if not issues:                                   # zero-issue path
                    write_fix_log(build_fix_log([], [], reqc_count=0), qc_log_path)
                    output_file = _finalize_video(video_fut.result())
                else:
                    bundle = build_evidence_bundle(issues, config)
                    context = build_shared_context(config)
                    playbook = load_playbook("config/qc_playbook.json")
                    # Stash of lines soft-dropped by song detection; lets the agent
                    # promote a hedged "restore this window" propose into restore_line.
                    try:
                        dropped_lines = json.loads(
                            Path(config.dropped_song_lines_file).read_text())
                    except (FileNotFoundError, json.JSONDecodeError):
                        dropped_lines = {}

                    decisions = t_qc_decide.submit(
                        issues, bundle, playbook,
                        config.audio_result_file, context=context,
                        dropped_lines=dropped_lines).result()
                    auto, proposals = split_auto_and_proposals(decisions)

                    if not auto:                                 # all proposals -> no regen
                        write_fix_log(build_fix_log([], proposals, reqc_count=0),
                                      qc_log_path)
                        output_file = _finalize_video(video_fut.result())
                    else:
                        # A fix is coming — the pass-1 video is superseded. Await it
                        # first so its write completes before the pass-2 render targets
                        # the same output file (it overlapped QC, so likely already done).
                        video_fut.result()
                        auto = resolve_collisions(auto)          # dedupe by idx FIRST
                        proposals += [d for d in decisions
                                      if not d.auto_apply and d not in proposals]
                        changed = apply_fixes(
                            auto, subtitles_file=config.subtitles_retranslated_file,
                            emotions_file=config.emotions_tags_file,
                            dropped_lines_file=config.dropped_song_lines_file)

                        _regen_and_combine(
                            config=config, speakers_array=speakers_array,
                            emotions_tags=emotions_tags,
                            subtitle_visibility_analysis=subtitle_visibility_analysis,
                            dst_language=dst_language, ttsmodel=ttsmodel,
                            speaker_base_atempo=speaker_base_atempo,
                            timing_overflow_threshold=timing_overflow_threshold,
                            timing_max_speed_factor=timing_max_speed_factor,
                            is_dubbed=is_dubbed, mix_gains=mix_gains,
                            use_non_speech=use_non_speech, video_file=video_file,
                            changed_list=changed, qc_fix=True)

                        # Pass-2 audio is now written. Render the pass-2 video and run
                        # re-QC concurrently — re-QC only listens to the audio, so the
                        # render (which just reads that same audio) can overlap it
                        # instead of waiting behind it, mirroring the pass-1 pattern.
                        pass2_video_fut = t_generate_videos.submit(
                            config, video_file, config.audio_result_file,
                            preview=False)

                        reqc_issues = t_qc_listen_check.submit(
                            config.subtitles_retranslated_file,
                            config.audio_result_file, label="QC re-check").result()
                        write_qc_issues(
                            reqc_issues,
                            os.path.join(config.data_output_folder,
                                         "qc_issues_reqc.json"))
                        diff_reqc(auto, issues, reqc_issues)
                        append_promotion_queue(
                            auto, "config/qc_promotion_queue.json")
                        write_fix_log(build_fix_log(auto, proposals, reqc_count=1),
                                      qc_log_path)
                        # Await the pass-2 render that overlapped re-QC.
                        output_file = _finalize_video(pass2_video_fut.result())
            except Exception as e:  # noqa: BLE001 — QC-fix must never fail the dub
                print(f"⚠️  QC-fix tail failed ({e}); shipping render pass 1")
                write_fix_log({"summary": {"error": str(e)}, "decisions": [],
                               "proposals": []}, qc_log_path)
                output_file = _finalize_video(video_fut.result())

        with timer("Move output to storagebox"):
            storagebox = Path("/mnt/storagebox")
            if local_dir.exists() and storagebox.is_dir():
                shutil.move(str(local_dir), str(storagebox / "output" / run_id))
            elif not storagebox.is_dir():
                print(f"ℹ️  /mnt/storagebox not mounted; leaving output at {local_dir}")


def generate_run_id():
    timestamp = datetime.now().strftime("%Y%m%d")
    rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{timestamp}_{rand}"

def preconfigure():
    from pydub import AudioSegment

    AudioSegment.converter = "/opt/homebrew/bin/ffmpeg"
    AudioSegment.ffprobe = "/opt/homebrew/bin/ffprobe"
    os.environ["PREFECT_HOME"] = "/tmp/.prefect"

# # # === Entry Point ===
if __name__ == "__main__":
    preconfigure()
    dubbing_flow("input/rapuntsel.mp4",
                 dst_language="ru",
                 trans_type='default',
                 emotions_flag=True,
                 ttsmodel=TTS_MODEL.QWEN3TTS.value,
                 elevenlabs_emotions=ELEVENLABS_EMOTIONS.HIGH.value,
                 # num_speakers=1,
                 test_mode=False,
                 changed_list=[],
                 run_id='20260922_rapuntsel_CUT_05',
                 # test_duration_sec=120,
                 is_dubbed=False,
                 use_non_speech=True,
                 stage = STAGES.EMOTION.value)