from __future__ import annotations
import os
#before 1 starting execute prefect server start
API_URL = "http://127.0.0.1:4200/api"
os.environ["PREFECT_API_URL"] = API_URL
os.environ["PREFECT_LOGGING_LEVEL"] = "INFO"
from enum import Enum
from pathlib import Path
import httpx
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

import pickle

# === Import your existing functions ===
from audio import extract_audio, split_audio, split_vocal, cut_speaker_audio, combine_audio_files
from diarization import diarize
# from subtitles import translate  # replaced by bounded translation
from translate_bounded import translate_bounded
from translate_duration import translate_duration
from fix_timing_subs import rewrite_timing_mismatched_subtitles
from gender import detect_gender
from tts_v2 import tts_build_final, tts_generate_multivoice_elevenlabs_segments, tts_generate_multivoice_cartesia_segments
from video import generate_videos, cut_video, encode_preview_base
from openvoice_module import openvoice_convert_runpod
from subs_from_ocr import process_video_with_subs
from emotion_detect_module import emotion_detect_runpod
from emotion_extract_module import extract_emotions
from gemini_emotion_extract import extract_emotions_gemini
from detect_language import detect_language_for_routing
from voice_profiles import extract_voice_profiles
from assemblyai_transcribe import assemblyai_transcribe
from speechmatics_transcribe import speechmatics_transcribe
from alibabacloud_transcribe import alibabacloud_transcribe
from deepgram_transcribe import deepgram_transcribe
from loudness_adjust import run_line_loudness_stage
from detect_mouth_windows import detect_mouth_windows
from test_results import qc_check
from final_audio import build_audio
from prefect.cache_policies import NO_CACHE
from tts_inworld import tts_generate_multivoice_inworld_segments
from tts_fish_audio import tts_generate_multivoice_fish_segments
from tts_index_tts2 import tts_generate_index_tts2_segments


from whisper_transcribe import groq_whisper_large_v3_transcribe
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
def t_emotion_detect(config, audio_file, subtitles_file, emotions_flag, changed_list, run_id):
    with timer("Emotion detect"):
        boto_session = config.get_boto_session()
        openai_client = config.get_openai_client()
        if changed_list:
            speakers_array = json.loads(Path(config.speakers_file).read_text())
        else:
            speakers_array = None
        speakers = emotion_detect_runpod(openai_client=openai_client,
                                         openai_model=config.openai_emotion_model,
                                         audio_file=audio_file,
                                         subtitles_file=subtitles_file,
                                         boto_session=boto_session,
                                         bucket_name=config.s3_bucket_name,
                                         runpod_key=config.runpod_key,
                                         runpod_template_id=config.runpod_emotion_detect_id,
                                         emotions_flag=emotions_flag,
                                         changed_list = changed_list,
                                         old_speakers_array = speakers_array,
                                         run_id= run_id)
    return speakers


@task(cache_policy=NO_CACHE)
def t_extract_emotions(config, audio_file, subtitles_file, run_id):
    with timer("Extract emotions (WavLM)"):
        boto_session = config.get_boto_session()
        embeddings = extract_emotions(
            audio_file=audio_file,
            subtitles_file=subtitles_file,
            boto_session=boto_session,
            bucket_name=config.s3_bucket_name,
            run_id=run_id,
        )
    return embeddings


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
def t_assemblyai_transcribe(config, audio_file, speaker_segments, language, num_speakers):
    with timer("assemblyai_transcribe"):
        openai_client = config.get_openai_client()
        lang = assemblyai_transcribe(
            audio_file_raw=audio_file,
            subtitles_file=config.subtitles,
            speaker_segments=speaker_segments,
            assemblyai_api_key=config.assemblyai_api_key,
            openai_client=openai_client,
            openai_model=config.openai_diarization_model,
            num_speakers=num_speakers,
            language=language)
    return lang

@task(cache_policy=NO_CACHE)
def t_speechmatics_transcribe(config, audio_file, speaker_segments, language, num_speakers):
    with timer("speechmatics_transcribe"):
        openai_client = config.get_openai_client()
        lang = speechmatics_transcribe(
            audio_file_raw=audio_file,
            subtitles_file=config.subtitles,
            speaker_segments=speaker_segments,
            speechmatics_api_key=config.speechmatics_api_key,
            openai_client=openai_client,
            openai_model=config.openai_diarization_model,
            num_speakers=num_speakers,
            language=language)
    return lang

@task(cache_policy=NO_CACHE)
def t_whisper_transcribe(config, audio_file, speaker_segments, language, num_speakers):
    with timer("whisper_transcribe"):
        openai_client = config.get_openai_client()
        lang = groq_whisper_large_v3_transcribe(
            audio_file_raw=audio_file,
            subtitles_file=config.subtitles,
            speaker_segments=speaker_segments,
            groq_api_key=config.groq_api_key,
            openai_client=openai_client,
            openai_model=config.openai_diarization_model,
            num_speakers=num_speakers,
            language=language)
    return lang




@task(cache_policy=NO_CACHE)
def t_alibabacloud_transcribe(config, audio_file, speaker_segments, language, num_speakers, run_id):
    with timer("alibaba_transcribe"):
        boto_session = config.get_boto_session()
        openai_client = config.get_openai_client()

        lang = alibabacloud_transcribe(
            audio_file_raw=audio_file,
            subtitles_file=config.subtitles,
            speaker_segments=speaker_segments,
            alibaba_api_key=config.alibaba_api_key,
            s3_bucket_name=config.s3_bucket_name,
            boto_session=boto_session,
            openai_client=openai_client,
            openai_model=config.openai_diarization_model,
            num_speakers=num_speakers,
            language=language,
            run_id=run_id)
    return lang

@task(cache_policy=NO_CACHE)
def t_deepgram_transcribe(config, audio_file, speaker_segments, language, num_speakers):
    with timer("deepgram_transcribe"):
        openai_client = config.get_openai_client()

        lang = deepgram_transcribe(
            audio_file_raw=audio_file,
            subtitles_file=config.subtitles,
            speaker_segments=speaker_segments,
            deepgram_api_key=config.deepgram_api_key,
            openai_client=openai_client,
            openai_model=config.openai_diarization_model,
            num_speakers=num_speakers,
            language=language,
        )
    return lang

@task(cache_policy=NO_CACHE)
def t_process_video_with_subs(config, video_file, speaker_segments, language, num_speakers, run_id):
    with timer("process_video_with_subs"):
        boto_session = config.get_boto_session()
        openai_client = config.get_openai_client()
        lang = process_video_with_subs(video_file, boto_session,openai_client, config.openai_diarization_model,  config.s3_bucket_name, config.runpod_key, config.runpod_paddleocr_id,speaker_segments, config.subtitles, language, num_speakers, run_id)
    return lang

@task
def t_detect_gender(audio_file, subtitle_file):
    with timer("detect gender"):
        result = detect_gender(audio_file, subtitle_file)
    return result

# @task(cache_policy=NO_CACHE)
# def t_translate(config, subtitles, src_lang, translate_to_language, subtitles_translated, punctuation):
#     with timer("translate"):
#         openai_client = config.get_openai_client()
#         translate(openai_client, subtitles, src_lang, translate_to_language, config.openai_translate_model, subtitles_translated, punctuation)
#     return config.subtitles_translated_file

@task(cache_policy=NO_CACHE)
def t_translate(config, subtitles, src_lang, translate_to_language, subtitles_translated, punctuation, emotions_data=None, speakers_data=None):
    with timer("translate (duration-bounded)"):
        result = translate_duration(
            anthropic_api_key=config.anthropic_api_key,
            subtitles_file=subtitles,
            source_language=src_lang,
            target_language=translate_to_language,
            pass1_model=config.anthropic_translate_pass1_model,
            pass2_model=config.anthropic_translate_pass2_model,
            result_file=subtitles_translated,
            emotions_data=emotions_data,
            speakers_data=speakers_data,
        )
        import json
        from dataclasses import asdict
        translation_stats_path = os.path.join(os.path.dirname(subtitles_translated), "translation_stats.json")
        with open(translation_stats_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    return result

@task(cache_policy=NO_CACHE)
def t_rewrite_timing_mismatched_subtitles(config, original_subtitles,translated_subtitles, fixed_translated_subtitles, visibility_res_to_fix, src_lang, translate_to_language, punctuation,num_speakers, prev_visibility_res_to_fix = None, prev_translated_subs_file = None):
    with timer("translate"):
        openai_client = config.get_openai_client()
        result = rewrite_timing_mismatched_subtitles(
            client=openai_client,
            non_translated_subs_file = original_subtitles,
            translated_subs_file=translated_subtitles,
            visibility_res_to_fix=visibility_res_to_fix,
            source_language=src_lang,
            target_language=translate_to_language,
            model=config.openai_timing_model,
            result_file=fixed_translated_subtitles,
            punctuation=punctuation,
            context_radius=2 if num_speakers ==1 else 5,
            prev_visibility_res_to_fix=prev_visibility_res_to_fix,
            prev_translated_subs_file = prev_translated_subs_file,
            editable_radius= 1 if translate_to_language != "de" else 2)
    return result




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


@task(cache_policy=NO_CACHE)
def t_generate_indextts2_segments(config, translated_file, speakers, emotions_tags, changed_list=None, max_pods=2, duration_factors=None):
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
        )
    return result



@task
def t_combine_tts_segments(speakers, tts_segments_folder):
    combine_audio_files(speakers, tts_segments_folder, tts_segments_folder)
    return tts_segments_folder


@task(cache_policy=NO_CACHE)
def t_openvoice_convert(config, speakers,voice_profiles,changed_list:list| None =None,run_id:str='', ttsmodel:int = 0):
    with timer("openvoice convert"):
        boto_session = config.get_boto_session()

        openvoice_convert_runpod(speakers=speakers,
                                 voice_profile=voice_profiles,
                                 temp_output_folder=config.temp_output_folder,
                                 reference_speakers_folder=config.speakers_folder,
                                 base_speaker_folder=config.tts_segments_folder,
                                 input_files_folder=config.tts_segments_folder,
                                 output_files_folder=config.tts_segments_folder,
                                 boto_session=boto_session,
                                 bucket_name=config.s3_bucket_name,
                                 runpod_key=config.runpod_key,
                                 runpod_template_id=config.runpod_openvoice_id,
                                 changed_list=changed_list,
                                 run_id=run_id,
                                 ttsmodel=ttsmodel)
        shutil.rmtree(config.temp_output_folder, ignore_errors=True)
    return True

@task(cache_policy=NO_CACHE)
def t_tts_build_final(config, speakers, convert_flag, subtitle_visibility_analysis, testing = False, build_cache=None, changed_list=None):
    with timer("Build Final TTS"):
        result = tts_build_final(speakers, config.subtitles_translated_file, config.tts_segments_folder, config.tts_segments_folder, subtitle_visibility_analysis, testing, changed_list=changed_list, build_cache=build_cache)
    return result

@task
def t_get_voice_profiles(audio_file, subtitles_file):
    with timer("Get Voice Profiles"):
        segments = extract_voice_profiles( audio_path=audio_file, srt_path=subtitles_file)
    return segments


@task(cache_policy=NO_CACHE)
def t_build_audio(config, tts_build_final_flag, is_dubbed=False, mix_gains: list[float] | None = None, use_non_speech: bool = True):
    # mix_gains = [background_db, dialog_db, non_speech_db, original_underlay_db]
    gain_kwargs = {}
    if mix_gains and len(mix_gains) == 4:
        gain_kwargs = {
            "background_gain_db": mix_gains[0],
            "dialog_gain_db": mix_gains[1],
            "non_speech_gain_db": mix_gains[2],
            "original_underlay_gain_db": mix_gains[3],
        }
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
        generate_videos(video_file, audio_result_file, config.subtitles_translated_file, output_file, preview=preview,
                        pre_encoded_video=config.preview_base_video_file if preview else None)
    return output_file

@task(cache_policy=NO_CACHE)
def t_detect_language(config, speaker_segments, audio = None):
    with timer("Detect Language"):
        audio = audio if audio else config.vocal_file
        lang = detect_language_for_routing(audio, speaker_segments)
        return lang

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
    TEST_FIX_TIMING = 6
    TEST_FIX_TIMING_2 = 7
    CONVERT = 8
    COMBINE = 9

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
        fix_timing:bool=True,
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
        "fix_timing": fix_timing,
        "test_mode": test_mode,
        "is_dubbed": is_dubbed,
        "test_duration_sec": test_duration_sec,
        "use_non_speech": use_non_speech,
        "output_file": None
    }
    # data["start_time"] = datetime.now().isoformat()
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
        diar_fut = t_diarize.submit(config, audio_file, num_speakers, run_id)
        speakers_segments = diar_fut.result()
        Path(config.speakers_segments_file).write_text(json.dumps(speakers_segments, indent=4))
        initial_language_fut = t_detect_language.submit(config, speakers_segments)
        initial_language = initial_language_fut.result()
        path = Path(config.general_config_file)
        data = json.loads(path.read_text()) if path.exists() else {}
        data["initial_src_language"] = initial_language
        path.write_text(json.dumps(data, indent=4))
    else:
        speakers_segments = json.loads(Path(config.speakers_segments_file).read_text())
        initial_language = json.loads(Path(config.general_config_file).read_text())["initial_src_language"]

    if stage <= STAGES.TRANSCRIBE:
        if trans_type == 'ocr':
            transcribe_fut = t_process_video_with_subs.submit(config, video_file, speakers_segments, initial_language, num_speakers, run_id) # no speakers support yet
        else:
            if initial_language == 'zh':
                transcribe_fut = t_alibabacloud_transcribe.submit(config, vocal_asr_file, speakers_segments, initial_language, num_speakers, run_id)
            elif initial_language in ('ja'):
                transcribe_fut = t_deepgram_transcribe.submit(config, vocal_asr_file, speakers_segments, initial_language, num_speakers)
            else:
                if trans_type == 'speechmatics':
                    transcribe_fut = t_speechmatics_transcribe.submit(config, audio_file, speakers_segments,
                                                                  initial_language, num_speakers)
                else:
                    transcribe_fut = t_assemblyai_transcribe.submit(config, audio_file, speakers_segments,
                                                                    initial_language,
                                                                    num_speakers)

        src_language = transcribe_fut.result()
        path = Path(config.general_config_file)
        data = json.loads(path.read_text()) if path.exists() else {}
        data["src_language"] = src_language
        path.write_text(json.dumps(data, indent=4))
    else:
        src_language = json.loads(Path(config.general_config_file).read_text())["src_language"]
    # --- EMOTION stage (must complete before TRANSLATE for duration estimation) ---
    gemini_emotions_fut = None
    detect_gender_fut = None
    mouth_windows_fut = None

    if stage <= STAGES.EMOTION:
        gemini_emotions_fut = t_gemini_extract_emotions.submit(config, vocal_asr_file, config.subtitles)
        detect_gender_fut = t_detect_gender.submit(audio_file=vocal_asr_file,
                                                   subtitle_file=config.subtitles)
        mouth_windows_fut = t_detect_mouth_windows.submit(video_file, config.subtitles)

    if gemini_emotions_fut and detect_gender_fut:
        emotions_tags = gemini_emotions_fut.result()
        speakers_array = detect_gender_fut.result()
        Path(config.emotions_tags_file).write_text(json.dumps(emotions_tags, indent=4))
        Path(config.speakers_file).write_text(json.dumps(speakers_array, indent=4))
    else:
        speakers_array = json.loads(Path(config.speakers_file).read_text())
        emotions_tags = json.loads(Path(config.emotions_tags_file).read_text())

    if mouth_windows_fut:
        subtitle_visibility_analysis = mouth_windows_fut.result()
        Path(config.subtitles_visibility_file).write_text(json.dumps(subtitle_visibility_analysis))
    else:
        subtitle_visibility_analysis = json.loads(Path(config.subtitles_visibility_file).read_text())

    # --- TRANSLATE stage (uses emotion vectors for duration scoring) ---
    translated_fut = None
    split_vocal_fut = None

    if stage <= STAGES.TRANSLATE:
        translated_fut = t_translate.submit(config, subtitles=config.subtitles,
                                            src_lang=src_language,
                                            translate_to_language=dst_language,
                                            subtitles_translated=config.subtitles_translated_file,
                                            punctuation=punctuation,
                                            emotions_data=emotions_tags,
                                            speakers_data=speakers_array)
        split_vocal_fut = t_split_vocal.submit(vocal_file, config.subtitles, config.non_speech_layer_file)

    if translated_fut:
        translate_stats = translated_fut.result()
        copyfile(config.subtitles_translated_file, config.subtitles_fixed_translated_file)
    if split_vocal_fut:
        split_vocal_fut.result()
    translated_file = config.subtitles_translated_file

    if stage <= STAGES.GENERATE:
        # if len(speakers_array) <=2 and dst_language != "ru":
        #     print ("using fishaudio model because <  2 speakers")
        #     ttsmodel = TTS_MODEL.FISHAUDIO.value
        cut_speakers_fut = t_cut_speakers.submit(vocal_file=vocal_asr_file, subtitles_file = config.subtitles, speakers_array=speakers_array, emotions_tags=emotions_tags, speakers_folder = config.speakers_folder)
        speakers_array = cut_speakers_fut.result()
        Path(config.speakers_file).write_text(json.dumps(speakers_array, indent=4))

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
                force_delete=False if fix_timing else True,
                force_no_batch=True if fix_timing or changed_list else False,
                )
        elif ttsmodel == TTS_MODEL.INDEXTTS2.value:
            t_generate_segments_fut = t_generate_indextts2_segments.submit(
                config=config,
                translated_file=translated_file,
                speakers=speakers_array,
                emotions_tags=emotions_tags,
                changed_list=changed_list,
                duration_factors=None,
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
    else:
        speakers_array=json.loads(Path(config.speakers_file).read_text())
        # voices = json.loads(Path(config.general_config_file).read_text())["voices"]

    if stage <= STAGES.TEST_FIX_TIMING and fix_timing:
        tts_test_build_fut = t_tts_build_final.submit(config, speakers= speakers_array, convert_flag = True, subtitle_visibility_analysis = subtitle_visibility_analysis, testing = True)
        tts_test_build = tts_test_build_fut.result()
        build_cache = tts_test_build["build_cache"]
        visibility_res_to_fix = tts_test_build['visibility_res']
        changed_idx_fut = t_rewrite_timing_mismatched_subtitles.submit(
            config,
            original_subtitles = config.subtitles,
            translated_subtitles  =config.subtitles_translated_file,
            fixed_translated_subtitles = config.subtitles_fixed_translated_file,
            visibility_res_to_fix = visibility_res_to_fix,
            src_lang = src_language,
            translate_to_language = dst_language,
            punctuation = punctuation,
            num_speakers=num_speakers)
        changed_idx = changed_idx_fut.result()
        if changed_idx:
            if ttsmodel == TTS_MODEL.ELEVENLABS.value:
                t_generate_elevenlab_segments.submit(
                    config,
                    speakers=speakers_array,
                    translated_file=config.subtitles_fixed_translated_file,
                    voices=voices,
                    language_code=dst_language,
                    changed_list=changed_idx
                ).result()
            elif ttsmodel == TTS_MODEL.INWORLD.value:
                t_generate_inworld_segments.submit(
                    config,
                    speakers=speakers_array,
                    translated_file=config.subtitles_fixed_translated_file,
                    voices=voices,
                    language_code=dst_language,
                    changed_list=changed_idx
                ).result()
            elif ttsmodel == TTS_MODEL.CARTESIA.value:
                t_generate_cartesia_segments.submit(
                    config,
                    speakers=speakers_array,
                    translated_file=config.subtitles_fixed_translated_file,
                    voices=voices,
                    language_code=dst_language,
                    changed_list=changed_idx
                ).result()
            elif ttsmodel == TTS_MODEL.FISHAUDIO.value:
                t_generate_fishaudio_segments.submit(
                    config=config,
                    subtitles_file=config.subtitles,
                    translated_file=config.subtitles_fixed_translated_file,
                    voice_audio=vocal_file,
                    speakers=speakers_array,
                    voices=voices,
                    language_code=dst_language,
                    changed_list=changed_idx,
                    run_id = run_id,
                    force_delete=False,
                    force_no_batch=True).result()
            elif ttsmodel == TTS_MODEL.INDEXTTS2.value:
                t_generate_indextts2_segments.submit(
                    config=config,
                    translated_file=config.subtitles_fixed_translated_file,
                    speakers=speakers_array,
                    emotions_tags=emotions_tags,
                    changed_list=changed_idx,
                ).result()
            else:
                raise ValueError("Unknown ttsmodel {}".format(ttsmodel))
    else:
        visibility_res_to_fix = None
        build_cache = None
        changed_idx = None
    if stage <= STAGES.TEST_FIX_TIMING_2 and fix_timing:
        tts_test_build_fut = t_tts_build_final.submit(config, speakers=speakers_array, convert_flag=True, subtitle_visibility_analysis=subtitle_visibility_analysis, testing=True, build_cache=build_cache, changed_list=changed_idx)
        tts_test_build = tts_test_build_fut.result()
        current_visibility_res_to_fix = tts_test_build['visibility_res']
        changed_idx_fut = t_rewrite_timing_mismatched_subtitles.submit(
            config,
            original_subtitles = config.subtitles,
            translated_subtitles  =config.subtitles_fixed_translated_file,
            fixed_translated_subtitles = config.subtitles_fixed_translated_file,
            visibility_res_to_fix = current_visibility_res_to_fix,
            src_lang = src_language,
            translate_to_language = dst_language,
            punctuation = punctuation,
            num_speakers=num_speakers,
            prev_visibility_res_to_fix = visibility_res_to_fix,
            prev_translated_subs_file = config.subtitles_translated_file)
        changed_idx = changed_idx_fut.result()
        if changed_idx:
            if ttsmodel == TTS_MODEL.ELEVENLABS.value:
                t_generate_elevenlab_segments.submit(
                    config,
                    speakers=speakers_array,
                    translated_file=config.subtitles_fixed_translated_file,
                    voices=voices,
                    language_code=dst_language,
                    changed_list=changed_idx
                ).result()
            elif ttsmodel == TTS_MODEL.INWORLD.value:
                t_generate_inworld_segments.submit(
                    config,
                    speakers=speakers_array,
                    translated_file=config.subtitles_fixed_translated_file,
                    voices=voices,
                    language_code=dst_language,
                    changed_list=changed_idx
                ).result()
            elif ttsmodel == TTS_MODEL.CARTESIA.value:
                t_generate_cartesia_segments.submit(
                    config,
                    speakers=speakers_array,
                    translated_file=config.subtitles_fixed_translated_file,
                    voices=voices,
                    language_code=dst_language,
                    changed_list=changed_idx
                ).result()
            elif ttsmodel == TTS_MODEL.FISHAUDIO.value:
                t_generate_fishaudio_segments.submit(
                    config=config,
                    subtitles_file=config.subtitles,
                    translated_file=config.subtitles_fixed_translated_file,
                    voice_audio=vocal_file,
                    speakers=speakers_array,
                    voices=voices,
                    language_code=dst_language,
                    changed_list=changed_idx,
                    run_id = run_id,
                    force_delete=True,
                    force_no_batch=False).result()
            elif ttsmodel == TTS_MODEL.INDEXTTS2.value:
                t_generate_indextts2_segments.submit(
                    config=config,
                    translated_file=config.subtitles_fixed_translated_file,
                    speakers=speakers_array,
                    emotions_tags=emotions_tags,
                    changed_list=changed_idx,
                ).result()
            else:
                raise ValueError("Unknown ttsmodel {}".format(ttsmodel))
    else:
        pass

    # ---------------- Two-pass TTS regen (IndexTTS2 only) ----------------
    if ttsmodel == TTS_MODEL.INDEXTTS2.value and stage < STAGES.COMBINE:
        regen_build = t_tts_build_final.submit(
            config, speakers=speakers_array, convert_flag=True,
            subtitle_visibility_analysis=subtitle_visibility_analysis,
            testing=True, build_cache=build_cache, changed_list=changed_idx,
        ).result()
        regen_candidates = regen_build.get("regen_candidates", [])
        build_cache = regen_build["build_cache"]

        if regen_candidates:
            regen_indices = [r["idx"] for r in regen_candidates]
            regen_factors = {r["idx"]: r["measured_factor"] for r in regen_candidates}
            regen_log_path = os.path.join(config.data_output_folder, "regen_factors.json")
            with open(regen_log_path, "w", encoding="utf-8") as f:
                json.dump(regen_candidates, f, indent=2, ensure_ascii=False)
            t_generate_indextts2_segments.submit(
                config=config,
                translated_file=config.subtitles_fixed_translated_file,
                speakers=speakers_array,
                emotions_tags=emotions_tags,
                changed_list=regen_indices,
                duration_factors=regen_factors,
            ).result()

    if stage <= STAGES.COMBINE:
        loudness_adjust_fut = t_loudness_adjust.submit(subtitles_file=config.subtitles_fixed_translated_file, vocal_file=config.vocal_file, tts_segments_folder = config.tts_segments_folder)
        loudness_adjust_fut.result()
        # subtitle_visibility_analysis=[]
        openvoice_convert_flag = True
        tts_build_final_fut = t_tts_build_final.submit(config, speakers= speakers_array, convert_flag = openvoice_convert_flag, subtitle_visibility_analysis = subtitle_visibility_analysis, testing = False)
        build_audio_fut = t_build_audio.submit(config, tts_build_final_flag=tts_build_final_fut.result(),
                                            is_dubbed=is_dubbed, mix_gains=mix_gains, use_non_speech=use_non_speech)
        copyfile(config.subtitles_fixed_translated_file,config.subtitles_translated_file)
        audio_result_file = build_audio_fut.result()
        if preview_proc is not None:
            preview_proc.wait()
            if preview_proc.returncode != 0:
                Path(config.preview_base_video_file).unlink(missing_ok=True)
        generate_videos_fut = t_generate_videos.submit(config, video_file, audio_result_file, preview=True)
        # test_results = t_test_results.submit(audio_result_file)
        output_file = generate_videos_fut.result()
        path = Path(config.general_config_file)
        data = json.loads(path.read_text()) if path.exists() else {}
        data["output_file"] = output_file
        path.write_text(json.dumps(data, indent=4))
        with timer("Move output to storagebox"):
            if local_dir.exists():
                shutil.move(str(local_dir), str(Path("/mnt/storagebox") / "output" / run_id))
        # test_results.result()


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
    dubbing_flow("input/vor_rus.mp4",
                 dst_language="en",
                 trans_type='default',
                 emotions_flag=True,
                 ttsmodel=TTS_MODEL.INDEXTTS2.value,
                 elevenlabs_emotions=ELEVENLABS_EMOTIONS.HIGH.value,
                 # num_speakers=1,
                 fix_timing=False,
                 test_mode=False,
                 changed_list=[],
                 run_id='20260507_vor_rus',
                 test_duration_sec=120,
                 is_dubbed=False,
                 stage = STAGES.EMOTION.value)