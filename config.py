from pathlib import Path
import os
from dotenv import load_dotenv
import boto3
import httpx
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from prefect import get_run_logger
from fishaudio import FishAudio
# Load environment variables from .env file
load_dotenv()

class Configuration:
    def __init__(self, run_id: str):
        self.run_id = str(run_id)
        self.global_output_folder = f"output/{run_id}"
        self.temp_output_folder = f"{self.global_output_folder}/temp"
        self.temp_input_folder = f"{self.global_output_folder}/input"
        self.audio_output_folder = f"{self.global_output_folder}/audio"
        self.data_output_folder = f"{self.global_output_folder}/data"
        self.tts_segments_folder = f"{self.audio_output_folder}/tts_segments"
        self.audio_parts = f"{self.audio_output_folder}/audio_parts"
        self.speakers_folder = f"{self.audio_output_folder}/speakers"


        self.test_video_file = f"{self.global_output_folder}/test_video.mp4"
        self.audio_file = f"{self.audio_output_folder}/audio.wav"
        self.audio_result_file = f"{self.audio_output_folder}/final_audio.wav"

        self.speakers_segments_file = f"{self.data_output_folder}/speakers_segments_data.json"
        self.permanent_voices_file = "config/permanent_voices.json"
        self.speakers_file = f"{self.data_output_folder}/speakers_data.json"
        self.general_config_file = f"{self.data_output_folder}/general_config.json"
        self.subtitles_visibility_file = f"{self.data_output_folder}/subtitles_visibility.json"

        self.voice_profiles_file = f"{self.data_output_folder}/voice_profiles.pkl"
        self.vocal_file = f"{self.audio_output_folder}/vocal.wav"
        self.vocal_asr_file = f"{self.audio_output_folder}/vocal_asr.wav"
        self.music_file = f"{self.audio_output_folder}/music.wav"

        self.non_speech_layer_file = f"{self.audio_output_folder}/non_speech_layer.wav"

        self.stem_background = f"{self.audio_output_folder}/stem_background.mp3"
        self.stem_dialog = f"{self.audio_output_folder}/stem_dialog.mp3"
        self.stem_original = f"{self.audio_output_folder}/stem_original.mp3"

        self.subtitles_translated_file = f"{self.data_output_folder}/subtitles_translated.srt"
        self.subtitles_fixed_translated_file = f"{self.data_output_folder}/subtitles_fixed_translated.srt"
        self.subtitles = f"{self.data_output_folder}/subtitles.srt"

        self.final_video_file = f"{self.global_output_folder}/final_video.mp4"
        self.final_video_preview_file = f"{self.global_output_folder}/final_video_preview.mp4"
        self.preview_base_video_file = f"{self.global_output_folder}/preview_base.mp4"

        # Load sensitive data from environment variables
        self.boto3_profile = os.getenv("BOTO3_PROFILE", "research")
        self.s3_bucket_name = os.getenv("S3_BUCKET_NAME", "fishmanresearch")

        self.runpod_key = os.getenv("RUNPOD_KEY")
        self.runpod_split_id = os.getenv("RUNPOD_SPLIT_ID")
        self.runpod_paddleocr_id = os.getenv("RUNPOD_PADDLEOCR_ID")
        self.runpod_openvoice_id = os.getenv("RUNPOD_OPENVOICE_ID")
        self.runpod_emotion_detect_id = os.getenv("RUNPOD_EMOTION_DETECT_ID")
        self.pyannote_key = os.getenv("PYANNOTE_KEY")

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_diarization_model = "gpt-5.4-mini"  # diarization.py / whisper / assemblyai / alibaba / OCR speaker repair
        self.openai_emotion_model = "gpt-5.4-nano"      # emotion label fixes
        self.openai_timing_model = "gpt-5.4"            # timing rewrites
        self.openai_translate_model = "gpt-5.4"         # translation

        self.tts_model = 'tts-1-hd'

        self.elevenlabs_key = os.getenv("ELEVENLABS_KEY")
        self.fishaudio_key = os.getenv("FISHAUDIO_API_KEY")

        self.inworld_key = os.getenv("INWORLD_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")


        self.assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.alibaba_api_key = os.getenv("ALIBABA_API_KEY")
        self.cartesia_api_key = os.getenv("CARTESIA_API_KEY")
        self._boto_session = None
        self._openai_client = None
        self._elevenlabs_client = None
        self._fishaudio_client = None

    # 🔥 create all directories
    def create_dirs(self):
        dirs = [
            self.global_output_folder,
            self.audio_output_folder,
            self.data_output_folder,
            self.tts_segments_folder,
            self.audio_parts,
            self.speakers_folder,
            self.temp_output_folder,
            self.temp_input_folder
        ]

        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def get_boto_session(self):
        if self._boto_session is None:
            log = get_run_logger()
            log.info("🔐 Initializing boto3 session")

            self._boto_session = boto3.Session(
                profile_name=self.boto3_profile,
                region_name="us-east-1"
            )

        return self._boto_session

    def get_elevenlabs_client(self):
        if self._elevenlabs_client is None:
            log = get_run_logger()
            log.info("🔐 Initializing elevenlabs client")

            self._elevenlabs_client = ElevenLabs(
                api_key=self.elevenlabs_key
            )

        return self._elevenlabs_client

    def get_fishaudio_client(self):
        if self._fishaudio_client is None:
            log = get_run_logger()
            log.info("🔐 Initializing fishaudio client")

            self._fishaudio_client =     client = FishAudio(api_key=self.fishaudio_key)


        return self._fishaudio_client

    def get_openai_client(self):
        if self._openai_client is None:
            log = get_run_logger()
            log.info("🔐 Initializing openai client")

            self._openai_client = OpenAI(
                timeout=600,
                http_client=httpx.Client(
                    timeout=httpx.Timeout(600.0, connect=30.0)
                ),
                api_key=self.openai_api_key
            )

        return self._openai_client