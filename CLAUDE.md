# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

**Dubbed** is a video dubbing and localization platform. It automatically extracts audio from a video, transcribes it, translates it, generates dubbed audio using multiple TTS engines, optionally applies voice conversion, and renders a final video with the new audio track.

## Running the Services

Three processes must run together. Start them in order:

```bash
# 1. Prefect orchestration server
export PREFECT_API_URL=http://127.0.0.1:4200/api
prefect server start

# 2. Register the Prefect flow
python api/serve.py

# 3. FastAPI backend
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Frontend:
```bash
cd new_ui
pnpm install
pnpm dev       # dev server
pnpm build     # production build
pnpm lint      # ESLint
```

## Pipeline Architecture

The core logic lives in `main_prefect_dag.py` as a Prefect flow (`dubbing_flow`). The API in `api/app.py` invokes this flow via Prefect's API. Each pipeline run is identified by a `run_id` and all files are stored under `output/{run_id}/`.

The pipeline has 10 stages (configurable start stage):

| Stage | Name | What it does |
|-------|------|-------------|
| 0 | SPLIT | Extract audio from video, separate speech/music via RunPod |
| 1 | DIARIZE | Speaker diarization via Pyannote API |
| 2 | TRANSCRIBE | STT via Groq/Whisper, AssemblyAI, or Alibaba Cloud |
| 3 | TRANSLATE | Translation via OpenAI GPT |
| 4 | EMOTION | Gender/emotion detection, voice profile extraction |
| 5 | GENERATE | TTS generation (ElevenLabs, Inworld, Cartesia, or Fish Audio) |
| 6–7 | TEST_FIX_TIMING | Timing validation and correction using mouth detection |
| 8 | CONVERT | Voice conversion via OpenVoice on RunPod |
| 9 | COMBINE | Audio mixing, loudness normalization, video rendering |

To run the flow directly for testing:
```bash
python main_prefect_dag.py  # uses the __main__ block
```

## Configuration (`config.py`)

Every run gets a `Config(run_id)` object that:
- Manages all input/output paths under `output/{run_id}/`
- Lazy-loads API clients (OpenAI, ElevenLabs, Fish Audio, boto3/S3)
- Reads credentials from environment variables (see `.env`)

Key paths on a `Config` instance: `.audio_path`, `.vocal_path`, `.music_path`, `.subtitles_path`, `.translated_subs_path`, `.tts_dir`, `.speakers_dir`, `.data_dir`.

## Key Module Map

| Module | Role |
|--------|------|
| `audio.py` | Audio extraction, VAD (Silero), silence detection |
| `diarization.py` | Pyannote speaker diarization |
| `whisper_transcribe.py` | Groq Whisper STT |
| `assemblyai_transcribe.py` | AssemblyAI STT |
| `subtitles.py` | SRT parsing + OpenAI translation |
| `tts_v2.py` | ElevenLabs TTS |
| `tts_fish_audio.py` | Fish Audio TTS with voice cloning |
| `tts_inworld.py` | Inworld TTS |
| `openvoice_module.py` | Voice conversion via RunPod |
| `final_audio.py` | Final audio mixing and composition |
| `video.py` | Video rendering |
| `loudness_adjust.py` | Per-line loudness normalization |
| `fix_timing_subs.py` | Subtitle timing correction |
| `detect_mouth_windows.py` | Mouth detection for lip-sync |
| `runpod_utils.py` | RunPod serverless GPU calls |

## API Layer (`api/`)

- `api/app.py` — FastAPI app. Key endpoints:
  - `POST /runs/upload` — upload video, returns `run_id`
  - `POST /runs/start` — kick off pipeline from a given stage
  - `GET /runs/{run_id}/status` — SSE stream of real-time progress
  - `POST /runs/{run_id}/regenerate` — re-run from edited subtitles
- `api/db.py` — PostgreSQL connection and queries (users, projects, sessions)
- `api/auth.py` — Session auth + Google OAuth
- `api/serve.py` — Registers `dubbing_flow` with the Prefect server

## Per-Run File Layout

```
output/{run_id}/
├── input/video.mp4
├── audio/
│   ├── audio.wav, vocal.wav, music.wav
│   ├── tts_segments/SPEAKER_00_0.wav ...
│   └── speakers/          # original speaker voice samples
├── data/
│   ├── general_config.json
│   ├── speakers_segments_data.json
│   ├── speakers_data.json
│   ├── subtitles.srt / subtitles_translated.srt
│   └── voice_profiles.pkl
└── temp/
```

## Permanent Voice Profiles

`config/permanent_voices.json` stores pre-configured voice mappings (ElevenLabs voice IDs keyed by speaker label). These persist across runs and are merged with per-run speaker data at the GENERATE stage.

## External Services Required

The platform depends on several paid/external APIs whose keys must be in `.env`:

- **RunPod** — audio separation (SPLIT), OCR, voice conversion (CONVERT), emotion detection
- **Pyannote** — speaker diarization
- **OpenAI** — translation and LLM tasks
- **ElevenLabs / Fish Audio / Inworld / Cartesia** — TTS engines
- **Groq** — Whisper STT
- **AssemblyAI** — alternative STT
- **AWS S3** (`fishmanresearch` bucket) — file storage
- **PostgreSQL** (`replidub` database) — user/project state
