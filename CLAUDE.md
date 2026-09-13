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
| 4 | EMOTION | Gender detection (Gemini, per-speaker), emotion detection, voice profile extraction |
| 5 | GENERATE | TTS generation — engine selected by the `ttsmodel` flow arg (see below) |
| 6–7 | TEST_FIX_TIMING | Timing validation and correction using mouth detection |
| 8 | CONVERT | Voice conversion via OpenVoice on RunPod |
| 9 | COMBINE | Audio mixing, loudness normalization, video rendering |

To run the flow directly for testing:
```bash
python main_prefect_dag.py  # uses the __main__ block
```

## TTS Engines (`ttsmodel` flow argument)

`TTS_MODEL` in `main_prefect_dag.py` selects the GENERATE engine:

| Value | Name | Module | Runs on |
|-------|------|--------|---------|
| 0 | ELEVENLABS | `tts_v2.py` | API |
| 1 | INWORLD | `tts_inworld.py` | API |
| 2 | CARTESIA | `tts_v2.py` | API |
| 3 | FISHAUDIO | `tts_fish_audio.py` | API |
| 4 | INDEXTTS2 | `tts_index_tts2.py` | Modal GPU |
| 5 | QWEN3TTS | `tts_qwen3_tts.py` | Modal GPU |

The two Modal engines shard work across pods with LPT bin-packing sized by
`CHARS_PER_SEC_PER_GPU * TARGET_POD_SECONDS`, then dispatch via `.map()`. IndexTTS2 additionally pre-warms
containers during the EMOTION stage; Qwen3 does not.

**Timing caps are per-engine** (`main_prefect_dag.py`, search `is_qwen3`): `timing_speaker_base_cap`,
`timing_overflow_threshold`, `timing_max_speed_factor`. Qwen3 uses tighter values (1.2 / 1.25 / 1.25) than
IndexTTS2 (1.35 / 1.35 / 1.45). Atempo clamping does not truncate audio — an over-long line spills into the
next and is flagged `warn_too_long`.

## Modal Apps (`docker/`)

Each subdirectory is a self-contained Modal app deployed with `modal deploy docker/<name>/app.py`:

| App | Purpose |
|-----|---------|
| `index-tts2-generator` | IndexTTS2 inference |
| `qwen3-tts-generator` | Qwen3-TTS inference (+ Seed-VC voice conversion); loads the Base checkpoint only, LoRA adapters on top |
| `qwen3-tts-lora-train` | Qwen3-TTS LoRA fine-tuning; writes to the `dubbing-transfer` volume |
| `emotion_estract_modal` | Emotion tag extraction |
| `forced-alignment-modal` | Forced alignment (MFA) |
| `split_audio_runpod`, `emotion_runpod`, `padleocr_gpu_runpod`, `myshell-openvoice-docker` | RunPod-era workers |

The `dubbing-transfer` Modal volume carries LoRA checkpoints and training data between the trainer and the
generator. LoRA checkpoints live at `qwen3_lora_output/{speaker}/checkpoint-epoch-{n}/`.

Vendored upstream sources (`Qwen3-TTS/`, `seed-vc/`, `index-tts/`) are mounted into the images via
`add_local_dir`. Local edits to them ship on the next deploy — read them rather than guessing at their APIs.

### Qwen3 voice selection

`config/qwen3_lora_map.json` maps `"{LANG}_{gender}"` to a LoRA adapter name on the Base checkpoint.
Unmapped pairs fall back to the top-level `fallback` adapter; an unsupported *language* still raises.
Every voice is a LoRA — CustomVoice's builtin timbres were dropped so only one 1.7B checkpoint sits in VRAM
alongside Seed-VC. Speaker identity comes from Seed-VC downstream, so the adapter only supplies language and
prosody quality.

`generate_lora_vc_multi` groups a pod's lines by adapter name, so a mixed pod costs one adapter swap per
distinct LoRA. Adding a language is a config edit plus a trained adapter, not a code change.

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
| `gender.py` | Per-speaker gender detection via Gemini (audio clip + transcript) |
| `whisper_transcribe.py` | Groq Whisper STT |
| `assemblyai_transcribe.py` | AssemblyAI STT |
| `subtitles.py` | SRT parsing + OpenAI translation |
| `tts_v2.py` | ElevenLabs TTS + the shared final audio build / atempo logic |
| `tts_fish_audio.py` | Fish Audio TTS with voice cloning |
| `tts_inworld.py` | Inworld TTS |
| `tts_index_tts2.py` | IndexTTS2 dispatch to Modal |
| `tts_qwen3_tts.py` | Qwen3-TTS dispatch to Modal |
| `natural_tts_timing.py` | Per-speaker base atempo (p80 percentile) |
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
