import os
import shutil
import subprocess
import srt
import stripe
import yt_dlp
from pathlib import Path
os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, UTC, timezone
from unidecode import unidecode
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

from enum import IntEnum
from typing import Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends, Cookie
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, PlainTextResponse, RedirectResponse

from pydantic import BaseModel, Field
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterId
from prefect.deployments import run_deployment
from main_prefect_dag import generate_run_id
from video import generate_videos, EXPORT_RESOLUTIONS
from pathlib import Path
import json
import asyncio
from fastapi import HTTPException
from api.db import fetch_one, fetch_all, execute, execute_returning
from api.pricing import PRICING, calculate_run_cost_cents, calculate_regen_cost_cents
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi.middleware.cors import CORSMiddleware


from fastapi import Response, Cookie
from api.auth import (
    hash_password,
    verify_password,
    create_session,
    clear_session_cookie,
    verify_google_credential,
    get_current_user,
    get_current_user_id_from_cookie,
    create_password_reset_token,
    create_email_verification_token,
    consume_password_reset_token,
    consume_email_verification_token,
    send_email,
    check_disposable_email,
    check_ip_registration_limit,
    record_registration_attempt,
    verify_turnstile_token,
)

_scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _scheduler.add_job(
        reconcile_pending_jobs,
        "interval",
        minutes=15,
        id="reconcile_jobs",
        replace_existing=True,
    )
    _scheduler.start()
    yield
    _scheduler.shutdown(wait=False)


app = FastAPI(title="Dubbing API", lifespan=lifespan)
PUBLIC_APP_URL = os.getenv("PUBLIC_APP_URL", "http://localhost")
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", "http://localhost")
ENV = os.getenv("ENV", "development")

if ENV == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
elif ENV == "production":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://verbox.ai",
            "https://www.verbox.ai"
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
DEPLOYMENT_NAME = "video-dubbing-pipeline/video-dubbing-deployment"


class STAGES(IntEnum):
    SPLIT = 0
    DIARIZE = 1
    TRANSCRIBE = 2
    TRANSLATE = 3
    EMOTION = 4
    GENERATE = 5
    TEST_FIX_TIMING = 6
    TEST_FIX_TIMING_2 = 7
    CONVERT = 8
    COMBINE = 9


class ELEVENLABS_EMOTIONS(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2

class YoutubeImportRequest(BaseModel):
    url: str
    project_id: int


FINAL_STAGE_MAP = {
    "PREPARE": {
        "t_cut_video",
        "t_extract_audio",
        "t_split_audio",
        "t_split_vocal",
        "t_diarize",
        "t_detect_language",
    },
    "TRANSCRIBE": {
        "t_process_video_with_subs",
        "t_alibabacloud_transcribe",
        "t_assemblyai_transcribe",
        "t_whisper_transcribe",
    },
    "TRANSLATE": {
        "t_translate",
        "t_detect_mouth_windows",
        "t_get_voice_profiles",
        "t_emotion_detect",
        "t_detect_gender",
    },
    "GENERATE": {
        "t_cut_speakers",
        "t_generate_elevenlab_segments",
        "t_generate_cartesia_segments",
        "t_generate_inworld_segments",
        "t_generate_fishaudio_segments",
        "t_combine_tts_segments",
    },
    "SYNC": {
        "t_rewrite_timing_mismatched_subtitles",
        "t_openvoice_convert",
    },
    "RENDER": {
        "t_loudness_adjust",
        "t_build_audio",
        "t_generate_videos",
        "t_test_results",
    },
}

# Weights for percent calculation (must sum to 100)
STAGE_WEIGHTS = {
    "PREPARE":    8,
    "TRANSCRIBE": 12,
    "TRANSLATE":  20,
    "GENERATE":   35,
    "SYNC":       15,
    "RENDER":     10,
}


class StartDubRequest(BaseModel):
    video_file: str | None = None
    dst_language: str
    trans_type: str = "default"
    elevenlabs: bool = False
    num_speakers: int | None = None
    emotions_flag: bool = True
    ttsmodel: int
    elevenlabs_emotions: int = ELEVENLABS_EMOTIONS.MEDIUM.value
    fix_timing: bool = True
    changed_list: list[int] | None = None
    is_dubbed: bool = False
    run_id: str = ""
    stage: int = STAGES.TRANSLATE.value
    use_non_speech: bool = True

STORAGEBOX_ROOT = Path("/mnt/storagebox")

def _run_dir(run_id: str) -> Path:
    sb = STORAGEBOX_ROOT / "output" / run_id
    print (sb, flush=True)
    if sb.exists():
        return sb
    return Path("output") / run_id

def read_run_config(run_id: str) -> dict:
    path = _run_dir(run_id) / "data/general_config.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Config not found for run_id={run_id}")
    return json.loads(path.read_text(encoding="utf-8"))

def update_config_flow_id(run_id: str, config : dict) -> None:
    path = _run_dir(run_id) / "data/general_config.json"
    path.write_text(json.dumps(config, indent=4))

def _state_name(obj) -> str | None:
    name = getattr(obj, "state_name", None)
    if name:
        return name
    state = getattr(obj, "state", None)
    return getattr(state, "name", None)


def compute_final_stage_status(stages: list[dict]) -> dict:
    # Normalize: keep strongest state per task (Completed > Running/Pending > Failed)
    STATE_RANK = {"Completed": 3, "Running": 2, "Pending": 1, "Failed": 0, "Crashed": 0}
    task_states: dict[str, str] = {}
    for s in stages:
        name = s["task_name"]
        state = s["state_name"]
        if name not in task_states or STATE_RANK.get(state, 0) > STATE_RANK.get(task_states[name], 0):
            task_states[name] = state

    def get_stage_status(task_names: set[str]) -> str:
        relevant = [task_states[t] for t in task_names if t in task_states]
        if not relevant:
            return "not_started"
        if all(st == "Completed" for st in relevant):
            return "done"
        if any(st in ("Failed", "Crashed") for st in relevant):
            return "failed"
        return "in_progress"

    result: dict[str, str] = {}

    result["PREPARE"]    = get_stage_status(FINAL_STAGE_MAP["PREPARE"])
    result["TRANSCRIBE"] = get_stage_status(FINAL_STAGE_MAP["TRANSCRIBE"])
    result["TRANSLATE"]  = get_stage_status(FINAL_STAGE_MAP["TRANSLATE"])
    result["GENERATE"]   = get_stage_status(FINAL_STAGE_MAP["GENERATE"])

    # SYNC depends on GENERATE being done
    if result["GENERATE"] != "done":
        result["SYNC"] = "not_started"
    else:
        result["SYNC"] = get_stage_status(FINAL_STAGE_MAP["SYNC"])

    # SYNC completion is gated on t_openvoice_convert (it also runs in RENDER via t_tts_build_final
    # which we intentionally exclude from SYNC to avoid false positives)
    convert_state = task_states.get("t_openvoice_convert", "")
    if result["SYNC"] != "not_started":
        if convert_state == "Completed":
            result["SYNC"] = "done"
        elif convert_state in ("Failed", "Crashed"):
            result["SYNC"] = "failed"
        else:
            result["SYNC"] = "in_progress"

    # RENDER depends on SYNC being done, unless no SYNC tasks ran at all (e.g. remix-only run)
    sync_tasks_ran = any(t in task_states for t in FINAL_STAGE_MAP["SYNC"])
    if sync_tasks_ran and result["SYNC"] != "done":
        result["RENDER"] = "not_started"
    else:
        result["RENDER"] = get_stage_status(FINAL_STAGE_MAP["RENDER"])

    # --- percent calculation ---
    percent = 0
    for stage_name, weight in STAGE_WEIGHTS.items():
        status = result[stage_name]
        if status == "done":
            percent += weight
        elif status == "in_progress":
            percent += weight // 2
    result["percent"] = percent

    return result




async def fetch_prefect_status(flow_run_id: str) -> dict:
    async with get_client() as client:
        flow_run = await client.read_flow_run(flow_run_id)
        flow_state = _state_name(flow_run)

        if flow_state in ("Completed", 'Failed'):
           return {
                "flow_run_id": flow_run_id,
                "flow_state_name": flow_state
           }

        task_runs = await client.read_task_runs(
            flow_run_filter=FlowRunFilter(
                id=FlowRunFilterId(any_=[flow_run_id])
            )
        )

    task_runs = sorted(
        task_runs,
        key=lambda t: (
            getattr(t, "start_time", None)
            or getattr(t, "expected_start_time", None)
            or getattr(t, "created", None)
        ),
    )

    stages = []
    for t in task_runs:
        stages.append(
            {
                "task_name": t.name.rsplit("-", 1)[0],
                "state_name": _state_name(t),
            }
        )
    stage_result = compute_final_stage_status(stages)
    percent = stage_result.pop("percent", 0)

    return {
        "flow_run_id": str(flow_run.id),
        "flow_state_name": _state_name(flow_run),
        "stages": stage_result,
        "percent": percent,
    }

def require_user_id(session_token: str | None = Cookie(default=None)):
    return get_current_user(session_token)["id"]

def check_project_access(run_id: str, user_id: int):
    project = fetch_one(
        """
        SELECT id
        FROM projects
        WHERE run_id = %s AND user_id = %s
        """,
        (run_id, user_id),
    )

    if not project:
        raise HTTPException(status_code=403, detail="Access denied")

    return project


def _get_video_duration_minutes(video_path: str) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ],
            capture_output=True,
            text=True,
            check=True
        )

        duration = float(result.stdout.strip())
        return float(duration) / 60
    except Exception:
        return None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/pricing")
async def get_pricing():
    return PRICING


# ---------------------------------------------------------------------------
# Billing helpers
# ---------------------------------------------------------------------------

def _charge_user(user_id: int, amount_cents: int, run_id: str, job_type: str) -> int:
    """
    Deduct amount_cents from user balance and insert a pending job record.
    Returns the payments_for_jobs row id.
    Raises HTTPException(402) on insufficient balance.
    Raises HTTPException(409) if a pending job for this run_id already exists.
    """
    if fetch_one("SELECT id FROM payments_for_jobs WHERE run_id = %s AND status = 'pending'", (run_id,)):
        raise HTTPException(status_code=409, detail="A pending job for this run already exists")

    updated = execute(
        "UPDATE users SET balance = balance - %s WHERE id = %s AND balance >= %s",
        (amount_cents, user_id, amount_cents),
    )
    if updated == 0:
        row = fetch_one("SELECT balance FROM users WHERE id = %s", (user_id,))
        raise HTTPException(
            status_code=402,
            detail={
                "error": "insufficient_balance",
                "required_cents": amount_cents,
                "balance_cents": row["balance"] if row else 0,
            },
        )

    record = execute_returning(
        """
        INSERT INTO payments_for_jobs (user_id, run_id, job_type, amount_cents, status)
        VALUES (%s, %s, %s, %s, 'pending')
        RETURNING id
        """,
        (user_id, run_id, job_type, amount_cents),
    )
    return record["id"]


def _refund_job(job_id: int, user_id: int, amount_cents: int) -> None:
    """Return amount_cents to the user balance and mark the job as refunded."""
    execute("UPDATE users SET balance = balance + %s WHERE id = %s", (amount_cents, user_id))
    execute(
        "UPDATE payments_for_jobs SET status = 'refunded', updated_at = NOW() WHERE id = %s",
        (job_id,),
    )


async def reconcile_pending_jobs() -> None:

    pending = fetch_all(
        "SELECT id, user_id, run_id, amount_cents, prefect_flow_run_id, created_at "
        "FROM payments_for_jobs WHERE status = 'pending'"
    )
    if not pending:
        return

    failure_threshold = timedelta(hours=2)
    now = datetime.now(UTC)

    for job in pending:
        job_id: int = job["id"]
        user_id: int = job["user_id"]
        amount_cents: int = job["amount_cents"]
        flow_run_id: str | None = job.get("prefect_flow_run_id")
        created_at = job.get("created_at")

        if created_at is not None and getattr(created_at, "tzinfo", None) is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        age = (now - created_at) if created_at else timedelta(hours=999)

        if not flow_run_id:
            if age > failure_threshold:
                _refund_job(job_id, user_id, amount_cents)
            continue

        try:
            status = await fetch_prefect_status(flow_run_id)
            flow_state = (status.get("flow_state_name") or "").upper()
            print (flow_state)
        except Exception:
            if age > failure_threshold:
                _refund_job(job_id, user_id, amount_cents)
            continue

        if flow_state == "COMPLETED":
            execute(
                "UPDATE payments_for_jobs SET status = 'succeeded', updated_at = NOW() WHERE id = %s",
                (job_id,),
            )
        elif flow_state in ("FAILED", "CRASHED", "CANCELLED") or age > failure_threshold:
            _refund_job(job_id, user_id, amount_cents)


def build_password_reset_email(reset_link: str, user_name: str | None = None) -> tuple[str, str]:
    greeting = f"Hi {user_name}," if user_name else "Hi,"

    text = f"""\
        {greeting}
        
        We received a request to reset your Verbox password.
        
        Reset your password using the link below:
        {reset_link}
        
        If you did not request this, you can safely ignore this email.
        
        For security reasons, this link should expire soon.
        
        — Verbox
        """

    html = f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Reset your password</title>
        </head>
        <body style="margin:0; padding:0; background-color:#f4f7fb; font-family:Arial,Helvetica,sans-serif; color:#1f2937;">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color:#f4f7fb; margin:0; padding:24px 0;">
            <tr>
              <td align="center">
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="max-width:600px; background:#ffffff; border-radius:14px; overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,0.06);">
        
                  <tr>
                    <td align="center" style="padding:32px 24px 16px 24px;">
                      <img src="https://verbox.ai/logo.png" alt="Verbox" width="140" style="display:block; border:0; outline:none; text-decoration:none;">
                    </td>
                  </tr>
        
                  <tr>
                    <td style="padding:8px 32px 0 32px; text-align:center;">
                      <h1 style="margin:0; font-size:26px; line-height:1.3; color:#111827;">Reset your password</h1>
                    </td>
                  </tr>
        
                  <tr>
                    <td style="padding:20px 32px 0 32px; font-size:16px; line-height:1.7; color:#374151;">
                      <p style="margin:0 0 16px 0;">{greeting}</p>
                      <p style="margin:0 0 16px 0;">
                        We received a request to reset your password for your <strong>Verbox</strong> account.
                      </p>
                      <p style="margin:0 0 24px 0;">
                        Click the button below to choose a new password.
                      </p>
                    </td>
                  </tr>
        
                  <tr>
                    <td align="center" style="padding:0 32px 8px 32px;">
                      <a href="{reset_link}"
                         style="display:inline-block; background-color:#2563eb; color:#ffffff; text-decoration:none; font-size:16px; font-weight:bold; padding:14px 28px; border-radius:10px;">
                        Reset Password
                      </a>
                    </td>
                  </tr>
        
                  <tr>
                    <td style="padding:24px 32px 0 32px; font-size:14px; line-height:1.7; color:#6b7280;">
                      <p style="margin:0 0 12px 0;">
                        If the button does not work, copy and paste this link into your browser:
                      </p>
                      <p style="margin:0 0 16px 0; word-break:break-all;">
                        <a href="{reset_link}" style="color:#2563eb; text-decoration:none;">{reset_link}</a>
                      </p>
                      <p style="margin:0 0 12px 0;">
                        If you did not request a password reset, you can safely ignore this email.
                      </p>
                      <p style="margin:0;">
                        For security reasons, this link should expire shortly.
                      </p>
                    </td>
                  </tr>
        
                  <tr>
                    <td style="padding:28px 32px 32px 32px; font-size:13px; line-height:1.6; color:#9ca3af; text-align:center;">
                      © Verbox. All rights reserved.
                    </td>
                  </tr>
        
                </table>
              </td>
            </tr>
          </table>
        </body>
        </html>
        """
    return text, html

def build_verify_email(verify_link: str, user_name: str | None = None) -> tuple[str, str]:
    greeting = f"Hi {user_name}," if user_name else "Hi,"

    text = f"""\
        {greeting}
        
        Welcome to Verbox!
        
        Please verify your email address by clicking the link below:
        {verify_link}
        
        If you did not create this account, you can safely ignore this email.
        
        — Verbox
        """

    html = f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Verify your email</title>
        </head>
        <body style="margin:0; padding:0; background-color:#f4f7fb; font-family:Arial,Helvetica,sans-serif; color:#1f2937;">
          <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f7fb; padding:24px 0;">
            <tr>
              <td align="center">
                <table width="100%" cellpadding="0" cellspacing="0" style="max-width:600px; background:#ffffff; border-radius:14px; overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,0.06);">
        
                  <tr>
                    <td align="center" style="padding:32px 24px 16px;">
                      <img src="https://verbox.ai/logo.png" width="140" alt="Verbox">
                    </td>
                  </tr>
        
                  <tr>
                    <td style="padding:8px 32px 0; text-align:center;">
                      <h1 style="margin:0; font-size:26px;">Verify your email</h1>
                    </td>
                  </tr>
        
                  <tr>
                    <td style="padding:20px 32px 0; font-size:16px; line-height:1.7;">
                      <p>{greeting}</p>
                      <p>Welcome to <strong>Verbox</strong> 👋</p>
                      <p>
                        To start using your account, please confirm your email address.
                      </p>
                    </td>
                  </tr>
        
                  <tr>
                    <td align="center" style="padding:24px 32px;">
                      <a href="{verify_link}"
                         style="background:#10b981; color:#fff; text-decoration:none; padding:14px 28px; border-radius:10px; font-weight:bold;">
                        Verify Email
                      </a>
                    </td>
                  </tr>
        
                  <tr>
                    <td style="padding:0 32px 24px; font-size:14px; color:#6b7280;">
                      <p>
                        If the button doesn’t work, use this link:
                      </p>
                      <p style="word-break:break-all;">
                        <a href="{verify_link}">{verify_link}</a>
                      </p>
                      <p>
                        If you didn’t create an account, you can safely ignore this email.
                      </p>
                    </td>
                  </tr>
        
                  <tr>
                    <td style="padding:24px; text-align:center; font-size:13px; color:#9ca3af;">
                      © Verbox
                    </td>
                  </tr>
        
                </table>
              </td>
            </tr>
          </table>
        </body>
        </html>
        """
    return text, html


@app.post("/runs/start")
async def start_dubbing_flow(req: StartDubRequest, user_id: int = Depends(require_user_id)):
    run_id = req.run_id
    check_project_access(run_id, user_id)

    config = read_run_config(run_id)
    run_params = config.get("run_params", {})
    video_file = run_params.get("video_file", "")
    duration_minutes = float(run_params.get("video_duration_minutes") or
                             _get_video_duration_minutes(str(_run_dir(run_id) / Path(video_file).relative_to(Path("output") / run_id)) if video_file else "") or 0)
    cost_cents = calculate_run_cost_cents(duration_minutes, bool(req.fix_timing))

    job_id: int | None = None
    if cost_cents > 0:
        await reconcile_pending_jobs()
        job_id = _charge_user(user_id, cost_cents, run_id, "dub")  # raises 402/409

    try:
        video_path = config.get("run_params", {}).get("video_file")
        flow_run = await run_deployment(
            name=DEPLOYMENT_NAME,
            parameters={
                "video_file": video_path,
                "dst_language": req.dst_language,
                "trans_type": req.trans_type,
                "ttsmodel": req.ttsmodel,
                "num_speakers": req.num_speakers,
                "emotions_flag": req.emotions_flag,
                "elevenlabs_emotions": int(req.elevenlabs_emotions),
                "fix_timing": req.fix_timing,
                "changed_list": req.changed_list,
                "is_dubbed": req.is_dubbed,
                "run_id": run_id,
                "stage": int(req.stage),
                "use_non_speech": req.use_non_speech,
            },
            timeout=0,
        )
    except Exception as e:
        if job_id is not None:
            _refund_job(job_id, user_id, cost_cents)
        raise HTTPException(status_code=500, detail=f"Failed to start deployment: {e}") from e

    flow_run_id = str(flow_run.id)

    if job_id is not None:
        execute(
            "UPDATE payments_for_jobs SET prefect_flow_run_id = %s, updated_at = NOW() WHERE id = %s",
            (flow_run_id, job_id),
        )

    data = read_run_config(run_id)
    data["flow_run_id"] = flow_run_id
    data["start_time"] = datetime.now(UTC).isoformat()
    data.setdefault("run_params", {})["dst_language"] = req.dst_language
    data["run_params"]["ttsmodel"] = req.ttsmodel
    data["run_params"]["use_non_speech"] = req.use_non_speech
    update_config_flow_id(run_id, data)

    return flow_run.id


@app.get("/runs/{run_id}/status")
async def stream_run_events(run_id: str,user_id: int = Depends(require_user_id)):
    check_project_access(run_id, user_id)
    flow_run_id = read_run_config(run_id)["flow_run_id"]

    async def event_generator():
        last_payload = None
        completed_states = {"COMPLETED"}
        error_states = {"FAILED", "CRASHED", "CANCELLED"}
        seen_running = False  # don't treat COMPLETED as done until we've seen the flow actually run
        while True:
            try:
                status = await fetch_prefect_status(flow_run_id)
                payload = {
                    "run_id": run_id,
                    **status,
                }
                serialized = json.dumps(payload, sort_keys=True)

                if serialized != last_payload:
                    yield f"event: status\ndata: {serialized}\n\n"
                    last_payload = serialized

                state_upper = (status["flow_state_name"] or "").upper()
                if state_upper in ("RUNNING",):
                    seen_running = True
                if state_upper in completed_states and seen_running:
                    yield f"event: done\ndata: {serialized}\n\n"
                    break
                elif state_upper in error_states:
                    yield f"event: error\ndata: {serialized}\n\n"
                    break

            except Exception as e:
                err = json.dumps({"run_id": run_id, "error": str(e)})
                yield f"event: error\ndata: {err}\n\n"
                break

            await asyncio.sleep(5.0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/projects/create")
async def create_project(req: Request, user_id: int = Depends(require_user_id)):
    data = await req.json()

    project_name = data.get("project_name")

    if not user_id or not project_name:
        raise HTTPException(status_code=400, detail="user_id and project_name are required")

    project = execute_returning(
        """
        INSERT INTO projects (project_name, user_id, updated_at)
        VALUES (%s, %s, NOW())
        RETURNING id, project_name, user_id
        """,
        (project_name, user_id),
    )

    if not project:
        raise HTTPException(status_code=500, detail="Failed to create project")

    return project


@app.get("/projects")
async def list_projects(user_id: int = Depends(require_user_id)):
    projects = fetch_all(
        """
        SELECT id, project_name, run_id, created_at, updated_at
        FROM projects
        WHERE user_id = %s
        ORDER BY updated_at DESC
        """,
        (user_id,),
    )

    return projects


@app.delete("/projects/{project_id}")
async def delete_project(project_id: int, user_id: int = Depends(require_user_id)):
    project = fetch_one(
        "SELECT id, run_id FROM projects WHERE id = %s AND user_id = %s",
        (project_id, user_id),
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    run_id = project.get("run_id")

    execute(
        "DELETE FROM projects WHERE id = %s AND user_id = %s",
        (project_id, user_id),
    )

    if run_id:
        for run_dir in [Path("output") / run_id, STORAGEBOX_ROOT / "output" / run_id]:
            if run_dir.exists():
                shutil.rmtree(run_dir)

    return {"ok": True}


@app.get("/runs/{run_id}/video")
async def get_video(run_id: str, user_id: int = Depends(require_user_id)):
    try:
        check_project_access(run_id, user_id)
        config = read_run_config(run_id)
        video_path = config.get("output_file")
    except Exception:
        raise HTTPException(status_code=404, detail="Run not found")

    if not video_path:
        raise HTTPException(status_code=404, detail="Video not ready")

    path = _run_dir(run_id) / Path(video_path).relative_to(Path("output") / run_id)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        path=path,
        media_type="video/mp4",
        filename=path.name,
    )

@app.get("/runs/{run_id}/subtitles/original")
async def get_subtitles(run_id: str, user_id: int = Depends(require_user_id)):
    check_project_access(run_id, user_id)
    srt_path = _run_dir(run_id) / "data/subtitles.srt"

    if not srt_path.exists():
        raise HTTPException(status_code=404, detail="Subtitles not found")

    content = srt_path.read_text(encoding="utf-8")

    return PlainTextResponse(content, media_type="text/plain")

@app.get("/runs/{run_id}/subtitles/translated")
async def get_subtitles(run_id: str, user_id: int = Depends(require_user_id)):
    check_project_access(run_id, user_id)
    srt_path = _run_dir(run_id) / "data/subtitles_translated.srt"

    if not srt_path.exists():
        raise HTTPException(status_code=404, detail="Subtitles not found")

    content = srt_path.read_text(encoding="utf-8")

    return PlainTextResponse(content, media_type="text/plain")


_STEM_FILES = {
    "background": "audio/stem_background.mp3",
    "dialog":     "audio/stem_dialog.mp3",
    "original":   "audio/stem_original.mp3",
}

@app.get("/runs/{run_id}/stems")
async def get_stems(run_id: str, user_id: int = Depends(require_user_id)):
    check_project_access(run_id, user_id)
    available = {}
    for name, rel in _STEM_FILES.items():
        if (_run_dir(run_id) / rel).exists():
            available[name] = f"/runs/{run_id}/stems/{name}"
    if not available:
        raise HTTPException(status_code=404, detail="Stems not ready")
    return available

@app.get("/runs/{run_id}/stems/{stem}")
async def get_stem(run_id: str, stem: str, user_id: int = Depends(require_user_id)):
    check_project_access(run_id, user_id)
    if stem not in _STEM_FILES:
        raise HTTPException(status_code=404, detail="Unknown stem")
    path = _run_dir(run_id) / _STEM_FILES[stem]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Stem not ready")
    return FileResponse(path=path, media_type="audio/mpeg", filename=f"{stem}.mp3")


@app.get("/runs/{run_id}/speaker-names")
async def get_speaker_names(run_id: str, user_id: int = Depends(require_user_id)):
    check_project_access(run_id, user_id)
    config = read_run_config(run_id)
    return config.get("speaker_names", {})


@app.post("/runs/{run_id}/speaker-names")
async def set_speaker_names(run_id: str, req: Request, user_id: int = Depends(require_user_id)):
    check_project_access(run_id, user_id)
    names = await req.json()
    config_path = _run_dir(run_id) / "data/general_config.json"
    config = json.loads(config_path.read_text())
    config["speaker_names"] = names
    config_path.write_text(json.dumps(config, indent=4))
    return names


@app.get("/runs/{run_id}/input_video")
async def get_input_video(run_id: str, user_id: int = Depends(require_user_id)):
    try:
        check_project_access(run_id, user_id)
        config = read_run_config(run_id)
        video_path = config.get("run_params").get("video_file")
    except Exception:
        raise HTTPException(status_code=404, detail="Run not found")

    if not video_path:
        raise HTTPException(status_code=404, detail="Input video not found")

    path = _run_dir(run_id) / Path(video_path).relative_to(Path("output") / run_id)

    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing")

    return FileResponse(
        path=path,
        media_type="video/mp4",
        filename=path.name,
    )


@app.post("/runs/upload")
async def upload_video(project_id: int, file: UploadFile = File(...), user_id: int = Depends(require_user_id)):
    try:
        project = fetch_one(
            """
            SELECT id
            FROM projects
            WHERE id = %s
              AND user_id = %s
            """,
            (project_id, user_id),
        )

        if not project:
            raise HTTPException(status_code=403, detail="Access denied")
        run_id = generate_run_id()

        run_dir = Path("output") / run_id
        input_dir = run_dir / "input"
        data_dir = run_dir / "data"

        input_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        file_name = unidecode(file.filename)
        file_path = input_dir / file_name

        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)

        duration_minutes = _get_video_duration_minutes(str(file_path))

        config = {
            "run_params": {
                "video_file": str(file_path),
                "video_duration_minutes": duration_minutes,
            }
        }

        config_path = data_dir / "general_config.json"
        config_path.write_text(json.dumps(config, indent=4), encoding="utf-8")

        shutil.move(str(run_dir), str(STORAGEBOX_ROOT / "output" / run_id))

        updated = execute(
            """
            UPDATE projects
            SET run_id = %s, updated_at = NOW()
            WHERE id = %s
            """,
            (run_id, project_id),
        )

        if updated == 0:
            raise HTTPException(status_code=404, detail="Project not found")

        return {
            "run_id": run_id,
            "project_id": project_id,
            "duration_minutes": duration_minutes,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")



@app.post("/runs/youtube")
async def import_youtube(req: YoutubeImportRequest, user_id: int = Depends(require_user_id)):
    project = fetch_one(
        "SELECT id FROM projects WHERE id = %s AND user_id = %s",
        (req.project_id, user_id),
    )
    if not project:
        raise HTTPException(status_code=403, detail="Access denied")

    run_id = generate_run_id()

    run_dir = Path("output") / run_id
    input_dir = run_dir / "input"
    data_dir = run_dir / "data"
    input_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    progress_file = data_dir / "yt_progress.json"

    def _write_progress(payload: dict) -> None:
        progress_file.write_text(json.dumps(payload), encoding="utf-8")

    # Associate run_id with the project NOW so check_project_access works
    # when the FE opens the SSE stream immediately after this response.
    execute(
        "UPDATE projects SET run_id = %s, updated_at = NOW() WHERE id = %s",
        (run_id, req.project_id),
    )

    _write_progress({"status": "downloading", "percent": 0, "filename": None, "run_id": run_id, "error": None})

    async def _download():
        def _progress_hook(d: dict):
            if d.get("status") == "downloading":
                raw = d.get("_percent_str", "").strip().replace("%", "")
                try:
                    _write_progress({"status": "downloading", "percent": min(99, float(raw)), "filename": None, "run_id": run_id, "error": None})
                except ValueError:
                    pass
            elif d.get("status") == "finished":
                _write_progress({"status": "downloading", "percent": 99, "filename": None, "run_id": run_id, "error": None})

        BASE_DIR = Path(__file__).resolve().parent.parent

        COOKIE_FILE = BASE_DIR / "config" / "youtube_cookies.txt"

        ydl_opts = {
            "format": "bestvideo[vcodec^=avc1]+bestaudio[acodec^=mp4a]/bestvideo[vcodec^=avc1]+bestaudio/bestvideo+bestaudio/best",
            "outtmpl": str(input_dir / "%(title)s.%(ext)s"),
            "merge_output_format": "mp4",
            "progress_hooks": [_progress_hook],
            "quiet": True,
            "no_warnings": True,
            "cookiefile": str(COOKIE_FILE),
            "js_runtimes": {
                "node": {}
            },
            "source_address": "0.0.0.0",
        }
        try:
            loop = asyncio.get_event_loop()

            def _blocking():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(req.url, download=True)
                    downloaded = ydl.prepare_filename(info)
                    if not downloaded.endswith(".mp4"):
                        downloaded = downloaded.rsplit(".", 1)[0] + ".mp4"
                    safe_name = unidecode(Path(downloaded).name)
                    safe_path = Path(downloaded).parent / safe_name
                    if Path(downloaded) != safe_path:
                        Path(downloaded).rename(safe_path)
                    # Re-encode to H.264/AAC if video codec is not iOS-compatible
                    probe = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-show_entries", "stream=codec_name",
                         "-select_streams", "v:0", "-of", "default=nw=1", str(safe_path)],
                        capture_output=True, text=True
                    )
                    vcodec = probe.stdout.strip().replace("codec_name=", "")
                    if vcodec not in ("h264", "hevc"):
                        reencoded = safe_path.parent / ("h264_" + safe_path.name)
                        subprocess.run([
                            "ffmpeg", "-y", "-i", str(safe_path),
                            "-vcodec", "libx264", "-acodec", "aac",
                            "-movflags", "+faststart", str(reencoded)
                        ], check=True, capture_output=True)
                        safe_path.unlink()
                        reencoded.rename(safe_path)
                    return safe_path

            file_path = await loop.run_in_executor(None, _blocking)

            duration_minutes = _get_video_duration_minutes(str(file_path))

            config = {
                "run_params": {
                    "video_file": str(file_path),
                    "video_duration_minutes": duration_minutes,
                }
            }
            (data_dir / "general_config.json").write_text(json.dumps(config, indent=4), encoding="utf-8")

            shutil.move(str(run_dir), str(STORAGEBOX_ROOT / "output" / run_id))

            # Write done progress to storagebox path so SSE stream picks it up after the move
            done_progress_file = STORAGEBOX_ROOT / "output" / run_id / "data" / "yt_progress.json"
            done_progress_file.write_text(json.dumps({"status": "done", "percent": 100, "filename": file_path.name, "run_id": run_id, "error": None, "duration_minutes": duration_minutes}), encoding="utf-8")
        except Exception as e:
            _write_progress({"status": "error", "percent": 0, "filename": None, "run_id": run_id, "error": str(e)})

    asyncio.create_task(_download())

    return {"run_id": run_id}


@app.get("/runs/{run_id}/youtube-progress")
async def youtube_progress(run_id: str, user_id: int = Depends(require_user_id)):
    check_project_access(run_id, user_id)
    local_progress_file = Path(f"output/{run_id}/data/yt_progress.json")
    sb_progress_file = STORAGEBOX_ROOT / "output" / run_id / "data" / "yt_progress.json"

    def _progress_file() -> Path | None:
        if sb_progress_file.exists():
            return sb_progress_file
        if local_progress_file.exists():
            return local_progress_file
        return None

    async def _stream():
        # Wait up to 10 s for the file to appear (FE may open the SSE
        # stream before the background task writes it)
        for _ in range(20):
            if _progress_file() is not None:
                break
            await asyncio.sleep(0.5)

        while True:
            try:
                pf = _progress_file()
                job = json.loads(pf.read_text(encoding="utf-8")) if pf else {}
            except Exception:
                job = {}
            yield f"data: {json.dumps(job)}\n\n"
            if job.get("status") in ("done", "error"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/projects/{project_id}/status")
async def get_project_status(project_id: int, user_id: int = Depends(require_user_id)):
    project = fetch_one(
        """
        SELECT id, project_name, user_id, run_id, created_at, updated_at
        FROM projects
        WHERE id = %s and user_id = %s
        """,
        (project_id,user_id),
    )

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    run_id = project.get("run_id")

    if not run_id:
        return {
            "project_id": project_id,
            "run_id": None,
            "status": "initial",
            "file_name": None,
        }

    config = read_run_config(run_id)

    run_params = config.get("run_params", {})
    full_path = run_params.get("video_file")
    file_name = Path(full_path).name if full_path else None
    duration_minutes = run_params.get("video_duration_minutes")
    dst_language = run_params.get("dst_language")
    ttsmodel = run_params.get("ttsmodel")
    use_non_speech = run_params.get("use_non_speech", True)
    is_dubbed = run_params.get("is_dubbed", False)
    emotions_flag = run_params.get("emotions_flag", True)
    trans_type = run_params.get("trans_type", "default")
    fix_timing = run_params.get("fix_timing", True)
    mix_gains = config.get("mix_gains")
    # src_language is written by the pipeline after transcription
    src_language = config.get("src_language") or run_params.get("src_language")

    start_time = config.get("start_time")
    output_file = config.get("output_file")

    if not start_time:
        return {
            "project_id": project_id,
            "run_id": run_id,
            "status": "uploaded",
            "file_name": file_name,
            "duration_minutes": duration_minutes,
            "dst_language": dst_language,
            "ttsmodel": ttsmodel,
        }

    run_config_fields = {
        "src_language": src_language,
        "use_non_speech": use_non_speech,
        "is_dubbed": is_dubbed,
        "emotions_flag": emotions_flag,
        "trans_type": trans_type,
        "fix_timing": fix_timing,
    }

    if output_file:
        return {
            "project_id": project_id,
            "run_id": run_id,
            "status": "finished",
            "file_name": file_name,
            "duration_minutes": duration_minutes,
            "dst_language": dst_language,
            "ttsmodel": ttsmodel,
            "mix_gains": mix_gains,
            **run_config_fields,
        }

    # start_time set but no output_file — query Prefect for live state
    flow_run_id = config.get("flow_run_id")
    if not flow_run_id:
        return {
            "project_id": project_id,
            "run_id": run_id,
            "status": "failed",
            "file_name": file_name,
            "duration_minutes": duration_minutes,
            "dst_language": dst_language,
            "ttsmodel": ttsmodel,
            "mix_gains": mix_gains,
            **run_config_fields,
        }

    try:
        prefect_status = await fetch_prefect_status(flow_run_id)
        flow_state = (prefect_status.get("flow_state_name") or "").upper()
        error_states = {"FAILED", "CRASHED", "CANCELLED"}
        run_status = "failed" if flow_state in error_states else "processing"
    except Exception:
        run_status = "failed"

    return {
        "project_id": project_id,
        "run_id": run_id,
        "status": run_status,
        "file_name": file_name,
        "duration_minutes": duration_minutes,
        "dst_language": dst_language,
        "ttsmodel": ttsmodel,
        "mix_gains": mix_gains,
        **run_config_fields,
    }




@app.post("/runs/{run_id}/regenerate")
async def regenerate_from_subtitles(run_id: str, req: Request, user_id: int = Depends(require_user_id)):
    check_project_access(run_id, user_id)
    data = await req.json()

    subtitles_text = data.get("subtitles")
    changed_list = data.get("changed_list", [])
    ttsmodel = int(data.get("ttsmodel", 1))

    if not subtitles_text:
        raise HTTPException(status_code=400, detail="subtitles are required")

    changed_set = set(changed_list) if changed_list else set()
    parsed_subs = list(srt.parse(subtitles_text))

    if ttsmodel == 3:
        # Original Voice: flat cost based on full video duration
        config = read_run_config(run_id)
        duration_minutes = float(config.get("run_params", {}).get("video_duration_minutes") or 0)
        cost_cents = calculate_regen_cost_cents(duration_minutes, ttsmodel)
    else:
        total_seconds = sum(
            (sub.end - sub.start).total_seconds()
            for sub in parsed_subs
            if sub.index in changed_set
        )
        cost_cents = calculate_regen_cost_cents(total_seconds / 60, ttsmodel)

    job_id: int | None = None
    if cost_cents > 0:
        await reconcile_pending_jobs()
        job_id = _charge_user(user_id, cost_cents, run_id, "regen")  # raises 402/409

    try:
        srt_paths = [_run_dir(run_id) / "data/subtitles_translated.srt",
                     _run_dir(run_id) / "data/subtitles_fixed_translated.srt"]

        for srt_path in srt_paths:
            srt_path.write_text(subtitles_text, encoding="utf-8")

        # Clear stale export files so FE won't show Download for old renders
        for p in (_run_dir(run_id) / "data").glob("export_status_*.json"):
            p.unlink(missing_ok=True)

        config = read_run_config(run_id)
        run_params = config.get("run_params", {})
        if not run_params:
            raise HTTPException(status_code=400, detail="run_params missing in config")

        flow_run = await run_deployment(
            name=DEPLOYMENT_NAME,
            parameters={
                **run_params,
                "run_id": run_id,
                "fix_timing": False,
                "changed_list": changed_list,
                "stage": int(STAGES.EMOTION),
            },
            timeout=0,
        )
    except HTTPException:
        if job_id is not None:
            _refund_job(job_id, user_id, cost_cents)
        raise
    except Exception as e:
        if job_id is not None:
            _refund_job(job_id, user_id, cost_cents)
        raise HTTPException(status_code=500, detail=f"Regeneration failed: {e}") from e

    flow_run_id = str(flow_run.id)

    if job_id is not None:
        execute(
            "UPDATE payments_for_jobs SET prefect_flow_run_id = %s, updated_at = NOW() WHERE id = %s",
            (flow_run_id, job_id),
        )

    config["flow_run_id"] = flow_run_id
    config["start_time"] = datetime.now().isoformat()
    config.pop("output_file", None)
    update_config_flow_id(run_id, config)

    execute("UPDATE projects SET updated_at = NOW() WHERE run_id = %s", (run_id,))

    return {"run_id": run_id, "flow_run_id": flow_run_id}


class RemixRequest(BaseModel):
    mix_gains: list[float] = Field(..., min_length=4, max_length=4,
                                   description="[background_db, dialog_db, non_speech_db, original_underlay_db]")

@app.post("/runs/{run_id}/remix")
async def remix_audio(run_id: str, req: RemixRequest, user_id: int = Depends(require_user_id)):
    check_project_access(run_id, user_id)

    config = read_run_config(run_id)
    run_params = config.get("run_params", {})
    if not run_params:
        raise HTTPException(status_code=400, detail="run_params missing in config")

    for p in (_run_dir(run_id) / "data").glob("export_status_*.json"):
        p.unlink(missing_ok=True)

    flow_run = await run_deployment(
        name=DEPLOYMENT_NAME,
        parameters={
            **run_params,
            "run_id": run_id,
            "fix_timing": False,
            "stage": int(STAGES.COMBINE),
            "mix_gains": req.mix_gains,
        },
        timeout=0,
    )

    flow_run_id = str(flow_run.id)
    config["flow_run_id"] = flow_run_id
    config["start_time"] = datetime.now().isoformat()
    config["mix_gains"] = req.mix_gains
    config.pop("output_file", None)  # clear so status returns "processing" until new render completes
    update_config_flow_id(run_id, config)

    execute("UPDATE projects SET updated_at = NOW() WHERE run_id = %s", (run_id,))

    return {"run_id": run_id, "flow_run_id": flow_run_id}
















def _export_status_path(run_id: str, resolution: str) -> Path:
    return _run_dir(run_id) / f"data/export_status_{resolution}.json"


def _write_export_status(run_id: str, resolution: str, status: str, output_file: str | None) -> None:
    _export_status_path(run_id, resolution).write_text(
        json.dumps({"status": status, "output_file": output_file}), encoding="utf-8"
    )


def _read_export_status(run_id: str, resolution: str) -> dict:
    path = _export_status_path(run_id, resolution)
    if not path.exists():
        return {"status": "not_started", "output_file": None}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "not_started", "output_file": None}


@app.post("/runs/{run_id}/export")
async def start_export(run_id: str, resolution: str = "original", user_id: int = Depends(require_user_id)):
    if resolution not in EXPORT_RESOLUTIONS:
        raise HTTPException(400, f"resolution must be one of {sorted(EXPORT_RESOLUTIONS)}")
    check_project_access(run_id, user_id)

    cfg_data = read_run_config(run_id)
    run_dir = _run_dir(run_id)
    video_in = str(run_dir / Path(cfg_data["run_params"]["video_file"]).relative_to(Path("output") / run_id))
    audio_in = str(run_dir / "audio/final_audio.wav")
    subs_in  = str(run_dir / "data/subtitles_translated.srt")
    final_output = str(run_dir / f"final_video_{resolution}.mp4")

    _write_export_status(run_id, resolution, "processing", None)

    async def _run():
        import tempfile
        tmp_file = tempfile.NamedTemporaryFile(suffix=f"_{resolution}.mp4", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, generate_videos, video_in, audio_in, subs_in, tmp_path, False, resolution)
            shutil.move(tmp_path, final_output)
            _write_export_status(run_id, resolution, "ready", final_output)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            _write_export_status(run_id, resolution, "error", None)

    asyncio.create_task(_run())
    return {"status": "processing"}


@app.get("/runs/{run_id}/export/status")
async def get_export_status(run_id: str, resolution: str = "original", user_id: int = Depends(require_user_id)):
    check_project_access(run_id, user_id)
    return {"status": _read_export_status(run_id, resolution)["status"]}


@app.get("/runs/{run_id}/export/file")
async def download_export(run_id: str, resolution: str = "original", user_id: int = Depends(require_user_id)):
    check_project_access(run_id, user_id)
    task = _read_export_status(run_id, resolution)
    if task["status"] != "ready":
        raise HTTPException(404, "Export not ready")
    path = Path(task["output_file"])
    if not path.exists():
        raise HTTPException(404, "Export file missing")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/runs/{run_id}/audio")
async def download_audio(run_id: str, user_id: int = Depends(require_user_id)):
    check_project_access(run_id, user_id)
    path = _run_dir(run_id) / "audio/final_audio.wav"
    if not path.exists():
        raise HTTPException(404, "Audio not ready")
    return FileResponse(path, media_type="audio/wav", filename="final_audio.wav")


@app.put("/users/profile")
async def update_profile(req: Request, user_id: int = Depends(require_user_id)):
    data = await req.json()
    first_name = (data.get("first_name") or "").strip()
    last_name = (data.get("last_name") or "").strip()
    email = (data.get("email") or "").strip()

    if not first_name or not last_name:
        raise HTTPException(status_code=400, detail="first_name and last_name are required")

    user = fetch_one("SELECT auth_provider FROM users WHERE id = %s", (user_id,))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user["auth_provider"] == "google":
        execute(
            "UPDATE users SET first_name = %s, last_name = %s WHERE id = %s",
            (first_name, last_name, user_id),
        )
    else:
        if not email:
            raise HTTPException(status_code=400, detail="email is required")
        taken = fetch_one("SELECT id FROM users WHERE email = %s AND id != %s", (email, user_id))
        if taken:
            raise HTTPException(status_code=400, detail="Email already in use")
        execute(
            "UPDATE users SET first_name = %s, last_name = %s, email = %s WHERE id = %s",
            (first_name, last_name, email, user_id),
        )

    return {"ok": True}


@app.post("/users/change-password")
async def change_password_endpoint(req: Request, user_id: int = Depends(require_user_id)):
    data = await req.json()
    current_password = data.get("current_password", "")
    new_password = data.get("new_password", "")

    if not current_password or not new_password:
        raise HTTPException(status_code=400, detail="current_password and new_password are required")

    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    user = fetch_one("SELECT password_hash, auth_provider FROM users WHERE id = %s", (user_id,))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user["auth_provider"] != "local":
        raise HTTPException(status_code=403, detail="Password change not available for Google accounts")

    if not verify_password(current_password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    execute(
        "UPDATE users SET password_hash = %s WHERE id = %s",
        (hash_password(new_password), user_id),
    )

    return {"ok": True}


@app.get("/users/balance")
async def get_user_balance(user_id: int = Depends(require_user_id)):
    row = fetch_one("SELECT balance FROM users WHERE id = %s", (user_id,))
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return {"balance": row["balance"]}


@app.post("/payments/create-checkout")
async def create_checkout_session(req: Request, user_id: int = Depends(require_user_id)):
    data = await req.json()
    amount_cents = data.get("amount_cents")

    if not isinstance(amount_cents, int) or amount_cents < 500:
        raise HTTPException(status_code=400, detail="Minimum amount is $5.00")

    user = fetch_one("SELECT id, email, stripe_customer_id FROM users WHERE id = %s", (user_id,))

    customer_id = user.get("stripe_customer_id")
    if not customer_id:
        customer = stripe.Customer.create(email=user["email"])
        customer_id = customer.id
        execute("UPDATE users SET stripe_customer_id = %s WHERE id = %s", (customer_id, user_id))

    payment = execute_returning(
        """
        INSERT INTO payments (user_id, amount_cents, credits, status)
        VALUES (%s, %s, %s, 'pending')
        RETURNING id
        """,
        (user_id, amount_cents, amount_cents),
    )

    session = stripe.checkout.Session.create(
        customer=customer_id,
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency": "usd",
                "unit_amount": amount_cents,
                "product_data": {
                    "name": f"Verbox.ai Credits — ${amount_cents // 100}",
                },
            },
            "quantity": 1,
        }],
        mode="payment",
        success_url=f"{PUBLIC_APP_URL}/?payment=success",
        cancel_url=f"{PUBLIC_APP_URL}/?payment=cancelled",
        metadata={
            "user_id": str(user_id),
            "payment_id": str(payment["id"]),
        },
    )

    execute(
        "UPDATE payments SET stripe_checkout_session_id = %s WHERE id = %s",
        (session.id, payment["id"]),
    )

    return {"checkout_url": session.url}


@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    if event.type == "checkout.session.completed":
        session = event.data.object

        if session.payment_status == "paid":
            metadata = session.metadata or {}

            payment_id =  metadata["payment_id"] if "payment_id" in metadata else None
            user_id = metadata["user_id"]  if "user_id" in metadata else None

            if payment_id and user_id:
                updated = execute(
                    """
                    UPDATE payments
                    SET status = 'succeeded',
                        stripe_payment_intent_id = %s,
                        updated_at = NOW()
                    WHERE id = %s AND status = 'pending'
                    """,
                    (session.payment_intent, int(payment_id)),
                )

                if updated > 0:
                    row = fetch_one(
                        "SELECT credits FROM payments WHERE id = %s",
                        (int(payment_id),)
                    )
                    if row:
                        execute(
                            "UPDATE users SET balance = balance + %s WHERE id = %s",
                            (row["credits"], int(user_id)),
                        )

    return {"ok": True}


@app.post("/auth/register/local")
async def register_local(req: Request):
    data = await req.json()

    user_name = data.get("user_name")
    email = data.get("email")
    password = data.get("password")
    first_name = data.get("first_name")
    last_name = data.get("last_name")
    turnstile_token = data.get("cf_turnstile_token")

    if not user_name or not email or not password:
        raise HTTPException(status_code=400, detail="user_name, email, password are required")

    if not turnstile_token:
        raise HTTPException(status_code=400, detail="CAPTCHA token is required")

    await verify_turnstile_token(turnstile_token)

    ip = req.client.host if req.client else "unknown"
    check_ip_registration_limit(ip)
    check_disposable_email(email)

    existing = fetch_one(
        "SELECT id FROM users WHERE user_name = %s OR email = %s",
        (user_name, email),
    )
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    record_registration_attempt(ip)

    user = execute_returning(
        """
        INSERT INTO users (user_name, email, password_hash, first_name, last_name,
                           auth_provider, created_at, status, balance, email_verified)
        VALUES (%s, %s, %s, %s, %s, 'local', NOW(), 'active', 500,
                FALSE) RETURNING id, user_name, email, first_name, last_name, email_verified
        """,
        (
            user_name,
            email,
            hash_password(password),
            first_name,
            last_name,
        ),
    )

    token = create_email_verification_token(user["id"])
    verify_link = f"{PUBLIC_API_URL}/verify-email?token={token}"
    text_body, html_body = build_verify_email(
        verify_link,
        user_name
    )
    send_email(
        user["email"],
        "Verify your Verbox email",
        text_body,
        html_body,
    )

    return user

@app.post("/auth/google")
async def auth_google(req: Request, response: Response):
    data = await req.json()
    credential = data.get("credential")
    if not credential:
        raise HTTPException(status_code=400, detail="credential is required")

    info = verify_google_credential(credential)

    google_sub = info.get("sub")
    email = info.get("email")
    first_name = info.get("given_name")
    last_name = info.get("family_name")
    full_name = info.get("name") or email or f"google_{google_sub}"

    if not google_sub or not email:
        raise HTTPException(status_code=400, detail="Google credential missing required fields")

    user = fetch_one(
        "SELECT id, user_name, email, first_name, last_name FROM users WHERE google_sub = %s",
        (google_sub,),
    )

    if not user:
        # fallback by email
        user = fetch_one(
            "SELECT id, user_name, email, first_name, last_name FROM users WHERE email = %s",
            (email,),
        )

        if user:
            execute(
                """
                UPDATE users
                SET google_sub     = %s,
                    auth_provider  = 'google',
                    last_login     = NOW(),
                    email_verified = TRUE
                WHERE id = %s
                """,
                (google_sub, user["id"]),
            )
            user = fetch_one(
                "SELECT id, user_name, email, first_name, last_name FROM users WHERE id = %s",
                (user["id"],),
            )
        else:
            user = execute_returning(
                """
                INSERT INTO users (user_name, email, google_sub, password_hash, first_name, last_name,
                                   auth_provider, created_at, status, balance, last_login, email_verified)
                VALUES (%s, %s, %s, NULL, %s, %s, 'google', NOW(), 'active', 500, NOW(),
                        TRUE) RETURNING id, user_name, email, first_name, last_name, email_verified
                """,
                (
                    full_name,
                    email,
                    google_sub,
                    first_name,
                    last_name,
                ),
            )
    else:
        execute("UPDATE users SET last_login = NOW() WHERE id = %s", (user["id"],))

    create_session(user["id"], req, response)

    return {
        "user": user
    }


@app.post("/auth/login/local")
async def login_local(req: Request, response: Response):
    data = await req.json()

    login = data.get("login")  # user_name or email
    password = data.get("password")

    if not login or not password:
        raise HTTPException(status_code=400, detail="login and password are required")

    user = fetch_one(
        """
        SELECT id,
               user_name,
               email,
               password_hash,
               first_name,
               last_name,
               status,
               email_verified
        FROM users
        WHERE user_name = %s
           OR email = %s
            AND auth_provider='local'
        """,
        (login, login),
    )

    if not user or not user.get("password_hash"):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user["email_verified"]:
        raise HTTPException(status_code=403, detail="Email not verified")

    if user["status"] != "active":
        raise HTTPException(status_code=403, detail="User inactive")

    if not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    execute("UPDATE users SET last_login = NOW() WHERE id = %s", (user["id"],))
    create_session(user["id"], req, response)

    return {
        "user": {
            "id": user["id"],
            "user_name": user["user_name"],
            "email": user["email"],
            "first_name": user["first_name"],
            "last_name": user["last_name"],
        }
    }


@app.post("/auth/logout")
async def logout(response: Response, session_token: str | None = Cookie(default=None)):
    if session_token:
        execute("DELETE FROM user_sessions WHERE session_token = %s", (session_token,))
    clear_session_cookie(response)
    return {"ok": True}


@app.get("/auth/me")
async def auth_me(session_token: str | None = Cookie(default=None)):
    user = get_current_user(session_token)
    return {"user": user}

@app.post("/auth/request-password-reset")
async def request_password_reset(req: Request):
    data = await req.json()
    email = data.get("email")

    if not email:
        raise HTTPException(status_code=400, detail="email is required")

    user = fetch_one(
        """
        SELECT id, user_name, email, auth_provider, status
        FROM users
        WHERE email = %s
        """,
        (email,),
    )

    # Always return ok to avoid leaking whether email exists
    if not user:
        return {"ok": True}

    if user["status"] != "active":
        return {"ok": True}

    if user["auth_provider"] != "local":
        return {"ok": True}

    token = create_password_reset_token(user["id"])
    reset_link = f"{PUBLIC_API_URL}/reset-password?token={token}"
    text_body, html_body = build_password_reset_email(reset_link, user.get("user_name"))
    send_email(
        user["email"],
        "Reset your Verbox password",
        text_body,
        html_body
    )

    return {"ok": True}


@app.get("/reset-password")
async def reset_password_redirect(token: str):
    if not token:
        raise HTTPException(status_code=400, detail="token is required")
    return RedirectResponse(url=f"{PUBLIC_APP_URL}/reset-password?token={token}")


@app.post("/auth/reset-password")
async def reset_password(req: Request):
    data = await req.json()
    token = data.get("token")
    new_password = data.get("new_password")

    if not token or not new_password:
        raise HTTPException(status_code=400, detail="token and new_password are required")

    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    row = consume_password_reset_token(token)

    execute(
        """
        UPDATE users
        SET password_hash = %s
        WHERE id = %s
        """,
        (hash_password(new_password), row["user_id"]),
    )

    # Optional: invalidate all existing sessions for safety
    execute(
        "DELETE FROM user_sessions WHERE user_id = %s",
        (row["user_id"],),
    )

    return {"ok": True}


#
# @app.post("/auth/request-email-verification")
# async def request_email_verification(req: Request):
#     data = await req.json()
#     email = data.get("email")
#
#     if not email:
#         raise HTTPException(status_code=400, detail="email is required")
#
#     user = fetch_one(
#         """
#         SELECT id, email, email_verified, status
#         FROM users
#         WHERE email = %s
#         """,
#         (email,),
#     )
#
#     if not user:
#         return {"ok": True}
#
#     if user["status"] != "active" or user["email_verified"]:
#         return {"ok": True}
#
#     token = create_email_verification_token(user["id"])
#     verify_link = f"{PUBLIC_APP_URL}/verify-email?token={token}"
#
#     send_email(
#         user["email"],
#         "Verify your email",
#         f"Use this link to verify your email:\n{verify_link}",
#     )
#
#     return {"ok": True}


@app.get("/verify-email")
async def verify_email(token: str):

    if not token:
        raise HTTPException(status_code=400, detail="token is required")

    row = consume_email_verification_token(token)
    if not row:
        return RedirectResponse(url=f"{PUBLIC_APP_URL}/verify-email?status=error")
    execute(
        """
        UPDATE users
        SET email_verified = TRUE
        WHERE id = %s
        """,
        (row["user_id"],),
    )

    return RedirectResponse(url=f"{PUBLIC_APP_URL}/verify-email?status=success")

@app.post("/auth/resend-email-verification")
async def resend_email_verification(req: Request):
    data = await req.json()
    email = data.get("email")

    if not email:
        raise HTTPException(status_code=400, detail="email is required")

    user = fetch_one(
        """
        SELECT id, user_name,email, email_verified, status
        FROM users
        WHERE email = %s
        """,
        (email,),
    )

    # Always return ok (security: don't reveal if email exists)
    if not user:
        return {"ok": True}

    if user["status"] != "active":
        return {"ok": True}

    if user["email_verified"]:
        return {"ok": True}

    # --- generate new verification token ---
    token = create_email_verification_token(user["id"])

    verify_link = f"{PUBLIC_API_URL}/verify-email?token={token}"

    text_body, html_body = build_verify_email(
        verify_link,
        user["status"]
    )
    send_email(
        user["user_name"],
        "Verify your Verbox email",
        text_body,
        html_body,
    )

    return {"ok": True}