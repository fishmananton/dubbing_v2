import time
from typing import Any

import requests


RUNPOD_BASE_URL = "https://api.runpod.ai/v2"

# Retry-worthy HTTP status codes per practical API behavior.
TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}

# Known terminal job statuses from Runpod docs and common serverless behavior.
TERMINAL_SUCCESS = {"COMPLETED"}
TERMINAL_FAILURE = {"FAILED", "CANCELLED", "TIMED_OUT", "ERROR"}

_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def _request_with_retry(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, Any] | None = None,
    timeout: tuple[float, float] = (10.0, 60.0),
    max_attempts: int = 4,
    base_backoff_sec: float = 1.5,
) -> requests.Response:
    session = _get_session()
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.request(
                method=method,
                url=url,
                headers=headers,
                json=json_body,
                timeout=timeout,
            )

            # Fail fast on permanent client-side request/auth issues.
            if resp.status_code in {400, 401}:
                raise RuntimeError(
                    f"Permanent Runpod HTTP error {resp.status_code}: {resp.text[:500]}"
                )

            # 404 is nuanced:
            # - for /status it may mean TTL expiry or bad endpoint/job id
            # - for /run it likely means bad endpoint id
            if resp.status_code == 404:
                raise requests.HTTPError(
                    f"Runpod returned 404: {resp.text[:500]}",
                    response=resp,
                )

            if resp.status_code in TRANSIENT_HTTP_CODES:
                last_error = RuntimeError(
                    f"Transient Runpod HTTP {resp.status_code}: {resp.text[:500]}"
                )
            else:
                resp.raise_for_status()
                return resp

        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
        except requests.HTTPError as exc:
            # Keep 404 handling to caller when needed; everything else fail fast here.
            if exc.response is not None and exc.response.status_code == 404:
                raise
            raise RuntimeError(f"HTTP error calling Runpod: {exc}") from exc

        if attempt < max_attempts:
            sleep_s = base_backoff_sec ** (attempt - 1)
            time.sleep(sleep_s)

    raise RuntimeError(
        f"Runpod request failed after {max_attempts} attempts: {last_error}"
    )


def run_runpod_job(
    runpod_key: str,
    runpod_template_id: str,
    payload: dict,
    job_name: str,
    timeout_minutes: int = 10,
    poll_delay_sec: int = 5,
) -> dict:
    """
    Submit an async Runpod job and poll until terminal state or timeout.

    Returns a normalized dict:
    {
        "status": "...",
        "job_id": "...",
        "data": {...},
        "output": {...} | None,
        "elapsed_seconds": float,
    }
    """
    headers = {
        "Authorization": f"Bearer {runpod_key}",
        "Content-Type": "application/json",
    }

    submit_url = f"{RUNPOD_BASE_URL}/{runpod_template_id}/run"
    submit_resp = _request_with_retry(
        "POST",
        submit_url,
        headers=headers,
        json_body={"input": payload},
        timeout=(10.0, 60.0),
        max_attempts=4,
    )

    submit_data = submit_resp.json()
    job_id = submit_data.get("id")
    initial_status = submit_data.get("status")

    if not job_id:
        raise RuntimeError(
            f"[{job_name}] Runpod submit response missing job id: {submit_data}"
        )

    print(f"[{job_name}] Runpod started. job_id={job_id}, initial_status={initial_status}")

    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    last_minute_logged = 0

    status_url = f"{RUNPOD_BASE_URL}/{runpod_template_id}/status/{job_id}"

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"[{job_name}] Timeout after {timeout_minutes} minutes. job_id={job_id}"
            )

        try:
            status_resp = _request_with_retry(
                "GET",
                status_url,
                headers=headers,
                timeout=(10.0, 30.0),
                max_attempts=3,
            )
            data = status_resp.json()

        except requests.HTTPError as exc:
            # Runpod docs note that /status can return 404 if TTL expired.
            if exc.response is not None and exc.response.status_code == 404:
                raise RuntimeError(
                    f"[{job_name}] Runpod status returned 404 for job_id={job_id}. "
                    "This can mean the job/result expired (TTL/result retention) or the endpoint/job id is wrong."
                ) from exc
            raise

        except RuntimeError as exc:
            # After bounded retries, keep polling until the overall timeout budget is exhausted.
            print(f"[{job_name}] Polling error: {exc}")
            time.sleep(poll_delay_sec)
            continue

        status = data.get("status", "UNKNOWN")

        if status in TERMINAL_SUCCESS:
            print(f"[{job_name}] Runpod finished with status: {status}")
            return {
                "status": status,
                "job_id": job_id,
                "data": data,
                "output": data.get("output"),
                "elapsed_seconds": round(elapsed, 2),
            }

        if status in TERMINAL_FAILURE:
            print(f"[{job_name}] Runpod finished with status: {status}")
            return {
                "status": status,
                "job_id": job_id,
                "data": data,
                "output": data.get("output"),
                "elapsed_seconds": round(elapsed, 2),
            }

        elapsed_minutes = int(elapsed // 60)
        if elapsed_minutes >= 1 and elapsed_minutes > last_minute_logged:
            last_minute_logged = elapsed_minutes
            print(f"[{job_name}] Runpod still waiting... ({elapsed_minutes} min elapsed, status={status})")

        time.sleep(poll_delay_sec)