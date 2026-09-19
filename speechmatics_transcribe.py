"""Speechmatics STT front-end. Returns raw word data (ms timings) for the shared
assembler in transcribe_common. Default engine for all non-en/zh/ja languages."""
import json
import os
import subprocess
import tempfile
import time

import requests

MAX_PARALLEL_JOBS = 30

# Parallel-transcription policy (tunable via env). Audio longer than
# CHUNK_MIN_FILE_S is split once per CHUNK_TARGET_S, cutting at the midpoint of
# the first silence gap >= CHUNK_MIN_GAP_S past each mark, with CHUNK_OVERLAP_S of
# neighbour context deduplicated afterward. Shorter audio uses a single job.
CHUNK_MIN_FILE_S = float(os.environ.get("SPEECHMATICS_CHUNK_MIN_FILE_S", "900"))   # 15 min
CHUNK_TARGET_S = float(os.environ.get("SPEECHMATICS_CHUNK_TARGET_S", "600"))       # 10 min
CHUNK_MIN_GAP_S = float(os.environ.get("SPEECHMATICS_CHUNK_MIN_GAP_S", "5"))
CHUNK_OVERLAP_S = float(os.environ.get("SPEECHMATICS_CHUNK_OVERLAP_S", "20"))

# HTTP resilience for the Speechmatics REST API.
HTTP_MAX_ATTEMPTS = int(os.environ.get("SPEECHMATICS_HTTP_MAX_ATTEMPTS", "5"))
HTTP_BASE_DELAY_S = float(os.environ.get("SPEECHMATICS_HTTP_BASE_DELAY_S", "2"))
POLL_MAX_S = float(os.environ.get("SPEECHMATICS_POLL_MAX_S", "1800"))              # 30 min cap
UPLOAD_TIMEOUT_S = float(os.environ.get("SPEECHMATICS_UPLOAD_TIMEOUT_S", "600"))


def _retryable(exc: Exception | None = None, status: int | None = None) -> bool:
    """Transient failures worth retrying: network timeouts/connection drops, and
    HTTP 429 (rate limit) or 5xx (server) responses. Client 4xx (bad request,
    auth) are not retried."""
    if exc is not None:
        return isinstance(exc, (requests.exceptions.Timeout,
                                requests.exceptions.ConnectionError))
    if status is not None:
        return status == 429 or 500 <= status < 600
    return False


def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
    """Exponential backoff (base * 2^(attempt-1)), overridden by a numeric
    Retry-After header when the server supplies one."""
    if retry_after:
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            pass
    return HTTP_BASE_DELAY_S * (2 ** (attempt - 1))


def _post_job_with_retries(url, headers, upload_path, job_config):
    """Create a job, REOPENING the upload file on every attempt. A retry must not
    reuse an exhausted file handle: the earlier version did, so a transient upload
    error (e.g. write timeout on a large file) caused the retry to send an empty
    body and get a 400. Retries on transient network/HTTP failures with backoff."""
    last_exc = None
    for attempt in range(1, HTTP_MAX_ATTEMPTS + 1):
        try:
            with open(upload_path, "rb") as f:
                files = {
                    "data_file": (os.path.basename(upload_path), f, "application/octet-stream"),
                    "config": (None, json.dumps(job_config), "application/json"),
                }
                resp = requests.post(url, headers=headers, files=files, timeout=(10, UPLOAD_TIMEOUT_S))
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < HTTP_MAX_ATTEMPTS and _retryable(exc=e):
                print(f"  [speechmatics] create job upload error ({type(e).__name__}), "
                      f"retry {attempt}/{HTTP_MAX_ATTEMPTS - 1}")
                time.sleep(_retry_delay(attempt))
                continue
            raise
        if resp.status_code >= 400:
            if _retryable(status=resp.status_code) and attempt < HTTP_MAX_ATTEMPTS:
                print(f"  [speechmatics] create job HTTP {resp.status_code}, "
                      f"retry {attempt}/{HTTP_MAX_ATTEMPTS - 1}")
                time.sleep(_retry_delay(attempt, resp.headers.get("Retry-After")))
                continue
            raise requests.exceptions.HTTPError(
                f"Speechmatics create job failed: HTTP {resp.status_code} — {resp.text[:800]}",
                response=resp,
            )
        return resp
    raise last_exc  # pragma: no cover


def _request_with_retries(fn, what: str):
    """Call fn() (which issues one request and returns a Response), retrying on
    transient network/HTTP failures with backoff. Raises on non-retryable errors
    or once attempts are exhausted."""
    last_exc = None
    for attempt in range(1, HTTP_MAX_ATTEMPTS + 1):
        try:
            resp = fn()
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < HTTP_MAX_ATTEMPTS and _retryable(exc=e):
                time.sleep(_retry_delay(attempt))
                continue
            raise
        if resp.status_code >= 400:
            if _retryable(status=resp.status_code) and attempt < HTTP_MAX_ATTEMPTS:
                print(f"  [speechmatics] {what} HTTP {resp.status_code}, retry {attempt}/{HTTP_MAX_ATTEMPTS - 1}")
                time.sleep(_retry_delay(attempt, resp.headers.get("Retry-After")))
                continue
            # Non-retryable (or exhausted): surface the API's explanation, not just the status line.
            raise requests.exceptions.HTTPError(
                f"Speechmatics {what} failed: HTTP {resp.status_code} — {resp.text[:800]}",
                response=resp,
            )
        return resp
    raise last_exc  # pragma: no cover

BASE_URL = "https://asr.api.speechmatics.com/v2"

LANGUAGE_MAP = {
    "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it", "pt": "pt",
    "nl": "nl", "sv": "sv", "da": "da", "no": "no", "fi": "fi", "pl": "pl",
    "ru": "ru", "zh": "cmn", "ja": "ja", "ko": "ko", "ar": "ar", "hi": "hi",
    "tr": "tr", "uk": "uk", "cs": "cs", "ro": "ro", "hu": "hu", "el": "el",
    "he": "he", "th": "th", "vi": "vi", "id": "id", "ms": "ms", "bg": "bg",
    "hr": "hr", "sk": "sk", "sl": "sl", "lt": "lt", "lv": "lv", "et": "et",
    "ca": "ca", "gl": "gl", "eu": "eu",
}


def speechmatics_transcribe_raw(audio_file_raw: str, speechmatics_api_key: str, language: str = "auto"):
    """Returns (words_data, trans_language) with ms start/end. Punctuation tokens
    are appended to the preceding word so the shared assembler can split on
    sentence ends."""
    t0 = time.time()
    headers = {"Authorization": f"Bearer {speechmatics_api_key}"}

    transcription_config = {"language": "auto", "model": "enhanced", "enable_entities": True}
    if language != "auto":
        transcription_config["language"] = LANGUAGE_MAP.get(language, language)

    job_config = {"type": "transcription", "transcription_config": transcription_config}

    # Downsample to 16k mono + FLAC before upload. Speechmatics downsamples to
    # 16k mono internally anyway, so on the full mix this is effectively lossless
    # (same transcript) while shrinking the upload ~12x (123MB -> ~10MB). One
    # ffmpeg pass, no extra step: large slices now upload fast enough to avoid
    # write timeouts and concurrent chunk uploads don't starve each other.
    upload_flac = tempfile.mktemp(suffix=".flac")
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", audio_file_raw,
                        "-ar", "16000", "-ac", "1", upload_flac],
                       check=True, capture_output=True)
        response = _post_job_with_retries(f"{BASE_URL}/jobs", headers, upload_flac, job_config)
    finally:
        if os.path.exists(upload_flac):
            os.remove(upload_flac)
    job_id = response.json()["id"]

    poll_count = 0
    while True:
        response = _request_with_retries(
            lambda: requests.get(f"{BASE_URL}/jobs/{job_id}", headers=headers, timeout=(10, 60)),
            "poll status",
        )
        job_status = response.json()["job"]
        if job_status["status"] == "done":
            break
        elif job_status["status"] == "rejected":
            raise RuntimeError(f"Speechmatics rejected: {job_status.get('errors', '')}")
        if time.time() - t0 > POLL_MAX_S:
            raise TimeoutError(f"Speechmatics job {job_id} not done after {POLL_MAX_S:.0f}s")
        poll_count += 1
        time.sleep(3)

    response = _request_with_retries(
        lambda: requests.get(
            f"{BASE_URL}/jobs/{job_id}/transcript",
            headers=headers,
            params={"format": "json-v2"},
            timeout=(10, 120),
        ),
        "get transcript",
    )
    transcript = response.json()

    try:
        trans_language = transcript["metadata"]["language_identification"]["predicted_language"]
    except (KeyError, TypeError):
        trans_language = language
    if trans_language == "auto":
        trans_language = "en"
    reverse_map = {v: k for k, v in LANGUAGE_MAP.items()}
    trans_language = reverse_map.get(trans_language, trans_language.split("-")[0])

    words_data = []
    for result in transcript.get("results", []):
        rtype = result.get("type")
        alt = result["alternatives"][0]
        content = alt["content"]
        if rtype == "word":
            words_data.append({
                "text": content,
                "start": int(round(float(result["start_time"]) * 1000)),
                "end": int(round(float(result["end_time"]) * 1000)),
                "speaker": None,
            })
        elif rtype == "entity":
            # enable_entities emits numbers/times/dates as a single "entity" whose
            # alternatives[0].content is the written form ("8:45"). We want the
            # spoken form the TTS can read ("eight forty five"), matching how
            # AssemblyAI behaves with format_text=False. spoken_form is a list of
            # word tokens with their own timings.
            spoken = result.get("spoken_form")
            if spoken:
                for tok in spoken:
                    words_data.append({
                        "text": tok["alternatives"][0]["content"],
                        "start": int(round(float(tok["start_time"]) * 1000)),
                        "end": int(round(float(tok["end_time"]) * 1000)),
                        "speaker": None,
                    })
            else:
                words_data.append({
                    "text": content,
                    "start": int(round(float(result["start_time"]) * 1000)),
                    "end": int(round(float(result["end_time"]) * 1000)),
                    "speaker": None,
                })
        elif rtype == "punctuation" and words_data:
            words_data[-1]["text"] += content
            words_data[-1]["end"] = int(round(float(result["end_time"]) * 1000))

    print(f"  [speechmatics] got {len(words_data)} words, language: {trans_language} "
          f"({poll_count} polls, {time.time() - t0:.1f}s)")
    return words_data, trans_language


def plan_transcribe_chunks(speech_segments, total_dur_s, target_s=300.0, min_gap_s=2.0):
    """Plan silence-aligned chunk boundaries for parallel transcription.

    Splits the audio into ~target_s parts, cutting at the midpoint of the FIRST
    silence gap >= min_gap_s that starts at/after each running mark (previous cut
    + target_s). Every chunk is therefore >= target_s. If no qualifying gap
    remains before the end, the final chunk runs to total_dur_s.

    speech_segments: list of {"start": s, "end": s} (seconds), speech regions.
    Returns a list of (start_s, end_s) tuples covering [0, total_dur_s].
    """
    segs = sorted(speech_segments, key=lambda s: s["start"])
    # interior silence gaps between consecutive speech regions
    gaps = [(segs[i]["end"], segs[i + 1]["start"]) for i in range(len(segs) - 1)]

    boundaries = [0.0]
    prev = 0.0
    gi = 0
    while True:
        mark = prev + target_s
        if mark >= total_dur_s:
            break
        cut = None
        while gi < len(gaps):
            gstart, gend = gaps[gi]
            gi += 1
            if gstart >= mark and (gend - gstart) >= min_gap_s:
                cut = (gstart + gend) / 2.0
                break
        if cut is None:
            break
        boundaries.append(cut)
        prev = cut
    boundaries.append(total_dur_s)

    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def speechmatics_transcribe_raw_chunked(audio_file_raw, speechmatics_api_key,
                                        speech_segments, language="auto",
                                        target_s=CHUNK_TARGET_S, min_gap_s=CHUNK_MIN_GAP_S,
                                        overlap_s=CHUNK_OVERLAP_S,
                                        max_parallel=MAX_PARALLEL_JOBS):
    """Split the audio at silence, transcribe chunks concurrently, merge on the
    absolute timeline. Each chunk's transcription window is extended by overlap_s
    on its interior sides for cross-boundary context; words are then kept by
    core-region ownership so the overlap is deduplicated losslessly. Falls back
    to a single job when there is only one chunk (short audio or no qualifying
    silence). Same (words_data, trans_language) contract as
    speechmatics_transcribe_raw."""
    import soundfile as sf
    from parallel import parallel_map

    total_dur_s = sf.info(audio_file_raw).duration

    # Only split long files; short audio isn't worth the overlap cost / boundary risk.
    if total_dur_s < CHUNK_MIN_FILE_S:
        return speechmatics_transcribe_raw(audio_file_raw, speechmatics_api_key, language)

    core = plan_transcribe_chunks(speech_segments or [], total_dur_s,
                                  target_s=target_s, min_gap_s=min_gap_s)

    if len(core) <= 1:
        return speechmatics_transcribe_raw(audio_file_raw, speechmatics_api_key, language)

    print(f"  [speechmatics] chunked transcribe: {len(core)} parts, overlap={overlap_s}s "
          f"(<= {min(max_parallel, len(core))} parallel)")

    def _do_chunk(chunk):
        core_start_s, core_end_s = chunk
        win_start = max(0.0, core_start_s - overlap_s)
        win_end = min(total_dur_s, core_end_s + overlap_s)
        slice_path = tempfile.mktemp(suffix=".wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", audio_file_raw,
                 "-ss", f"{win_start}", "-to", f"{win_end}", "-c:a", "pcm_s16le", slice_path],
                check=True, capture_output=True,
            )
            words, lang = speechmatics_transcribe_raw(slice_path, speechmatics_api_key, language)
        finally:
            if os.path.exists(slice_path):
                os.remove(slice_path)
        return (int(round(core_start_s * 1000)), int(round(core_end_s * 1000)),
                int(round(win_start * 1000)), words, lang)

    results = parallel_map(_do_chunk, core, max_workers=min(max_parallel, len(core)))
    merged = dedup_overlap_words([(cs, ce, off, words) for cs, ce, off, words, _ in results])
    trans_language = results[0][4]
    print(f"  [speechmatics] merged {len(merged)} words from {len(core)} chunks")
    return merged, trans_language


def merge_chunk_words(chunk_results):
    """Merge per-chunk word lists into one absolute-timeline list.

    chunk_results: list of (offset_ms, words_data) in chunk order. Each word's
    ms start/end is shifted by its chunk's offset, then all words are
    concatenated. Extra fields on each word are preserved.
    """
    merged = []
    for offset_ms, words in chunk_results:
        for w in words:
            shifted = dict(w)
            shifted["start"] = w["start"] + offset_ms
            shifted["end"] = w["end"] + offset_ms
            merged.append(shifted)
    return merged


def dedup_overlap_words(chunk_results):
    """Merge overlapping-window chunks, keeping each word by core-region ownership.

    chunk_results: list of (core_start_ms, core_end_ms, offset_ms, words_data) in
    order. Each chunk was transcribed over a window that overlaps its neighbours
    (for context), but a word is kept only if its absolute start falls in that
    chunk's [core_start_ms, core_end_ms) region. Since core boundaries sit in
    silence, no word straddles a boundary, so this is lossless and
    duplicate-free. Word ms times are shifted to the absolute timeline.
    """
    merged = []
    for core_start_ms, core_end_ms, offset_ms, words in chunk_results:
        for w in words:
            abs_start = w["start"] + offset_ms
            if core_start_ms <= abs_start < core_end_ms:
                shifted = dict(w)
                shifted["start"] = abs_start
                shifted["end"] = w["end"] + offset_ms
                merged.append(shifted)
    return merged
