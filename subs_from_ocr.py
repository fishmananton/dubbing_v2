


import os, subprocess
from langdetect import detect
from difflib import SequenceMatcher
from openai import OpenAI
import shutil
import tarfile
from pathlib import Path
import boto3
import re
import unicodedata
import json
from runpod_utils import run_runpod_job
from modal_utils import run_modal_job


SYSTEM_PROMPT = """You are given subtitles with imperfect speaker diarization.

Your task:
1. Correct speaker assignments based on conversation context and flow.
2. Replace generic speaker labels (e.g., SPEAKER_01, Person_1) with consistent, meaningful role names in English.
3. Ensure the same character always uses the same role name throughout the entire dialogue.
4. Minimize the number of unique roles. Do NOT create unnecessary new roles.
5. Number of speakers is known to be {num_speakers} or less.

Correction rules:
- If a speaker appears for less than 5 seconds total, assume it is a misclassification and reassign those lines to the most contextually appropriate existing role.
- Use dialogue logic (who responds to whom, tone, relationships, etc.) to infer correct speakers.
- Preserve narrative consistency (e.g., family roles, hierarchy, tone).

Output requirements:
- Return STRICT JSON ONLY.
- Do not include explanations or extra text.
- Output format:

[
  {
    "index": <subtitle index>,
    "speaker": "<role_name>"
  }
]
"""



def make_tar_archive(file_name:str, file_list: list, output_dir:str="tmp"):
    output_dir = Path(output_dir)
    archive_path = output_dir / f"{file_name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for file_path in file_list:
            tar.add(file_path, arcname=Path(file_path).name)
    return archive_path


def extract_frames(video_path, interval=0.2, start_time=0, end_time=None,image_quality:int=2,crop_bottom_fraction: float = 0.33):
    # Clean output directory if it exists
    output_folder = tempfile.mkdtemp(prefix="ocr_video_")
    output_pattern = os.path.join(output_folder, "frame_%05d.jpg")

    # Build -vf filter dynamically
    crop_h_expr = f"ih*{crop_bottom_fraction}"
    crop_y_expr = f"ih*(1-{crop_bottom_fraction})"
    vf_parts = [f"fps=1/{interval}", f"crop=iw:{crop_h_expr}:0:{crop_y_expr}"]
    vf_filter = ",".join(vf_parts)

    # Base command
    cmd = ["ffmpeg", "-y"]
    cmd.extend(["-ss", str(start_time)])
    cmd.extend(["-i", video_path])

    # Add end time if given (still useful for stopping decoding)
    if end_time is not None:
        cmd += ["-t", str(end_time - start_time)]

    cmd.extend([
        "-vf", vf_filter,
        "-vsync", "vfr",
        "-qscale:v", str(image_quality),
        output_pattern,
        "-hide_banner",
        "-loglevel", "error"
    ])

    subprocess.run(cmd, check=True)
    print(f"✅ Extracted frames every {interval}s from t={start_time}s to t={end_time or 'EOF'}s into '{output_folder}/'")
    return output_folder



def get_ocr_results(boto_session: boto3.Session, bucket_name:str, runpod_key, runpod_paddleocr_id,  video_path, interval=0.5,run_id='', start_time=0.0, end_time=None):
    s3 = boto_session.client("s3")
    s3_path = f"{run_id}/ocr/input/input.mp4"
    s3.upload_file(video_path, bucket_name, s3_path)
    input_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket_name, "Key": s3_path},
        ExpiresIn=3600)

    runpod_payload = {'input_url': input_url,
                      "interval": interval,
                      "crop_bottom_fraction": 0.33,
                      "image_quality": 2,
                      "batch_size": 20,
                      "start_time": start_time,
                      "end_time": end_time,
                      "visual_diff_threshold": 2,
                      "use_doc_orientation_classify": False,
                      "use_doc_unwarping": False,
                      "use_textline_orientation": False
                      }

    # result = run_runpod_job(
    #     runpod_key=runpod_key,
    #     runpod_template_id=runpod_paddleocr_id,
    #     payload=runpod_payload,
    #     job_name="paddle_ocr",
    #     timeout_minutes=10
    # )
    result = run_modal_job(
        app_name="get_ocr",
        function_name="get_ocr_job",
        timeout_minutes=30,
        poll_delay_sec=5,
        input_url=input_url,
        interval=interval,
        crop_bottom_fraction=0.33,
        image_quality=2,
        batch_size=20,
        start_time=start_time,
        end_time=end_time,
        visual_diff_threshold=2,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )

    if result["status"] == "COMPLETED":
        ocr_result = result['output']['result']
    s3.delete_object(Bucket=bucket_name, Key=s3_path)
    if result["status"] != "COMPLETED":
        raise Exception(f"Didn't get ocr results. job_id {result['job_id']}")
    return ocr_result

def align_subs_with_speakers(
    subs,
    speaker_segments,
    buffer=0.3,
    fallback_nearest=True
):
    diar = list(speaker_segments)
    aligned = []

    for k, sub in enumerate(subs, 1):
        s0 = sub["start"]
        e0 = sub["end"]
        text = sub["text"]

        win_start = s0 - buffer
        win_end = e0 + buffer

        speaker_overlap = {}

        for seg in diar:
            if seg["end"] < win_start or seg["start"] > win_end:
                continue

            overlap_start = max(seg["start"], s0)
            overlap_end = min(seg["end"], e0)
            overlap = max(0.0, overlap_end - overlap_start)

            # if only buffer overlap exists, still allow tiny contribution
            if overlap <= 0:
                overlap_start = max(seg["start"], win_start)
                overlap_end = min(seg["end"], win_end)
                overlap = max(0.0, overlap_end - overlap_start) * 0.25

            if overlap > 0:
                spk = seg["speaker"]
                speaker_overlap[spk] = speaker_overlap.get(spk, 0.0) + overlap

        speaker = None

        if speaker_overlap:
            speaker = max(speaker_overlap.items(), key=lambda x: x[1])[0]

        elif fallback_nearest and diar:
            center = (s0 + e0) / 2
            nearest = min(
                diar,
                key=lambda seg: min(abs(center - seg["start"]), abs(center - seg["end"]))
            )
            speaker = nearest["speaker"]

        if speaker is None:
            speaker = "UNKNOWN"

        aligned.append({
            "index": k,
            "speaker": speaker,
            "start": s0,
            "end": e0,
            "text": text
        })

    return aligned

def format_timestamp(seconds):
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    millis = int((secs - int(secs)) * 1000)
    return f"{int(hrs):02}:{int(mins):02}:{int(secs):02},{millis:03}"

def fix_subtitle_timings(subs, min_duration=0.3):
    fixed = []

    for i, sub in enumerate(sorted(subs, key=lambda x: x["start"])):
        start = sub["start"]
        end = sub["end"]

        # ensure minimum duration
        if end - start < min_duration:
            end = start + min_duration

        # prevent overlap with previous
        if fixed:
            prev = fixed[-1]
            if start < prev["end"]:
                start = prev["end"]
                if end <= start:
                    end = start + min_duration

        fixed.append({
            **sub,
            "start": start,
            "end": end
        })

    return fixed

def write_srt(aligned, path="final.srt"):
    aligned.sort(key=lambda x: x["start"])
    with open(path, "w", encoding="utf-8") as f:
        for i, entry in enumerate(aligned, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(entry['start'])} --> {format_timestamp(entry['end'])}\n")
            f.write(f"{entry['speaker']}: {entry['text']}\n\n")
    print(f"✅ SRT written to {path}")


def detect_language_from_subs(subs, min_chars=200):
    """
    Detect the overall language from a list of subtitle dicts with a 'text' key.
    Combines all text until at least `min_chars` characters are gathered.
    """
    combined = []
    total_len = 0

    for sub in subs:
        # txt = re.sub(r'[^A-Za-zÀ-ÿ\u0400-\u04FF\u0600-\u06FF\s]', '', sub.get("text", ""))
        # txt = re.sub(r'\s+', ' ', txt.strip())
        txt = sub.get("text", "").strip()
        if txt:
            combined.append(txt)
            total_len += len(txt)
        if total_len >= min_chars:
            break

    if not combined:
        return None

    text_block = " ".join(combined)
    try:
        lang_code = detect(text_block)
    except Exception:
        lang_code = None

    return lang_code

def check_language(language):
    lang_conv = {
        # --- Latin / common ---
        "en": "en",
        "fr": "fr",
        "de": "de",
        "es": "es",
        "pt": "pt",
        "it": "it",
        "nl": "nl",
        "sv": "sv",
        "no": "no",
        "da": "da",
        "pl": "pl",
        "cs": "cs",
        "sk": "sk",
        "sl": "sl",
        "hr": "hr",
        "hu": "hu",
        "ro": "ro",
        "tr": "tr",
        "et": "et",
        "lv": "lv",
        "lt": "lt",
        "sq": "sq",
        "sw": "sw",

        # --- Cyrillic ---
        "ru": "ru",
        "uk": "uk",
        "bg": "bg",
        "sr": "rs_cyrillic",  # default to Cyrillic
        "be": "be",

        # --- CJK ---
        "zh": "ch",  # simplified by default
        "ja": "japan",
        "ko": "korean",

        # --- Arabic script ---
        "ar": "ar",
        "fa": "fa",
        "ur": "ur",
        "ug": "ug",

        # --- Indic ---
        "hi": "hi",
        "ta": "ta",
        "te": "te",
        "mr": "mr",
        "ne": "ne",

        # --- Others ---
        "vi": "vi",
        "id": "id",
        "ms": "ms",
        "auto": "auto"
    }
    paddle_lang = lang_conv.get(language)

    if paddle_lang is None:
        raise RuntimeError(f"Language '{language}' not supported for OCR processing.")


def chunk_matches_language(chunk: str, language: str) -> bool:
    if not chunk:
        return False

    if language == "zh":
        return all('\u4e00' <= c <= '\u9fff' for c in chunk)

    if language == "ko":
        return all('\uac00' <= c <= '\ud7af' for c in chunk)

    if language == "ja":
        return all(
            ('\u3040' <= c <= '\u309f') or
            ('\u30a0' <= c <= '\u30ff') or
            ('\u4e00' <= c <= '\u9fff')
            for c in chunk
        )

    if language in {"ru", "uk", "bg", "be", "sr"}:
        return all('\u0400' <= c <= '\u04ff' for c in chunk)

    if language in {"ar", "fa", "ur", "ug"}:
        return all(('\u0600' <= c <= '\u06ff') or ('\u0750' <= c <= '\u077f') for c in chunk)

    if language in {
        "en", "fr", "de", "es", "pt", "it", "nl", "sv", "no", "da",
        "pl", "cs", "sk", "sl", "hr", "hu", "ro", "tr", "et", "lv",
        "lt", "sq", "sw", "vi", "id", "ms"
    }:
        return all("LATIN" in unicodedata.name(c, "") for c in chunk if c.isalpha())

    return False

def has_real_text(text: str, language: str) -> bool:
    if language == "zh":
        return any('\u4e00' <= c <= '\u9fff' for c in text)
    if language == "ko":
        return any('\uac00' <= c <= '\ud7af' for c in text)
    if language == "ja":
        return any(
            ('\u3040' <= c <= '\u309f') or
            ('\u30a0' <= c <= '\u30ff') or
            ('\u4e00' <= c <= '\u9fff')
            for c in text
        )
    return any(c.isalpha() for c in text)

def filter_text_by_language(text: str, language: str) -> str:
    if not text:
        return ""

    chunks = text.split()
    if len(chunks) < 2:
        return ""
    kept = [chunk for chunk in chunks if chunk_matches_language(chunk, language)]
    kept = [chunk for chunk in kept if has_real_text(chunk, language)]
    return " ".join(kept).strip()


def filter_ocr_results_by_language(results, language: str):
    filtered = []

    for item in results:
        clean_text = filter_text_by_language(item.get("text", ""), language)
        if clean_text:
            filtered.append({
                **item,
                "text": clean_text
            })

    return filtered

def normalize_sub_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", "", text)  # remove all spaces
    text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    return text

def should_merge_subs(a, b, max_gap=0.25, sim_thresh=0.9):
    if a["speaker"] != b["speaker"]:
        return False

    gap = b["start"] - a["end"]

    # allow overlap or tiny gap
    if gap > max_gap:
        return False

    ta = normalize_sub_text(a["text"])
    tb = normalize_sub_text(b["text"])

    if not ta or not tb:
        return False

    sim = SequenceMatcher(None, ta, tb).ratio()
    return sim >= sim_thresh

def choose_better_text(text_a: str, text_b: str) -> str:
    na = normalize_sub_text(text_a)
    nb = normalize_sub_text(text_b)

    # same normalized text -> prefer more readable version
    if na == nb:
        if " " in text_b and " " not in text_a:
            return text_b.strip()
        if len(text_b.strip()) > len(text_a.strip()):
            return text_b.strip()
        return text_a.strip()

    # otherwise prefer longer visible text
    return text_b.strip() if len(text_b.strip()) > len(text_a.strip()) else text_a.strip()

def merge_two_subs(a, b):
    return {
        "index": a["index"],
        "speaker": a["speaker"],
        "start": min(a["start"], b["start"]),
        "end": max(a["end"], b["end"]),
        "text": choose_better_text(a["text"], b["text"]),
    }



def merge_duplicate_subs(subs, max_gap=0.25, sim_thresh=0.9):
    if not subs:
        return []

    subs = sorted(subs, key=lambda x: (x["start"], x["end"]))
    merged = [subs[0]]

    for cur in subs[1:]:
        prev = merged[-1]

        if should_merge_subs(prev, cur, max_gap=max_gap, sim_thresh=sim_thresh):
            merged[-1] = merge_two_subs(prev, cur)
        else:
            merged.append(cur)

    for i, sub in enumerate(merged, 1):
        sub["index"] = i

    return merged

def fix_sub_diarization_with_ai(client:OpenAI, model,  srt_res:list,num_speakers:int):
    text = f"Subtitles:\n{json.dumps(srt_res, ensure_ascii=False, indent=2)}"
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.replace("{num_speakers}", f"{num_speakers}")},
            {"role": "user", "content": text},
        ],
    )
    fix_map = {item["index"]: item["speaker"] for item in json.loads(response.choices[0].message.content.strip())}
    corrected = []
    for row in srt_res:
        new_row = row.copy()
        if row.get("index") in fix_map:
            new_row["speaker"] = fix_map[row["index"]]
        corrected.append(new_row)

    return corrected

def filter_speakable_subs(subs: list[dict]) -> list[dict]:
    result = []

    for sub in subs:
        text = sub.get("text", "")
        if text and any(ch.isalpha() for ch in text):
            result.append(sub)

    return result

def process_video_with_subs(video_path:str,boto_session: boto3.Session,client: OpenAI,model: str, bucket_name:str, runpod_key:str, runpod_paddleocr_id:str, speaker_segments:list, subtitles_file:str, language:str,num_speakers:int| None, run_id:str='', interval=0.128, start_time=0.0, end_time=None):
    check_language(language)

    # extract_frames(video_path, frames_folder,interval=interval, start_time=start_time, end_time=end_time)
    ocr_results = get_ocr_results(boto_session=boto_session,
                                  bucket_name=bucket_name,
                                  runpod_key=runpod_key,
                                  runpod_paddleocr_id=runpod_paddleocr_id,
                                  video_path=video_path,
                                  interval=interval,
                                  run_id=run_id,
                                  start_time=start_time,
                                  end_time=end_time)
    filtered_ocr_original_results = filter_ocr_results_by_language(ocr_results, language)
    filtered_ocr_fallback_results = filter_ocr_results_by_language(ocr_results, 'en')
    filtered_ocr_results = filtered_ocr_original_results if len(filtered_ocr_original_results) >= 0.5*len(filtered_ocr_fallback_results) else filtered_ocr_fallback_results

    # lang_code = detect_language_from_subs(ocr_results)
    srt_res = align_subs_with_speakers(filtered_ocr_results, speaker_segments)

    srt_res = merge_duplicate_subs(srt_res, max_gap=0.5, sim_thresh=0.9)
    srt_res = fix_subtitle_timings(srt_res)
    srt_res = filter_speakable_subs(srt_res)
    unique_speakers = len(set(seg["speaker"] for seg in srt_res))
    actual_num_speakers = num_speakers if num_speakers is not None else unique_speakers

    if actual_num_speakers > 1:
        srt_res = fix_sub_diarization_with_ai(client, model,srt_res,num_speakers)

    write_srt(srt_res, subtitles_file)
    return language
