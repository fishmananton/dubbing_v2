from paddleocr import PaddleOCR
import runpod
import requests
import glob
import os
from difflib import SequenceMatcher
import paddle
import subprocess
import tempfile
import cv2
import numpy as np

# global variable, initially None
ocr_cache = {}

def frame_signature(image_path, size=(32, 32)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def visual_diff(a, b):
    return float(np.mean(np.abs(a - b)))


def filter_visually_duplicate_frames(img_paths, diff_threshold=2.0):
    print (f"🔍 Filtering visually duplicate frames. Total frames before filtering: {len(img_paths)}")
    if not img_paths:
        return []

    kept = []
    prev_sig = None

    for img_path in img_paths:
        sig = frame_signature(img_path)

        if prev_sig is None:
            kept.append(img_path)
            prev_sig = sig
            continue

        diff = visual_diff(prev_sig, sig)
        if diff >= diff_threshold:
            kept.append(img_path)
            prev_sig = sig
    print (f"✅ Frames after filtering: {len(kept)}")
    return kept


def get_ocr_instance(params):
    key = tuple(sorted(params.items()))
    if key not in ocr_cache:
        print(f"🔧 Initializing OCR with params: {params}")
        ocr_cache[key] = PaddleOCR(**params)
    else:
        print(f"✅ Reusing OCR instance for params: {params}")
    return ocr_cache[key]


def download_file(url, path):
    print(f"Starting file download from {url} to {path}...")
    with requests.get(url, stream=True, timeout=(10, 300)) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"File downloaded successfully to {path}")



def batch(iterable, batch_size):
    """Yield successive batches of size `batch_size`."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]





def process_batch(images_batch, ocr_instance, interval=0.5, start_time=0.0):

    ocr_results = ocr_instance.predict(images_batch)

    results = []
    for block in ocr_results:
        if isinstance(block, dict) and "input_path" in block:
            frame_index = int(os.path.basename(block["input_path"]).split("_")[1].split(".")[0])
            frame_sec = start_time + (frame_index - 1) * interval
        else:
            frame_sec = 0.0

        if isinstance(block, dict) and "rec_texts" in block:
            # rec_texts is typically a list of strings
            rec_texts = block["rec_texts"]
            if isinstance(rec_texts, list):
                text = " ".join(rec_texts).strip()
            elif isinstance(rec_texts, str):
                text = rec_texts.strip()
            else:
                text = ""
        elif isinstance(block, str):
            # mobile model output
            text = block.strip()
        else:
            text = ""

        if text:
            results.append({"time": frame_sec, "text": text})

    return results

def _norm(s: str) -> str:
    # Normalize text for comparison
    return " ".join(s.strip().lower().split())
def deduplicate_srts(srt_list, similarity_thresh: float = 0.9, frame_interval: float = 0.5):
    if not srt_list:
        return []

    merged = []

    cur = {
        "start": srt_list[0]["time"],
        "end": srt_list[0]["time"] + frame_interval,
        "text": srt_list[0]["text"],
    }
    cur_norm = _norm(cur["text"])

    for snap in srt_list[1:]:
        t = snap["time"]
        txt = snap["text"]
        txt_norm = _norm(txt)

        # Fast path for exact normalized match
        if cur_norm == txt_norm:
            cur["end"] = t + frame_interval
            if len(txt) > len(cur["text"]):
                cur["text"] = txt
                cur_norm = txt_norm
            continue

        sim = SequenceMatcher(None, cur_norm, txt_norm).ratio()

        if sim >= similarity_thresh:
            # Extend duration through the current sampled frame window
            cur["end"] = t + frame_interval

            # Keep the more complete / longer text variant
            if len(txt) > len(cur["text"]):
                cur["text"] = txt
                cur_norm = txt_norm
        else:
            merged.append(cur)
            cur = {
                "start": t,
                "end": t + frame_interval,
                "text": txt,
            }
            cur_norm = txt_norm

    merged.append(cur)
    return merged



def extract_frames(video_path,output_folder, interval=0.2, start_time=0, end_time=None,image_quality:int=2,crop_bottom_fraction: float = 0.33):
    print (f"🎬 Extracting frames...")
    # Clean output directory if it exists
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


def handler(job):
    if paddle.device.is_compiled_with_cuda():
        gpu_count = paddle.device.cuda.device_count()
        print(f"🚀 Paddle is using CUDA. {gpu_count} GPU(s) detected.")
    else:
        print("⚠️ Paddle is running on CPU — no CUDA detected.")


    payload = job["input"]
    input_url = payload.get("input_url", '')
    interval = payload.get("interval", 0.125)
    crop_bottom_fraction = payload.get("crop_bottom_fraction", 0.33)
    image_quality = payload.get("image_quality", 2)
    batch_size = payload.get("batch_size", 20)
    start_time = payload.get("start_time", 0.0)
    end_time = payload.get("end_time", None)
    visual_diff_threshold = payload.get("visual_diff_threshold", 2)
    use_doc_orientation_classify = payload.get("use_doc_orientation_classify", False)
    use_doc_unwarping = payload.get("use_doc_unwarping", False)
    use_textline_orientation = payload.get("use_textline_orientation", False)


    output_folder = tempfile.mkdtemp(prefix="ocr_video_")
    frames_folder = os.path.join(output_folder, "frames")
    os.makedirs(frames_folder, exist_ok=True)
    file_path = os.path.join(output_folder, "input.mp4")
    download_file(input_url, file_path)
    params = {
        "use_doc_orientation_classify": use_doc_orientation_classify,
        "use_doc_unwarping": use_doc_unwarping,
        "use_textline_orientation": use_textline_orientation,
    }
    if "lang" in payload and payload["lang"] is not None:
        params["lang"] = payload["lang"]
    extract_frames(video_path=file_path,
                                   output_folder=frames_folder,
                                   image_quality=image_quality,
                                   interval=interval,
                                   start_time=start_time,
                                   end_time=end_time,
                                   crop_bottom_fraction=crop_bottom_fraction)
    ocr_instance = get_ocr_instance(params)
    img_paths = sorted(glob.glob(os.path.join(frames_folder, "*.jpg")))
    img_paths = filter_visually_duplicate_frames(img_paths, diff_threshold=visual_diff_threshold)
    all_results = []
    for batch_num, img_batch in enumerate(batch(img_paths, batch_size), start=1):
        print(f"📦 Processing batch {batch_num} from {((len(img_paths)-1)//batch_size)+1}...")
        batch_results = process_batch(img_batch, ocr_instance, interval, start_time=start_time)
        all_results.extend(batch_results)
    clear_res = deduplicate_srts(srt_list = all_results, frame_interval= interval)
    return {"message": "Processing done", "result": clear_res}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

