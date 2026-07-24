from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import torch
import shutil
import runpod
import requests
import tarfile
from pathlib import Path
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


ckpt_converter = 'openvoice/checkpoints_v2/converter'
device="cuda:0" if torch.cuda.is_available() else "cpu"

_converter = None
_converter_lock = threading.Lock()

def get_converter():
    global _converter
    if _converter is None:
        with _converter_lock:
            if _converter is None:
                log("Loading OpenVoice ToneColorConverter (once)")
                conv = ToneColorConverter(f"{ckpt_converter}/config.json", device=device)
                conv.load_ckpt(f"{ckpt_converter}/checkpoint.pth")
                conv.watermark_model = None
                _converter = conv
    return _converter

def log(msg):
    print(msg, flush=True)  # still goes to RunPod logs

def download_file(url, path):
    with requests.get(url, stream=True, timeout=(10, 300)) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def upload_file(url, path):
    with open(path, "rb") as f:
        r = requests.put(url, data=f, timeout=(10, 300))
        r.raise_for_status()

def extract_tar_archive(archive_path, output_dir):
    """
    Extracts a .tar.gz archive to a new directory.
    """
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=output_dir)
    return output_dir

def make_tar_archive(speaker_name:str, file_list: list, output_dir:str="tmp"):
    output_dir = Path(output_dir)
    archive_path = output_dir / f"{speaker_name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for file_path in file_list:
            tar.add(file_path, arcname=Path(file_path).name)
    return archive_path

def get_se_with_fallback(path, converter):
    try:
        return se_extractor.get_se(path, converter, vad=False)
    except Exception as e:
        log(f"get_se vad=False failed for {path}: {e}. Retrying with vad=True")
        return se_extractor.get_se(path, converter, vad=True)

def process_speaker(speaker: str, emotion: str, value: dict, temp_folder: str, source_se, tone_color_converter):
    log(f"process_speaker started for {speaker} with emotion {emotion}")

    reference_speaker = f"{temp_folder}/{speaker}/{speaker}_{emotion}.wav"

    target_se, _ = get_se_with_fallback(reference_speaker, tone_color_converter)

    file_list = []

    for idx in value["idxs"]:
        input_path = f"{temp_folder}/{speaker}/{idx}.wav"
        output_path = f"{temp_folder}/{speaker}/{idx}_out.wav"

        tone_color_converter.convert(
            audio_src_path=input_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path
        )

        file_list.append(output_path)

    return file_list

def process_speaker_bundle(speaker_name: str, speaker_data: dict, total_sec_thresh: float):
    tone_color_converter = get_converter()

    temp_folder = tempfile.mkdtemp(prefix="openvoice_")
    try:
        file_path = f"{temp_folder}/{speaker_name}.tar.gz"

        download_file(speaker_data["s3_input_obj"], file_path)
        extract_tar_archive(file_path, f"{temp_folder}/{speaker_name}")

        base_speaker = f"{temp_folder}/{speaker_name}/{speaker_name}_combined.wav"
        source_se, _ = get_se_with_fallback(base_speaker, tone_color_converter)

        groups = speaker_data.get("groups", {})
        result_array = []

        for emotion, group_data in groups.items():
            if group_data["total_sec"] >= total_sec_thresh:
                result_array.extend(
                    process_speaker(
                        speaker_name,
                        emotion,
                        group_data,
                        temp_folder,
                        source_se,
                        tone_color_converter,
                    )
                )

        archive_path = make_tar_archive(speaker_name, result_array, temp_folder)
        upload_file(speaker_data["s3_output_obj"], archive_path)

        return {"speaker": speaker_name, "status": "ok"}

    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)



def handler(job):
    payload = job["input"]
    total_sec_thresh = payload.get("total_sec_thresh", 5)
    speakers = payload["speakers"]
    max_workers = payload.get("max_workers", 2)

    valid_speakers = [
        (speaker_name, speaker_data)
        for speaker_name, speaker_data in speakers.items()
        if "s3_input_obj" in speaker_data
    ]

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_speaker = {
            executor.submit(process_speaker_bundle, speaker_name, speaker_data, total_sec_thresh): speaker_name
            for speaker_name, speaker_data in valid_speakers
        }

        for future in as_completed(future_to_speaker):
            speaker_name = future_to_speaker[future]
            try:
                results.append(future.result())
                log(f"Finished speaker {speaker_name}")
            except Exception as e:
                log(f"Speaker {speaker_name} failed: {e}")
                raise

    return {"message": "Processing done", "results": results}





if __name__ == "__main__":
    # import multiprocessing
    # multiprocessing.set_start_method("spawn", force=True)
    runpod.serverless.start({"handler": handler})
