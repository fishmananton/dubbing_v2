import requests
import runpod
import torch
import srt
import numpy as np
from scipy.signal import resample_poly
import os, sys, logging, warnings, builtins
import tempfile
import shutil
import soundfile as sf

_EMO_PIPELINE = None

def get_emo_pipeline():
    global _EMO_PIPELINE
    if _EMO_PIPELINE is None:
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["FUNASR_RTF_SHOW"] = "0"
        warnings.filterwarnings("ignore")
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger("modelscope").setLevel(logging.CRITICAL)
        logging.getLogger("funasr").setLevel(logging.CRITICAL)

        try:
            import tqdm
            def silent_tqdm(iterable=None, *args, **kwargs):
                return iterable if iterable is not None else []
            tqdm.tqdm = silent_tqdm
            builtins.tqdm = silent_tqdm
        except ImportError:
            pass

        class DevNull:
            def write(self, *_): pass
            def flush(self): pass

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = DevNull()
        sys.stderr = DevNull()
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            device = "cuda" if torch.cuda.is_available() else "cpu"
            _EMO_PIPELINE = pipeline(
                task=Tasks.emotion_recognition,
                model="/root/.cache/modelscope/iic/emotion2vec_plus_large",
                device=device,
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        log("✅ Emotion model loaded and cached.")

    return _EMO_PIPELINE

def log(msg):
    print(msg, flush=True)  # still goes to RunPod logs

def load_audio_wav(path, target_sr=16000):
    y, sr = sf.read(path, dtype="float32")

    if y.ndim == 2:
        y = np.mean(y, axis=1)

    if sr != target_sr:
        y = resample_poly(y, target_sr, sr).astype(np.float32)
        sr = target_sr

    return y, sr

def download_file(url, path):
    with requests.get(url, stream=True, timeout=(10, 300)) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def modelscope_emotions(audio_file: str, subtitles_file: str):
    # quiet all model output
    emo_pipeline = get_emo_pipeline()



    # load audio and subtitles
    y, sr = load_audio_wav(audio_file, target_sr=16000)
    with open(subtitles_file, encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))

    # ======================
    # INFERENCE PER SUBTITLE
    # ======================
    speakers = {}
    for sub in subs:
        start, end = sub.start.total_seconds(), sub.end.total_seconds()
        clip = y[int(start * sr):int(end * sr)]


        if len(clip) < 0.25 * sr:
            emotion_label = "neutral"
        else:
            result = emo_pipeline(input=clip, sample_rate=sr)[0]
            labels = result["labels"]
            scores = result["scores"]
            top_idx = int(np.argmax(scores))
            emotion = labels[top_idx].split("/")[-1]  # remove any Chinese text like "中立/"
            emotion_label = emotion if emotion != "<unk>" else "neutral"

        # update subtitle text: prepend [emotion]
        if ":" not in sub.content:
            continue
        speaker, _ = sub.content.strip().split(":", 1)
        idx = sub.index
        duration = end - start
        speakers.setdefault(speaker, {"groups": {}})
        speakers[speaker]["groups"].setdefault(
            emotion_label,
            {"idxs": [], "initial_total_sec": 0.0},
        )
        speakers[speaker]["groups"][emotion_label]["idxs"].append(idx)
        speakers[speaker]["groups"][emotion_label]["initial_total_sec"] += duration

    return speakers

def handler(job):
    log("Job started")
    payload = job["input"]
    srt_url = payload["srt_url"]
    audio_url = payload["audio_url"]

    temp_folder = tempfile.mkdtemp(prefix="emotion_")
    try:
        srt_path = os.path.join(temp_folder, "subtitle.srt")
        audio_path = os.path.join(temp_folder, "audio.wav")

        download_file(srt_url, srt_path)
        download_file(audio_url, audio_path)
        log("Downloaded files")

        speakers = modelscope_emotions(audio_path, srt_path)

        log("Emotion detection complete")
        return {"message": "Processing done", "speakers": speakers}
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

