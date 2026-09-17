import modal
import os
import subprocess
import tempfile
import shutil

app = modal.App("audio-dubbing-separator")

vol = modal.Volume.from_name("dubbing-transfer", create_if_missing=True)

audio_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("audio-separator[gpu]", "requests", "soundfile")
    .run_commands(
        "mkdir -p /opt/audio_separator_models",
        "python -c \"from audio_separator.separator import Separator; "
        "sep = Separator(model_file_dir='/opt/audio_separator_models'); "
        "sep.load_model('mel_band_roformer_kim_ft_unwa.ckpt');\""
    )
    .env({"MODELS_DIR": "/opt/audio_separator_models"})
)

VOL_MOUNT = "/vol"


@app.function(
    image=audio_image,
    gpu=["L40S", "A10G"],
    timeout=60 * 30,
    volumes={VOL_MOUNT: vol},
)
def split_audio_job(run_id: str):
    from audio_separator.separator import Separator

    vol_dir = os.path.join(VOL_MOUNT, run_id)
    input_flac = os.path.join(vol_dir, "input.flac")

    temp_dir = tempfile.mkdtemp(prefix="split_")
    try:
        input_wav = os.path.join(temp_dir, "input.wav")
        print("Decompressing FLAC -> WAV...")
        subprocess.run(["ffmpeg", "-y", "-i", input_flac, input_wav], check=True, capture_output=True)

        print("Initializing Separator...")
        separator = Separator(
            model_file_dir=os.environ["MODELS_DIR"],
            output_dir=temp_dir,
            output_format="WAV",
        )
        separator.load_model("mel_band_roformer_kim_ft_unwa.ckpt")

        print("Separating audio...")
        # Sample GPU memory + utilization during separation so we can size batch_size.
        # peak mem/total -> how large a batch fits (size for the smaller GPU, A10G 24GB);
        # peak util% -> whether batching helps (low util = idle GPU = batching wins).
        import threading as _thr
        _peak = {"mem": 0, "total": 0, "util": 0, "util_sum": 0, "n": 0}
        _stop = _thr.Event()

        def _sample_gpu():
            while not _stop.is_set():
                try:
                    out = subprocess.run(
                        ["nvidia-smi",
                         "--query-gpu=memory.used,memory.total,utilization.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
                    used, total, util = [int(x.strip()) for x in out.split(",")]
                    _peak["mem"] = max(_peak["mem"], used)
                    _peak["total"] = total
                    _peak["util"] = max(_peak["util"], util)
                    _peak["util_sum"] += util
                    _peak["n"] += 1
                except Exception:
                    pass
                _stop.wait(0.5)

        _sampler = _thr.Thread(target=_sample_gpu, daemon=True)
        _sampler.start()
        try:
            output_files = separator.separate(input_wav)
        finally:
            _stop.set()
            _sampler.join(timeout=2)
        pct = 100 * _peak["mem"] / max(1, _peak["total"])
        avg_util = _peak["util_sum"] / max(1, _peak["n"])
        print(f"📊 GPU during separate: {_peak['mem']}/{_peak['total']} MiB "
              f"({pct:.0f}% VRAM) | util avg={avg_util:.0f}% peak={_peak['util']}%")

        vocal_filename = next((f for f in output_files if "vocal" in f.lower()), output_files[1])
        music_filename = next((f for f in output_files if "other" in f.lower()), output_files[0])

        vocal_wav = os.path.join(temp_dir, vocal_filename)
        music_wav = os.path.join(temp_dir, music_filename)

        vocal_flac = os.path.join(vol_dir, "vocal.flac")
        music_flac = os.path.join(vol_dir, "music.flac")

        print("Compressing outputs -> FLAC...")
        subprocess.run(["ffmpeg", "-y", "-i", vocal_wav, vocal_flac], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", music_wav, music_flac], check=True, capture_output=True)

        vol.commit()

        os.remove(input_flac)
        vol.commit()

        return {"message": "Processing done"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
