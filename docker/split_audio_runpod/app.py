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
        output_files = separator.separate(input_wav)

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
