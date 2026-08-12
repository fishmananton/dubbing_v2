import modal
import os
import requests
import tempfile
import shutil

# 1. Define the Modal App
app = modal.App("audio-dubbing-separator")

# 2. Build the Image Natively in Modal (Replaces your Dockerfile!)
# Modal caches these steps on their backend, so it only builds once.
audio_image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install system dependencies
    .apt_install("ffmpeg", "libsndfile1")
    # Install Python dependencies
    .pip_install("audio-separator[gpu]", "requests", "soundfile")
    # Pre-download the models directly into the Modal image cache
    .run_commands(
        "mkdir -p /opt/audio_separator_models",
        "python -c \"from audio_separator.separator import Separator; "
        "sep = Separator(model_file_dir='/opt/audio_separator_models'); "
        "sep.load_model('mel_band_roformer_kim_ft_unwa.ckpt');\""
        # "sep.load_model('UVR-MDX-NET-Inst_HQ_3.onnx'); "
        # "sep.load_model('Kim_Vocal_2.onnx');\""
    )
    # Set the environment variable so our code knows where the models are
    .env({"MODELS_DIR": "/opt/audio_separator_models"})
)


# Helper functions for downloading/uploading
def download_file(url, path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


def upload_file(url, path):
    with open(path, "rb") as f:
        requests.put(url, data=f)


# 3. Define the Serverless GPU Function
@app.function(
    image=audio_image,
    gpu="A10G",  # 24GB VRAM, perfect for RoFormer
    timeout=60 * 30,  # 30 minute timeout
    startup_timeout=60 * 10,
)
def split_audio_job(input_url: str, output_vocal_url: str, output_music_url: str):
    import tempfile
    import shutil
    from audio_separator.separator import Separator

    # Create a temporary directory for processing
    temp_dir = tempfile.mkdtemp(prefix="split_")

    try:
        input_path = os.path.join(temp_dir, "input_audio.wav")
        print("Downloading audio file...")
        download_file(input_url, input_path)

        print("Initializing Separator...")
        # Point to the models we cached during the image build step
        separator = Separator(
            model_file_dir=os.environ["MODELS_DIR"],
            output_dir=temp_dir,
            output_format="WAV"
        )

        # Load the best model for dubbing
        separator.load_model('mel_band_roformer_kim_ft_unwa.ckpt')

        print("Separating audio...")
        # Output is a tuple of (Vocals, Instrumental)
        output_files = separator.separate(input_path)

        vocal_filename = next((f for f in output_files if "vocal" in f.lower()), output_files[1])
        music_filename = next((f for f in output_files if "other" in f.lower()), output_files[0])

        vocal_path = os.path.join(temp_dir, vocal_filename)
        music_path = os.path.join(temp_dir, music_filename)
        print("Uploading results...")
        upload_file(output_vocal_url, vocal_path)
        upload_file(output_music_url, music_path)

        return {"message": "Processing done", "vocals": output_files[0], "music": output_files[1]}

    finally:
        # Clean up temp files to prevent disk space leaks
        shutil.rmtree(temp_dir, ignore_errors=True)