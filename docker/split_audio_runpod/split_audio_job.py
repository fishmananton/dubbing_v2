import os
import requests
import tempfile
import runpod
from audio_separator.separator import Separator

# Global separator instance to keep the model loaded in GPU memory across invocations
_SEPARATOR = None

def get_separator(model_name='mel_band_roformer_kim_ft_unwa.ckpt'):
    global _SEPARATOR
    if _SEPARATOR is None:
        print(f"Initializing Separator and loading {model_name} to VRAM...")

        # Grab the pre-downloaded models directory from the Docker ENV
        model_dir = os.environ.get("MODELS_DIR", "/opt/audio_separator_models")

        _SEPARATOR = Separator(
            model_file_dir=model_dir,  # <-- This is the crucial addition
            output_dir=tempfile.gettempdir(),
            output_format="WAV",
            normalization_threshold=0.9,
            mdxc_overlap=8,
        )
        _SEPARATOR.load_model(model_name)

    return _SEPARATOR


def download_file(url, path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


def upload_file(url, path):
    with open(path, "rb") as f:
        requests.put(url, data=f)


def handler(job):
    temp_folder = tempfile.mkdtemp()
    print("Job started")

    payload = job["input"]
    input_url = payload["input_url"]
    input_path = os.path.join(temp_folder, "input_audio.wav")

    # 1. Download
    download_file(input_url, input_path)
    print("Downloaded file")

    # 2. Separate (This automatically outputs the Vocal and Instrumental stems)
    separator = get_separator()
    print("Separating audio...")

    # audio-separator returns a tuple of the output filenames (Vocals, Instrumental)
    # Output files are automatically saved to the temp_folder (output_dir)
    output_files = separator.separate(input_path)

    vocal_filename = output_files[0]
    music_filename = output_files[1]

    vocal_path = os.path.join(temp_folder, vocal_filename)
    music_path = os.path.join(temp_folder, music_filename)
    print("Audio successfully split")

    # 3. Upload
    output_vocal_url = payload["output_vocal_url"]
    output_music_url = payload["output_music_url"]

    upload_file(output_vocal_url, vocal_path)
    upload_file(output_music_url, music_path)

    # Cleanup local temp files to free space
    os.remove(input_path)
    os.remove(vocal_path)
    os.remove(music_path)

    return {"message": "Processing done", "vocals": vocal_filename, "background": music_filename}


if __name__ == "__main__":
    # Pre-load the model before the server starts receiving requests
    get_separator()
    runpod.serverless.start({"handler": handler})