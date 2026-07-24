import modal

app = modal.App("split-audio")

# Public Docker Hub image example:
image = modal.Image.from_registry("docker.io/delascorpion/split-audio-runpod:v2")

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 30,
    startup_timeout=60 * 10,
)
def split_audio_job(input_url: str, output_vocal_url: str, output_music_url: str):
    from split_audio_job import split_audio, download_file, upload_file
    import os
    import shutil
    import tempfile

    temp_dir = tempfile.mkdtemp(prefix="split_")
    try:
        input_path = os.path.join(temp_dir, "input.wav")
        vocal_path = os.path.join(temp_dir, "vocals.wav")
        music_path = os.path.join(temp_dir, "music.wav")

        download_file(input_url, input_path)
        split_audio(input_path, vocal_path, music_path)
        upload_file(output_vocal_url, vocal_path)
        upload_file(output_music_url, music_path)

        return {"message": "Processing done"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)