import modal

app = modal.App("emotion_detect")

# Public Docker Hub image example:
image = modal.Image.from_registry("docker.io/delascorpion/emotion-runpod")

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 30,
    startup_timeout=60 * 10,
)
def emotion_job(srt_url: str, audio_url: str):
    from test_emotion import modelscope_emotions, download_file
    import os
    import shutil
    import tempfile

    temp_folder = tempfile.mkdtemp(prefix="emotion_")
    try:
        srt_path = os.path.join(temp_folder, "subtitle.srt")
        audio_path = os.path.join(temp_folder, "audio.wav")

        download_file(srt_url, srt_path)
        download_file(audio_url, audio_path)
        speakers = modelscope_emotions(audio_path, srt_path)
        print("Emotion detection complete")
        return {"message": "Processing done", "speakers": speakers}
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)