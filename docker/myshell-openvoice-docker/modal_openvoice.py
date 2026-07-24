import modal

app = modal.App("openvoice")

image = modal.Image.from_registry("docker.io/delascorpion/openvoice-runpod:2")


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 30,
    startup_timeout=60 * 10,
)
def openvoice_job(
    speakers: dict,
    total_sec_thresh: float = 5,
    max_workers: int = 2,
):
    from app import process_speaker_bundle, log
    from concurrent.futures import ThreadPoolExecutor, as_completed

    valid_speakers = [
        (speaker_name, speaker_data)
        for speaker_name, speaker_data in speakers.items()
        if "s3_input_obj" in speaker_data
    ]

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_speaker = {
            executor.submit(
                process_speaker_bundle,
                speaker_name,
                speaker_data,
                total_sec_thresh,
            ): speaker_name
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