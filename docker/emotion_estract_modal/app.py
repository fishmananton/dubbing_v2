import modal
import os
import requests

app = modal.App("audio-dubbing-emotion-batch")

wavlm_image = (
    modal.Image.debian_slim(python_version="3.11")
    # Added soundfile and libsndfile1 to handle audio loading backends smoothly
    .apt_install("libsndfile1")
    .pip_install("torch", "torchaudio", "soundfile", "transformers", "requests", "numpy")
    .env({"HF_HUB_CACHE": "/opt/hf_cache"})
    .run_commands(
        "mkdir -p /opt/hf_cache",
        "python -c \"from transformers import WavLMModel, Wav2Vec2FeatureExtractor; "
        "Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-large'); "
        "WavLMModel.from_pretrained('microsoft/wavlm-large');\""
    )
)


def download_file(url, path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


@app.function(
    image=wavlm_image,
    gpu="A10G",
    timeout=60 * 15,
)
def extract_batch_emotions(full_audio_url: str, segments: list[dict]):
    """
    segments format:
    [
      {"sub_id": "001", "start": 1.25, "end": 4.10},
      {"sub_id": "002", "start": 4.50, "end": 8.00},
      ...
    ]
    """
    import tempfile
    import shutil
    import torch
    import torchaudio
    import io
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel

    # Explicitly enforce soundfile backend to prevent torchcodec fallback errors
    temp_dir = tempfile.mkdtemp()

    try:
        input_path = os.path.join(temp_dir, "vocal_16k.wav")
        print("Downloading full vocal stem...")
        download_file(full_audio_url, input_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load from pre-built local image cache to prevent HF unauthenticated warnings
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            'microsoft/wavlm-large',
            local_files_only=True
        )
        model = WavLMModel.from_pretrained(
            'microsoft/wavlm-large',
            local_files_only=True
        ).to(device)
        model.eval()

        # Load full audio into PyTorch RAM ONCE
        import soundfile as sf
        import numpy as np
        audio_np, sample_rate = sf.read(input_path, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        waveform = torch.from_numpy(audio_np).unsqueeze(0)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        embeddings_dict = {}

        print(f"Processing {len(segments)} subtitle segments in memory...")

        with torch.no_grad():
            for seg in segments:
                sub_id = seg["sub_id"]
                start_frame = int(seg["start"] * 16000)
                end_frame = int(seg["end"] * 16000)

                # Direct memory slicing (instantaneous)
                chunk = waveform[:, start_frame:end_frame]

                # Failsafe for empty or tiny chunks (< 0.1s)
                if chunk.shape[1] < 1600:
                    continue

                inputs = feature_extractor(
                    chunk.squeeze().numpy(),
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs)
                # Output shape: [1, sequence_length, 1024]
                emb = outputs.last_hidden_state.cpu()

                # Serialize tensor to memory bytes (No S3/Disk Upload Needed)
                buffer = io.BytesIO()
                torch.save(emb, buffer)
                embeddings_dict[sub_id] = buffer.getvalue()

        print("Batch extraction complete!")
        # Returns a dict of { "001": b'...binary tensor...', "002": b'...' } directly over API
        return {"status": "success", "embeddings": embeddings_dict}

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)