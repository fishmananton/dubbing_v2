import os
import modal

app = modal.App("index-tts-2-5-generator")


# ------------------------------------------------------------------------
# 1. DOWNLOAD & CACHE FUNCTIONS
# ------------------------------------------------------------------------
def download_indextts2_5_weights():
    from huggingface_hub import snapshot_download
    print("Downloading IndexTTS-2.5 weights...")
    snapshot_download(
        repo_id="IndexTeam/IndexTTS-2.5",
        local_dir="/model_cache/indextts2_5",
        ignore_patterns=["*.md", "*.txt"]
    )
    print("Download complete and baked into image!")


def prewarm_indextts2_5_cache():
    from indextts.infer_v2_5 import IndexTTS2
    import numpy as np
    import soundfile as sf

    print("Pre-warming auxiliary models and CUDA graphs...")
    tts = IndexTTS2(
        cfg_path="/model_cache/indextts2_5/config.yaml",
        model_dir="/model_cache/indextts2_5",
        use_bf16=True,
        use_qwen_emo=False
    )

    # Create a tiny dummy audio file to trigger CUDA compilation
    dummy_spk = "/tmp/dummy_spk.wav"
    sf.write(dummy_spk, np.zeros(16000, dtype=np.float32), 16000)

    # Warm-up inference pass
    try:
        tts.infer(
            spk_audio_prompt=dummy_spk,
            text="Warmup pass.",
            lang="EN",
            emo_vector=[0.0] * 8,
            output_path="/tmp/warmup.wav"
        )
    except Exception as e:
        print(f"Warmup pass skipped: {e}")
    print("Pre-warm complete!")


# ------------------------------------------------------------------------
# 2. MODAL ENVIRONMENT SETUP
# ------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch",
        "torchaudio",
        "transformers",
        "soundfile",
        "numpy",
        "huggingface_hub",
        "hf_transfer",
        "ninja",
        "git+https://github.com/index-tts/index-tts.git@main"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Uncomment to reduce flow-matching steps (25→15) for ~40% faster s2mel at slight quality cost:
    # .run_commands("sed -i 's/diffusion_steps = 25/diffusion_steps = 15/' /usr/local/lib/python3.11/site-packages/indextts/infer_v2_5.py")
    .run_function(download_indextts2_5_weights)
    .run_function(prewarm_indextts2_5_cache)
)


# ------------------------------------------------------------------------
# 3. STATEFUL CLOUD GPU CLASS
# ------------------------------------------------------------------------
@app.cls(gpu="L4", image=image, scaledown_window=2)
class IndexTTSGenerator:

    @modal.enter()
    def load_model(self):
        from indextts.infer_v2_5 import IndexTTS2

        os.environ["HF_HUB_OFFLINE"] = "1"

        print("Loading IndexTTS-2.5 into GPU VRAM...")
        self.tts = IndexTTS2(
            cfg_path="/model_cache/indextts2_5/config.yaml",
            model_dir="/model_cache/indextts2_5",
            use_bf16=True,
        )
        print("Model loaded and ready for inference.")

    @modal.method()
    def generate(self, ref_audios: dict, subtitles: list[dict]) -> list[dict]:
        import tempfile
        import time
        import soundfile as sf

        ref_paths = {}
        for i, (speaker, ref_bytes) in enumerate(ref_audios.items()):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix=f"ref_{i}_")
            tmp.write(ref_bytes)
            tmp.close()
            ref_paths[speaker] = tmp.name

        generated = []

        try:
            t0 = time.perf_counter()
            for sub in subtitles:
                out_path = f"/tmp/output_{sub['idx']}.wav"
                spk_path = ref_paths[sub["speaker"]]
                lang = sub.get("lang", "EN")

                try:
                    self.tts.infer(
                        spk_audio_prompt=spk_path,
                        text=sub["text"],
                        lang=lang,
                        emo_vector=sub.get("emo_vector"),
                        duration_factor=sub.get("duration_factor", 1.0),
                        output_path=out_path,
                        use_random=False,
                    )

                    info = sf.info(out_path)
                    with open(out_path, "rb") as f:
                        audio_bytes = f.read()

                    generated.append({
                        "idx": sub["idx"],
                        "audio_bytes": audio_bytes,
                        "audio_len_sec": round(info.duration, 2),
                    })
                except Exception as e:
                    print(f"Error processing sub {sub['idx']}: {e}")
                finally:
                    if os.path.exists(out_path):
                        os.remove(out_path)

            print(f"Processed {len(subtitles)} lines in {time.perf_counter() - t0:.2f}s")

        finally:
            for p in ref_paths.values():
                if os.path.exists(p):
                    os.remove(p)

        return sorted(generated, key=lambda x: x["idx"])