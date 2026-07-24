# ==============================================================
#  OpenVoice RunPod Dockerfile (Slim Optimized)
#  Base: nvidia/cuda:12.8.0-base-ubuntu22.04
# ==============================================================
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# --------------------------------------------------------------
# System setup
# --------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app/openvoice
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN


RUN apt-get update && apt-get install -y \
    ffmpeg git pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libavfilter-dev libswresample-dev \
    aria2 unzip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

WORKDIR /app

# --------------------------------------------------------------
# Clone OpenVoice
# --------------------------------------------------------------
RUN git clone https://github.com/myshell-ai/OpenVoice openvoice
WORKDIR /app/openvoice

# Dependency Fixes (AV, NumPy, faster-whisper)
RUN pip3 install --no-cache-dir "Cython<3.0" "faster-whisper>=1.0.0" "numpy>=1.26.0" && \
    sed -i 's/"faster-whisper==0.9.0",//g' pyproject.toml setup.py || true && \
    sed -i 's/faster-whisper==0.9.0//g' pyproject.toml setup.py || true && \
    sed -i 's/numpy==1.22.0/numpy>=1.26.0/g' pyproject.toml setup.py || true
# --------------------------------------------------------------
# Install OpenVoice + runtime deps
# --------------------------------------------------------------
# Install OpenVoice + runtime deps safely
RUN pip3 install -e . && \
    pip3 install --no-cache-dir soundfile jieba silero runpod

# --------------------------------------------------------------
# Download model checkpoints + resources
# --------------------------------------------------------------
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip \
    -d /app/openvoice -o checkpoints_v2_0417.zip && \
    unzip checkpoints_v2_0417.zip && \
    mv /app/openvoice/checkpoints_v2 /app/openvoice/openvoice/checkpoints_v2 && \
    mv /app/openvoice/resources /app/openvoice/openvoice/resources && \
    rm checkpoints_v2_0417.zip

# --------------------------------------------------------------
# Preload Silero-VAD (for torch.hub cache)
# --------------------------------------------------------------
RUN mkdir -p /root/.cache/torch/hub && \
    cd /root/.cache/torch/hub && \
    aria2c -x 8 -s 8 -k 1M -o silero-vad-master.zip https://github.com/snakers4/silero-vad/archive/master.zip && \
    unzip silero-vad-master.zip && \
    mv silero-vad-master snakers4_silero-vad_master && \
    rm silero-vad-master.zip

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='M4869/WavMark', local_dir_use_symlinks=False); \
snapshot_download(repo_id='Systran/faster-whisper-medium', local_dir_use_symlinks=False)"

# --------------------------------------------------------------
# Cleanup to minimize final size
# --------------------------------------------------------------
RUN apt-get purge -y git pkg-config libavformat-dev libavcodec-dev libavdevice-dev \
                     libavutil-dev libswscale-dev libavfilter-dev libswresample-dev manpages man-db tzdata locales || true && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /usr/share/doc /usr/share/man /tmp/* && \
    rm -rf ~/.cache/pip /root/.cache/pip-tools /root/.cache/torch/hub/logs

# --------------------------------------------------------------
# App setup
# --------------------------------------------------------------
COPY app.py /app/openvoice/
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# --------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------
CMD ["python", "app.py"]