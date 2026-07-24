FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/demucs/.cached_models

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libsndfile1 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN git clone --depth=1 https://github.com/facebookresearch/demucs.git /opt/demucs && \
    pip install --no-deps /opt/demucs && \
    pip install --no-cache-dir \
        scikit-learn \
        dora-search \
        openunmix \
        julius \
        einops \
        lameenc

RUN mkdir -p $TORCH_HOME/checkpoints && \
    python3 -c "from demucs.pretrained import get_model; get_model('htdemucs')"

COPY split_audio_job.py .

RUN rm -rf /root/.cache/pip /opt/demucs/.git && \
    find /opt/conda/lib/python3.*/site-packages -name '__pycache__' -exec rm -rf {} + || true && \
    find /opt/conda/lib/python3.*/site-packages -name 'tests' -type d -exec rm -rf {} + || true && \
    find /opt/conda/lib/python3.*/site-packages -name '*.pyc' -delete || true

CMD ["python", "split_audio_job.py"]