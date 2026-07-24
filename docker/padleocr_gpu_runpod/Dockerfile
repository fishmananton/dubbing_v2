FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY wheels/paddlepaddle_gpu-3.2.1-cp310-cp310-linux_x86_64.whl /tmp/

RUN python -m pip install --no-cache-dir /tmp/paddlepaddle_gpu-3.2.1-cp310-cp310-linux_x86_64.whl \
 && python -m pip install --no-cache-dir paddleocr opencv-python numpy requests runpod \
 && rm -rf /root/.cache/pip /tmp/*

COPY paddlex_models.tar.gz /root/paddlex_models.tar.gz
RUN mkdir -p /root/.paddlex && \
    tar -xzf /root/paddlex_models.tar.gz -C /root/.paddlex && \
    rm /root/paddlex_models.tar.gz && \
    echo "✅ PaddleX official models pre-cached"

COPY getocr.py .

CMD ["python", "getocr.py"]