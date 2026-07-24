FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    MODEL_HOME=/root/.cache/modelscope \
    MODELSCOPE_CACHE=/root/.cache/modelscope \
    TQDM_DISABLE=1 \
    FUNASR_RTF_SHOW=0

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 git \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        "numpy<2" \
        soundfile \
        scipy \
        srt \
        tqdm \
        requests \
        runpod

RUN pip install --no-cache-dir editdistance-s && \
    PYTHON_SITE=$(python -c 'import site; print(site.getsitepackages()[0])') && \
    mkdir -p $PYTHON_SITE/editdistance && \
    echo "from editdistance_s import *" > $PYTHON_SITE/editdistance/__init__.py && \
    echo '0.8.0' > $PYTHON_SITE/editdistance/VERSION

RUN pip install --no-cache-dir \
        "modelscope==1.14.0" \
        "funasr===1.0.27" --no-deps && \
    pip install --no-cache-dir \
        torch_complex \
        omegaconf \
        hydra-core \
        protobuf==3.20.3 \
        sentencepiece \
        "transformers>=4.30" \
        scikit-learn \
        onnx \
        onnxruntime \
        pandas \
        addict \
        yapf \
        gast \
        simplejson \
        "datasets==2.16.1" \
        oss2 \
        "pyarrow==16.1.0" \
        kaldiio \
        "huggingface_hub>=0.24.0,<1.0" \
        librosa

RUN python -c "\
from modelscope.pipelines import pipeline; \
from modelscope.utils.constant import Tasks; \
print('Pre-loading iic/emotion2vec_plus_large...'); \
pipe = pipeline(task=Tasks.emotion_recognition, model='iic/emotion2vec_plus_large'); \
_ = pipe.model; \
print('Model cached successfully at:', pipe.model.model_dir)"

RUN rm -rf /root/.cache/modelscope/ast_indexer || true

RUN rm -rf /root/.cache/pip && \
    find /opt/conda/lib/python3.*/site-packages -name '__pycache__' -exec rm -rf {} + || true

COPY test_emotion.py .

CMD ["python", "test_emotion.py"]