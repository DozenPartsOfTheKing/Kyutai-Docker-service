FROM python:3.12-slim

# System deps for audio processing
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy dependency list & install first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard] \
    && pip install --no-cache-dir huggingface_hub \
    && python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("kyutai/tts-1.6b-en_fr", local_dir="/opt/models/tts", local_dir_use_symlinks=False)
snapshot_download("kyutai/tts-voices", local_dir="/opt/models/voices", local_dir_use_symlinks=False)
PY

# Tell HF libraries to look here first
ENV HF_HOME=/opt/models

# Copy source
COPY . .

# Default port
ENV PORT=8000
ENV WORKERS=2
# Tune Python & Uvicorn
ENV UVICORN_TIMEOUT=120

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers $WORKERS --loop uvloop --timeout-keep-alive ${UVICORN_TIMEOUT}"] 