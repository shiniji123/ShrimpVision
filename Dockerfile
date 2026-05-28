FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_DEVICE=cpu \
    PORT=7860 \
    UPLOAD_LIMIT_MB=75 \
    VIDEO_FRAME_SKIP=5 \
    MAX_VIDEO_FRAMES=300 \
    VIDEO_MAX_DIM=960 \
    STREAM_FRAME_EVERY=20 \
    STREAM_JPEG_QUALITY=55 \
    MAX_CACHED_MODELS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY webapp/requirements.txt /app/webapp/requirements.txt
RUN pip install --no-cache-dir -r /app/webapp/requirements.txt

COPY . /app

EXPOSE 7860

CMD ["sh", "-c", "gunicorn --chdir webapp app:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 1 --timeout 300 --max-requests 20 --max-requests-jitter 5"]
