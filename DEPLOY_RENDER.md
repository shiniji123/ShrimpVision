# Deploying ShrimpVision to Render

This project is configured for Render with `render.yaml`.

## Render Blueprint

1. Push this repository to GitHub.
2. In Render, create a new Blueprint from the repository.
3. Render will use:
   - Build command: `pip install -r webapp/requirements.txt`
   - Start command: `gunicorn --chdir webapp app:app --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 300 --max-requests 20 --max-requests-jitter 5`
   - Environment variable: `YOLO_DEVICE=cpu`
   - The free Render plan has a 512 MB memory limit. This app is configured to
     keep video processing lightweight by limiting upload size, sampling frames,
     resizing large frames before inference, and caching only one model at a time.

## Manual Web Service

If creating a normal Web Service instead of a Blueprint:

- Runtime: Python 3
- Build Command: `pip install -r webapp/requirements.txt`
- Start Command: `gunicorn --chdir webapp app:app --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 300 --max-requests 20 --max-requests-jitter 5`
- Environment Variable: `YOLO_DEVICE=cpu`
- Recommended memory controls:
  - `UPLOAD_LIMIT_MB=75`
  - `VIDEO_FRAME_SKIP=5`
  - `MAX_VIDEO_FRAMES=300`
  - `VIDEO_MAX_DIM=960`
  - `MAX_CACHED_MODELS=1`

If video inference still crashes with "used over 512MB", upgrade the Render
service plan. YOLO plus PyTorch/OpenCV can exceed 512 MB even with small videos.

The `.gitignore` keeps datasets, training runs, reports, and extra model files out of git. The runtime model files kept for the app are:

- `yolov5su.pt`
- `yolov8s.pt`
- `yolov10s.pt`
