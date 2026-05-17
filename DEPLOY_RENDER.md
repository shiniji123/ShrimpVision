# Deploying ShrimpVision to Render

This project is configured for Render with `render.yaml`.

## Render Blueprint

1. Push this repository to GitHub.
2. In Render, create a new Blueprint from the repository.
3. Render will use:
   - Build command: `pip install -r webapp/requirements.txt`
   - Start command: `gunicorn --chdir webapp app:app --bind 0.0.0.0:$PORT --timeout 300`
   - Environment variable: `YOLO_DEVICE=cpu`

## Manual Web Service

If creating a normal Web Service instead of a Blueprint:

- Runtime: Python 3
- Build Command: `pip install -r webapp/requirements.txt`
- Start Command: `gunicorn --chdir webapp app:app --bind 0.0.0.0:$PORT --timeout 300`
- Environment Variable: `YOLO_DEVICE=cpu`

The `.gitignore` keeps datasets, training runs, reports, and extra model files out of git. The runtime model files kept for the app are:

- `yolov5su.pt`
- `yolov8s.pt`
- `yolov10s.pt`
