# ShrimpVision

ShrimpVision is a Flask web application for shrimp detection, counting, and behavior activity analysis using YOLO models. It supports image uploads, video uploads, model switching, real-time video processing, and a simple dashboard for reviewing detection results.

## Live Demo

Try the deployed app here:

```text
https://shrimpvision.onrender.com
```

## Features

- Detect and count shrimp in images and videos.
- Switch between YOLOv5s, YOLOv8s, and YOLOv10s models.
- Stream video analysis results with Server-Sent Events.
- Show annotated image and video frames.
- Estimate shrimp activity from frame-to-frame centroid movement.
- Display summary metrics, charts, and exportable JSON results.
- Ready for deployment on Render with `render.yaml`.

## Project Structure

```text
.
+-- webapp/
|   +-- app.py                  # Flask routes and API entry point
|   +-- inference_engine.py     # YOLO model loading and inference
|   +-- behavior_analyzer.py    # Activity and behavior metrics
|   +-- config.py               # Model paths and app settings
|   +-- requirements.txt        # Python dependencies
|   +-- static/                 # CSS and JavaScript
|   +-- templates/              # HTML template
+-- yolov5su.pt                 # YOLOv5 fallback model
+-- yolov8s.pt                  # YOLOv8 fallback model
+-- yolov10s.pt                 # YOLOv10 fallback model
+-- render.yaml                 # Render deployment config
+-- DEPLOY_RENDER.md            # Render deployment notes
```

## Models

The app uses the model registry in `webapp/config.py`.

| Key | Name | Role |
| --- | --- | --- |
| `yolov5s` | YOLOv5s | Baseline model |
| `yolov8s` | YOLOv8s | Alternative comparison model |
| `yolov10s` | YOLOv10s | Default selected model |

If a trained model exists under `runs/.../weights/best.pt`, the app uses it. Otherwise, it falls back to the `.pt` files stored in the repository root.

## Requirements

- Python 3.11 recommended
- pip
- The model files listed above

Install dependencies from:

```bash
pip install -r webapp/requirements.txt
```

## Local Setup

1. Clone the repository.

```bash
git clone https://github.com/shiniji123/ShrimpVision.git
cd ShrimpVision
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r webapp/requirements.txt
```

4. Choose the inference device.

For CPU on Windows PowerShell:

```powershell
$env:YOLO_DEVICE="cpu"
```

For CPU on macOS/Linux:

```bash
export YOLO_DEVICE=cpu
```

For GPU on Windows PowerShell:

```powershell
$env:YOLO_DEVICE="0"
```

If `YOLO_DEVICE` is not set, the app defaults to GPU device `0`.

5. Start the app.

```bash
python webapp/app.py
```

Open:

```text
http://localhost:5000
```

## API Overview

### Health Check

```http
GET /api/health
```

Returns app status and model availability.

### List Models

```http
GET /api/models
```

Returns available model metadata and cache status.

### Image Detection

```http
POST /api/detect/image
```

Form data:

- `file`: image file
- `model_key`: optional, defaults to `yolov10s`
- `confidence`: optional, defaults to `0.25`

### Video Detection

```http
POST /api/detect/video
```

Form data:

- `file`: video file
- `model_key`: optional, defaults to `yolov10s`
- `confidence`: optional, defaults to `0.25`

The response includes a `job_id` and `stream_url`.

### Video Stream

```http
GET /api/stream/<job_id>
```

Streams frame-level detection and activity data using Server-Sent Events.

### Job Results

```http
GET /api/results/<job_id>
```

Returns the final summary for a completed video job.

## Supported Uploads

Images:

- JPG
- JPEG
- PNG
- WEBP
- BMP
- TIFF

Videos:

- MP4
- MOV
- AVI
- MKV
- WEBM

Maximum upload size is 500 MB.

## Behavior Analysis Notes

ShrimpVision estimates activity from detection centroids across video frames. It calculates metrics such as active ratio, average velocity, velocity variance, and spatial spread.

The behavior output is an activity indicator, not a clinical disease diagnosis. Real shrimp health assessment should be confirmed with proper aquaculture and veterinary methods.

## Deploy to Render

This repository includes `render.yaml`.

Render uses:

```bash
pip install -r webapp/requirements.txt
```

Start command:

```bash
gunicorn --chdir webapp app:app --bind 0.0.0.0:$PORT --timeout 300
```

Environment variables:

```text
PYTHON_VERSION=3.11.9
YOLO_DEVICE=cpu
```

For more details, see `DEPLOY_RENDER.md`.

## Notes

- Uploaded files are saved temporarily and removed after processing.
- Annotated image results are saved in `webapp/results`.
- Video processing is streamed live and summarized after completion.
- Large videos can take longer to process, especially on CPU-only deployments.
