# Deploying ShrimpVision to Hugging Face Spaces

Hugging Face Spaces is a better free target for this app than Render Free
because the default CPU Basic Space has much more memory for PyTorch/OpenCV
video inference.

## Create the Space

1. Go to Hugging Face and create a new Space.
2. Choose:
   - Space SDK: Docker
   - Visibility: Public or Private
   - Hardware: CPU Basic
3. Push this repository to the Space repository, or copy these files into it:
   - `Dockerfile`
   - `webapp/`
   - `yolov5su.pt`
   - `yolov8s.pt`
   - `yolov10s.pt`

The Dockerfile runs Flask through Gunicorn on port `7860`, which is the port
expected by Spaces.

## Recommended Settings

The Dockerfile already sets conservative defaults:

```text
YOLO_DEVICE=cpu
UPLOAD_LIMIT_MB=75
VIDEO_FRAME_SKIP=5
MAX_VIDEO_FRAMES=300
VIDEO_MAX_DIM=960
MAX_CACHED_MODELS=1
```

If the Space works well and you want better video coverage, raise these slowly:

```text
VIDEO_FRAME_SKIP=3
MAX_VIDEO_FRAMES=600
VIDEO_MAX_DIM=1280
```

Avoid loading many models at once unless you know memory is stable.

## Why Not Render Free?

Render Free is useful for lightweight Flask apps, but this project loads
Ultralytics YOLO, PyTorch, OpenCV, video frames, and model weights. That stack
can exceed 512 MB even before processing a large video. The Render config is
still kept for simple demos, but Hugging Face Spaces is the recommended free
deployment target for video inference.
