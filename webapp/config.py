"""
ShrimpVision AI — Application Configuration
=============================================
Central configuration for the Flask app, model paths, and inference settings.
"""

import os
from pathlib import Path


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_optional_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw in (None, "", "none", "None"):
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


class Config:
    """Base configuration."""

    # ── Paths ──────────────────────────────────────────────────
    BASE_DIR = Path(__file__).resolve().parent.parent        # shrime_project/
    WEBAPP_DIR = Path(__file__).resolve().parent              # shrime_project/webapp/
    UPLOAD_FOLDER = WEBAPP_DIR / "uploads"
    RESULTS_FOLDER = WEBAPP_DIR / "results"

    # ── Multi-Model Registry ───────────────────────────────────
    # Each entry maps a model_key -> trained best.pt path (with pretrained fallback)
    MODELS = {
        "yolov5s": {
            "name": "YOLOv5s",
            "trained_path": str(BASE_DIR / "runs" / "shrimp_yolov5s_gpu" / "weights" / "best.pt"),
            "fallback_path": str(BASE_DIR / "yolov5su.pt"),
            "description": "Literature baseline for shrimp post-larvae detection and counting.",
            "architecture": "YOLOv5 one-stage detector",
            "params": "~7.2M",
            "speed_tier": "fast",
            "badge": "Baseline",
            "year": 2020,
        },
        "yolov8s": {
            "name": "YOLOv8s",
            "trained_path": str(BASE_DIR / "runs" / "shrimp_yolov8s_gpu" / "weights" / "best.pt"),
            "fallback_path": str(BASE_DIR / "yolov8s.pt"),
            "description": "Ultralytics YOLOv8 small model used as a practical alternative comparator.",
            "architecture": "YOLOv8 anchor-free detector",
            "params": "~11.2M",
            "speed_tier": "balanced",
            "badge": "Alternative",
            "year": 2023,
        },
        "yolov10s": {
            "name": "YOLOv10s",
            "trained_path": str(BASE_DIR / "runs" / "shrimp_yolov10s_gpu" / "weights" / "best.pt"),
            "fallback_path": str(BASE_DIR / "yolov10s.pt"),
            "description": "Selected model for the current ShrimpVision version based on mAP, counting error, and model size.",
            "architecture": "CSPNet + Dual-label assignment",
            "params": "~8.0M",
            "speed_tier": "fast",
            "badge": "Selected",
            "year": 2024,
        },
    }

    DEFAULT_MODEL = "yolov10s"

    # Legacy aliases (for backward-compat)
    @classmethod
    def get_model_path(cls, model_key: str) -> str:
        """Return the best trained path, falling back if not found."""
        entry = cls.MODELS.get(model_key, cls.MODELS[cls.DEFAULT_MODEL])
        trained = entry["trained_path"]
        if Path(trained).exists():
            return trained
        return entry["fallback_path"]

    # ── Inference Defaults ─────────────────────────────────────
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    IMG_SIZE = 640
    DEVICE = os.environ.get("YOLO_DEVICE", "0")  # Render uses "cpu"; local can use GPU "0"

    # ── Flask ──────────────────────────────────────────────────
    SECRET_KEY = os.environ.get("SECRET_KEY", "shrimp-vision-dev-key-2026")
    UPLOAD_LIMIT_MB = _env_int("UPLOAD_LIMIT_MB", 75, minimum=1)
    MAX_CONTENT_LENGTH = UPLOAD_LIMIT_MB * 1024 * 1024
    ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png", "webp", "bmp", "tiff"}
    ALLOWED_VIDEO_EXT = {"mp4", "mov", "avi", "mkv", "webm"}

    # ── Behavior Analysis ──────────────────────────────────────
    MOVEMENT_THRESHOLD_PX = 1.0     # min pixel displacement to consider "active"
    HEALTH_THRESHOLDS = {
        "healthy": 0.80,             # >=80% active → high-activity behavior
        "warning": 0.50,             # >=50% active → reduced-activity behavior
    }                                # <50% → low-activity behavior

    # ── Video Processing ───────────────────────────────────────
    VIDEO_FRAME_SKIP = _env_int("VIDEO_FRAME_SKIP", 5, minimum=1)
    MAX_VIDEO_FRAMES = _env_optional_int("MAX_VIDEO_FRAMES", 300)
    VIDEO_MAX_DIM = _env_int("VIDEO_MAX_DIM", 960, minimum=320)
    STREAM_FRAME_EVERY = _env_int("STREAM_FRAME_EVERY", 20, minimum=1)
    STREAM_JPEG_QUALITY = _env_int("STREAM_JPEG_QUALITY", 55, minimum=25)
    MAX_CACHED_MODELS = _env_int("MAX_CACHED_MODELS", 1, minimum=1)

    @classmethod
    def init_dirs(cls):
        """Ensure upload/results directories exist."""
        cls.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
