"""
ShrimpVision — Flask Application
=====================================
Main entry point for the web application.
Serves the SPA, handles uploads, runs multi-model inference, and streams real-time results.
"""

import os
import json
import uuid
import logging
import threading
import time
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    Response,
)
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
from inference_engine import InferenceEngine
from behavior_analyzer import BehaviorAnalyzer

# ── App Setup ──────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)
Config.init_dirs()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# In-memory job store (for video processing status)
_jobs: dict = {}
_jobs_lock = threading.Lock()

# Shared inference engine (multi-model cache)
engine = InferenceEngine()


# ── Utility ────────────────────────────────────────────────────

def _allowed_file(filename: str) -> str | None:
    """Return 'image' or 'video' if the extension is allowed, else None."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in Config.ALLOWED_IMAGE_EXT:
        return "image"
    if ext in Config.ALLOWED_VIDEO_EXT:
        return "video"
    return None


def _save_upload(file) -> tuple[str, str]:
    """Save uploaded file with a unique name. Returns (path, file_type)."""
    original = file.filename
    file_type = _allowed_file(original)
    if not file_type:
        raise ValueError(f"Unsupported file type: {original}")

    ext = original.rsplit(".", 1)[-1].lower()
    unique_name = f"{uuid.uuid4().hex[:12]}.{ext}"
    save_path = str(Config.UPLOAD_FOLDER / unique_name)
    file.save(save_path)
    return save_path, file_type


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error):
    return jsonify({
        "error": f"File too large. Maximum upload size is {Config.UPLOAD_LIMIT_MB} MB.",
    }), 413


def _parse_model_key(request_obj) -> str:
    """Extract and validate model_key from form data or JSON body."""
    model_key = (
        request_obj.form.get("model_key")
        or (request_obj.get_json(silent=True) or {}).get("model_key")
        or Config.DEFAULT_MODEL
    )
    if model_key not in Config.MODELS:
        logger.warning("Unknown model_key '%s', falling back to default.", model_key)
        model_key = Config.DEFAULT_MODEL
    return model_key


def _parse_confidence(request_obj) -> float:
    """Extract confidence threshold from form data with bounds check."""
    try:
        val = float(
            request_obj.form.get("confidence")
            or (request_obj.get_json(silent=True) or {}).get("confidence")
            or Config.CONFIDENCE_THRESHOLD
        )
        return max(0.05, min(0.95, val))   # clamp to [0.05, 0.95]
    except (TypeError, ValueError):
        return Config.CONFIDENCE_THRESHOLD


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

# ── Pages ──────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the single-page application."""
    return render_template("index.html", static_version=int(time.time()))


# ── Model Metadata ─────────────────────────────────────────────

@app.route("/api/models")
def list_models():
    """
    Return available models with metadata and availability status.

    Response:
        {
            "default": "yolo26s",
            "models": [
                {
                    "key": "yolo26s",
                    "name": "YOLO26s",
                    "description": "...",
                    "architecture": "...",
                    "params": "~10.2M",
                    "speed_tier": "balanced",
                    "badge": "Aquaculture",
                    "year": 2025,
                    "available": true,
                },
                ...
            ]
        }
    """
    models_out = []
    for key, meta in Config.MODELS.items():
        trained_exists = Path(meta["trained_path"]).exists()
        fallback_exists = Path(meta["fallback_path"]).exists()
        available = trained_exists or fallback_exists
        active_path = meta["trained_path"] if trained_exists else meta["fallback_path"]

        models_out.append({
            "key": key,
            "name": meta["name"],
            "description": meta["description"],
            "architecture": meta["architecture"],
            "params": meta["params"],
            "speed_tier": meta["speed_tier"],
            "badge": meta["badge"],
            "year": meta["year"],
            "available": available,
            "trained": trained_exists,
            "active_path": active_path if available else None,
            "cached": key in engine.loaded_models(),
        })

    return jsonify({
        "default": Config.DEFAULT_MODEL,
        "models": models_out,
    })


# ── Static Results ─────────────────────────────────────────────

@app.route("/api/results/image/<filename>")
def serve_result_image(filename):
    """Serve annotated result images."""
    return send_from_directory(str(Config.RESULTS_FOLDER), filename)


# ── Health Check ───────────────────────────────────────────────

@app.route("/api/health")
def health_check():
    """API health check endpoint — reports status of all models."""
    model_status = {}
    for key, meta in Config.MODELS.items():
        model_status[key] = {
            "trained_found": Path(meta["trained_path"]).exists(),
            "fallback_found": Path(meta["fallback_path"]).exists(),
            "cached": key in engine.loaded_models(),
        }

    return jsonify({
        "status": "ok",
        "default_model": Config.DEFAULT_MODEL,
        "models": model_status,
    })


# ── Image Detection ────────────────────────────────────────────

@app.route("/api/detect/image", methods=["POST"])
def detect_image():
    """
    Upload an image → run YOLO detection with selected model → return results.

    Form fields:
        file       — image file
        model_key  — model to use (optional, default: yolo26s)
        confidence — confidence threshold (optional, default: 0.25)

    Returns JSON:
        {
            "type": "image",
            "detection": { count, detections, annotated_image_url, model_key, model_name, ... },
            "behavior": { total_shrimp, density, health_status, ... }
        }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    model_key = _parse_model_key(request)
    conf = _parse_confidence(request)

    try:
        file_path, file_type = _save_upload(file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if file_type != "image":
        return jsonify({"error": "Please use /api/detect/video for video files"}), 400

    try:
        # Run detection with selected model
        detection = engine.detect_image(file_path, model_key=model_key, conf=conf)

        # Behavior analysis
        analyzer = BehaviorAnalyzer()
        behavior = analyzer.analyze_image(detection["detections"])

        detection["annotated_image_url"] = (
            f"/api/results/image/{detection['annotated_image_name']}"
        )

        return jsonify({
            "type": "image",
            "detection": detection,   # already contains inference_ms
            "behavior": behavior,
        })

    except Exception as e:
        logger.exception("Image detection failed")
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass


# ── Video Detection ────────────────────────────────────────────

@app.route("/api/detect/video", methods=["POST"])
def detect_video():
    """
    Upload a video → create a job → return job_id.

    Form fields:
        file       — video file
        model_key  — model to use (optional, default: yolo26s)
        confidence — confidence threshold (optional, default: 0.25)

    Client connects to /api/stream/<job_id> for SSE results.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    model_key = _parse_model_key(request)
    conf = _parse_confidence(request)

    try:
        file_path, file_type = _save_upload(file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if file_type != "video":
        return jsonify({"error": "Please use /api/detect/image for image files"}), 400

    job_id = uuid.uuid4().hex[:16]
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "pending",
            "video_path": file_path,
            "model_key": model_key,
            "confidence": conf,
            "summary": None,
        }

    return jsonify({
        "job_id": job_id,
        "stream_url": f"/api/stream/{job_id}",
        "model_key": model_key,
        "model_name": Config.MODELS.get(model_key, {}).get("name", model_key),
    })


# ── SSE Stream ─────────────────────────────────────────────────

@app.route("/api/stream/<job_id>")
def stream_video_results(job_id):
    """
    Server-Sent Events endpoint for real-time video detection results.
    Yields one SSE event per processed frame, plus a final 'complete' event.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        video_path = job["video_path"]
        model_key = job.get("model_key", Config.DEFAULT_MODEL)
        conf = job.get("confidence", Config.CONFIDENCE_THRESHOLD)
        analyzer = BehaviorAnalyzer()

        with _jobs_lock:
            job["status"] = "processing"

        try:
            final_avg_fps = 0.0

            for frame_data in engine.detect_video_stream(video_path, model_key=model_key, conf=conf):
                if "error" in frame_data:
                    yield f"data: {json.dumps({'type': 'error', 'message': frame_data['error']})}\n\n"
                    return

                # Diagnostic warning from engine (e.g. codec issue)
                if frame_data.get("type") == "warn":
                    yield f"data: {json.dumps(frame_data)}\n\n"
                    continue

                behavior = analyzer.analyze_frame(
                    frame_data["centroids"],
                    frame_data["frame_number"],
                )

                event = {
                    "type": "frame",
                    "frame_number": frame_data["frame_number"],
                    "total_frames": frame_data["total_frames"],
                    "fps": frame_data["fps"],
                    "count": frame_data["count"],
                    "active_count": behavior["active_count"],
                    "inactive_count": behavior["inactive_count"],
                    "active_ratio": behavior["active_ratio"],
                    "health_status": behavior["health_status"],
                    "health_score": behavior["health_score"],
                    "annotated_frame_b64": frame_data["annotated_frame_b64"],
                    "model_key": frame_data.get("model_key", model_key),
                    "model_name": frame_data.get("model_name", model_key),
                    "inference_ms": frame_data.get("inference_ms", 0),
                    "avg_fps_inference": frame_data.get("avg_fps_inference", 0),
                }

                final_avg_fps = frame_data.get("avg_fps_inference", 0)

                yield f"data: {json.dumps(event)}\n\n"

            # Final summary
            summary = analyzer.get_summary()
            summary["model_key"] = model_key
            summary["model_name"] = Config.MODELS.get(model_key, {}).get("name", model_key)
            summary["avg_inference_fps"] = final_avg_fps

            with _jobs_lock:
                job["status"] = "complete"
                job["summary"] = summary

            yield f"data: {json.dumps({'type': 'complete', 'summary': summary})}\n\n"

        except Exception as e:
            logger.exception("Video streaming error for job %s", job_id)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        finally:
            try:
                os.remove(video_path)
            except OSError:
                pass

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Job Results ────────────────────────────────────────────────

@app.route("/api/results/<job_id>")
def get_job_results(job_id):
    """Get the final aggregated results for a completed video job."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "model_key": job.get("model_key"),
        "model_name": Config.MODELS.get(job.get("model_key", ""), {}).get("name"),
        "summary": job.get("summary"),
    })


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  ShrimpVision — Multi-Model Server Starting")
    logger.info("=" * 55)

    # Pre-load default model at startup
    logger.info("  Warming up default model: %s ...", Config.DEFAULT_MODEL)
    try:
        _ = engine.get_model(Config.DEFAULT_MODEL)
        logger.info("  Default model loaded successfully!")
    except Exception as e:
        logger.error("  Default model load failed: %s", e)

    logger.info("  Available models: %s", list(Config.MODELS.keys()))
    logger.info("  URL: http://localhost:5000")
    logger.info("=" * 55)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False,
    )
