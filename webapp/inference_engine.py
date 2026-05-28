"""
ShrimpVision AI — Multi-Model Inference Engine
================================================
Supports dynamic model switching across YOLOv9s, YOLOv10s, YOLO11s, YOLO12s, YOLO26s.
Uses a per-key model cache to avoid reloading the same model on repeated requests.
Thread-safe via per-key locks.

VIDEO BUG FIX (v2.1):
  Some video codecs on Windows cause OpenCV to return frames that are valid
  (ret=True) but contain all-zero or corrupted pixel data. The fix:
  1. Open video with cv2.CAP_FFMPEG backend explicitly
  2. Validate each frame (dtype, shape, mean pixel value)
  3. Convert frame to uint8 BGR explicitly before passing to YOLO
  4. Emit a diagnostic 'warn' event if too many consecutive bad frames detected

SPEED TIMING (v2.1):
  All detections now record inference_ms (wall-clock time of model.predict).
  Image detection returns inference_ms.
  Video stream yields inference_ms and running avg_fps_inference per frame.
"""

import cv2
import time
import numpy as np
import threading
import logging
import gc
from collections import OrderedDict
from pathlib import Path
from typing import Generator

from config import Config

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Thread-safe multi-model inference engine.
    Models are lazily loaded on first use and cached in memory.
    """

    _instance = None
    _class_lock = threading.Lock()

    def __new__(cls):
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._model_cache: OrderedDict[str, object] = OrderedDict()
        self._key_locks: dict = {}
        self._cache_lock = threading.Lock()
        self._initialized = True

    # ── Model Loading ──────────────────────────────────────────

    def _get_key_lock(self, model_key: str) -> threading.Lock:
        with self._cache_lock:
            if model_key not in self._key_locks:
                self._key_locks[model_key] = threading.Lock()
            return self._key_locks[model_key]

    def _load_model(self, model_key: str):
        from ultralytics import YOLO
        model_path = Config.get_model_path(model_key)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model '{model_key}' not found. Tried: {model_path}"
            )
        logger.info("Loading model '%s' from %s ...", model_key, model_path)
        model = YOLO(model_path)
        logger.info("Model '%s' loaded on device=%s", model_key, Config.DEVICE)
        return model

    def get_model(self, model_key: str):
        if model_key in self._model_cache:
            self._model_cache.move_to_end(model_key)
            return self._model_cache[model_key]
        key_lock = self._get_key_lock(model_key)
        with key_lock:
            if model_key not in self._model_cache:
                self._model_cache[model_key] = self._load_model(model_key)
                self._model_cache.move_to_end(model_key)
                self._evict_old_models()
        return self._model_cache[model_key]

    def loaded_models(self) -> list[str]:
        return list(self._model_cache.keys())

    def _evict_old_models(self):
        max_cached = max(1, int(getattr(Config, "MAX_CACHED_MODELS", 1)))
        while len(self._model_cache) > max_cached:
            old_key, _old_model = self._model_cache.popitem(last=False)
            logger.info("Evicted model '%s' from cache to keep memory usage low.", old_key)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def model(self):
        return self.get_model(Config.DEFAULT_MODEL)

    # ── Image Detection ────────────────────────────────────────

    def detect_image(self, image_path: str, model_key: str = None, conf: float = None) -> dict:
        """
        Run detection on a single image using the specified model.

        Returns:
            {
                "detections": [...],
                "count": int,
                "annotated_image_name": str,
                "annotated_image_url": str,
                "image_width": int,
                "image_height": int,
                "model_key": str,
                "model_name": str,
                "confidence_threshold": float,
                "inference_ms": float,       # ← NEW: wall-clock inference time
            }
        """
        model_key = model_key or Config.DEFAULT_MODEL
        conf = conf if conf is not None else Config.CONFIDENCE_THRESHOLD
        model = self.get_model(model_key)

        t0 = time.perf_counter()
        results = model.predict(
            source=image_path,
            conf=conf,
            iou=Config.IOU_THRESHOLD,
            imgsz=Config.IMG_SIZE,
            device=Config.DEVICE,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - t0) * 1000.0

        result = results[0]
        detections = self._extract_detections(result)

        annotated = self._draw_clean_boxes(result, model_key)
        out_name = f"detected_{Path(image_path).stem}_{model_key}.jpg"
        out_path = str(Config.RESULTS_FOLDER / out_name)
        cv2.imwrite(out_path, annotated)

        h, w = result.orig_shape
        model_meta = Config.MODELS.get(model_key, {})

        logger.info(
            "IMAGE detect  model=%-10s  count=%3d  conf=%.2f  time=%.1f ms",
            model_key, len(detections), conf, inference_ms,
        )

        return {
            "detections": detections,
            "count": len(detections),
            "annotated_image_path": out_path,
            "annotated_image_name": out_name,
            "image_width": w,
            "image_height": h,
            "model_key": model_key,
            "model_name": model_meta.get("name", model_key),
            "confidence_threshold": conf,
            "inference_ms": round(inference_ms, 1),
        }

    # ── Video Detection (Generator for SSE) ────────────────────

    def detect_video_stream(
        self,
        video_path: str,
        model_key: str = None,
        conf: float = None,
    ) -> Generator[dict, None, None]:
        """
        Process video frame-by-frame, yielding detection results per frame.

        Key fixes vs v1:
        - Opens VideoCapture with CAP_FFMPEG backend to support more codecs
        - Validates every frame before inference (shape, dtype, brightness)
        - Converts frame to uint8 BGR explicitly (handles unusual formats)
        - Emits 'warn' SSE events for diagnostic info if frames are bad
        - Reports inference_ms and rolling avg_fps_inference per frame

        Yields frame events:
            { type:"frame", frame_number, total_frames, fps, count, ...,
              inference_ms, avg_fps_inference }

        Yields diagnostic event (once if needed):
            { type:"warn", message: "..." }
        """
        import base64

        model_key = model_key or Config.DEFAULT_MODEL
        conf = conf if conf is not None else Config.CONFIDENCE_THRESHOLD
        model = self.get_model(model_key)
        model_meta = Config.MODELS.get(model_key, {})
        model_name = model_meta.get("name", model_key)

        # ── Open with FFMPEG backend explicitly (fix codec issues on Windows) ──
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            # Fallback to default backend
            cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield {"error": f"Cannot open video: {video_path}"}
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_skip = Config.VIDEO_FRAME_SKIP
        max_frames = Config.MAX_VIDEO_FRAMES
        stream_frame_every = max(1, int(getattr(Config, "STREAM_FRAME_EVERY", 1)))
        jpeg_quality = max(25, min(95, int(getattr(Config, "STREAM_JPEG_QUALITY", 80))))

        logger.info(
            "VIDEO start  model=%-10s  res=%dx%d  total_frames=%d  fps=%.1f  conf=%.2f",
            model_key, vid_w, vid_h, total_frames, fps, conf,
        )

        frame_idx = 0
        processed = 0
        consecutive_bad = 0
        warned_codec = False

        # Rolling inference timing
        inference_times: list[float] = []

        while cap.isOpened() and (max_frames is None or processed < max_frames):
            ret, frame = cap.read()
            if not ret:
                logger.info("VIDEO end-of-stream at frame %d (processed %d)", frame_idx, processed)
                break

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            # ── Frame validation ─────────────────────────────────
            frame = self._sanitize_frame(frame)
            if frame is None:
                consecutive_bad += 1
                logger.warning("Bad frame #%d  (consecutive bad: %d)", frame_idx, consecutive_bad)

                # Emit a one-time diagnostic warning to the client
                if consecutive_bad == 5 and not warned_codec:
                    warned_codec = True
                    yield {
                        "type": "warn",
                        "message": (
                            "5 consecutive unreadable frames detected. "
                            "The video codec may not be fully supported. "
                            "Try converting the video to H.264 MP4 for best results."
                        ),
                    }
                frame_idx += 1
                continue

            consecutive_bad = 0  # reset counter on a good frame
            frame = self._resize_for_inference(frame)

            # ── Inference + timing ────────────────────────────────
            t0 = time.perf_counter()
            results = model.predict(
                source=frame,
                conf=conf,
                iou=Config.IOU_THRESHOLD,
                imgsz=Config.IMG_SIZE,
                device=Config.DEVICE,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - t0) * 1000.0
            inference_times.append(inference_ms)

            # Rolling average FPS (last 30 frames)
            recent = inference_times[-30:]
            avg_fps = 1000.0 / (sum(recent) / len(recent)) if recent else 0.0

            result = results[0]
            detections = self._extract_detections(result)
            centroids = self._compute_centroids(detections)

            logger.debug(
                "FRAME %d  count=%d  det_conf_min=%.2f  time=%.1f ms",
                frame_idx, len(detections),
                min((d["confidence"] for d in detections), default=0.0),
                inference_ms,
            )

            # Annotated frame → base64
            b64_frame = None
            is_preview_frame = (
                processed % stream_frame_every == 0
                or (total_frames and frame_idx >= total_frames - 1)
            )
            if is_preview_frame:
                annotated = self._draw_clean_boxes(result, model_key)
                _, buffer = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                b64_frame = base64.b64encode(buffer).decode("utf-8")

            yield {
                "type": "frame",
                "frame_number": frame_idx,
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "detections": detections,
                "count": len(detections),
                "centroids": centroids,
                "annotated_frame_b64": b64_frame,
                "preview_frame": is_preview_frame,
                "model_key": model_key,
                "model_name": model_name,
                "inference_ms": round(inference_ms, 1),
                "avg_fps_inference": round(avg_fps, 1),
            }

            frame_idx += 1
            processed += 1

        cap.release()

        total_inf_time = sum(inference_times)
        overall_fps = 1000.0 / (total_inf_time / len(inference_times)) if inference_times else 0.0
        logger.info(
            "VIDEO done  model=%-10s  processed=%d  avg_fps=%.1f  total_time=%.1f s",
            model_key, processed, overall_fps, total_inf_time / 1000.0,
        )

    # ── Frame Sanitizer (video bug fix) ─────────────────────────

    @staticmethod
    def _sanitize_frame(frame: np.ndarray) -> np.ndarray | None:
        """
        Validate and normalise a video frame before inference.

        Returns None if the frame is unusable (all-black, wrong shape, etc.)

        Issues this catches:
          1. frame is None
          2. Empty array
          3. Wrong number of channels (not 3 → BGR)
          4. Non-uint8 dtype (some codecs return float or uint16)
          5. All-zero frame (decoder produced silence / black frame)
          6. Frame mean too low (very dark / codec glitch)
        """
        if frame is None or frame.size == 0:
            return None

        # Must be H×W×C with C == 3
        if frame.ndim == 2:
            # Grayscale → convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            return None

        # Ensure uint8 (YOLO expects uint8 BGR)
        if frame.dtype != np.uint8:
            if frame.dtype in (np.float32, np.float64):
                # Normalised float [0,1] → scale to [0,255]
                if frame.max() <= 1.0:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                else:
                    frame = frame.clip(0, 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        # Check for all-black / decoder glitch: mean < 1 pixel value
        mean_val = float(frame.mean())
        if mean_val < 1.0:
            return None

        return frame

    @staticmethod
    def _resize_for_inference(frame: np.ndarray) -> np.ndarray:
        max_dim = int(getattr(Config, "VIDEO_MAX_DIM", 960))
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest <= max_dim:
            return frame
        scale = max_dim / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _draw_clean_boxes(result, model_key: str = "") -> np.ndarray:
        img = result.orig_img.copy()
        boxes = result.boxes
        count = 0 if (boxes is None or len(boxes) == 0) else len(boxes)

        show_labels = count <= 15
        line_w = 1 if count > 30 else 2
        color = (0, 255, 180)

        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                conf = float(box.conf[0].cpu().numpy())
                cv2.rectangle(img, (x1, y1), (x2, y2), color, line_w)
                # Labels removed as per user request
                # if show_labels:
                #     label = f"{conf:.0%}"
                #     font_scale = 0.35
                #     thickness = 1
                #     (tw, th), _ = cv2.getTextSize(
                #         label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                #     )
                #     cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
                #     cv2.putText(
                #         img, label, (x1 + 2, y1 - 2),
                #         cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA,
                #     )

        # Count overlay
        count_text = f"Shrimp: {count}"
        cv2.rectangle(img, (8, 8), (200, 42), (0, 0, 0), -1)
        cv2.putText(
            img, count_text, (14, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2, cv2.LINE_AA,
        )

        # Model watermark (bottom-right)
        if model_key:
            model_label = Config.MODELS.get(model_key, {}).get("name", model_key)
            h_img, w_img = img.shape[:2]
            wm_text = f"[{model_label}]"
            font_scale_wm = 0.45
            (wm_w, wm_h), _ = cv2.getTextSize(
                wm_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_wm, 1
            )
            wx = w_img - wm_w - 12
            wy = h_img - 12
            cv2.rectangle(img, (wx - 4, wy - wm_h - 6), (w_img - 6, wy + 4), (0, 0, 0), -1)
            cv2.putText(
                img, wm_text, (wx, wy),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_wm, (0, 200, 255), 1, cv2.LINE_AA,
            )

        return img

    @staticmethod
    def _extract_detections(result) -> list:
        detections = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = result.names.get(cls_id, f"class_{cls_id}")
            detections.append({
                "bbox": [round(v, 1) for v in xyxy],
                "confidence": round(conf, 3),
                "class": cls_name,
                "class_id": cls_id,
            })
        return detections

    @staticmethod
    def _compute_centroids(detections: list) -> list:
        centroids = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centroids.append([round(cx, 1), round(cy, 1)])
        return centroids
