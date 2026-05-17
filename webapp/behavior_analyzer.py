"""
ShrimpVision AI — Behavior Analyzer (v2 - Multi-metric)
=========================================================
Analyzes shrimp behavior from video detection data using centroid tracking.
The output is a behavior/activity indicator, not a clinical disease diagnosis.

Metrics implemented:
  1. Activity Rate    — ratio of shrimp that moved beyond threshold (binary)
  2. Avg Velocity     — mean displacement per frame (higher = more active)
  3. Velocity Variance— consistency of movement (erratic vs smooth)
  4. Spatial Spread   — how spread out the shrimp are (std dev of positions)
                        Unusual aggregation or edge gathering can be treated as
                        a possible stress/context indicator.
  5. Composite Activity Score — weighted combination of all metrics

Research basis:
  - Shrimp behavior studies commonly quantify movement speed, activity,
    and spatial distribution from video/tracking data.
  - WOAH aquatic animal health guidance describes lethargy, reduced feeding,
    abnormal swimming, and edge gathering as clinical/behavioral signs that
    require further confirmation rather than image-only diagnosis.
"""

import math
import logging
import statistics
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)


class BehaviorAnalyzer:
    """
    Tracks shrimp centroids across consecutive frames to determine
    behavioral metrics and compute a composite population activity score.

    Multi-metric approach (v2):
      - Binary activity (moved > threshold)
      - Mean velocity (avg displacement pixels/frame)
      - Velocity variance (erratic vs smooth movement)
      - Spatial spread (clustering vs even distribution)
    """

    def __init__(self, movement_threshold: float = None):
        self.threshold = movement_threshold or Config.MOVEMENT_THRESHOLD_PX
        self._prev_centroids: Optional[list] = None
        self._frame_stats: list = []

    def reset(self):
        """Reset state for a new video analysis."""
        self._prev_centroids = None
        self._frame_stats = []

    # ── Per-Frame Analysis ────────────────────────────────────

    def analyze_frame(self, centroids: list, frame_number: int) -> dict:
        """
        Compare current frame centroids against the previous frame.
        Computes multi-metric behavioral analysis per frame.

        Args:
            centroids: [[cx, cy], ...] from current frame detections
            frame_number: current frame index

        Returns dict with:
            frame_number, total_shrimp, active_count, inactive_count,
            active_ratio, avg_velocity, velocity_variance, spatial_spread,
            health_status, health_score, per_shrimp
        """
        total = len(centroids)
        if total == 0:
            result = self._empty_result(frame_number)
            self._prev_centroids = centroids
            self._frame_stats.append(self._summary_frame(result))
            return result

        # First frame → no previous data, assume all active, compute spatial spread only
        if self._prev_centroids is None or len(self._prev_centroids) == 0:
            spread = self._compute_spatial_spread(centroids)
            per_shrimp = [
                {"centroid": c, "status": "active", "displacement": 0.0}
                for c in centroids
            ]
            result = self._build_result(
                frame_number, total, total, 0,
                avg_velocity=0.0, velocity_variance=0.0,
                spatial_spread=spread, per_shrimp=per_shrimp
            )
            self._prev_centroids = centroids
            self._frame_stats.append(self._summary_frame(result))
            return result

        # ── Match centroids (nearest-neighbor greedy) ──────────
        per_shrimp = []
        used_prev = set()
        displacements = []

        for curr in centroids:
            min_dist = float("inf")
            match_idx = -1

            for i, prev in enumerate(self._prev_centroids):
                if i in used_prev:
                    continue
                dist = math.dist(curr, prev)
                if dist < min_dist:
                    min_dist = dist
                    match_idx = i

            if match_idx >= 0:
                used_prev.add(match_idx)
                displacements.append(min_dist)

            is_active = min_dist > self.threshold if match_idx >= 0 else True
            per_shrimp.append({
                "centroid": curr,
                "status": "active" if is_active else "inactive",
                "displacement": round(min_dist, 2) if match_idx >= 0 else 0.0,
            })

        active_count   = sum(1 for s in per_shrimp if s["status"] == "active")
        inactive_count = total - active_count

        # ── Velocity metrics ───────────────────────────────────
        avg_velocity = (
            sum(displacements) / len(displacements) if displacements else 0.0
        )
        velocity_variance = (
            statistics.variance(displacements)
            if len(displacements) >= 2 else 0.0
        )

        # ── Spatial distribution ───────────────────────────────
        spatial_spread = self._compute_spatial_spread(centroids)

        result = self._build_result(
            frame_number, total, active_count, inactive_count,
            avg_velocity=round(avg_velocity, 3),
            velocity_variance=round(velocity_variance, 3),
            spatial_spread=round(spatial_spread, 3),
            per_shrimp=per_shrimp,
        )
        self._prev_centroids = centroids
        self._frame_stats.append(self._summary_frame(result))
        return result

    # ── Aggregate Summary ─────────────────────────────────────

    def get_summary(self) -> dict:
        """
        Return aggregated multi-metric behavior statistics across all frames.
        """
        if not self._frame_stats:
            return {
                "total_frames_analyzed": 0,
                "avg_shrimp_count": 0,
                "avg_active_ratio": 0,
                "avg_velocity": 0,
                "avg_velocity_variance": 0,
                "avg_spatial_spread": 0,
                "overall_health_status": "unknown",
                "overall_health_score": 0,
                "frame_stats": [],
                "peak_count": 0,
                "min_count": 0,
            }

        counts   = [f["total_shrimp"] for f in self._frame_stats]
        ratios   = [f["active_ratio"] for f in self._frame_stats if f["total_shrimp"] > 0]
        vels     = [f["avg_velocity"] for f in self._frame_stats if f["total_shrimp"] > 0]
        variances= [f["velocity_variance"] for f in self._frame_stats if f["total_shrimp"] > 0]
        spreads  = [f["spatial_spread"] for f in self._frame_stats if f["total_shrimp"] > 0]

        avg_ratio    = sum(ratios)    / len(ratios)    if ratios    else 0
        avg_velocity = sum(vels)      / len(vels)      if vels      else 0
        avg_variance = sum(variances) / len(variances) if variances else 0
        avg_spread   = sum(spreads)   / len(spreads)   if spreads   else 0
        avg_count    = sum(counts)    / len(counts)    if counts    else 0

        health_status, health_score = self._compute_composite_health(
            avg_ratio, avg_velocity, avg_variance, avg_spread
        )

        return {
            "total_frames_analyzed": len(self._frame_stats),
            "avg_shrimp_count":      round(avg_count, 1),
            "avg_active_ratio":      round(avg_ratio, 3),
            "avg_velocity":          round(avg_velocity, 3),
            "avg_velocity_variance": round(avg_variance, 3),
            "avg_spatial_spread":    round(avg_spread, 3),
            "overall_health_status": health_status,
            "overall_health_score":  health_score,
            "frame_stats": [
                {
                    "frame":    f["frame_number"],
                    "count":    f["total_shrimp"],
                    "active":   f["active_count"],
                    "inactive": f["inactive_count"],
                    "ratio":    f["active_ratio"],
                    "velocity": f["avg_velocity"],
                    "variance": f["velocity_variance"],
                    "spread":   f["spatial_spread"],
                }
                for f in self._frame_stats
            ],
            "peak_count": max(counts) if counts else 0,
            "min_count":  min(counts) if counts else 0,
        }

    # ── Image-Only Analysis ───────────────────────────────────

    @staticmethod
    def analyze_image(detections: list) -> dict:
        """
        For a single image: count + spatial distribution only.
        """
        count = len(detections)

        if count == 0:
            density = "none"
        elif count <= 20:
            density = "low"
        elif count <= 60:
            density = "medium"
        else:
            density = "high"

        # Compute spatial spread if we have detections
        spatial_spread = 0.0
        if count >= 2:
            centroids = [[d["x"], d["y"]] for d in detections if "x" in d and "y" in d]
            if len(centroids) >= 2:
                spatial_spread = BehaviorAnalyzer._compute_spatial_spread(centroids)

        return {
            "total_shrimp":   count,
            "analysis_type":  "image",
            "note": "Behavior analysis requires video input (multiple frames for motion detection).",
            "health_status":  "unknown",
            "health_score":   None,
            "density":        density,
            "spatial_spread": round(spatial_spread, 3),
        }

    # ── Private Helpers ───────────────────────────────────────

    @staticmethod
    def _summary_frame(result: dict) -> dict:
        frame = result.copy()
        frame.pop("per_shrimp", None)
        return frame

    def _build_result(self, frame_number, total, active, inactive,
                      avg_velocity, velocity_variance, spatial_spread, per_shrimp):
        ratio = active / total if total > 0 else 0
        health_status, health_score = self._compute_composite_health(
            ratio, avg_velocity, velocity_variance, spatial_spread
        )
        return {
            "frame_number":      frame_number,
            "total_shrimp":      total,
            "active_count":      active,
            "inactive_count":    inactive,
            "active_ratio":      round(ratio, 3),
            "avg_velocity":      avg_velocity,
            "velocity_variance": velocity_variance,
            "spatial_spread":    spatial_spread,
            "health_status":     health_status,
            "health_score":      health_score,
            "per_shrimp":        per_shrimp,
        }

    @staticmethod
    def _compute_spatial_spread(centroids: list) -> float:
        """
        Compute standard deviation of shrimp positions as a measure of
        how spread out vs clustered they are.

        Higher value = more spread-out detections.
        Lower value  = more clustered detections, interpreted only as a
        possible context/stress indicator.
        """
        if len(centroids) < 2:
            return 0.0
        xs = [c[0] for c in centroids]
        ys = [c[1] for c in centroids]
        std_x = statistics.stdev(xs)
        std_y = statistics.stdev(ys)
        return math.sqrt(std_x ** 2 + std_y ** 2)

    @staticmethod
    def _compute_composite_health(
        active_ratio: float,
        avg_velocity: float,
        velocity_variance: float,
        spatial_spread: float,
    ) -> tuple:
        """
        Composite activity score using 4 behavioral metrics.

        Scoring rationale (based on aquaculture research):
          - Activity rate (40%): primary behavioral indicator
          - Avg velocity (30%): speed of movement (healthy shrimp move faster)
          - Velocity variance (15%): penalize erratic movement (stress sign)
          - Spatial spread (15%): reward even distribution

        Returns: (status_string, score_0_100)
        """
        # ── 1. Activity score (0-100) ──────────────────────────
        activity_score = active_ratio * 100

        # ── 2. Velocity score (0-100) ──────────────────────────
        # Normalize: 0px/frame = 0, >=10px/frame = 100 (PL shrimp normal range 1-8px)
        velocity_score = min(avg_velocity / 10.0, 1.0) * 100

        # ── 3. Variance penalty (0-100, lower variance = higher score) ─
        # High variance can indicate inconsistent movement, but is not diagnostic.
        # Normalize: 0 variance = 100, >=50 variance = 0
        variance_score = max(0.0, 1.0 - (velocity_variance / 50.0)) * 100

        # ── 4. Spatial spread score (0-100) ───────────────────
        # 0=clustered, >=200px spread = 100 for this image scale.
        spread_score = min(spatial_spread / 200.0, 1.0) * 100

        # ── Weighted composite ─────────────────────────────────
        composite = (
            0.40 * activity_score +
            0.30 * velocity_score +
            0.15 * variance_score +
            0.15 * spread_score
        )
        score = int(round(composite))

        # ── Map to status ──────────────────────────────────────
        thresholds = Config.HEALTH_THRESHOLDS
        if active_ratio >= thresholds["healthy"] and score >= 65:
            status = "healthy"
        elif active_ratio >= thresholds["warning"] or score >= 40:
            status = "warning"
        else:
            status = "critical"

        return status, score

    @staticmethod
    def _empty_result(frame_number):
        return {
            "frame_number":      frame_number,
            "total_shrimp":      0,
            "active_count":      0,
            "inactive_count":    0,
            "active_ratio":      0.0,
            "avg_velocity":      0.0,
            "velocity_variance": 0.0,
            "spatial_spread":    0.0,
            "health_status":     "unknown",
            "health_score":      0,
            "per_shrimp":        [],
        }
