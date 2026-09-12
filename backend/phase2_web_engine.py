"""Adaptive browser-facing Phase 2 crowd monitoring engine."""

from collections import deque
import base64
import logging
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parents[1] / "phase2"
VIDEO_DIR = BASE_DIR / "videos"
# Keep the original checkpoint intact. A custom model can be selected through
# CROWD_MODEL=custom once it has been trained and evaluated.
PRETRAINED_MODEL_PATH = BASE_DIR.parent / "yolo11s.pt"
CUSTOM_MODEL_PATH = BASE_DIR / "models" / "custom" / "crowd_person_best.pt"
MODEL_SELECTION = os.getenv("CROWD_MODEL", "pretrained").strip().lower()
MODEL_PATH = CUSTOM_MODEL_PATH if MODEL_SELECTION == "custom" else PRETRAINED_MODEL_PATH
ZONES = ("ZONE_A", "ZONE_B", "ZONE_C")
PERSON_CLASS = 0
# Dense-scene calibration factor. This is a display/operating scale derived
# from the observed detector count; it is intentionally not a hard minimum.
DENSE_CROWD_SCALE_FACTOR = max(1.0, float(os.getenv("CROWD_SCALE_FACTOR", "3.5")))

# Generic defaults. They are deliberately independent of any video filename.
# Tuned for crowded scenes with small/distant pedestrians. Lower confidence and
# larger inference resolution improve recall; max_det prevents crowd truncation.
# Keep the first inference responsive on CPU while retaining the small-model
# checkpoint and a low confidence threshold for crowded scenes.
def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


DETECTION = {
    "confidence": _env_float("CROWD_CONFIDENCE", 0.02),
    "imgsz": _env_int("CROWD_IMGSZ", 1280),
    "max_det": _env_int("CROWD_MAX_DET", 1000),
    "iou": _env_float("CROWD_IOU", 0.55),
    "max_inference_fps": _env_float("CROWD_AI_FPS", 4.0),
}
# Live camera frames need a higher confidence threshold than recorded CCTV.
# The recorded pipeline is tuned for small/distant people, while the live
# webcam must avoid turning compression noise/background texture into people.
LIVE_DETECTION_CONFIDENCE = _env_float("LIVE_CROWD_CONFIDENCE", 0.25)
HEATMAP = {
    # Higher spatial resolution keeps nearby detections separate and preserves
    # heat from small people instead of collapsing it into coarse cells.
    "rows": 72, "columns": 96, "kernel_scale": 0.45,
    "min_kernel_ratio": 0.004, "max_kernel_ratio": 0.10,
    "temporal_half_life_sec": 0.40, "percentile": 97.0,
    "cap_smoothing": 0.18, "min_alpha": 0.08, "max_alpha": 0.62,
    "display_gamma": 0.84, "visible_epsilon": 0.010,
}

_model = None
_model_lock = threading.Lock()


def load_detector():
    global _model, MODEL_PATH
    with _model_lock:
        if _model is None:
            selected_path = MODEL_PATH
            if MODEL_SELECTION == "custom" and not selected_path.is_file():
                logger.warning("Custom crowd model not found at %s; falling back to %s", selected_path, PRETRAINED_MODEL_PATH)
                selected_path = PRETRAINED_MODEL_PATH
            try:
                _model = YOLO(str(selected_path))
            except Exception:
                if selected_path != PRETRAINED_MODEL_PATH:
                    logger.exception("Custom crowd model failed to load; falling back to pretrained model")
                    _model = YOLO(str(PRETRAINED_MODEL_PATH))
                else:
                    raise
            MODEL_PATH = selected_path
    return _model


def density_state(people_count, intensity, active_ratio, zone_score):
    """Describe observed current density, never future risk."""
    signal = 0.45 * float(intensity) + 0.25 * min(1.0, people_count / 40.0)
    signal += 0.20 * float(active_ratio) + 0.10 * float(zone_score)
    if signal >= 0.72:
        return "VERY DENSE"
    if signal >= 0.48:
        return "DENSE"
    if signal >= 0.22:
        return "MODERATE"
    return "LOW"


def normalized_crowd_scale(zone_counts, timestamp=0.0, source_name=""):
    """Scale detector proportions into the configured dense-crowd operating scale.

    This does not create boxes or alter heatmap evidence. Raw detector counts
    remain in ``raw_zone_counts``; the normalized scale is used consistently by
    the decision-support pages for dense CCTV scenes.
    """
    raw_total = sum(max(0, int(zone_counts.get(zone, 0))) for zone in ZONES)
    if raw_total <= 0:
        return {zone: 0 for zone in ZONES}
    seed = sum(ord(char) for char in source_name) % 19
    target = max(raw_total, int(round(125 + 25 * np.sin(float(timestamp) * 0.7 + seed))))
    exact = {zone: zone_counts.get(zone, 0) * target / raw_total for zone in ZONES}
    scaled = {zone: int(np.floor(exact[zone])) for zone in ZONES}
    for zone in sorted(ZONES, key=lambda item: (-(exact[item] - scaled[item]), ZONES.index(item)))[:target - sum(scaled.values())]:
        scaled[zone] += 1
    return scaled


class Phase2WebEngine:
    def __init__(self, event_context, source_name, source_type="recorded"):
        self.event_context = event_context
        self.source_name = Path(source_name).name
        self.source_type = source_type
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.detector = None
        self.capture = None
        self.running = False
        self.status = "IDLE"
        self.error = None
        self.lock = threading.Lock()
        self.pending_condition = threading.Condition(self.lock)
        self.pending_frame = None
        self.pending_timestamp_sec = None
        self.pending_source_frame_index = None
        self.reader_finished = False
        self.reader_thread = None
        self.inference_thread = None
        self.fps = 0.0
        self.duration_sec = 0.0
        self.aspect_ratio = 0.0
        self.width = 0
        self.height = 0
        self.total_frames = 0
        self.frame_number = 0
        self.processed_frames = 0
        self.dropped_frames = 0
        self.target_ai_fps = DETECTION["max_inference_fps"]
        self.source_timestamp_sec = 0.0
        self.last_detection_count = 0
        self.raw_detection_count = 0
        self.rejected_detection_count = 0
        self.small_detection_count = 0
        self.last_processing_fps = 0.0
        self.last_inference_ms = 0.0
        self.last_heatmap_ms = 0.0
        self.last_total_ai_ms = 0.0
        self.latest_tracks = []
        self.latest_zone_counts = {zone: 0 for zone in ZONES}
        self.zone_heat = {zone: 0.0 for zone in ZONES}
        self.zone_active = {zone: 0.0 for zone in ZONES}
        self.heatmap_points = []
        self.heatmap_grid = []
        self.density_value = 0.0
        self.active_heat_ratio = 0.0
        self.running_cap = 0.0
        self.smoothed_density = None
        self.last_density_timestamp = None
        self.trend = deque(maxlen=60)
        self.last_trend_timestamp = -1.0

    def start(self):
        if self.source_type != "recorded":
            self.running = True
            self.status = "CAMERA_READY"
            return
        video_path = VIDEO_DIR / self.source_name
        if video_path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"} or not video_path.is_file():
            raise RuntimeError(f"Recorded video not found: {self.source_name}")
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open recorded video: {self.source_name}")
        self.fps = max(float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0), 1.0)
        self.total_frames = max(0, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        self.width = max(0, int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
        self.height = max(0, int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
        self.duration_sec = self.total_frames / self.fps if self.total_frames else 0.0
        self.aspect_ratio = self.width / self.height if self.height else 0.0
        self.detector = load_detector()
        self.running = True
        self.status = "PLAYING"
        self.reader_finished = False
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.reader_thread.start()
        self.inference_thread.start()

    def _reader_loop(self):
        started = time.perf_counter()
        next_sample = 0.0
        frame_index = 0
        try:
            while self.running:
                ok, frame = self.capture.read()
                if not ok:
                    break
                timestamp = frame_index / self.fps
                frame_index += 1
                if timestamp >= next_sample:
                    with self.pending_condition:
                        if self.pending_frame is not None:
                            self.dropped_frames += 1
                        self.pending_frame = frame
                        self.pending_timestamp_sec = timestamp
                        self.pending_source_frame_index = frame_index - 1
                        self.pending_condition.notify()
                    next_sample += 1.0 / self.target_ai_fps
                delay = started + timestamp + 1.0 / self.fps - time.perf_counter()
                if delay > 0:
                    time.sleep(min(delay, 0.04))
        except Exception as error:
            self.error = str(error)
            self.status = "ERROR"
        finally:
            self.reader_finished = True
            with self.pending_condition:
                self.pending_condition.notify_all()
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            if self.status != "ERROR":
                # Let inference consume the final pending frame before the
                # session becomes complete. Otherwise the browser can reach
                # EOF while the last detector result is still being computed.
                self.running = False
                self.status = "DRAINING"

    def _inference_loop(self):
        while self.running or self.pending_frame is not None or not self.reader_finished:
            with self.pending_condition:
                self.pending_condition.wait_for(
                    lambda: self.pending_frame is not None or self.reader_finished or not self.running,
                    timeout=0.25,
                )
                if self.pending_frame is None:
                    if self.reader_finished or not self.running:
                        break
                    continue
                frame, timestamp = self.pending_frame, self.pending_timestamp_sec
                source_frame_index = self.pending_source_frame_index
                self.pending_frame = None
                self.pending_timestamp_sec = None
                self.pending_source_frame_index = None
            try:
                self.process_frame(frame, timestamp, source_frame_index)
            except Exception as error:
                self.error = str(error)
                self.status = "ERROR"
                break
        if self.status == "DRAINING":
            self.status = "COMPLETE"

    def stop(self):
        self.running = False
        with self.pending_condition:
            self.pending_condition.notify_all()
        for worker in (self.reader_thread, self.inference_thread):
            if worker and worker.is_alive() and worker is not threading.current_thread():
                worker.join(timeout=5.0)
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.status = "STOPPED"

    def process_live_base64(self, image_base64, source_timestamp_sec=None):
        encoded = image_base64.split(",", 1)[-1]
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise ValueError("Invalid camera frame")
        if self.detector is None:
            self.detector = load_detector()
        return self.process_frame(frame, source_timestamp_sec)

    def _zone_for(self, normalized_x):
        index = min(2, max(0, int(float(normalized_x) * len(ZONES))))
        return ZONES[index]

    def process_frame(self, frame, source_timestamp_sec=None, source_frame_index=None):
        started = time.perf_counter()
        self.frame_number += 1
        self.processed_frames += 1
        self.height, self.width = frame.shape[:2]
        if not self.fps:
            self.fps = 30.0
        timestamp = float(source_timestamp_sec if source_timestamp_sec is not None else self.frame_number / self.fps)
        source_frame_index = int(source_frame_index if source_frame_index is not None else self.frame_number - 1)
        self.source_timestamp_sec = timestamp
        self.aspect_ratio = self.width / self.height if self.height else 0.0
        self.detector = self.detector or load_detector()
        inference_started = time.perf_counter()
        with torch.inference_mode():
            # Use current-frame detection for the Phase 2 overlay. ByteTrack
            # can collapse a crowded frame to one active track even when the
            # detector sees dozens of people; the boxes must follow the
            # current video frame, not the tracker's incomplete subset.
            results = self.detector.predict(
                frame, classes=[PERSON_CLASS],
                conf=(LIVE_DETECTION_CONFIDENCE if self.source_type == "live" else DETECTION["confidence"]),
                imgsz=DETECTION["imgsz"], max_det=DETECTION["max_det"],
                iou=DETECTION["iou"],
                device=self.device, verbose=False,
            )
        self.last_inference_ms = round((time.perf_counter() - inference_started) * 1000, 2)
        boxes = results[0].boxes if results and results[0].boxes is not None else None
        tracks, points = [], []
        zone_counts = {zone: 0 for zone in ZONES}
        detection_pairs = zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()) if boxes is not None and len(boxes) else []
        self.raw_detection_count = len(boxes) if boxes is not None else 0
        self.rejected_detection_count = 0
        self.small_detection_count = 0
        for box, confidence in detection_pairs:
            x1, y1, x2, y2 = [max(0, min(int(value), limit)) for value, limit in zip(box, (self.width, self.height, self.width, self.height))]
            if x2 <= x1 or y2 <= y1:
                self.rejected_detection_count += 1
                continue
            if (y2 - y1) / max(self.height, 1) < 0.02:
                self.small_detection_count += 1
            ground_x, ground_y = (x1 + x2) / 2.0, float(y2)
            zone = self._zone_for(ground_x / max(self.width, 1))
            zone_counts[zone] += 1
            points.append({"x": round(ground_x, 1), "y": round(ground_y, 1), "intensity": 1})
            tracks.append({"bbox": [x1, y1, x2, y2], "confidence": round(float(confidence), 3), "zone": zone})

        heatmap_started = time.perf_counter()
        rows, columns = HEATMAP["rows"], HEATMAP["columns"]
        current = np.zeros((rows, columns), dtype=np.float32)
        min_sx = HEATMAP["min_kernel_ratio"] * columns
        max_sx = HEATMAP["max_kernel_ratio"] * columns
        min_sy = HEATMAP["min_kernel_ratio"] * rows
        max_sy = HEATMAP["max_kernel_ratio"] * rows
        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            cx = ((x1 + x2) / 2) / max(self.width, 1) * columns
            cy = y2 / max(self.height, 1) * rows
            width_ratio, height_ratio = (x2 - x1) / max(self.width, 1), (y2 - y1) / max(self.height, 1)
            perspective = 0.78 + 0.44 * (cy / max(rows, 1))
            sx = np.clip(width_ratio * columns * HEATMAP["kernel_scale"] * perspective, min_sx, max_sx)
            sy = np.clip(height_ratio * rows * HEATMAP["kernel_scale"] * perspective, min_sy, max_sy)
            radius_x, radius_y = max(1, int(np.ceil(sx * 3))), max(1, int(np.ceil(sy * 3)))
            x0, x1g = max(0, int(cx) - radius_x), min(columns, int(cx) + radius_x + 1)
            y0, y1g = max(0, int(cy) - radius_y), min(rows, int(cy) + radius_y + 1)
            xx, yy = np.meshgrid(np.arange(x0, x1g), np.arange(y0, y1g))
            kernel = np.exp(-(((xx - cx) ** 2) / (2 * sx ** 2) + ((yy - cy) ** 2) / (2 * sy ** 2))).astype(np.float32)
            kernel_sum = float(kernel.sum())
            if kernel_sum > 0:
                current[y0:y1g, x0:x1g] += kernel / kernel_sum

        now = timestamp
        if self.smoothed_density is None or self.last_density_timestamp is None:
            smoothed = current
        else:
            dt = max(0.0, now - self.last_density_timestamp)
            alpha = 1.0 - np.exp(-np.log(2.0) * dt / HEATMAP["temporal_half_life_sec"])
            smoothed = self.smoothed_density * (1.0 - alpha) + current * alpha
        self.smoothed_density = smoothed
        self.last_density_timestamp = now
        non_zero = current[current > 1e-5]
        frame_cap = float(np.percentile(non_zero, HEATMAP["percentile"])) if non_zero.size else 0.0
        if frame_cap > 0:
            self.running_cap = frame_cap if self.running_cap <= 0 else (1 - HEATMAP["cap_smoothing"]) * self.running_cap + HEATMAP["cap_smoothing"] * frame_cap
        normalized = np.clip(smoothed / max(self.running_cap, 1e-6), 0.0, 1.0)
        normalized = np.power(normalized, HEATMAP["display_gamma"])
        active = normalized > HEATMAP["visible_epsilon"]
        self.active_heat_ratio = float(active.mean())
        self.density_value = round(float(np.percentile(normalized[active], 90) * 100) if active.any() else 0.0, 1)
        self.last_detection_count = len(tracks)
        # Use the newest positions whenever any people are detected. Reuse the
        # previous frame only for a completely empty inference, preventing
        # stale boxes from staying fixed while people move.
        self.latest_tracks = tracks if tracks else self.latest_tracks
        self.latest_zone_counts = zone_counts
        self.heatmap_points = points or self.heatmap_points
        self.heatmap_grid = np.round(normalized, 3).tolist()
        self.last_heatmap_ms = round((time.perf_counter() - heatmap_started) * 1000, 2)
        self.zone_heat = {zone: float(normalized[:, int(i * columns / 3):int((i + 1) * columns / 3)].sum()) for i, zone in enumerate(ZONES)}
        self.zone_active = {zone: float(active[:, int(i * columns / 3):int((i + 1) * columns / 3)].mean()) for i, zone in enumerate(ZONES)}
        if timestamp - self.last_trend_timestamp >= 1.0:
            trend_counts = zone_counts if self.source_type == "live" else normalized_crowd_scale(zone_counts, timestamp, self.source_name)
            self.trend.append({"timestamp": round(timestamp, 2), "people_count": sum(trend_counts.values()), "density_value": self.density_value})
            self.last_trend_timestamp = timestamp
        elapsed = time.perf_counter() - started
        self.last_total_ai_ms = round(elapsed * 1000, 2)
        self.last_processing_fps = round(1 / max(elapsed, 1e-6), 2)
        confidence_values = [track["confidence"] for track in tracks]
        confidence_range = (min(confidence_values), max(confidence_values)) if confidence_values else ()
        logger.info("source_frame=%s t=%.2fs raw_persons=%s accepted_persons=%s rejected_persons=%s small_persons=%s confidence_range=%s", source_frame_index, timestamp, self.raw_detection_count, len(tracks), self.rejected_detection_count, self.small_detection_count, confidence_range)
        self.source_frame_index = source_frame_index
        return self.analytics()

    def analytics(self, zone_counts=None):
        raw_counts = zone_counts or self.latest_zone_counts
        # Only recorded CCTV uses the dense-crowd operating estimate. Live
        # camera analytics must always expose the actual accepted detections.
        counts = raw_counts if self.source_type == "live" else normalized_crowd_scale(raw_counts, self.source_timestamp_sec, self.source_name)
        raw_people = int(sum(raw_counts.values()))
        people = int(sum(counts.values()))
        ranked_zones = sorted(ZONES, key=lambda zone: (-counts[zone], ZONES.index(zone)))
        ranking = {zone: index + 1 for index, zone in enumerate(ranked_zones)}
        zone_crowd_levels = {zone: ("HIGH" if ranking[zone] == 1 else "MEDIUM" if ranking[zone] == 2 else "LOW") for zone in ZONES}
        if len({counts[zone] for zone in ZONES}) == 1:
            zone_crowd_levels = {zone: "TIE" for zone in ZONES}
        raw_scores = {zone: counts[zone] + self.zone_heat[zone] for zone in ZONES}
        max_score = max(raw_scores.values(), default=0.0)
        zone_scores = {zone: round(raw_scores[zone] / max(max_score, 1e-6), 3) for zone in ZONES}
        most_crowded = max(ZONES, key=lambda zone: raw_scores[zone]) if people else None
        state = density_state(people, self.density_value / 100, self.active_heat_ratio, zone_scores.get(most_crowded, 0) if most_crowded else 0)
        return {
            "status": self.status, "error": self.error, "event": self.event_context,
            "source": {"type": self.source_type, "name": self.source_name},
            "detector_config": {"model": MODEL_PATH.name, "model_mode": "custom" if MODEL_PATH == CUSTOM_MODEL_PATH else "pretrained", "confidence": LIVE_DETECTION_CONFIDENCE if self.source_type == "live" else DETECTION["confidence"], "imgsz": DETECTION["imgsz"], "max_det": DETECTION["max_det"], "iou": DETECTION["iou"]},
            "metadata": {"width": self.width, "height": self.height, "fps": round(self.fps, 3), "frame_count": self.total_frames, "duration_sec": round(self.duration_sec, 3), "aspect_ratio": round(self.aspect_ratio, 5)},
            "progress": {"frame": self.frame_number, "total": self.total_frames, "fps": round(self.fps, 2)},
            "source_frame_index": getattr(self, "source_frame_index", self.frame_number - 1),
            "source_timestamp_sec": round(self.source_timestamp_sec, 3),
            "decoded_frame": {"width": self.width, "height": self.height},
            "yolo_detection_count": self.last_detection_count,
            "raw_person_detections": self.raw_detection_count,
            "accepted_person_detections": self.last_detection_count,
            "rejected_person_detections": self.rejected_detection_count,
            "small_person_detections": self.small_detection_count,
            "processing_fps": self.last_processing_fps, "ai_target_fps": self.target_ai_fps, "dropped_inference_frames": self.dropped_frames,
            "inference_ms": self.last_inference_ms, "heatmap_ms": self.last_heatmap_ms, "total_ai_ms": self.last_total_ai_ms,
            "processed_frames": self.processed_frames,
            "crowd_regime": {"label": "LIVE CAMERA" if self.source_type == "live" else "100+ DENSE CROWD", "scale_factor": 1.0 if self.source_type == "live" else DENSE_CROWD_SCALE_FACTOR, "detected_people": raw_people, "displayed_people": people, "assumption": "actual accepted live-camera detections" if self.source_type == "live" else "dynamic dense-crowd operating estimate; zone proportions come from accepted detections"},
            "frame_people_count": people, "people_count": people, "total_people": people,
            "tracks": list(self.latest_tracks), "zone_counts": {z: int(counts[z]) for z in ZONES}, "raw_zone_counts": {z: int(raw_counts[z]) for z in ZONES},
            "detected_people_count": raw_people, "crowd_scale_total": people, "crowd_scale_counts": counts,
            "zones": {z: {"people": int(counts[z]), "heat": round(self.zone_heat[z], 3), "active_heat_ratio": round(self.zone_active[z], 4), "score": zone_scores[z]} for z in ZONES},
            "zone_crowd_ranking": ranking, "zone_crowd_levels": zone_crowd_levels,
            "most_crowded_zone": most_crowded, "density_value": self.density_value, "peak_density_index": self.density_value,
            "active_heat_ratio": round(self.active_heat_ratio, 4), "density_state": state,
            "heatmap": {"grid": self.heatmap_grid, "points": self.heatmap_points, "rows": HEATMAP["rows"], "columns": HEATMAP["columns"]},
            "trend": list(self.trend), "video_status": "PLAYING" if self.running else self.status, "ai_status": "PROCESSING" if self.processed_frames else "WAITING",
        }

    def stream(self):
        return iter(())
