"""Browser-facing Phase 2 engine.

This module reuses the integrated Phase 2 feature, risk, forecast, and XAI
functions. It adds only a web transport layer: JPEG frames and analytics JSON.
"""

from collections import deque
import base64
import math
import threading
import time

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from phase2.phase2_demo import (
    BASE_DIR,
    CONFIDENCE,
    CURRENT_MODEL_PATH,
    FORECAST_MODEL_PATH,
    IMAGE_SIZE,
    ZONES,
    build_global_window,
    build_zone_features,
    calculate_current_risk,
    calculate_zone_flow,
    direction_name,
    load_package,
    predict_forecast,
    risk_color,
)

# Keep the browser-facing API responsive while CPU inference runs in its
# background worker. The default Torch thread pool can otherwise starve the
# single Uvicorn worker during a detector call.
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


class Phase2WebEngine:
    """One model-loaded inference engine per browser monitoring session."""

    def __init__(self, event_context, source_name, source_type="recorded"):
        self.event_context = event_context
        self.source_name = source_name
        self.source_type = source_type
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.detector = YOLO(str(BASE_DIR.parent / "yolo11s.pt"))
        self.current_model, self.current_package = load_package(CURRENT_MODEL_PATH)
        self.forecast_model, self.forecast_package = load_package(FORECAST_MODEL_PATH)
        self.current_columns = self.current_package.get("feature_columns", [])
        self.track_memory = {}
        self.history = deque(maxlen=3)
        self.previous_features = {zone: None for zone in ZONES}
        self.displayed_scores = {zone: None for zone in ZONES}
        self.current_results = {zone: {"level": "COLLECTING", "risk_score": None, "explanation": []} for zone in ZONES}
        self.current_flows = {zone: {"flow": "NO DATA", "motion_percent": 0.0, "avg_speed": 0.0} for zone in ZONES}
        self.forecast_result = {"status": "COLLECTING", "probability": None, "explanation": []}
        self.window_rows = {zone: [] for zone in ZONES}
        self.window_counts = {zone: [] for zone in ZONES}
        self.window_frames = 60
        self.window_sequence = 0
        self.frame_number = 0
        self.total_frames = 0
        self.fps = 30.0
        self.width = 0
        self.height = 0
        self.latest_jpeg = None
        self.latest_zone_counts = {zone: 0 for zone in ZONES}
        self.raw_zone_counts = {zone: 0 for zone in ZONES}
        self.stable_zone_counts = {zone: 0 for zone in ZONES}
        self.grace_frames = 24
        self.status = "IDLE"
        self.error = None
        self.running = False
        self.lock = threading.Lock()
        self.pending_condition = threading.Condition(self.lock)
        self.pending_frame = None
        self.pending_timestamp_sec = None
        self.reader_thread = None
        self.inference_thread = None
        self.reader_finished = False
        self.capture = None
        self.thread = None
        self.last_detection_count = 0
        self.last_processing_fps = 0.0
        self.target_ai_fps = 8.0
        self.dropped_frames = 0
        self.processed_frames = 0
        self.source_timestamp_sec = 0.0
        self.window_start_timestamp = None
        self.last_inference_metrics = {}
        self.latest_tracks = []
        self.latest_frame_people_count = 0
        print(f"AI Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    def start(self):
        if self.source_type == "recorded":
            video_path = BASE_DIR / "videos" / self.source_name
            self.capture = cv2.VideoCapture(str(video_path))
            if not self.capture.isOpened():
                raise RuntimeError(f"Unable to open recorded video: {self.source_name}")
            self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.window_frames = max(1, int(round(self.fps * 2.0)))
            self.running = True
            self.status = "LOADING"
            self.reader_finished = False
            self.reader_thread = threading.Thread(target=self._recorded_reader_loop, daemon=True)
            self.inference_thread = threading.Thread(target=self._recorded_inference_loop, daemon=True)
            self.thread = self.reader_thread
            self.reader_thread.start()
            self.inference_thread.start()
        else:
            self.fps = 5.0
            self.window_frames = 10
            self.running = True
            self.status = "CAMERA_READY"

    def _recorded_reader_loop(self):
        """Decode at source speed and publish only the newest AI sample."""
        try:
            started_at = time.perf_counter()
            next_sample_timestamp = 0.0
            frame_index = 0
            while self.running:
                ok, frame = self.capture.read()
                if not ok:
                    break
                timestamp = frame_index / max(self.fps, 1.0)
                frame_index += 1
                if timestamp + 1e-6 >= next_sample_timestamp:
                    with self.pending_condition:
                        if self.pending_frame is not None:
                            self.dropped_frames += 1
                        self.pending_frame = frame
                        self.pending_timestamp_sec = timestamp
                        self.pending_condition.notify()
                    next_sample_timestamp += 1.0 / max(self.target_ai_fps, 1.0)
                wait_for = started_at + timestamp + (1.0 / max(self.fps, 1.0)) - time.perf_counter()
                if wait_for > 0:
                    time.sleep(min(wait_for, 0.04))
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
            self.running = False

    def _recorded_inference_loop(self):
        """Run AI asynchronously; stale pending frames are discarded."""
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
                frame = self.pending_frame
                timestamp = self.pending_timestamp_sec
                self.pending_frame = None
                self.pending_timestamp_sec = None
            try:
                self.process_frame(frame, timestamp)
            except Exception as error:
                self.error = str(error)
                self.status = "ERROR"
                break

    def stop(self):
        self.running = False
        with self.pending_condition:
            self.pending_condition.notify_all()
        worker = self.thread
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=8.0)
        inference_worker = self.inference_thread
        if inference_worker is not None and inference_worker.is_alive() and inference_worker is not threading.current_thread():
            inference_worker.join(timeout=8.0)
        if self.capture is not None and (worker is None or not worker.is_alive()):
            self.capture.release()
            self.capture = None
        self.status = "STOPPED"

    def process_live_base64(self, image_base64, source_timestamp_sec=None):
        encoded = image_base64.split(",", 1)[-1]
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0 or frame.dtype != np.uint8:
            raise ValueError("Invalid camera frame")
        return self.process_frame(frame, source_timestamp_sec)

    def process_frame(self, frame, source_timestamp_sec=None):
        started_at = time.perf_counter()
        self.frame_number += 1
        self.processed_frames += 1
        if source_timestamp_sec is None:
            source_timestamp_sec = self.frame_number / max(self.fps, 1.0)
        source_timestamp_sec = float(source_timestamp_sec)
        self.source_timestamp_sec = source_timestamp_sec
        if self.window_start_timestamp is None:
            self.window_start_timestamp = source_timestamp_sec
        self.height, self.width = frame.shape[:2]
        self.fps = self.fps if self.fps > 0 else 30.0
        diagonal = math.hypot(self.width, self.height)
        zone_people = {zone: [] for zone in ZONES}
        zone_counts = {zone: 0 for zone in ZONES}
        current_tracks = []
        self.raw_zone_counts = dict(zone_counts)
        yolo_started = time.perf_counter()
        with torch.inference_mode():
            results = self.detector.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=CONFIDENCE, imgsz=IMAGE_SIZE, device=self.device, verbose=False)
        yolo_ms = (time.perf_counter() - yolo_started) * 1000
        boxes = results[0].boxes if results and results[0].boxes is not None else None
        self.last_detection_count = int(len(boxes)) if boxes is not None else 0
        if boxes is not None and len(boxes) > 0:
            coordinates = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(coordinates)
            for box, track_id, confidence in zip(coordinates, ids, confidences):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                ratio = cx / self.width if self.width else 0.0
                zone = "ZONE_A" if ratio < 1 / 3 else "ZONE_B" if ratio < 2 / 3 else "ZONE_C"
                zone_counts[zone] += 1
                previous = self.track_memory.get(track_id) if track_id is not None else None
                motion_valid = 0
                dx = dy = distance_px = speed_norm = acceleration = direction_change = angle = 0.0
                direction_sin, direction_cos = 0.0, 1.0
                if previous and source_timestamp_sec > previous.get("timestamp", -1):
                    dt = max(1e-3, source_timestamp_sec - previous.get("timestamp", source_timestamp_sec))
                    dx, dy = cx - previous["x"], cy - previous["y"]
                    distance_px = math.hypot(dx, dy)
                    distance_norm = distance_px / diagonal if diagonal else 0.0
                    speed_norm = distance_norm / dt
                    acceleration = (speed_norm - previous.get("speed_norm", 0.0)) / dt if previous.get("motion_valid") else 0.0
                    angle = math.degrees(math.atan2(-dy, dx)) % 360
                    old_angle = previous.get("angle", angle)
                    difference = abs(angle - old_angle)
                    direction_change = min(difference, 360 - difference)
                    radians = math.radians(angle)
                    direction_sin, direction_cos = math.sin(radians), math.cos(radians)
                    motion_valid = 1
                else:
                    distance_norm = 0.0
                area_ratio = ((x2 - x1) * (y2 - y1)) / (self.width * self.height) if self.width and self.height else 0.0
                zone_people[zone].append({"track_id": track_id, "motion_valid": motion_valid, "movement_distance_px": distance_px, "movement_distance_norm": distance_norm, "speed_norm_s": speed_norm, "acceleration_norm_s2": acceleration, "direction_change_deg": direction_change, "direction_sin": direction_sin, "direction_cos": direction_cos, "bbox_area_ratio": area_ratio, "detection_confidence": float(confidence)})
                current_tracks.append({
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(float(confidence), 3),
                    "zone": zone,
                    "motion_valid": bool(motion_valid),
                    "direction": direction_name(dx, dy) if motion_valid else "STILL",
                })
                if track_id is not None:
                    self.track_memory[track_id] = {"x": cx, "y": cy, "frame": self.frame_number, "timestamp": source_timestamp_sec, "zone": zone, "speed_norm": speed_norm, "angle": angle, "motion_valid": motion_valid}
                cv2.rectangle(frame, (x1, y1), (x2, y2), (54, 180, 110), 2)
                cv2.putText(frame, f"ID {track_id}" if track_id is not None else "ID --", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .42, (54, 180, 110), 1, cv2.LINE_AA)
                if motion_valid and distance_px >= 1:
                    length = max(14, min(30, 14 + speed_norm * 240))
                    end = (int(cx + dx / distance_px * length), int(cy + dy / distance_px * length))
                    cv2.arrowedLine(frame, (cx, cy), end, (40, 190, 220), 1, cv2.LINE_AA, tipLength=.28)

        for zone in ZONES:
            self.window_rows[zone].extend(zone_people[zone])
            self.window_counts[zone].append(len(zone_people[zone]))
        a, b = self.width // 3, (2 * self.width) // 3
        shade = frame.copy()
        cv2.rectangle(shade, (0, 0), (a, self.height), (210, 160, 120), -1)
        cv2.rectangle(shade, (a, 0), (b, self.height), (180, 120, 210), -1)
        cv2.rectangle(shade, (b, 0), (self.width, self.height), (190, 210, 120), -1)
        frame = cv2.addWeighted(shade, 0.07, frame, 0.93, 0)
        cv2.line(frame, (a, 0), (a, self.height), (80, 190, 220), 2, cv2.LINE_AA)
        cv2.line(frame, (b, 0), (b, self.height), (80, 190, 220), 2, cv2.LINE_AA)
        cv2.putText(frame, "ZONE A", (max(8, a // 2 - 25), 24), cv2.FONT_HERSHEY_SIMPLEX, .42, (80, 190, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, "ZONE B", (a + max(8, (b - a) // 2 - 25), 24), cv2.FONT_HERSHEY_SIMPLEX, .42, (80, 190, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, "ZONE C", (b + max(8, (self.width - b) // 2 - 25), 24), cv2.FONT_HERSHEY_SIMPLEX, .42, (80, 190, 220), 1, cv2.LINE_AA)

        grace = max(3, int(self.fps * 0.8))
        active_tracks = [item for item in self.track_memory.values() if self.frame_number - item["frame"] <= grace]
        if active_tracks:
            self.stable_zone_counts = {zone: sum(item.get("zone") == zone for item in active_tracks) for zone in ZONES}
        else:
            self.stable_zone_counts = dict(zone_counts)
        self.latest_zone_counts = dict(self.stable_zone_counts)
        self.latest_tracks = current_tracks
        self.latest_frame_people_count = len(current_tracks)
        stable_people = int(sum(self.stable_zone_counts.values()))
        cv2.rectangle(frame, (10, 34), (190, 64), (15, 23, 42), -1)
        cv2.putText(frame, f"PEOPLE: {stable_people}", (18, 55), cv2.FONT_HERSHEY_SIMPLEX, .58, (235, 245, 255), 1, cv2.LINE_AA)

        if source_timestamp_sec - self.window_start_timestamp >= 2.0:
            self.window_sequence += 1
            expected_window_samples = max(1, len(next(iter(self.window_counts.values()))))
            features = {zone: build_zone_features(self.window_rows[zone], self.window_counts[zone], expected_window_samples) for zone in ZONES}
            available = [item for item in features.values() if item is not None]
            for zone in ZONES:
                feature_row = features[zone]
                self.current_flows[zone] = calculate_zone_flow(self.window_rows[zone], feature_row)
                if feature_row is not None:
                    previous = self.previous_features[zone]
                    self.current_flows[zone].update({
                        "avg_acceleration": feature_row["avg_acceleration_norm_s2"],
                        "avg_direction_change": feature_row["avg_direction_change_deg"],
                        "crowd_trend": (feature_row["avg_person_count_per_frame"] - previous["avg_person_count_per_frame"]) if previous else 0.0,
                        "motion_trend": (feature_row["movement_activity_ratio"] - previous["movement_activity_ratio"]) if previous else 0.0,
                    })
                probability = None
                if feature_row is not None and any(item["track_id"] is not None for item in self.window_rows[zone]):
                    model_input = pd.DataFrame([[feature_row[column] for column in self.current_columns]], columns=self.current_columns)
                    probability = float(self.current_model.predict_proba(model_input)[0][1])
                result = calculate_current_risk(feature_row, available, self.previous_features[zone], probability)
                if result["risk_score"] is not None:
                    old = self.displayed_scores[zone]
                    result["raw_score"] = result["risk_score"]
                    result["risk_score"] = result["risk_score"] if old is None else .45 * result["risk_score"] + .55 * old
                    result["level"] = "LOW" if result["risk_score"] < 33 else "MEDIUM" if result["risk_score"] < 66 else "HIGH"
                    self.displayed_scores[zone] = result["risk_score"]
                    self.previous_features[zone] = feature_row
                self.current_results[zone] = result
            global_window = build_global_window(features)
            if global_window is not None:
                self.history.append(global_window)
                self.forecast_result = predict_forecast(self.forecast_model, self.forecast_package, list(self.history))
            self.window_rows = {zone: [] for zone in ZONES}
            self.window_counts = {zone: [] for zone in ZONES}
            self.window_start_timestamp = source_timestamp_sec
            self.status = "ANALYZING"
        stale = [track_id for track_id, item in self.track_memory.items() if self.frame_number - item["frame"] > max(self.grace_frames, int(self.fps * 2))]
        for track_id in stale:
            del self.track_memory[track_id]
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            with self.lock:
                self.latest_jpeg = encoded.tobytes()
        elapsed = max(1e-6, time.perf_counter() - started_at)
        self.last_processing_fps = 1.0 / elapsed
        self.last_inference_metrics = {
            "decode_ms": 0.0,
            "yolo_ms": round(yolo_ms, 2),
            "total_inference_ms": round(elapsed * 1000, 2),
            "dropped_frames": self.dropped_frames,
        }
        if self.processed_frames % 50 == 0:
            print(
                f"AI FPS: {self.last_processing_fps:.1f} | "
                f"YOLO: {yolo_ms:.0f}ms | "
                f"Total inference: {elapsed * 1000:.0f}ms | "
                f"Dropped frames: {self.dropped_frames}"
            )
        return self.analytics(self.stable_zone_counts)

    def analytics(self, zone_counts=None):
        zone_counts = zone_counts or self.latest_zone_counts
        scored = [(zone, result) for zone, result in self.current_results.items() if result.get("risk_score") is not None]
        focus = max(scored, key=lambda item: item[1]["risk_score"]) if scored else None
        with self.lock:
            latest_tracks = list(self.latest_tracks)
        return {
            "status": self.status,
            "error": self.error,
            "event": self.event_context,
            "source": {"type": self.source_type, "name": self.source_name},
            "progress": {"frame": self.frame_number, "total": self.total_frames, "fps": self.fps},
            "source_timestamp_sec": round(self.source_timestamp_sec, 3),
            "decoded_frame": {"width": self.width, "height": self.height, "dtype": "uint8", "channels": 3},
            "yolo_detection_count": self.last_detection_count,
            "processing_fps": round(self.last_processing_fps, 2),
            "ai_target_fps": self.target_ai_fps,
            "dropped_inference_frames": self.dropped_frames,
            "performance": self.last_inference_metrics,
            "frame_people_count": int(self.latest_frame_people_count),
            "tracks": latest_tracks,
            "total_people": int(sum(zone_counts.values())),
            "raw_detected_people": int(sum(self.raw_zone_counts.values())),
            "active_track_count": int(sum(zone_counts.values())),
            "stable_people_count": int(sum(zone_counts.values())),
            "stable_zone_counts": {zone: int(zone_counts[zone]) for zone in ZONES},
            "focus_zone": focus[0] if focus else None,
            "current_risk": focus[1] if focus else {"level": "COLLECTING", "risk_score": None, "explanation": []},
            "zones": {zone: {"people": int(zone_counts[zone]), **self.current_flows[zone], **self.current_results[zone]} for zone in ZONES},
            "forecast": self.forecast_result,
            "history_windows": len(self.history),
            "window_sequence": self.window_sequence,
        }

    def stream(self):
        while self.running or self.latest_jpeg is not None:
            with self.lock:
                frame = self.latest_jpeg
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
            if not self.running and frame is None:
                break
            time.sleep(0.04)
