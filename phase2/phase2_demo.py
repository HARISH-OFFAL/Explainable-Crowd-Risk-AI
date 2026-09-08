"""Integrated Phase 2 inference demo.

The demo is inference-only: it never regenerates datasets, labels, or models.
It preserves the training-time zone and compact 6-second forecast formulas.
LOW/MEDIUM/HIGH are transparent decision-support bands, not validated
three-class stampede-risk predictions. The forecast is an experimental
research baseline because its held-out unseen-sequence performance was poor.
"""

from collections import deque
from pathlib import Path
import argparse
import math
import time
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "videos"
MODEL_DIR = BASE_DIR / "models"
YOLO_MODEL = BASE_DIR.parent / "yolo11s.pt"
CURRENT_MODEL_PATH = MODEL_DIR / "umn_zone_abnormal_classifier_v2.joblib"
FORECAST_MODEL_PATH = MODEL_DIR / "umn_final_6sec_forecast_model.joblib"

MODEL_NAME = str(YOLO_MODEL) if YOLO_MODEL.exists() else "yolo11s.pt"
CONFIDENCE = 0.15
IMAGE_SIZE = 960
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WINDOW_SECONDS = 2.0
HISTORY_WINDOWS = 3
DISPLAY_MAX_WIDTH = 1280
DISPLAY_MAX_HEIGHT = 720
VIDEO_DISPLAY_WIDTH = 1000
SIDEBAR_WIDTH = DISPLAY_MAX_WIDTH - VIDEO_DISPLAY_WIDTH
SMOOTHING_ALPHA = 0.45
ZONES = ("ZONE_A", "ZONE_B", "ZONE_C")

V2_FEATURES = (
    "zone_presence_ratio", "unique_person_count",
    "avg_person_count_per_frame", "max_person_count_per_frame",
    "avg_movement_distance_norm", "avg_speed_norm_s", "max_speed_norm_s",
    "speed_std_norm_s", "avg_acceleration_norm_s2",
    "acceleration_std_norm_s2", "avg_direction_change_deg",
    "direction_consistency", "movement_activity_ratio",
)

GLOBAL_FEATURES = (
    "global_unique_person_count", "global_avg_person_count_per_frame",
    "global_max_zone_person_count", "global_avg_speed_norm_s",
    "global_speed_std_norm_s", "global_avg_acceleration_norm_s2",
    "global_avg_direction_change_deg", "global_direction_consistency",
    "global_movement_activity_ratio",
)
DELTA_FEATURES = GLOBAL_FEATURES[:2] + GLOBAL_FEATURES[3:]


def load_package(path):
    if not path.exists():
        raise FileNotFoundError(f"Model package not found: {path}")
    package = joblib.load(path)
    if isinstance(package, dict):
        model = package.get("model") or package.get("pipeline") or package.get("classifier")
        if model is None:
            raise ValueError(f"No model object found in {path}")
        return model, package
    return package, {}


def safe_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def clip01(value):
    return max(0.0, min(1.0, float(value)))


def get_available_videos():
    return sorted(p for p in VIDEOS_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}) if VIDEOS_DIR.exists() else []


def select_video(preferred_name=None):
    videos = get_available_videos()
    if not videos:
        print(f"No supported videos found in {VIDEOS_DIR}")
        return None
    if preferred_name:
        preferred = Path(preferred_name).name
        selected = next((path for path in videos if path.name == preferred), None)
        if selected is None:
            raise FileNotFoundError(f"Selected demo video was not found in {VIDEOS_DIR}: {preferred_name}")
        return selected
    print("\nPHASE 2 - SPATIO-TEMPORAL CROWD ANALYSIS")
    for index, path in enumerate(videos, 1):
        print(f"{index}. {path.name}")
    print("0. Exit")
    while True:
        choice = input("\nSelect video number: ").strip()
        if choice.isdigit() and 0 <= int(choice) <= len(videos):
            return None if int(choice) == 0 else videos[int(choice) - 1]
        print("Enter a valid number.")


def fetch_event_context(event_id, api_url):
    request = Request(f"{api_url.rstrip('/')}/events/{event_id}", method="GET")
    with urlopen(request, timeout=3) as response:
        import json
        return json.loads(response.read().decode("utf-8"))


class MonitoringReporter:
    """Best-effort periodic bridge from the inference demo to FastAPI."""

    def __init__(self, session_id=None, api_url="http://127.0.0.1:8000"):
        self.session_id = session_id
        self.api_url = api_url.rstrip("/")

    def post(self, path, payload):
        if not self.session_id:
            return
        try:
            import json
            request = Request(
                f"{self.api_url}{path}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=3):
                return
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            print(f"Monitoring result sync skipped: {error}")

    def risk_update(self, payload):
        self.post(f"/monitoring-sessions/{self.session_id}/risk-update", payload)

    def stop(self):
        self.post(f"/monitoring-sessions/{self.session_id}/stop", {})


def zone_for_x(center_x, width):
    ratio = center_x / width if width else 0.0
    return "ZONE_A" if ratio < 1 / 3 else "ZONE_B" if ratio < 2 / 3 else "ZONE_C"


def direction_name(dx, dy):
    if abs(dx) < 1 and abs(dy) < 1:
        return "STILL"
    angle = math.degrees(math.atan2(-dy, dx)) % 360
    labels = ("RIGHT", "UP-RIGHT", "UP", "UP-LEFT", "LEFT", "DOWN-LEFT", "DOWN", "DOWN-RIGHT")
    return labels[int((angle + 22.5) // 45) % 8]


def direction_from_angle(angle):
    """Map a circular mean image-space angle to a readable flow label."""
    labels = ("RIGHT", "UP-RIGHT", "UP", "UP-LEFT", "LEFT", "DOWN-LEFT", "DOWN", "DOWN-RIGHT")
    return labels[int((angle % 360 + 22.5) // 45) % 8]


def calculate_zone_flow(rows, features):
    """Summarise only valid tracked motion; never infer flow from new tracks."""
    valid = [row for row in rows if row.get("motion_valid")]
    if not valid:
        return {"flow": "NO DATA", "motion_percent": 0.0, "avg_speed": 0.0, "direction_consistency": 0.0}

    moving = [row for row in valid if row.get("movement_distance_px", 0.0) >= 1.0]
    motion_percent = 100.0 * len(moving) / len(valid)
    avg_speed = float(np.mean([row.get("speed_norm_s", 0.0) for row in valid]))
    if motion_percent < 10.0:
        return {"flow": "STILL", "motion_percent": motion_percent, "avg_speed": avg_speed, "direction_consistency": 1.0}

    sin_mean = float(np.mean([row["direction_sin"] for row in moving]))
    cos_mean = float(np.mean([row["direction_cos"] for row in moving]))
    consistency = float(math.hypot(sin_mean, cos_mean))
    if consistency < 0.55:
        flow = "MIXED"
    else:
        flow = direction_from_angle(math.degrees(math.atan2(sin_mean, cos_mean)))
    return {"flow": flow, "motion_percent": motion_percent, "avg_speed": avg_speed, "direction_consistency": consistency}


def risk_color(label):
    return {"LOW": (80, 205, 110), "MEDIUM": (0, 215, 245), "HIGH": (70, 80, 235)}.get(label, (190, 195, 200))


def build_zone_features(rows, frame_counts, expected_frames):
    if not rows:
        return None
    speeds = np.asarray([r["speed_norm_s"] for r in rows], dtype=float)
    accelerations = np.asarray([r["acceleration_norm_s2"] for r in rows], dtype=float)
    changes = np.asarray([r["direction_change_deg"] for r in rows], dtype=float)
    movement = np.asarray([r["movement_distance_norm"] for r in rows], dtype=float)
    sin_values = np.asarray([r["direction_sin"] for r in rows], dtype=float)
    cos_values = np.asarray([r["direction_cos"] for r in rows], dtype=float)
    presence = sum(n > 0 for n in frame_counts) / expected_frames if expected_frames else 0.0
    track_ids = {r["track_id"] for r in rows if r["track_id"] is not None}
    return {
        "zone_presence_ratio": float(presence),
        "unique_person_count": float(len(track_ids)),
        "avg_person_count_per_frame": float(np.mean(frame_counts)),
        "max_person_count_per_frame": float(np.max(frame_counts)),
        "avg_movement_distance_norm": float(np.mean(movement)),
        "avg_speed_norm_s": float(np.mean(speeds)),
        "max_speed_norm_s": float(np.max(speeds)),
        "speed_std_norm_s": safe_std(speeds),
        "avg_acceleration_norm_s2": float(np.mean(accelerations)),
        "acceleration_std_norm_s2": safe_std(accelerations),
        "avg_direction_change_deg": float(np.mean(changes)),
        "direction_consistency": float(math.sqrt(float(np.mean(sin_values)) ** 2 + float(np.mean(cos_values)) ** 2)),
        "movement_activity_ratio": float(np.mean(speeds > 0)),
    }


def build_global_window(zone_features):
    available = [f for f in zone_features.values() if f is not None]
    if not available:
        return None
    return {
        "global_unique_person_count": float(sum(f["unique_person_count"] for f in available)),
        "global_avg_person_count_per_frame": float(sum(f["avg_person_count_per_frame"] for f in available)),
        "global_max_zone_person_count": float(max(f["max_person_count_per_frame"] for f in available)),
        "global_avg_speed_norm_s": float(np.mean([f["avg_speed_norm_s"] for f in available])),
        "global_speed_std_norm_s": float(np.mean([f["speed_std_norm_s"] for f in available])),
        "global_avg_acceleration_norm_s2": float(np.mean([f["avg_acceleration_norm_s2"] for f in available])),
        "global_avg_direction_change_deg": float(np.mean([f["avg_direction_change_deg"] for f in available])),
        "global_direction_consistency": float(np.mean([f["direction_consistency"] for f in available])),
        "global_movement_activity_ratio": float(np.mean([f["movement_activity_ratio"] for f in available])),
    }


def relative_scale(value, peers, neutral=0.0):
    if not peers:
        return neutral
    low, high = min(peers), max(peers)
    return clip01((value - low) / (high - low)) if high > low else neutral


def calculate_current_risk(features, all_features, previous_features=None, behavior_probability=None):
    """Return an inspectable decision-support risk band, not a trained 3-class prediction."""
    if features is None:
        return {"level": "NO DATA", "risk_score": None, "crowd_evidence": 0.0, "behavior_evidence": 0.0, "trend_evidence": 0.0, "explanation": ["No tracked people in this zone/window"]}
    occupied = [f for f in all_features if f is not None]
    counts = [f["avg_person_count_per_frame"] for f in occupied]
    shares_total = sum(counts) or 1.0
    share = features["avg_person_count_per_frame"] / shares_total
    relative_count = relative_scale(features["avg_person_count_per_frame"], counts)
    crowd = 100 * (0.45 * clip01(share * len(occupied)) + 0.35 * relative_count + 0.20 * features["zone_presence_ratio"])
    speed_instability = relative_scale(features["speed_std_norm_s"], [f["speed_std_norm_s"] for f in occupied])
    accel_instability = relative_scale(features["acceleration_std_norm_s2"], [f["acceleration_std_norm_s2"] for f in occupied])
    direction_instability = clip01(features["avg_direction_change_deg"] / 90.0)
    activity = clip01(features["movement_activity_ratio"])
    model_signal = clip01(behavior_probability if behavior_probability is not None else 0.0)
    behavior = 100 * (0.25 * speed_instability + 0.20 * accel_instability + 0.20 * direction_instability + 0.15 * activity + 0.20 * model_signal)
    trend = 0.0
    if previous_features is not None:
        count_change = (features["avg_person_count_per_frame"] - previous_features["avg_person_count_per_frame"]) / max(previous_features["avg_person_count_per_frame"], 1.0)
        activity_change = features["movement_activity_ratio"] - previous_features["movement_activity_ratio"]
        trend = 100 * (0.65 * clip01((count_change + 0.25) / 0.75) + 0.35 * clip01((activity_change + 0.25) / 0.75))
    score = 0.35 * crowd + 0.45 * behavior + 0.20 * trend
    level = "LOW" if score < 33 else "MEDIUM" if score < 66 else "HIGH"
    reasons = []
    if crowd >= 55: reasons.append("Relative crowd concentration is elevated")
    elif crowd < 30: reasons.append("Crowd load is low relative to the scene")
    if behavior >= 55: reasons.append("Movement variability indicates unstable behaviour")
    elif features["movement_activity_ratio"] < 0.25: reasons.append("Crowd remains dense but movement is orderly")
    if trend >= 55: reasons.append("Recent 2-second window shows increasing load or activity")
    elif trend < 25 and previous_features is not None: reasons.append("Recent crowd and activity trend is stable")
    if behavior_probability is not None:
        reasons.append(f"Behaviour baseline evidence: {behavior_probability * 100:.0f}% abnormal")
    return {"level": level, "risk_score": float(score), "crowd_evidence": float(crowd), "behavior_evidence": float(behavior), "trend_evidence": float(trend), "explanation": reasons[:3] or ["Evidence is limited; interpret this band cautiously"]}


def build_forecast_row(history, feature_columns):
    if len(history) < HISTORY_WINDOWS:
        return None
    result = {}
    for index, row in enumerate(history):
        for feature in GLOBAL_FEATURES:
            result[f"H{index}_{feature}"] = float(row[feature])
    for feature in DELTA_FEATURES:
        result[f"DELTA_{feature}"] = float(history[-1][feature] - history[0][feature])
    missing = [c for c in feature_columns if c not in result]
    if missing:
        raise ValueError(f"Forecast feature construction missing columns: {missing}")
    return pd.DataFrame([[result[c] for c in feature_columns]], columns=feature_columns)


def predict_forecast(model, package, history):
    columns = package.get("feature_columns")
    if not columns:
        raise ValueError("Forecast model package has no feature_columns")
    row = build_forecast_row(history, columns)
    if row is None:
        return {"status": "COLLECTING", "probability": None, "explanation": ["Waiting for three 2-second history windows"]}
    probability = float(model.predict_proba(row)[0][1])
    latest, oldest = history[-1], history[0]
    explanation = []
    if latest["global_movement_activity_ratio"] > oldest["global_movement_activity_ratio"] + 0.10:
        explanation.append("Movement activity increased across the 6-second history")
    if latest["global_speed_std_norm_s"] > oldest["global_speed_std_norm_s"] + 0.01:
        explanation.append("Speed variability increased across the 6-second history")
    if latest["global_avg_person_count_per_frame"] > oldest["global_avg_person_count_per_frame"] + 1:
        explanation.append("Average crowd count increased across the 6-second history")
    return {"status": "EXPERIMENTAL", "probability": probability, "explanation": explanation[:2] or ["No strong directional change in compact forecast features"]}


def overlay_panel(frame, x1, y1, x2, y2, alpha=0.72):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 25, 35), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def put(frame, text, origin, size=0.48, color=(235, 240, 245), thickness=1):
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)


def compose_display(video_frame, video_name, frame_number, total_frames, zone_counts, current_results, current_flows, forecast_result, history_count=0, event_context=None):
    """Fit the complete annotated source frame into the video area, then add a sidebar."""
    canvas = np.zeros((DISPLAY_MAX_HEIGHT, DISPLAY_MAX_WIDTH, 3), dtype=np.uint8)
    canvas[:, VIDEO_DISPLAY_WIDTH:] = (24, 30, 39)

    source_height, source_width = video_frame.shape[:2]
    scale = min(VIDEO_DISPLAY_WIDTH / source_width, DISPLAY_MAX_HEIGHT / source_height)
    fitted_width = max(1, round(source_width * scale))
    fitted_height = max(1, round(source_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    fitted = cv2.resize(video_frame, (fitted_width, fitted_height), interpolation=interpolation)
    offset_x = (VIDEO_DISPLAY_WIDTH - fitted_width) // 2
    offset_y = (DISPLAY_MAX_HEIGHT - fitted_height) // 2
    canvas[offset_y:offset_y + fitted_height, offset_x:offset_x + fitted_width] = fitted
    cv2.rectangle(canvas, (0, 0), (VIDEO_DISPLAY_WIDTH - 1, DISPLAY_MAX_HEIGHT - 1), (90, 100, 112), 1)

    panel_x = VIDEO_DISPLAY_WIDTH + 14
    context = event_context or {}
    event_name = str(context.get("event_name") or "Unlinked demo")
    location = str(context.get("location") or "—")
    event_id = context.get("id") or context.get("event_id") or "—"
    pre_event_risk = context.get("pre_event_risk") or {}
    pre_event_level = str(pre_event_risk.get("risk_level") or "—")
    short_event = event_name if len(event_name) <= 22 else event_name[:19] + "..."
    short_location = location if len(location) <= 22 else location[:19] + "..."
    put(canvas, "PHASE 2", (panel_x, 26), 0.56, (245, 248, 250), 2)
    put(canvas, "SPATIO-TEMPORAL AI", (panel_x, 44), 0.31, (165, 190, 205), 1)
    put(canvas, f"Event: {short_event}", (panel_x, 63), 0.31, (225, 230, 235))
    put(canvas, f"Location: {short_location}", (panel_x, 80), 0.31, (225, 230, 235))
    put(canvas, f"Pre-event risk: {pre_event_level}", (panel_x, 97), 0.29, risk_color(pre_event_level))
    put(canvas, f"ID: {event_id}  Video: {video_name}", (panel_x, 113), 0.27, (225, 230, 235))
    put(canvas, f"People: {sum(zone_counts.values())}  {frame_number}/{total_frames}", (panel_x, 127), 0.27, (225, 230, 235))

    card_x1 = VIDEO_DISPLAY_WIDTH + 8
    card_x2 = DISPLAY_MAX_WIDTH - 8
    # Explicit 720px-safe vertical budget. XAI is reserved before drawing cards.
    zone_top = 134
    card_height = 102
    card_gap = 5
    for index, zone in enumerate(ZONES):
        card_y1 = zone_top + index * (card_height + card_gap)
        card_y2 = card_y1 + card_height
        overlay_panel(canvas, card_x1, card_y1, card_x2, card_y2, alpha=0.9)
        result = current_results[zone]
        flow = current_flows[zone]
        label = result["level"]
        put(canvas, zone, (panel_x, card_y1 + 20), 0.43, (245, 248, 250), 2)
        put(canvas, f"People: {zone_counts[zone]}", (panel_x, card_y1 + 40), 0.33)
        put(canvas, f"Motion: {flow['motion_percent']:.0f}% active", (panel_x, card_y1 + 58), 0.32, (205, 225, 235))
        put(canvas, f"Flow: {flow['flow']}", (panel_x, card_y1 + 76), 0.32, (40, 220, 245), 1)
        score_text = "waiting" if result.get("risk_score") is None else f"{result['risk_score']:.0f}/100"
        put(canvas, f"Risk: {label}  {score_text}", (panel_x, card_y1 + 95), 0.32, risk_color(label), 2)

    forecast_y = 459
    overlay_panel(canvas, card_x1, forecast_y, card_x2, 505, alpha=0.9)
    put(canvas, "FUTURE RISK", (panel_x, forecast_y + 22), 0.40, (245, 248, 250), 2)
    if forecast_result["probability"] is None:
        forecast_text = f"Experimental: collecting {history_count}/3"
    else:
        forecast_text = f"Experimental: {forecast_result['probability'] * 100:.0f}% evidence"
    put(canvas, forecast_text, (panel_x, forecast_y + 48), 0.32, (245, 205, 110), 1)

    xai_y = 544
    xai_bottom = 708
    overlay_panel(canvas, card_x1, xai_y, card_x2, xai_bottom, alpha=0.9)
    put(canvas, "XAI / TOP FACTORS", (panel_x, xai_y + 21), 0.38, (245, 248, 250), 2)
    scored = [(zone, result) for zone, result in current_results.items() if result.get("risk_score") is not None]
    if scored:
        focus_zone, focus_result = max(scored, key=lambda item: item[1]["risk_score"])
        put(canvas, f"Focus: {focus_zone}", (panel_x, xai_y + 40), 0.32, risk_color(focus_result["level"]), 1)
        top_reasons = focus_result.get("explanation") or ["No additional evidence explanation"]
        for index, reason in enumerate(top_reasons[:3]):
            short_reason = reason if len(reason) <= 31 else reason[:28] + "..."
            put(canvas, f"{index + 1}. {short_reason}", (panel_x, xai_y + 64 + index * 18), 0.27, (220, 225, 232))
    elif any(result.get("level") == "NO DATA" for result in current_results.values()):
        put(canvas, "No current crowd", (panel_x, xai_y + 45), 0.31, (190, 195, 200))
        put(canvas, "evidence available", (panel_x, xai_y + 63), 0.31, (190, 195, 200))
    else:
        put(canvas, "Waiting for first", (panel_x, xai_y + 45), 0.31, (190, 195, 200))
        put(canvas, "risk update...", (panel_x, xai_y + 63), 0.31, (190, 195, 200))
    put(canvas, "Q: exit", (panel_x, 716), 0.29, (145, 155, 165))
    return canvas


def analyze_video(video_path, event_context=None, session_id=None, api_url="http://127.0.0.1:8000"):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Video has invalid dimensions")
    fps = fps if fps > 0 else 30.0
    diagonal = math.hypot(width, height)
    detector = YOLO(MODEL_NAME)
    current_model, current_package = load_package(CURRENT_MODEL_PATH)
    forecast_model, forecast_package = load_package(FORECAST_MODEL_PATH)
    reporter = MonitoringReporter(session_id=session_id, api_url=api_url)
    current_columns = current_package.get("feature_columns", list(V2_FEATURES))
    missing_current = [c for c in V2_FEATURES if c not in current_columns]
    if missing_current:
        raise ValueError(f"Current-risk package is incompatible; missing {missing_current}")
    print(f"\nVideo: {video_path.name} | {width}x{height} | {fps:.2f} FPS | {total_frames} frames")
    print(f"Device: {'CUDA' if DEVICE != 'cpu' else 'CPU'} | Current risk: V2 behaviour baseline | Forecast: experimental")
    print("Press Q to stop.")
    window_title = "Phase 2 - Explainable Crowd Risk AI"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT)

    track_memory = {}
    history = deque(maxlen=HISTORY_WINDOWS)
    previous_window_features = {zone: None for zone in ZONES}
    displayed_scores = {zone: None for zone in ZONES}
    current_results = {zone: {"level": "COLLECTING", "risk_score": None, "explanation": []} for zone in ZONES}
    current_flows = {zone: {"flow": "NO DATA", "motion_percent": 0.0, "avg_speed": 0.0} for zone in ZONES}
    forecast_result = {"status": "COLLECTING", "probability": None, "explanation": []}
    window_frames = max(1, int(round(fps * WINDOW_SECONDS)))
    window_rows = {zone: [] for zone in ZONES}
    window_counts = {zone: [] for zone in ZONES}
    frame_number = 0
    last_processing_time = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_number += 1
        zone_people = {zone: [] for zone in ZONES}
        zone_counts = {zone: 0 for zone in ZONES}
        results = detector.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=CONFIDENCE, imgsz=IMAGE_SIZE, device=DEVICE, verbose=False)
        boxes = results[0].boxes if results and results[0].boxes is not None else None
        if boxes is not None and len(boxes) > 0:
            coordinates = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(coordinates)
            for box, track_id, confidence in zip(coordinates, ids, confidences):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                zone = zone_for_x(cx, width)
                zone_counts[zone] += 1
                motion_valid = 0
                previous_x = previous_y = None
                dx = dy = 0.0
                movement_distance_px = 0.0
                distance_norm = speed_px_s = speed_norm = acceleration = direction_change = 0.0
                direction_angle = 0.0
                direction_sin, direction_cos, direction_text = 0.0, 1.0, "NEW"
                if track_id is not None and track_id in track_memory:
                    old = track_memory[track_id]
                    previous_x, previous_y = old["x"], old["y"]
                    if frame_number - old["frame"] == 1 and fps > 0:
                        dt = 1.0 / fps
                        dx, dy = cx - previous_x, cy - previous_y
                        movement_distance_px = math.hypot(dx, dy)
                        distance_norm = movement_distance_px / diagonal if diagonal else 0.0
                        speed_px_s = movement_distance_px / dt
                        speed_norm = distance_norm / dt
                        if old.get("motion_valid"):
                            acceleration = (speed_norm - old.get("speed_norm_s", 0.0)) / dt
                        direction_angle = math.degrees(math.atan2(-dy, dx)) % 360
                        difference = abs(direction_angle - old.get("direction_angle_deg", direction_angle))
                        direction_change = min(difference, 360 - difference)
                        radians = math.radians(direction_angle)
                        direction_sin, direction_cos = math.sin(radians), math.cos(radians)
                        direction_text = direction_name(dx, dy)
                        motion_valid = 1
                bbox_ratio = ((x2 - x1) * (y2 - y1)) / (width * height) if width and height else 0.0
                zone_people[zone].append({"track_id": track_id, "normalized_x": cx / width if width else 0.0, "normalized_y": cy / height if height else 0.0, "bbox_area_ratio": bbox_ratio, "detection_confidence": float(confidence), "motion_valid": motion_valid, "previous_x": previous_x, "previous_y": previous_y, "dx": dx, "dy": dy, "movement_distance_px": movement_distance_px, "movement_distance_norm": distance_norm, "speed_px_s": speed_px_s, "speed_norm_s": speed_norm, "acceleration_norm_s2": acceleration, "direction_angle_deg": direction_angle, "direction_text": direction_text, "direction_change_deg": direction_change, "direction_sin": direction_sin, "direction_cos": direction_cos})
                if track_id is not None:
                    track_memory[track_id] = {"x": cx, "y": cy, "frame": frame_number, "speed_norm_s": speed_norm, "direction_angle_deg": direction_angle, "motion_valid": motion_valid}
                cv2.rectangle(frame, (x1, y1), (x2, y2), (75, 215, 120), 2)
                put(frame, f"ID {track_id}" if track_id is not None else "ID --", (x1, max(18, y1 - 6)), 0.42, (75, 215, 120), 1)
                cv2.circle(frame, (cx, cy), 3, (80, 100, 235), -1)
                if motion_valid and movement_distance_px >= 1.0:
                    magnitude = movement_distance_px
                    display_length = max(14, min(30, 14 + speed_norm * 240))
                    end = (int(cx + dx / magnitude * display_length), int(cy + dy / magnitude * display_length))
                    cv2.arrowedLine(frame, (cx, cy), end, (40, 220, 245), 1, cv2.LINE_AA, tipLength=0.28)
        for zone in ZONES:
            window_rows[zone].extend(zone_people[zone])
            window_counts[zone].append(len(zone_people[zone]))

        if frame_number % window_frames == 0:
            features = {z: build_zone_features(window_rows[z], window_counts[z], window_frames) for z in ZONES}
            available = [f for f in features.values() if f is not None]
            for zone in ZONES:
                feature_row = features[zone]
                current_flows[zone] = calculate_zone_flow(window_rows[zone], feature_row)
                probability = None
                has_track_id = bool(feature_row is not None and any(r["track_id"] is not None for r in window_rows[zone]))
                if feature_row is not None and has_track_id:
                    model_input = pd.DataFrame([[feature_row[c] for c in current_columns]], columns=current_columns)
                    probability = float(current_model.predict_proba(model_input)[0][1])
                result = calculate_current_risk(feature_row, available, previous_window_features[zone], probability)
                if result["risk_score"] is not None:
                    old_score = displayed_scores[zone]
                    result["raw_score"] = result["risk_score"]
                    result["risk_score"] = result["risk_score"] if old_score is None else SMOOTHING_ALPHA * result["risk_score"] + (1 - SMOOTHING_ALPHA) * old_score
                    result["level"] = "LOW" if result["risk_score"] < 33 else "MEDIUM" if result["risk_score"] < 66 else "HIGH"
                    displayed_scores[zone] = result["risk_score"]
                    previous_window_features[zone] = feature_row
                current_results[zone] = result
                if result["risk_score"] is not None:
                    print(f"{zone} | people: {feature_row['avg_person_count_per_frame']:.1f} | crowd evidence: {result['crowd_evidence']:.1f} | behaviour evidence: {result['behavior_evidence']:.1f} | trend evidence: {result['trend_evidence']:.1f} | final score: {result['risk_score']:.1f} | risk: {result['level']}")
            global_window = build_global_window(features)
            if global_window is not None:
                history.append(global_window)
                forecast_result = predict_forecast(forecast_model, forecast_package, list(history))
                if forecast_result["probability"] is not None:
                    print(f"Experimental Future Risk | abnormal-evidence: {forecast_result['probability'] * 100:.1f}% | history: 6 sec | horizon: 6 sec")
                scored = [(zone, result) for zone, result in current_results.items() if result.get("risk_score") is not None]
                focus_result = max(scored, key=lambda item: item[1]["risk_score"])[1] if scored else None
                reporter.risk_update({
                    "zone_a_people": int(round(features["ZONE_A"]["avg_person_count_per_frame"])) if features["ZONE_A"] else 0,
                    "zone_b_people": int(round(features["ZONE_B"]["avg_person_count_per_frame"])) if features["ZONE_B"] else 0,
                    "zone_c_people": int(round(features["ZONE_C"]["avg_person_count_per_frame"])) if features["ZONE_C"] else 0,
                    "zone_a_risk_score": current_results["ZONE_A"].get("risk_score"),
                    "zone_a_risk_level": current_results["ZONE_A"].get("level"),
                    "zone_b_risk_score": current_results["ZONE_B"].get("risk_score"),
                    "zone_b_risk_level": current_results["ZONE_B"].get("level"),
                    "zone_c_risk_score": current_results["ZONE_C"].get("risk_score"),
                    "zone_c_risk_level": current_results["ZONE_C"].get("level"),
                    "future_risk_score": forecast_result.get("probability") * 100 if forecast_result.get("probability") is not None else None,
                    "future_risk_level": "EXPERIMENTAL" if forecast_result.get("probability") is not None else "COLLECTING",
                    "xai_summary": " | ".join((focus_result or {}).get("explanation", [])[:3]),
                })
            window_rows, window_counts = ({z: [] for z in ZONES}, {z: [] for z in ZONES})

        stale = [tid for tid, value in track_memory.items() if frame_number - value["frame"] > int(fps * 2)]
        for tid in stale:
            del track_memory[tid]
        a, b = width // 3, (2 * width) // 3
        cv2.line(frame, (a, 0), (a, height), (80, 210, 235), 2, cv2.LINE_AA)
        cv2.line(frame, (b, 0), (b, height), (80, 210, 235), 2, cv2.LINE_AA)
        put(frame, "ZONE A", (max(8, a // 2 - 28), 24), 0.42, (100, 220, 240), 1)
        put(frame, "ZONE B", (a + max(8, (b - a) // 2 - 28), 24), 0.42, (100, 220, 240), 1)
        put(frame, "ZONE C", (b + max(8, (width - b) // 2 - 28), 24), 0.42, (100, 220, 240), 1)
        display = compose_display(frame, video_path.name, frame_number, total_frames, zone_counts, current_results, current_flows, forecast_result, len(history), event_context)
        cv2.imshow(window_title, display)
        key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
    capture.release()
    reporter.stop()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Phase 2 crowd monitoring demo")
    parser.add_argument("--event-id", type=int, default=None, help="Existing Phase 1 event ID")
    parser.add_argument("--session-id", type=int, default=None, help="Prepared monitoring session ID")
    parser.add_argument("--video", default=None, help="Recorded video filename from phase2/videos")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="FastAPI base URL")
    args = parser.parse_args()
    try:
        event_context = None
        if args.event_id is not None:
            try:
                event_context = fetch_event_context(args.event_id, args.api_url)
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
                raise RuntimeError(f"Unable to load event {args.event_id} from FastAPI: {error}") from error
        video = select_video(args.video)
        if video is not None:
            analyze_video(video, event_context=event_context, session_id=args.session_id, api_url=args.api_url)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nStopped.")
    except Exception as exc:
        cv2.destroyAllWindows()
        print(f"\nPhase 2 demo could not start: {exc}")


if __name__ == "__main__":
    main()
