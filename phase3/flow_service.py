"""Bounded, explainable recorded-video flow analysis for Phase 3."""

from collections import deque
from pathlib import Path
import hashlib
import json
import math
import threading
import time

import cv2
import numpy as np
from backend.phase2_web_engine import load_detector, VIDEO_DIR, PERSON_CLASS

FLOW_CONFIG = {
    # Recorded analysis is background work; keep enough temporal samples for
    # transitions while avoiding a long CPU inference for every video frame.
    "sample_fps": 3.0,
    # Keep recorded-session analysis responsive on CPU; the detector still
    # uses the small-person-friendly confidence threshold below.
    "inference_imgsz": 640,
    "confidence": 0.10,
    # The visible trail represents real movement history. Keep a long enough
    # window for a person travelling across the scene, while the timestamp
    # pruning below remains authoritative when sampling is irregular.
    "analysis_version": 9,
    "trail_history_seconds": 8.0,
    "trail_max_points": None,
    "max_track_gap_seconds": 0.75,
    "flow_window_seconds": 20.0,
    "movement_deadzone_ratio": 0.004,
    "max_jump_diagonal_ratio": 0.09,
    "position_smoothing": 0.35,
    "transition_confirmation_frames": 2,
    "stationary_speed_ratio": 0.012,
    "counterflow_min_tracks": 4,
    "counterflow_min_share": 0.30,
}
ZONES = ("ZONE_A", "ZONE_B", "ZONE_C")
CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _cache_path(source_name, cache_key=None):
    identity = str(cache_key or Path(source_name).name)
    key = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{key}.json"


def load_cached_flow(source_name, cache_key=None, expected_version=None):
    path = _cache_path(source_name, cache_key)
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if expected_version is not None and result.get("analysis_version") != expected_version:
            return None
        return result
    except (OSError, ValueError):
        return None


def save_cached_flow(source_name, result, cache_key=None):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(source_name, cache_key).write_text(json.dumps(result), encoding="utf-8")


def zone_for(x, width):
    return ZONES[min(2, max(0, int((x / max(width, 1)) * 3)))]


def direction(dx, dy, deadzone):
    if math.hypot(dx, dy) <= deadzone:
        return "Stationary"
    horizontal = "East" if dx > 0 else "West"
    vertical = "South" if dy > 0 else "North"
    if abs(dx) < abs(dy) * 0.45:
        return vertical
    if abs(dy) < abs(dx) * 0.45:
        return horizontal
    return f"{vertical}-{horizontal}"


def analyze_recorded_video(source_name, progress_callback=None, session_id=None):
    path = VIDEO_DIR / Path(source_name).name
    if not path.is_file():
        raise FileNotFoundError(f"Source video not found: {Path(source_name).name}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("Unable to open source video")
    fps = max(float(capture.get(cv2.CAP_PROP_FPS) or 30.0), 1.0)
    total_frames = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0), 1)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)
    detector = load_detector()
    track_history = {}
    next_track_id = 1
    zone_events = {zone: {"inflow": [], "outflow": []} for zone in ZONES}
    trend = []
    direction_counts = {}
    speed_samples = []
    zone_speed_samples = {zone: [] for zone in ZONES}
    zone_vectors = {zone: {"dx": [], "dy": []} for zone in ZONES}
    stationary_samples = []
    counterflow_samples = []
    decoded_frame_index = 0
    sampled_frames = 0
    last_nonempty_tracks = []
    last_snapshot_positions = []
    last_nonempty_timestamp = 0.0
    last_nonempty_frame_index = 0
    frame_step = max(int(round(fps / FLOW_CONFIG["sample_fps"])), 1)
    # Size the deque from the configured analysis cadence rather than baking
    # in a point count. The small margin tolerates a final irregular sample.
    trail_max_points = max(2, int(math.ceil(FLOW_CONFIG["sample_fps"] * FLOW_CONFIG["trail_history_seconds"])) + 2)
    total_samples = max(int(math.ceil(total_frames / frame_step)), 1)
    started = time.perf_counter()
    try:
        while decoded_frame_index < total_frames:
            ok, frame = capture.read()
            if not ok:
                break
            sampled_frame_index = decoded_frame_index
            decoded_frame_index += 1
            if sampled_frame_index % frame_step != 0 and sampled_frame_index != total_frames - 1:
                continue
            timestamp = sampled_frame_index / fps
            sampled_frames += 1
            result = detector.predict(frame, classes=[PERSON_CLASS], conf=FLOW_CONFIG["confidence"], imgsz=FLOW_CONFIG["inference_imgsz"], max_det=500, verbose=False)[0]
            boxes = result.boxes
            current_tracks = []
            if boxes is not None and len(boxes):
                coordinates = boxes.xyxy.cpu().tolist()
                previous_positions = {track_id: item[0:2] for track_id, item in ((key, value) for key, value in track_history.items())}
                used_ids = set()
                for box in coordinates:
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) / 2, y2
                    if not (0 <= cx <= width and 0 <= cy <= height):
                        continue
                    candidates = sorted((math.hypot(cx - px, cy - py), track_id) for track_id, (px, py) in previous_positions.items() if track_id not in used_ids)
                    max_match = max(width, height) * 0.12
                    if candidates and candidates[0][0] <= max_match:
                        track_id = candidates[0][1]
                    else:
                        track_id = next_track_id
                        next_track_id += 1
                    used_ids.add(track_id)
                    current_tracks.append((track_id, cx, cy))
            # The final decoded video frame can be empty even when the last
            # usable frame contains people (codec padding/end-of-file effect).
            # Keep the last measured non-empty frame as the completed-session
            # snapshot instead of replacing it with a false zero state.
            if current_tracks:
                last_nonempty_tracks = list(current_tracks)
                last_nonempty_timestamp = timestamp
                last_nonempty_frame_index = sampled_frame_index
            # ByteTrack may drop IDs when analysis samples are separated by
            # several source frames. Occupancy is still an observed detector
            # measurement, so refresh the final valid snapshot with the same
            # detector settings used by Phase 2; only ID-bearing tracks feed
            # trajectory history.
            if sampled_frame_index + frame_step >= total_frames:
                snapshot = detector.predict(frame, classes=[PERSON_CLASS], conf=FLOW_CONFIG["confidence"], imgsz=FLOW_CONFIG["inference_imgsz"], max_det=500, verbose=False)[0]
                snapshot_boxes = snapshot.boxes
                if snapshot_boxes is not None and len(snapshot_boxes):
                    last_snapshot_positions = []
                    for box in snapshot_boxes.xyxy.cpu().tolist():
                        x1, y1, x2, y2 = box
                        cx, cy = (x1 + x2) / 2, y2
                        if 0 <= cx <= width and 0 <= cy <= height:
                            last_snapshot_positions.append((cx, cy))
            speeds = []
            frame_directions = []
            for track_id, cx, cy in current_tracks:
                previous = track_history.get(track_id)
                current_zone = zone_for(cx, width)
                if previous:
                    px, py, previous_time, previous_zone, pending_zone, pending_count, history = previous
                    dt = max(timestamp - previous_time, 1.0 / fps)
                    if dt > FLOW_CONFIG["max_track_gap_seconds"]:
                        history = deque([(cx, cy, timestamp)], maxlen=trail_max_points)
                        track_history[track_id] = (cx, cy, timestamp, current_zone, None, 0, history)
                        continue
                    raw_dx, raw_dy = cx - px, cy - py
                    if math.hypot(raw_dx, raw_dy) / max(math.hypot(width, height), 1) > FLOW_CONFIG["max_jump_diagonal_ratio"]:
                        # Break the visual trail at an implausible association;
                        # never retain the old point for a later connection.
                        history = deque([(cx, cy, timestamp)], maxlen=trail_max_points)
                        track_history[track_id] = (cx, cy, timestamp, current_zone, None, 0, history)
                        continue
                    smooth = FLOW_CONFIG["position_smoothing"]
                    smoothed_cx, smoothed_cy = px + smooth * raw_dx, py + smooth * raw_dy
                    dx, dy = smoothed_cx - px, smoothed_cy - py
                    relative_speed = math.hypot(dx, dy) / dt
                    deadzone = max(width, height) * FLOW_CONFIG["movement_deadzone_ratio"]
                    movement = direction(dx, dy, deadzone)
                    speeds.append(relative_speed)
                    zone_speed_samples[current_zone].append(relative_speed)
                    zone_vectors[current_zone]["dx"].append(dx)
                    zone_vectors[current_zone]["dy"].append(dy)
                    frame_directions.append(movement)
                    if current_zone != previous_zone:
                        if pending_zone == current_zone:
                            pending_count += 1
                        else:
                            pending_zone, pending_count = current_zone, 1
                        if pending_count >= FLOW_CONFIG["transition_confirmation_frames"]:
                            zone_events[previous_zone]["outflow"].append(timestamp)
                            zone_events[current_zone]["inflow"].append(timestamp)
                            previous_zone, pending_zone, pending_count = current_zone, None, 0
                    else:
                        pending_zone, pending_count = None, 0
                    history.append((smoothed_cx, smoothed_cy, timestamp))
                    while history and timestamp - history[0][2] > FLOW_CONFIG["trail_history_seconds"]:
                        history.popleft()
                    cx, cy = smoothed_cx, smoothed_cy
                else:
                    previous_zone, pending_zone, pending_count = current_zone, None, 0
                    history = deque([(cx, cy, timestamp)], maxlen=trail_max_points)
                track_history[track_id] = (cx, cy, timestamp, previous_zone, pending_zone, pending_count, history)
            speed_samples.extend(speeds)
            stationary_samples.append(sum(1 for speed in speeds if speed <= max(width, height) * FLOW_CONFIG["stationary_speed_ratio"]) / max(len(speeds), 1))
            if frame_directions:
                direction_counts.update({item: direction_counts.get(item, 0) + 1 for item in frame_directions if item != "Stationary"})
                opposing = sum(1 for item in frame_directions if item in ("East", "West"))
                east = sum(1 for item in frame_directions if item == "East")
                west = sum(1 for item in frame_directions if item == "West")
                counterflow_samples.append(opposing >= FLOW_CONFIG["counterflow_min_tracks"] and min(east, west) / max(opposing, 1) >= FLOW_CONFIG["counterflow_min_share"])
            trend.append({"timestamp": round(timestamp, 2), "active_tracks": len(current_tracks), "average_speed": round(float(np.mean(speeds)) if speeds else 0.0, 2)})
            if progress_callback:
                progress_callback(min(99, round((sampled_frames / total_samples) * 100)))
    finally:
        capture.release()
    duration = total_frames / fps
    now = duration
    zone_metrics = {}
    for zone in ZONES:
        inflow = [value for value in zone_events[zone]["inflow"] if now - value <= FLOW_CONFIG["flow_window_seconds"]]
        outflow = [value for value in zone_events[zone]["outflow"] if now - value <= FLOW_CONFIG["flow_window_seconds"]]
        occupancy_positions = last_snapshot_positions or [(cx, cy) for _, cx, cy in last_nonempty_tracks]
        count = sum(1 for cx, _ in occupancy_positions if zone_for(cx, width) == zone)
        zone_metrics[zone] = {"people": count, "inflow": round(len(inflow) / max(min(duration, FLOW_CONFIG["flow_window_seconds"]), 1) * 60, 2), "outflow": round(len(outflow) / max(min(duration, FLOW_CONFIG["flow_window_seconds"]), 1) * 60, 2), "net": round((len(inflow) - len(outflow)) / max(min(duration, FLOW_CONFIG["flow_window_seconds"]), 1) * 60, 2), "average_movement": round(float(np.mean(zone_speed_samples[zone])) if zone_speed_samples[zone] else 0.0, 2), "state": "Accumulating" if len(inflow) > len(outflow) else "Dispersing" if len(outflow) > len(inflow) else "Balanced"}
    most_accumulating = max(ZONES, key=lambda zone: zone_metrics[zone]["net"])
    average_speed = float(np.mean(speed_samples)) if speed_samples else 0.0
    stationary_ratio = float(np.mean(stationary_samples)) if stationary_samples else 0.0
    counterflow = bool(sum(counterflow_samples) >= 2)
    density = max((item["people"] for item in zone_metrics.values()), default=0)
    if counterflow:
        flow_state = "COUNTER FLOW"
    elif density > 0 and stationary_ratio >= 0.55 and most_accumulating:
        flow_state = "STATIONARY CLUSTER"
    elif zone_metrics[most_accumulating]["net"] > 0 and average_speed < max(width, height) * 0.02:
        flow_state = "BOTTLENECK"
    elif zone_metrics[most_accumulating]["net"] > 0:
        flow_state = "ACCUMULATING"
    elif abs(sum(item["net"] for item in zone_metrics.values())) < 0.1:
        flow_state = "BALANCED FLOW"
    else:
        flow_state = "FREE FLOW"
    trajectories = []
    for track_id, (_, _, last_timestamp, _, _, _, history) in track_history.items():
        points = list(history)[-trail_max_points:]
        if len(points) >= 2:
            trajectories.append({"track_id": track_id, "points": [{"x": round(point[0], 1), "y": round(point[1], 1), "timestamp": round(point[2], 2)} for point in points]})
    aggregate_vectors = {zone: {"dx": round(float(np.mean(zone_vectors[zone]["dx"])) if zone_vectors[zone]["dx"] else 0.0, 2), "dy": round(float(np.mean(zone_vectors[zone]["dy"])) if zone_vectors[zone]["dy"] else 0.0, 2)} for zone in ZONES}
    result = {"status": "COMPLETE", "analysis_version": FLOW_CONFIG["analysis_version"], "duration_sec": round(duration, 2), "source": Path(source_name).name, "session_id": session_id, "snapshot_provenance": {"source": "Phase 3A completed recorded-session final detector snapshot", "session_id": session_id, "source_name": Path(source_name).name, "source_frame_index": last_nonempty_frame_index, "source_timestamp_sec": round(last_nonempty_timestamp, 3), "sampled_frames": sampled_frames}, "frame": {"width": width, "height": height, "fps": round(fps, 3)}, "analysis_fps": FLOW_CONFIG["sample_fps"], "sampled_frames": sampled_frames, "active_tracks": len(last_nonempty_tracks), "snapshot_people": len(last_snapshot_positions) or len(last_nonempty_tracks), "dominant_direction": max(direction_counts, key=direction_counts.get) if direction_counts else "Insufficient movement data", "average_relative_speed": round(average_speed, 2), "stationary_ratio": round(stationary_ratio, 3), "counter_flow": counterflow, "flow_state": flow_state, "most_accumulating_zone": most_accumulating if density else None, "zones": zone_metrics, "zone_vectors": aggregate_vectors, "trend": trend[-120:], "trajectories": trajectories, "trajectory_count": len(trajectories), "processing_seconds": round(time.perf_counter() - started, 2)}
    if progress_callback:
        progress_callback(100)
    save_cached_flow(source_name, result, cache_key=f"session-{session_id}" if session_id is not None else None)
    return result
