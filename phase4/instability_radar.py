"""Evidence-based Phase 4 instability analysis over Phase 3 track histories."""

from __future__ import annotations

import logging
import math
import os
from statistics import median

from .schemas import number

LOGGER = logging.getLogger(__name__)
ZONES = ("ZONE_A", "ZONE_B", "ZONE_C")
GRID_COLUMNS, GRID_ROWS = 20, 12
WINDOW_SECONDS = 3.0
EMA_ALPHA = 0.35
NEIGHBOR_RADIUS_RATIO = 0.075
MIN_TRACK_POINTS = 3
MIN_TRACK_DURATION = 0.5
MAX_TRACK_GAP = 0.75
MOVEMENT_RATIO = 0.004
MOVING_SPEED_FLOOR = 0.0
WEIGHTS = {
    "compression": 0.30,
    "direction_disorder": 0.20,
    "counter_flow": 0.20,
    "stop_go": 0.15,
    "speed_drop": 0.15,
}
THRESHOLDS = ((81, "SEVERE"), (66, "HIGH"), (46, "UNSTABLE"), (26, "WATCH"), (0, "STABLE"))
HYSTERESIS = 4.0


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def level(score, previous=None):
    score = float(score)
    current = next(name for threshold, name in THRESHOLDS if score >= threshold)
    if previous:
        for threshold, name in THRESHOLDS:
            if name == previous and score >= threshold - HYSTERESIS:
                return previous
    return current


def percentile(values, fraction=.9):
    values = sorted(float(value) for value in values)
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))] if values else 0.0


def _point(point):
    try:
        return {"x": number(point.get("x"), "track.x"), "y": number(point.get("y"), "track.y"), "timestamp": number(point.get("timestamp"), "track.timestamp")}
    except (AttributeError, ValueError, TypeError):
        return None


def _history(item, timestamp, window):
    points = sorted((point for point in (_point(value) for value in item.get("points") or []) if point), key=lambda value: value["timestamp"])
    points = [point for point in points if timestamp - window <= point["timestamp"] <= timestamp]
    if len(points) < MIN_TRACK_POINTS or points[-1]["timestamp"] < timestamp - 0.5:
        return None
    if points[-1]["timestamp"] - points[0]["timestamp"] < MIN_TRACK_DURATION:
        return None
    for left, right in zip(points[:-1], points[1:]):
        if right["timestamp"] - left["timestamp"] > MAX_TRACK_GAP:
            points = points[points.index(right):]
            break
    if len(points) < MIN_TRACK_POINTS:
        return None
    velocities = []
    for left, right in zip(points[:-1], points[1:]):
        dt = right["timestamp"] - left["timestamp"]
        if dt <= 0 or dt > MAX_TRACK_GAP:
            continue
        vx = (right["x"] - left["x"]) / dt
        vy = (right["y"] - left["y"]) / dt
        velocities.append({"vx": vx, "vy": vy, "speed": math.hypot(vx, vy), "timestamp": right["timestamp"]})
    if len(velocities) < 2:
        return None
    current = points[-1]
    velocity = velocities[-1]
    return {"id": item.get("track_id"), "x": current["x"], "y": current["y"], "history": points, "velocities": velocities, **velocity}


def _neighbors(track, tracks, radius):
    return [other for other in tracks if other["id"] != track["id"] and math.hypot(track["x"] - other["x"], track["y"] - other["y"]) <= radius]


def _cell_key(cell):
    return cell["row"], cell["column"]


def _cell_features(local, all_tracks, previous_cell, radius, reference_speed, diagonal):
    if not local:
        return None
    moving = [track for track in local if track["speed"] > MOVING_SPEED_FLOOR]
    units = [(track["vx"] / track["speed"], track["vy"] / track["speed"]) for track in moving if track["speed"] > diagonal * MOVEMENT_RATIO]
    mean_unit = (sum(unit[0] for unit in units) / len(units), sum(unit[1] for unit in units) / len(units)) if units else (0.0, 0.0)
    alignment = math.hypot(*mean_unit)
    disorder = 1.0 - clamp(alignment)

    opposing_pairs = 0
    pair_count = 0
    for index, left in enumerate(moving):
        for right in moving[index + 1:]:
            if left["speed"] <= diagonal * MOVEMENT_RATIO or right["speed"] <= diagonal * MOVEMENT_RATIO:
                continue
            cosine = (left["vx"] * right["vx"] + left["vy"] * right["vy"]) / max(left["speed"] * right["speed"], 1e-6)
            pair_count += 1
            opposing_pairs += cosine < -0.5
    counter_flow = clamp(opposing_pairs / max(pair_count, 1) * 2.0) if len(moving) >= 4 and opposing_pairs >= 2 else 0.0

    closing, converging, distance_samples = [], [], []
    for track in local:
        for other in _neighbors(track, all_tracks, radius):
            if track["id"] >= other["id"]:
                continue
            dx, dy = track["x"] - other["x"], track["y"] - other["y"]
            distance = math.hypot(dx, dy)
            if distance <= 0:
                continue
            previous_left, previous_right = track["history"][0], other["history"][0]
            before = math.hypot(previous_left["x"] - previous_right["x"], previous_left["y"] - previous_right["y"])
            distance_samples.append(clamp((before - distance) / max(before, 1e-6)))
            radial = ((track["vx"] - other["vx"]) * dx + (track["vy"] - other["vy"]) * dy) / distance
            converging.append(clamp(-radial / max(diagonal * .08, 1e-6)))
            closing.append(clamp((before - distance) / max(before, 1e-6)))
    neighbor_closing = sum(closing) / len(closing) if closing else 0.0
    convergence = sum(converging) / len(converging) if converging else 0.0
    previous_concentration = float((previous_cell or {}).get("concentration", len(local) / 8))
    concentration = clamp(len(local) / 8)
    concentration_growth = clamp(concentration - previous_concentration + .5)
    current_speed = sum(track["speed"] for track in local) / len(local)
    old_speed = float((previous_cell or {}).get("mean_speed", current_speed))
    speed_reduction = clamp((old_speed - current_speed) / max(old_speed, diagonal * .004, 1e-6))
    compression = clamp(.35 * concentration_growth + .30 * neighbor_closing + .20 * convergence + .15 * speed_reduction)

    transitions = 0
    transition_tracks = 0
    for track in local:
        states = ["MOVING" if item["speed"] > diagonal * .012 else "STOPPED" for item in track["velocities"]]
        changes = sum(left != right for left, right in zip(states, states[1:]))
        if changes >= 2:
            transition_tracks += 1
            transitions += changes
    stop_go = clamp(transitions / max(len(local) * 3, 1)) if transition_tracks >= 2 and len(local) >= 3 else 0.0
    speed_drop = sum(clamp((median(track["velocities"][:-1] and [item["speed"] for item in track["velocities"][:-1]] or [track["speed"]]) - track["speed"]) / max(median([item["speed"] for item in track["velocities"][:-1]] or [track["speed"]]), 1e-6)) for track in local) / len(local)
    mean_vx = sum(track["vx"] for track in moving) / max(len(moving), 1)
    mean_vy = sum(track["vy"] for track in moving) / max(len(moving), 1)
    score = 100 * (WEIGHTS["compression"] * compression + WEIGHTS["direction_disorder"] * disorder + WEIGHTS["counter_flow"] * counter_flow + WEIGHTS["stop_go"] * stop_go + WEIGHTS["speed_drop"] * speed_drop)
    return {"people": len(local), "concentration": round(concentration, 3), "mean_speed": round(current_speed, 3), "direction_disorder": round(disorder, 3), "compression": round(compression, 3), "counter_flow": round(counter_flow, 3), "stop_go": round(stop_go, 3), "speed_drop": round(speed_drop, 3), "mean_vx": round(mean_vx, 3), "mean_vy": round(mean_vy, 3), "instability_score": round(clamp(score, 0, 100), 1), "level": level(score)}


def analyze_at(flow_result, timestamp, window_seconds=WINDOW_SECONDS, previous_snapshot=None):
    frame = flow_result.get("frame") or {}
    width, height = max(number(frame.get("width") or 1280), 1), max(number(frame.get("height") or 720), 1)
    diagonal = math.hypot(width, height)
    radius = diagonal * NEIGHBOR_RADIUS_RATIO
    tracks = [track for item in flow_result.get("trajectories") or [] if (track := _history(item, timestamp, window_seconds))]
    if len(tracks) < 3:
        raise ValueError("Insufficient stable movement history.")
    reference_speed = median([track["speed"] for track in tracks])
    cells, previous_cells = [], {(item.get("row"), item.get("column")): item for item in (previous_snapshot or {}).get("field", [])}
    cell_width, cell_height = width / GRID_COLUMNS, height / GRID_ROWS
    for row in range(GRID_ROWS):
        for column in range(GRID_COLUMNS):
            local = [track for track in tracks if column * cell_width <= track["x"] < (column + 1) * cell_width and row * cell_height <= track["y"] < (row + 1) * cell_height]
            if not local:
                continue
            features = _cell_features(local, tracks, previous_cells.get((row, column)), radius, reference_speed, diagonal)
            zone = ZONES[min(2, int(((column + .5) / GRID_COLUMNS) * 3))]
            cells.append({"row": row, "column": column, "x": round((column + .5) * cell_width, 1), "y": round((row + .5) * cell_height, 1), "width": round(cell_width, 1), "height": round(cell_height, 1), "zone": zone, "vx": features.pop("mean_vx"), "vy": features.pop("mean_vy"), **features})
    zone_metrics, zone_centers = {}, {}
    for zone in ZONES:
        active = [cell for cell in cells if cell["zone"] == zone]
        scores = [cell["instability_score"] for cell in active]
        if not active:
            zone_metrics[zone] = {"people": 0, "active_cells": 0, "available": False, "state": "INACTIVE", "level": "INACTIVE", "instability_score": 0, "concentration": 0, "compression": 0, "counter_flow": 0, "stop_go": 0, "direction_disorder": 0, "speed_drop": 0, "mean_speed": 0, "mean_vx": 0, "mean_vy": 0}
            continue
        people = sum(cell["people"] for cell in active)
        raw_score = .60 * sum(scores) / len(scores) + .40 * percentile(scores, .90)
        previous_zone = (previous_snapshot or {}).get("zone_metrics", {}).get(zone, {})
        previous_score = float(previous_zone.get("instability_score", raw_score))
        score = EMA_ALPHA * raw_score + (1 - EMA_ALPHA) * previous_score
        zone_metrics[zone] = {"people": people, "active_cells": len(active), "available": True, "state": level(score, previous_zone.get("level")), "level": level(score, previous_zone.get("level")), "instability_score": round(score, 1), "raw_instability_score": round(raw_score, 1), **{key: round(sum(cell[key] * cell["people"] for cell in active) / max(people, 1), 3) for key in ("concentration", "compression", "counter_flow", "stop_go", "direction_disorder", "speed_drop", "mean_speed", "vx", "vy") if key in active[0]}}
        zone_metrics[zone]["mean_vx"] = round(sum(cell["vx"] * cell["people"] for cell in active) / max(people, 1), 3)
        zone_metrics[zone]["mean_vy"] = round(sum(cell["vy"] * cell["people"] for cell in active) / max(people, 1), 3)
        zone_centers[zone] = {"x": round(sum(cell["x"] * cell["people"] for cell in active) / people, 1), "y": round(sum(cell["y"] * cell["people"] for cell in active) / people, 1)}
    active_zones = [zone for zone in ZONES if zone_metrics[zone]["available"]]
    active_scores = [zone_metrics[zone]["instability_score"] for zone in active_zones]
    overall = .50 * (sum(active_scores) / len(active_scores)) + .50 * percentile(active_scores, .90) if active_scores else 0
    highest = max(active_zones, key=lambda zone: zone_metrics[zone]["instability_score"]) if active_zones else None
    hotspot_cells = [cell for cell in cells if cell["instability_score"] >= 26]
    total_weight = sum(cell["instability_score"] * max(cell["people"], 1) for cell in hotspot_cells)
    hotspot = {"x": round(sum(cell["x"] * cell["instability_score"] * max(cell["people"], 1) for cell in hotspot_cells) / total_weight, 1), "y": round(sum(cell["y"] * cell["instability_score"] * max(cell["people"], 1) for cell in hotspot_cells) / total_weight, 1), "score": round(max(cell["instability_score"] for cell in hotspot_cells), 1)} if hotspot_cells else None
    def centroid(predicate):
        selected = [cell for cell in cells if predicate(cell)]
        weight = sum(max(cell["people"], 1) for cell in selected)
        return {"x": round(sum(cell["x"] * max(cell["people"], 1) for cell in selected) / weight, 1), "y": round(sum(cell["y"] * max(cell["people"], 1) for cell in selected) / weight, 1)} if selected else None
    flow_weight = sum(max(cell["people"], 1) for cell in cells)
    flow = {"vx": round(sum(cell["vx"] * max(cell["people"], 1) for cell in cells) / max(flow_weight, 1), 3), "vy": round(sum(cell["vy"] * max(cell["people"], 1) for cell in cells) / max(flow_weight, 1), 3)}
    snapshot = {"timestamp": round(timestamp, 3), "window_seconds": window_seconds, "available": True, "grid": {"columns": GRID_COLUMNS, "rows": GRID_ROWS, "cells": cells}, "field": cells, "zone_metrics": zone_metrics, "zone_centers": zone_centers, "overall_instability": round(overall, 1), "overall_stability": round(100 - overall, 1), "level": level(overall, (previous_snapshot or {}).get("level")), "highest_instability_zone": highest, "hotspot": hotspot, "hotspot_centroid": hotspot, "compression_centroid": centroid(lambda cell: cell["compression"] >= .15), "stop_go_centroid": centroid(lambda cell: cell["stop_go"] > .1), "counter_flow_regions": [{"x": cell["x"], "y": cell["y"], "vx": cell["vx"], "vy": cell["vy"]} for cell in cells if cell["counter_flow"] > 0], "flow": flow, "relative_speed_index": round(sum(track["speed"] for track in tracks) / max(reference_speed, 1e-6), 2), "active_tracks": len(tracks), "stable_tracks": len(tracks), "occupied_cells": len(cells), "neighbor_radius_normalized": round(radius / diagonal, 4), "weights": WEIGHTS, "ema_alpha": EMA_ALPHA, "thresholds": {str(value): name for value, name in THRESHOLDS}, "hysteresis": HYSTERESIS}
    if os.getenv("CROWDGUARD_DEBUG_RADAR") == "1":
        LOGGER.info("radar t=%.2f detected=%s stable=%s active_cells=%s zones=%s", timestamp, len(flow_result.get("trajectories") or []), len(tracks), len(cells), {zone: zone_metrics[zone]["instability_score"] for zone in ZONES})
    return snapshot


def _propagation(snapshots):
    points = [item.get("hotspot_centroid") for item in snapshots[-6:] if item.get("hotspot_centroid")]
    if len(points) < 3:
        return {"established": False, "from": None, "to": None, "dx": 0, "dy": 0, "vector": {"x": 0, "y": 0}, "confidence": 0, "label": "Propagation not established"}
    dx = (points[-1]["x"] - points[0]["x"]) / max(len(points) - 1, 1)
    dy = (points[-1]["y"] - points[0]["y"]) / max(len(points) - 1, 1)
    consistency = sum(((point["x"] - points[0]["x"]) * dx + (point["y"] - points[0]["y"]) * dy) >= 0 for point in points) / len(points)
    confidence = clamp(consistency * min(1, math.hypot(dx, dy) / 25))
    if confidence < .45:
        return {"established": False, "from": None, "to": None, "dx": round(dx, 2), "dy": round(dy, 2), "vector": {"x": round(dx, 2), "y": round(dy, 2)}, "confidence": round(confidence, 2), "label": "Propagation not established"}
    return {"established": True, "from": points[0], "to": points[-1], "dx": round(dx, 2), "dy": round(dy, 2), "vector": {"x": round(dx, 2), "y": round(dy, 2)}, "confidence": round(confidence, 2), "label": f"{dx:+.0f}px, {dy:+.0f}px"}


def _forecast(snapshot):
    output = []
    for horizon in (0, 10, 20, 30):
        field = [{**cell, "x": round(cell["x"] + cell["vx"] * horizon, 1), "y": round(cell["y"] + cell["vy"] * horizon, 1)} for cell in snapshot["field"]]
        output.append({"horizon_seconds": horizon, "timestamp": round(snapshot["timestamp"] + horizon, 3), "zone_scores": {zone: round(snapshot["zone_metrics"][zone]["instability_score"], 1) for zone in ZONES}, "field": field, "propagation": snapshot["propagation"]})
    return output


def analyze(flow_result, window_seconds=WINDOW_SECONDS, sample_fps=10):
    timestamps = [number(point.get("timestamp"), "point.timestamp") for item in flow_result.get("trajectories") or [] for point in item.get("points") or [] if point.get("timestamp") is not None]
    if not timestamps:
        raise ValueError("No timestamped trajectory history is available for instability analysis.")
    start, end = min(timestamps), max(timestamps)
    snapshots, previous = [], None
    step = 1 / max(sample_fps, 1)
    current = start
    while current <= end + .0001:
        try:
            snapshot = analyze_at(flow_result, current, window_seconds, previous)
        except ValueError:
            current += step
            continue
        snapshots.append(snapshot)
        previous = snapshot
        current += step
    if not snapshots:
        raise ValueError("Insufficient stable movement history.")
    current = {**snapshots[-1]}
    current["snapshots"] = snapshots
    current["propagation"] = _propagation(snapshots)
    current["forecast"] = _forecast({**current, "propagation": current["propagation"]})
    return current
