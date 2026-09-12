"""Time-indexed, rolling-window crowd instability radar.

The radar is deliberately derived from the existing Phase 3 trajectories.  It
does not invent motion: every snapshot is calculated from the track points that
were visible at that video timestamp and the preceding temporal window.
"""
from math import hypot, sqrt
from statistics import median

from .schemas import number

ZONES = ("ZONE_A", "ZONE_B", "ZONE_C")
WEIGHTS = {"concentration": .10, "direction_disorder": .18, "counter_flow": .20, "compression": .24, "stop_go": .14, "speed_drop": .14}
THRESHOLDS = ((81, "SEVERE"), (66, "HIGH"), (46, "UNSTABLE"), (26, "WATCH"), (0, "STABLE"))


def level(score):
    return next(name for threshold, name in THRESHOLDS if score >= threshold)


def percentile(values, fraction=.9):
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def _point_at(points, timestamp):
    visible = [point for point in points if number(point.get("timestamp"), "point.timestamp") <= timestamp]
    return visible[-1] if visible else None


def _track_at(item, timestamp, window):
    points = item.get("points") or []
    current = _point_at(points, timestamp)
    if current is None:
        return None
    history = [point for point in points if timestamp - window <= number(point.get("timestamp"), "point.timestamp") <= timestamp]
    if len(history) < 2:
        history = points[: points.index(current) + 1] if current in points else [current]
    if len(history) < 2:
        return None
    previous = history[-2]
    dt = max(number(current.get("timestamp"), "point.timestamp") - number(previous.get("timestamp"), "point.timestamp"), .001)
    vx = (number(current.get("x"), "point.x") - number(previous.get("x"), "point.x")) / dt
    vy = (number(current.get("y"), "point.y") - number(previous.get("y"), "point.y")) / dt
    speed = hypot(vx, vy)
    speeds = []
    for left, right in zip(history[:-1], history[1:]):
        delta = max(number(right.get("timestamp"), "point.timestamp") - number(left.get("timestamp"), "point.timestamp"), .001)
        speeds.append(hypot(number(right.get("x"), "point.x") - number(left.get("x"), "point.x"), number(right.get("y"), "point.y") - number(left.get("y"), "point.y")) / delta)
    baseline = median(speeds[:-1]) if len(speeds) > 1 else speeds[-1]
    return {"id": item.get("track_id"), "x": number(current.get("x"), "track.x"), "y": number(current.get("y"), "track.y"), "vx": vx, "vy": vy, "speed": speed, "speeds": speeds, "baseline": baseline, "history": history}


def _region_features(local, all_speeds):
    moving = [item for item in local if item["speed"] > 1]
    units = [(item["vx"] / item["speed"], item["vy"] / item["speed"]) for item in moving]
    mean_x = sum(x for x, _ in units) / max(len(units), 1)
    mean_y = sum(y for _, y in units) / max(len(units), 1)
    direction_disorder = 1 - min(1, sqrt(mean_x * mean_x + mean_y * mean_y)) if units else 0
    opposing = sum(1 for x, y in units if x * mean_x + y * mean_y < -.3) / max(len(units), 1)
    counter_flow = min(1, opposing * 2) if len(moving) >= 2 else 0
    compression_values = []
    for index, left in enumerate(local):
        for right in local[index + 1:]:
            left_history = left["history"]
            right_history = right["history"]
            before_left = left_history[0]
            before_right = right_history[0]
            before = hypot(number(before_left.get("x"), "point.x") - number(before_right.get("x"), "point.x"), number(before_left.get("y"), "point.y") - number(before_right.get("y"), "point.y"))
            after = hypot(left["x"] - right["x"], left["y"] - right["y"])
            if before > 0:
                compression_values.append(max(0, min(1, (before - after) / before)))
    compression = sum(compression_values) / max(len(compression_values), 1)
    transitions = 0
    stop_count = 0
    speed_drops = []
    for item in local:
        states = [speed > 1 for speed in item["speeds"]]
        transitions += sum(first != second for first, second in zip(states, states[1:]))
        stop_count += sum(not state for state in states)
        speed_drops.append(max(0, min(1, (item["baseline"] - item["speed"]) / max(item["baseline"], 1))))
    stop_go = min(1, (transitions + stop_count * .35) / max(len(local) * 3, 1)) if local else 0
    speed_drop = sum(speed_drops) / max(len(speed_drops), 1)
    concentration = min(1, len(local) / 8)
    mean_speed = sum(item["speed"] for item in local) / max(len(local), 1)
    relative_speed = min(1, mean_speed / max(median(all_speeds) if all_speeds else 1, 1))
    score = 100 * (WEIGHTS["concentration"] * concentration + WEIGHTS["direction_disorder"] * direction_disorder + WEIGHTS["counter_flow"] * counter_flow + WEIGHTS["compression"] * compression + WEIGHTS["stop_go"] * stop_go + WEIGHTS["speed_drop"] * speed_drop)
    return {"people": len(local), "concentration": round(concentration, 3), "average_relative_speed": round(relative_speed, 3), "direction_disorder": round(direction_disorder, 3), "compression": round(compression, 3), "counter_flow": round(counter_flow, 3), "stop_go": round(stop_go, 3), "speed_drop": round(speed_drop, 3), "instability_score": round(min(100, max(0, score)), 1), "level": level(score)}


def analyze_at(flow_result, timestamp, window_seconds=3.0, previous_snapshot=None):
    frame = flow_result.get("frame") or {}
    width = max(number(frame.get("width"), "frame.width"), 1.0)
    height = max(number(frame.get("height"), "frame.height"), 1.0)
    trajectories = flow_result.get("trajectories") or []
    tracks = [track for item in trajectories if (track := _track_at(item, timestamp, window_seconds))]
    if len(tracks) < 2:
        raise ValueError("Insufficient tracked people in the current radar window.")
    all_speeds = [track["speed"] for track in tracks]
    cols, rows = 20, 12
    cells = []
    for row in range(rows):
        for col in range(cols):
            left, top = col * width / cols, row * height / rows
            local = [track for track in tracks if left <= track["x"] < (col + 1) * width / cols and top <= track["y"] < (row + 1) * height / rows]
            if not local:
                continue
            features = _region_features(local, all_speeds)
            zone = ZONES[min(2, max(0, int((col + .5) / cols * 3)))]
            cells.append({"x": round(left + width / cols / 2, 1), "y": round(top + height / rows / 2, 1), "width": round(width / cols, 1), "height": round(height / rows, 1), "zone": zone, "vx": round(sum(item["vx"] for item in local) / len(local), 2), "vy": round(sum(item["vy"] for item in local) / len(local), 2), **features})
    zone_metrics = {}
    zone_cells = {}
    population_counts = ((flow_result.get("phase2_snapshot") or {}).get("zone_counts") or {})
    for zone in ZONES:
        local_cells = [cell for cell in cells if cell["zone"] == zone]
        zone_cells[zone] = local_cells
        # Trajectories are intentionally sparse: stable tracks are used for
        # motion features, not for total occupancy.  When Phase 2 persisted a
        # final accepted-detection snapshot, use those reconciled counts here.
        people = int(number(population_counts.get(zone), f"phase2_snapshot.zone_counts.{zone}")) if zone in population_counts else sum(cell["people"] for cell in local_cells)
        active_scores = [cell["instability_score"] for cell in local_cells]
        mean_score = sum(active_scores) / max(len(active_scores), 1)
        score = .60 * mean_score + .40 * percentile(active_scores, .90)
        previous_score = number(((previous_snapshot or {}).get("zone_metrics") or {}).get(zone, {}).get("instability_score", score))
        trend = "RISING" if score - previous_score >= 1.5 else "FALLING" if previous_score - score >= 1.5 else "STABLE"
        zone_metrics[zone] = {"people": people, "concentration": round(sum(cell["concentration"] * cell["people"] for cell in local_cells) / max(people, 1), 3), "direction_disorder": round(sum(cell["direction_disorder"] * cell["people"] for cell in local_cells) / max(people, 1), 3), "compression": round(sum(cell["compression"] * cell["people"] for cell in local_cells) / max(people, 1), 3), "counter_flow": round(sum(cell["counter_flow"] * cell["people"] for cell in local_cells) / max(people, 1), 3), "stop_go": round(sum(cell["stop_go"] * cell["people"] for cell in local_cells) / max(people, 1), 3), "speed_drop": round(sum(cell["speed_drop"] * cell["people"] for cell in local_cells) / max(people, 1), 3), "average_relative_speed": round(sum(cell["average_relative_speed"] * cell["people"] for cell in local_cells) / max(people, 1), 3), "active_cell_mean": round(mean_score, 1), "active_cell_p90": round(percentile(active_scores, .90), 1), "instability_score": round(score, 1), "level": level(score), "trend": trend}
    highest = max(ZONES, key=lambda zone: (zone_metrics[zone]["instability_score"], -ZONES.index(zone)))
    overall = round(sum(zone_metrics[zone]["instability_score"] for zone in ZONES) / 3, 1)
    high_cells = [cell for cell in cells if cell["instability_score"] >= 46]
    center = {"x": round(sum(cell["x"] for cell in high_cells) / max(len(high_cells), 1), 1), "y": round(sum(cell["y"] for cell in high_cells) / max(len(high_cells), 1), 1)} if high_cells else None
    propagation = {"from": None, "to": None, "vector": {"x": 0, "y": 0}, "confidence": 0, "label": "Propagation not established"}
    if previous_snapshot and center and previous_snapshot.get("hotspot_centroid"):
        previous = previous_snapshot["hotspot_centroid"]
        dx, dy = center["x"] - previous["x"], center["y"] - previous["y"]
        if hypot(dx, dy) >= 8:
            propagation = {"from": previous, "to": {"x": round(center["x"] + dx, 1), "y": round(center["y"] + dy, 1)}, "vector": {"x": round(dx, 2), "y": round(dy, 2)}, "confidence": round(min(1, hypot(dx, dy) / 100), 2), "label": f"{dx:+.0f}px, {dy:+.0f}px"}
    return {"timestamp": round(timestamp, 3), "window_seconds": window_seconds, "grid": {"columns": cols, "rows": rows, "cells": cells}, "field": cells, "zone_metrics": zone_metrics, "zone_centers": {zone: {"x": round(sum(cell["x"] * cell["people"] for cell in zone_cells[zone]) / max(zone_metrics[zone]["people"], 1), 1), "y": round(sum(cell["y"] * cell["people"] for cell in zone_cells[zone]) / max(zone_metrics[zone]["people"], 1), 1)} for zone in ZONES}, "overall_instability": overall, "overall_stability": round(100 - overall, 1), "level": level(overall), "highest_instability_zone": highest, "relative_speed_index": round(sum(track["speed"] for track in tracks) / max(median(all_speeds), 1), 2), "counter_flow_regions": [{"x": cell["x"], "y": cell["y"], "vx": cell["vx"], "vy": cell["vy"]} for cell in cells if cell["counter_flow"] >= .15], "compression_centroid": next(({"x": cell["x"], "y": cell["y"]} for cell in cells if cell["compression"] >= .15), None), "stop_go_centroid": next(({"x": cell["x"], "y": cell["y"]} for cell in cells if cell["stop_go"] > .1), None), "hotspot_centroid": center, "propagation": propagation, "active_tracks": len(tracks), "occupied_cells": len(cells), "weights": WEIGHTS, "thresholds": {str(value): name for value, name in THRESHOLDS}}


def analyze(flow_result, window_seconds=3.0, sample_fps=10):
    timestamps = [number(point.get("timestamp"), "point.timestamp") for item in flow_result.get("trajectories") or [] for point in item.get("points") or []]
    if not timestamps:
        raise ValueError("No timestamped trajectory history is available for instability analysis.")
    start, end = min(timestamps), max(timestamps)
    step = 1 / max(1, sample_fps)
    snapshots = []
    previous = None
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
        raise ValueError("Insufficient timestamped movement history for instability analysis.")
    # Do not mutate the last time-series item when attaching the aggregate
    # response; otherwise the snapshot would contain itself recursively.
    current = {**snapshots[-1]}
    # Use a one-second trailing trend so the forecast is visibly responsive
    # without amplifying single-frame noise.
    previous_scores = snapshots[max(0, len(snapshots) - sample_fps - 1)]["zone_metrics"] if snapshots else {}
    for zone in ZONES:
        current["zone_metrics"][zone]["previous_score"] = previous_scores.get(zone, {}).get("instability_score", current["zone_metrics"][zone]["instability_score"])
    current.update({"snapshots": snapshots, "sample_fps": sample_fps, "duration_start": start, "duration_end": end, "forecast": _forecast(current)})
    for zone in ZONES:
        current["zone_metrics"][zone].pop("previous_score", None)
    return current


def _forecast(snapshot):
    current = snapshot
    output = []
    for horizon in (0, 10, 20, 30):
        seconds = horizon
        cells = []
        for cell in current["field"]:
            cells.append({**cell, "x": round(cell["x"] + cell["vx"] * seconds, 1), "y": round(cell["y"] + cell["vy"] * seconds, 1)})
        output.append({"horizon_seconds": horizon, "timestamp": current["timestamp"] + seconds, "zone_scores": {zone: round(min(100, max(0, current["zone_metrics"][zone]["instability_score"] + (seconds / 10) * (current["zone_metrics"][zone]["instability_score"] - current["zone_metrics"][zone].get("previous_score", current["zone_metrics"][zone]["instability_score"])) * .2)), 1) for zone in ZONES}, "field": cells, "propagation": current["propagation"]})
    return output
