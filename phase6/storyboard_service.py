"""Deterministic, evidence-led event extraction for AI Crowd Storyboard."""

from __future__ import annotations

from statistics import median

ZONES = ("ZONE_A", "ZONE_B", "ZONE_C")
STORY_DETAIL_INTERVAL_SEC = 3.0
STORY_CHANGE_THRESHOLDS = {
    "stable_relative": 0.05,
    "building_relative": 0.15,
}
ALGORITHM_VERSION = "storyboard-v2"


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(value, default=0):
    return int(round(_num(value, default)))


def _label(value):
    return str(value or "").replace("_", " ").title()


def _time(value):
    value = max(0.0, _num(value))
    return f"{int(value // 60):02d}:{int(value % 60):02d}"


def _optional_num(value):
    """Return None for unavailable metrics; never turn missing evidence into 0."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric(record, key, fallback=None):
    value = record.get(key)
    if value is None and fallback:
        value = fallback.get(key)
    return _optional_num(value)


def _risk_level(record):
    levels = [str(record.get(f"{zone.lower()}_risk_level") or "").upper() for zone in ZONES]
    priority = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "STABLE")
    return next((item for item in priority if item in levels), "STABLE")


def _counts(record):
    if record.get("zone_counts"):
        return {zone: _int(record["zone_counts"].get(zone)) for zone in ZONES}
    return {zone: _int(record.get(f"{zone.lower()}_people")) for zone in ZONES}


def _normalise_record(record, timestamp=0.0, fallback=None):
    fallback = fallback or {}
    has_people_data = any(key in record for key in ("total_people", "zone_counts", "zone_a_people", "zone_b_people", "zone_c_people"))
    counts = _counts(record) if has_people_data else _counts(fallback)
    total = _int(record.get("total_people"), sum(counts.values())) or sum(counts.values())
    highest = max(counts, key=counts.get) if any(counts.values()) else None
    return {
        "timestamp_sec": max(0.0, _num(record.get("timestamp_sec", record.get("timestamp", timestamp)), timestamp)),
        "people_available": has_people_data,
        "total_people": total,
        "zone_counts": counts,
        "highest_zone": record.get("most_crowded_zone") or highest,
        "density_state": record.get("density_state") or fallback.get("density_state"),
        "risk_level": str(record.get("risk_level") or _risk_level(record) or fallback.get("risk_level") or "STABLE").upper(),
        "instability_score": _metric(record, "instability_score", fallback),
        "active_tracks": _metric(record, "active_tracks", fallback),
        "average_speed": _metric(record, "average_speed", fallback),
        "compression": _metric(record, "compression", fallback),
        "counter_flow": _metric(record, "counter_flow", fallback),
        "stop_go": _metric(record, "stop_go", fallback),
        "direction_disorder": _metric(record, "direction_disorder", fallback),
    }


def _fallback_snapshot(flow_result, phase2_records):
    snapshot = flow_result.get("phase2_snapshot") or {}
    if snapshot:
        return _normalise_record(snapshot)
    if phase2_records:
        return _normalise_record(phase2_records[-1])
    return _normalise_record({})


def build_snapshots(flow_result, phase2_records=None, radar=None, time_result=None):
    """Build time-indexed observations from persisted source packets only."""
    phase2_records = sorted(phase2_records or [], key=lambda item: str(item.get("timestamp", "")))
    fallback = _fallback_snapshot(flow_result, phase2_records)
    trend = flow_result.get("trend") or []
    snapshots = []
    if trend:
        for item in trend:
            snapshots.append(_normalise_record(item, item.get("timestamp", 0), fallback))
    if phase2_records:
        for index, item in enumerate(phase2_records):
            timestamp_value = item.get("source_timestamp_sec")
            timestamp = _num(timestamp_value, index * 2) if timestamp_value is not None else _num(item.get("timestamp_sec"), index * 2)
            current = _normalise_record({**item, "timestamp_sec": timestamp}, timestamp, fallback)
            if snapshots:
                nearest = min(snapshots, key=lambda sample: abs(sample["timestamp_sec"] - timestamp))
                current["average_speed"] = nearest.get("average_speed", 0)
            snapshots.append(current)
    if not snapshots:
        snapshots = [fallback]
    snapshots.sort(key=lambda item: item["timestamp_sec"])
    deduped = []
    for item in snapshots:
        if deduped and abs(item["timestamp_sec"] - deduped[-1]["timestamp_sec"]) < 0.05:
            deduped[-1] = item
        else:
            deduped.append(item)
    if radar:
        for radar_snapshot in radar.get("snapshots") or []:
            timestamp = _num(radar_snapshot.get("timestamp"), 0)
            radar_record = {
                "timestamp_sec": timestamp,
                "instability_score": radar_snapshot.get("overall_instability"),
                "compression": max((_num(radar_snapshot.get("zone_metrics", {}).get(zone, {}).get("compression")) for zone in ZONES), default=0),
                "counter_flow": max((_num(radar_snapshot.get("zone_metrics", {}).get(zone, {}).get("counter_flow")) for zone in ZONES), default=0),
                "stop_go": max((_num(radar_snapshot.get("zone_metrics", {}).get(zone, {}).get("stop_go")) for zone in ZONES), default=0),
                "direction_disorder": max((_num(radar_snapshot.get("zone_metrics", {}).get(zone, {}).get("direction_disorder")) for zone in ZONES), default=0),
                "highest_zone": radar_snapshot.get("highest_instability_zone"),
            }
            nearest = min(deduped, key=lambda sample: abs(sample["timestamp_sec"] - timestamp), default=None)
            if nearest and abs(nearest["timestamp_sec"] - timestamp) <= 0.35:
                nearest.update({key: value for key, value in radar_record.items() if key != "timestamp_sec"})
            else:
                deduped.append(_normalise_record(radar_record, timestamp, fallback))
        last = deduped[-1]
        last.update({
            "instability_score": _num(radar.get("overall_instability")),
            "compression": max((_num(radar.get("zone_metrics", {}).get(zone, {}).get("compression")) for zone in ZONES), default=0),
            "counter_flow": max((_num(radar.get("zone_metrics", {}).get(zone, {}).get("counter_flow")) for zone in ZONES), default=0),
            "stop_go": max((_num(radar.get("zone_metrics", {}).get(zone, {}).get("stop_go")) for zone in ZONES), default=0),
            "direction_disorder": max((_num(radar.get("zone_metrics", {}).get(zone, {}).get("direction_disorder")) for zone in ZONES), default=0),
            "highest_zone": radar.get("highest_instability_zone") or last.get("highest_zone"),
        })
    duration = _num(flow_result.get("duration_sec"), max((item["timestamp_sec"] for item in deduped), default=0))
    return deduped, max(duration, max((item["timestamp_sec"] for item in deduped), default=0)), fallback


def _sample_at(snapshots, timestamp, tolerance=1.5):
    if not snapshots:
        return None
    exact = [item for item in snapshots if abs(item["timestamp_sec"] - timestamp) <= 0.05]
    if exact:
        return min(exact, key=lambda item: abs(item["timestamp_sec"] - timestamp))
    nearest = min(snapshots, key=lambda item: abs(item["timestamp_sec"] - timestamp))
    return nearest if abs(nearest["timestamp_sec"] - timestamp) <= tolerance else None


def _classification(start_people, end_people):
    delta = end_people - start_people
    relative = delta / max(start_people, 1)
    stable = STORY_CHANGE_THRESHOLDS["stable_relative"]
    building = STORY_CHANGE_THRESHOLDS["building_relative"]
    if relative >= building:
        return "BUILDING", delta, relative
    if relative > stable:
        return "SLIGHT INCREASE", delta, relative
    if relative <= -building:
        return "DISPERSING", delta, relative
    if relative < -stable:
        return "SLIGHT DECREASE", delta, relative
    return "STABLE", delta, relative


def _percent(value):
    value = _optional_num(value)
    return None if value is None else round(value * 100 if abs(value) <= 1 else value, 1)


def _round_optional(value, digits=1):
    value = _optional_num(value)
    return None if value is None else round(value, digits)


def build_detailed_timeline(snapshots, duration):
    """Create chronological interval evidence from the available time-indexed data."""
    duration = max(0.0, _num(duration))
    if duration <= 0:
        return {"interval_sec": STORY_DETAIL_INTERVAL_SEC, "interval_count": 0, "intervals": []}
    intervals = []
    people_snapshots = [item for item in snapshots if item.get("people_available")]
    interval_count = int((duration + STORY_DETAIL_INTERVAL_SEC - 1e-9) // STORY_DETAIL_INTERVAL_SEC)
    for index in range(interval_count):
        start = round(index * STORY_DETAIL_INTERVAL_SEC, 3)
        end = round(min(duration, start + STORY_DETAIL_INTERVAL_SEC), 3)
        inside = [item for item in people_snapshots if start <= item["timestamp_sec"] < end or (index == interval_count - 1 and start <= item["timestamp_sec"] <= end)]
        before_start = [item for item in people_snapshots if item["timestamp_sec"] <= start]
        before_end = [item for item in people_snapshots if item["timestamp_sec"] <= end]
        start_sample = _sample_at(before_start, start)
        end_sample = _sample_at(before_end, end, tolerance=1.5)
        representative = _sample_at(snapshots, (start + end) / 2, tolerance=max(1.5, (end - start) / 2 + 0.5))
        # Sparse history is still usable when the nearest real observations
        # bracket the interval. Do not manufacture a population value from a
        # final snapshot just because an interval has no exact sample.
        evidence = inside or [sample for sample in (start_sample, end_sample) if sample]
        data_source = "MONITORING_ANALYTICS_SNAPSHOTS" if inside else "NEAREST_REAL_SNAPSHOTS"
        if not evidence or not start_sample or not end_sample:
            movement = [item for item in snapshots if start <= item["timestamp_sec"] <= end]
            movement_speeds = [_optional_num(item.get("average_speed")) for item in movement if _optional_num(item.get("average_speed")) is not None]
            movement_text = ""
            if movement_speeds:
                movement_text = f" Tracked movement averaged {round(sum(movement_speeds) / len(movement_speeds), 1)} pixels per second."
            intervals.append({
                "index": index + 1, "start_sec": start, "end_sec": end,
                "representative_timestamp_sec": representative["timestamp_sec"] if representative else None,
                "analytics_snapshot_count": 0, "data_source": "VIDEO_ONLY" if not movement_speeds else "FLOW_TRACK_HISTORY",
                "population_available": False, "zones_available": False, "flow_available": bool(movement_speeds),
                "event_types": ["VIDEO_ONLY" if not movement_speeds else "FLOW_CHANGE"],
                "explanation": "Historical crowd analytics were not recorded for this interval." + movement_text,
                "narration_text": f"Between {_time(start)} and {_time(end)}, historical crowd analytics were not recorded for this interval." + movement_text,
                "image_url": None,
            })
            continue
        people = [item["total_people"] for item in evidence]
        zone_totals = {zone: sum(item["zone_counts"].get(zone, 0) for item in evidence) for zone in ZONES}
        dominant = max(zone_totals, key=zone_totals.get) if any(zone_totals.values()) else end_sample.get("highest_zone")
        start_people, end_people = start_sample["total_people"], end_sample["total_people"]
        change, delta, relative = _classification(start_people, end_people)
        start_zone = start_sample.get("highest_zone")
        event_types = []
        sentences = []
        if change == "BUILDING": sentences.append(f"Crowd increased from {start_people} to {end_people} people.")
        elif change == "DISPERSING": sentences.append(f"Crowd reduced from {start_people} to {end_people} people.")
        elif change == "SLIGHT INCREASE": sentences.append(f"Crowd increased slightly from {start_people} to {end_people} people.")
        elif change == "SLIGHT DECREASE": sentences.append(f"Crowd decreased slightly from {start_people} to {end_people} people.")
        else: sentences.append(f"Crowd remained relatively stable between {start_people} and {end_people} people.")
        if dominant:
            sentences.append(f"{_label(dominant)} was the dominant zone.")
        if start_zone and dominant and start_zone != dominant:
            sentences.append(f"Crowd concentration shifted from {_label(start_zone)} to {_label(dominant)}.")
            event_types.append("HOTSPOT_SHIFT")
        density_start, density_end = start_sample.get("density_state"), end_sample.get("density_state")
        if density_start and density_end and density_start != density_end:
            sentences.append(f"Density changed from {density_start} to {density_end}.")
            event_types.append("DENSITY_TRANSITION")
        speed_values = [_optional_num(item.get("average_speed")) for item in evidence if _optional_num(item.get("average_speed")) is not None]
        compression_values = [_percent(item.get("compression")) for item in evidence if _percent(item.get("compression")) is not None]
        counter_values = [_percent(item.get("counter_flow")) for item in evidence if _percent(item.get("counter_flow")) is not None]
        instability_values = [_optional_num(item.get("instability_score")) for item in evidence if _optional_num(item.get("instability_score")) is not None]
        if speed_values and len(speed_values) > 1 and speed_values[-1] < speed_values[0] * .8:
            sentences.append("Average movement speed reduced during this interval."); event_types.append("MOVEMENT_SLOWS")
        if counter_values and max(counter_values) > 10:
            sentences.append("Counter-flow was detected."); event_types.append("COUNTER_FLOW")
        if compression_values and compression_values[-1] > compression_values[0] + 10:
            sentences.append("Compression increased noticeably."); event_types.append("COMPRESSION_RISE")
        if instability_values and instability_values[-1] > instability_values[0] + 10:
            sentences.append(f"Instability rose from {round(instability_values[0])} to {round(instability_values[-1])}."); event_types.append("INSTABILITY_RISE")
        intervals.append({
            "index": index + 1, "start_sec": start, "end_sec": end,
            "representative_timestamp_sec": representative["timestamp_sec"] if representative else end,
            "start_people": start_people, "end_people": end_people, "minimum_people": min(people), "maximum_people": max(people), "average_people": round(sum(people) / len(people), 1), "min_people": min(people), "max_people": max(people), "avg_people": round(sum(people) / len(people), 1),
            "start_zone_counts": start_sample["zone_counts"], "end_zone_counts": end_sample["zone_counts"], "peak_zone_counts": {zone: max(item["zone_counts"].get(zone, 0) for item in evidence) for zone in ZONES}, "dominant_zone": dominant,
            "density_start": density_start, "density_end": density_end,
            "flow_summary": {"average_speed": round(sum(speed_values) / len(speed_values), 1) if speed_values else None, "speed_start": speed_values[0] if speed_values else None, "speed_end": speed_values[-1] if speed_values else None},
            "instability_summary": {"compression_start": compression_values[0] if compression_values else None, "compression_end": compression_values[-1] if compression_values else None, "counter_flow_peak": max(counter_values) if counter_values else None, "instability_start": instability_values[0] if instability_values else None, "instability_end": instability_values[-1] if instability_values else None},
            "change_classification": change, "relative_change": round(relative, 4), "event_types": event_types,
            "explanation": " ".join(sentences), "narration_text": f"Between {_time(start)} and {_time(end)}, " + " ".join(sentences),
            "image_url": None, "analytics_snapshot_count": len(inside), "data_source": data_source,
            "population_available": True, "zones_available": True, "flow_available": bool(speed_values),
            "instability_available": bool(instability_values),
        })
    return {"interval_sec": STORY_DETAIL_INTERVAL_SEC, "interval_count": len(intervals), "intervals": intervals}


def _candidate(event_type, index, timestamp, title, caption, explanation, score, snapshot, evidence):
    return {"event_type": event_type, "timestamp_sec": timestamp, "title": title, "caption": caption, "explanation": explanation, "score": round(max(0, min(100, score)), 1), "snapshot": snapshot, "evidence": evidence}


def detect_candidates(snapshots, duration):
    if not snapshots:
        return []
    candidates = []
    first = snapshots[0]
    candidates.append(_candidate("SESSION_START", 0, first["timestamp_sec"], "Monitoring begins", f"{first['total_people']} people observed at the start of the available record.", "This moment anchors the storyboard to the first synchronized monitoring observation.", 28, first, ["Phase 2 monitoring snapshot"]))
    for index in range(1, len(snapshots)):
        before, current = snapshots[index - 1], snapshots[index]
        timestamp = current["timestamp_sec"]
        delta = current["total_people"] - before["total_people"]
        old_zone, new_zone = before.get("highest_zone"), current.get("highest_zone")
        if delta and abs(delta) >= max(2, _num(median([abs(snapshots[i]["total_people"] - snapshots[i - 1]["total_people"]) for i in range(1, len(snapshots))]) if len(snapshots) > 1 else 1) * 1.5):
            direction = "builds" if delta > 0 else "eases"
            candidates.append(_candidate("CROWD_BUILDUP" if delta > 0 else "CROWD_DROP", index, timestamp, f"Crowd {direction}", f"Observed crowd changes from {before['total_people']} to {current['total_people']} people.", f"The synchronized Phase 2 counts show a {abs(delta)} person change at {_time(timestamp)}.", 35 + min(35, abs(delta)), current, ["Phase 2 risk record", "Monitoring timestamp"]))
        if old_zone and new_zone and old_zone != new_zone:
            candidates.append(_candidate("HOTSPOT_SHIFT", index, timestamp, "Crowd hotspot shifts", f"Highest observed zone changes from {_label(old_zone)} to {_label(new_zone)}.", "The most crowded zone changed between adjacent synchronized observations.", 58, current, ["Zone counts", "Phase 2 risk record"]))
        if before.get("risk_level") != current.get("risk_level") and current.get("risk_level") != "STABLE":
            candidates.append(_candidate("RISK_TRANSITION", index, timestamp, f"Risk moves to {current['risk_level']}", f"Risk status is now {current['risk_level']} in {_label(current.get('highest_zone'))}.", "The risk label changed in the recorded monitoring analytics.", 65, current, ["Phase 2 risk level"]))
        if _num(current.get("instability_score")) > _num(before.get("instability_score")) + 10:
            candidates.append(_candidate("INSTABILITY_RISE", index, timestamp, "Instability rises", f"Instability reaches {round(current['instability_score'])}/100.", "The synchronized instability score increased by more than ten points.", 70, current, ["Instability Radar snapshot"]))
        if _num(current.get("compression")) > _num(before.get("compression")) + 10:
            candidates.append(_candidate("COMPRESSION_RISE", index, timestamp, "Compression signal rises", f"Compression signal reaches {round(current['compression'])}/100.", "The Instability Radar signal increased between observations.", 72, current, ["Instability Radar snapshot"]))
        before_speed = _num(before.get("average_speed"))
        current_speed = _num(current.get("average_speed"))
        if before_speed < 10 <= current_speed:
            candidates.append(_candidate("MOVEMENT_ACTIVATES", index, timestamp, "Movement becomes active", f"Tracked movement speed rises from {before_speed:.1f} to {current_speed:.1f} pixels per second.", "Flow Intelligence detected the first sustained movement in the available video record.", 52, current, ["Flow Intelligence trend", "Video timestamp"]))
        elif before_speed >= 10 and current_speed < before_speed - 8:
            candidates.append(_candidate("MOVEMENT_SLOWS", index, timestamp, "Movement slows", f"Tracked movement speed eases from {before_speed:.1f} to {current_speed:.1f} pixels per second.", "Flow Intelligence detected a meaningful reduction in movement speed between adjacent video samples.", 48, current, ["Flow Intelligence trend", "Video timestamp"]))
    last = snapshots[-1]
    changed = any(
        last.get(key) != first.get(key)
        for key in ("total_people", "zone_counts", "highest_zone", "density_state", "risk_level", "instability_score", "compression", "counter_flow", "stop_go", "direction_disorder", "active_tracks", "average_speed")
    )
    if duration > first["timestamp_sec"] and changed:
        candidates.append(_candidate("SESSION_END", duration, duration, "Monitoring record ends", f"The available monitoring record ends with {last['total_people']} observed people.", "This is the last synchronized observation available for this storyboard.", 30, last, ["Flow Intelligence duration", "Monitoring snapshot"]))
    return candidates


def _suppress(candidates, duration):
    gap = max(1.0, min(12.0, duration * 0.04))
    result = []
    for item in sorted(candidates, key=lambda candidate: (candidate["timestamp_sec"], -candidate["score"])):
        duplicate = next((existing for existing in result if existing["event_type"] == item["event_type"] and abs(existing["timestamp_sec"] - item["timestamp_sec"]) <= gap), None)
        if duplicate:
            if item["score"] > duplicate["score"]:
                result[result.index(duplicate)] = item
        else:
            result.append(item)
    return result


def generate_storyboard(event, monitoring, flow_result, phase2_records=None, radar=None, time_result=None, lineage=None):
    snapshots, duration, fallback = build_snapshots(flow_result, phase2_records, radar, time_result)
    candidates = _suppress(detect_candidates(snapshots, duration), duration)
    max_cards = 4 if duration <= 15 else 5 if duration <= 30 else 6 if duration <= 60 else 8 if duration <= 180 else 10
    selected = sorted(sorted(candidates, key=lambda item: item["score"], reverse=True)[:max_cards], key=lambda item: item["timestamp_sec"])
    cards = []
    for order, item in enumerate(selected, 1):
        snapshot = item["snapshot"]
        previous = snapshots[max(0, min(len(snapshots) - 1, next((index for index, sample in enumerate(snapshots) if sample["timestamp_sec"] >= item["timestamp_sec"]), 0) - 1))]
        previous_counts = previous.get("zone_counts") or {}
        current_counts = snapshot.get("zone_counts") or {}
        delta_people = snapshot["total_people"] - previous.get("total_people", snapshot["total_people"])
        current_compression = _optional_num(snapshot.get("compression"))
        previous_compression = _optional_num(previous.get("compression"))
        delta_compression = round(current_compression - previous_compression, 1) if current_compression is not None and previous_compression is not None else None
        narration_text = f"At {_time(item['timestamp_sec'])}, {item['title'].lower()}. {item['caption']}"
        cards.append({
            "id": f"story-{order}-{int(item['timestamp_sec'] * 1000)}",
            "order": order,
            "timestamp_sec": round(item["timestamp_sec"], 3),
            "timestamp": _time(item["timestamp_sec"]),
            "frame_index": int(round(item["timestamp_sec"] * _num((flow_result.get("frame") or {}).get("fps"), 1))),
            "event_type": item["event_type"],
            "title": item["title"],
            "caption": item["caption"],
            "explanation": item["explanation"],
            "narration_text": narration_text,
            "significance_score": item["score"],
            "evidence": item["evidence"],
            "total_people": snapshot["total_people"],
            "zone_counts": snapshot["zone_counts"],
            "highest_zone": snapshot.get("highest_zone"),
            "density_state": snapshot.get("density_state"),
            "risk_level": snapshot.get("risk_level"),
            "instability_score": _round_optional(snapshot.get("instability_score")),
            "metrics": {"people_delta": delta_people, "active_tracks": _round_optional(snapshot.get("active_tracks")), "average_speed": _round_optional(snapshot.get("average_speed")), "compression": _round_optional(snapshot.get("compression")), "compression_delta": delta_compression, "counter_flow": _round_optional(snapshot.get("counter_flow")), "stop_go": _round_optional(snapshot.get("stop_go")), "direction_disorder": _round_optional(snapshot.get("direction_disorder")), "zone_people_before": previous_counts, "zone_people_after": current_counts},
            "frame_reference": {"video_source": monitoring.source_name, "timestamp_sec": round(item["timestamp_sec"], 3)},
            "source_provenance": {"monitoring_session_id": monitoring.id, **(lineage or {})},
        })
    population_history = [item for item in snapshots if item.get("people_available")]
    peak = max(population_history or snapshots, key=lambda item: item["total_people"])
    peak_instability = max(snapshots, key=lambda item: _num(item.get("instability_score")))
    detailed_timeline = build_detailed_timeline(snapshots, duration)
    narration = " ".join(f"At {card['timestamp']}, {card['title']}. {card['caption']}" for card in cards)
    return {
        "storyboard_id": None,
        "algorithm_version": ALGORITHM_VERSION,
        "event": {"id": event.id, "event_name": event.event_name, "location": event.location},
        "monitoring_session": {"id": monitoring.id, "source_name": monitoring.source_name, "source_type": monitoring.source_type, "status": monitoring.status},
        "duration_sec": round(duration, 3),
        "event_count": len(cards),
        "candidate_event_count": len(candidates),
        "cards": cards,
        "detailed_timeline": detailed_timeline,
        "narration": narration or "No significant transition was found in the available monitoring evidence.",
        "summary": {"peak_people": peak["total_people"], "peak_timestamp_sec": peak["timestamp_sec"], "highest_crowd_zone": peak.get("highest_zone"), "peak_instability": _round_optional(peak_instability.get("instability_score")), "peak_instability_timestamp_sec": peak_instability["timestamp_sec"], "final_density_state": snapshots[-1].get("density_state"), "final_risk_level": snapshots[-1].get("risk_level")},
        "data_quality": {
            "source": "synchronized monitoring analytics",
            "uses_real_timestamps": True,
            "timestamp_unit": "source-video-seconds",
            "yolo_rerun": False,
            "fallback_snapshot": len(snapshots) == 1 and not phase2_records,
            "people_snapshot_count": len(population_history),
            "first_people_timestamp_sec": population_history[0]["timestamp_sec"] if population_history else None,
            "last_people_timestamp_sec": population_history[-1]["timestamp_sec"] if population_history else None,
        },
        "source_lineage": {"monitoring_session_id": monitoring.id, **(lineage or {})},
        "video": {"source_name": monitoring.source_name, "url": f"/monitoring-videos/{monitoring.source_name}"},
        "enhanced_narrative": {"enabled": False, "provider": None},
    }
