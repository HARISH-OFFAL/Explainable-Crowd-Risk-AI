"""Data-derived explanation layer for the existing CrowdGuard analytics."""

from math import fabs

ZONES = ("ZONE_A", "ZONE_B", "ZONE_C")
LEVELS = ("STABLE", "WATCH", "WARNING", "HIGH", "CRITICAL")


def number(value, fallback=0.0):
    try:
        parsed = float(value)
        return parsed if parsed == parsed else fallback
    except (TypeError, ValueError):
        return fallback


def level_from_score(score):
    value = number(score)
    if value >= 81:
        return "CRITICAL"
    if value >= 66:
        return "HIGH"
    if value >= 46:
        return "WARNING"
    if value >= 26:
        return "WATCH"
    return "STABLE"


def score(value):
    return round(max(0.0, min(100.0, number(value))))


def _risk_score(prediction, radar=None):
    if radar:
        return score(radar.get("overall_instability"))
    level = str((prediction or {}).get("risk_level") or "STABLE").upper()
    return {"CRITICAL": 90, "HIGH": 75, "WARNING": 55, "MEDIUM": 55, "WATCH": 35, "LOW": 15, "STABLE": 10}.get(level, 10)


def _drivers(current, future, radar, flow):
    current_zones = current.get("zone_counts") or {}
    future_zones = (future or {}).get("zone_counts") or current_zones
    risk_zone = (future or {}).get("risk_zone") or max(ZONES, key=lambda zone: number(future_zones.get(zone)))
    drivers = []
    current_people = number(current.get("total_people"))
    future_people = number((future or {}).get("total_people"), current_people)
    growth = number(future_zones.get(risk_zone)) - number(current_zones.get(risk_zone))
    concentration = number(current_zones.get(risk_zone)) / max(current_people, 1) * 100
    drivers.append({"name": "Crowd accumulation", "score": score(max(0, growth) * 8 + concentration * .45), "evidence": f"{risk_zone.replace('_', ' ').title()} changes from {int(number(current_zones.get(risk_zone)))} to {int(number(future_zones.get(risk_zone)))} people by the selected forecast."})

    metrics = (radar or {}).get("zone_metrics", {}).get(risk_zone, {})
    if radar:
        drivers.extend([
            {"name": "Compression Proxy", "score": score(number(metrics.get("compression")) * 100), "evidence": f"Crowd compression proxy for {risk_zone.replace('_', ' ').title()} is {score(number(metrics.get('compression')) * 100)}%."},
            {"name": "Stop-Go behaviour", "score": score(number(metrics.get("stop_go")) * 100), "evidence": f"Repeated movement interruptions register at {score(number(metrics.get('stop_go')) * 100)}%."},
            {"name": "Motion disorder", "score": score(number(metrics.get("direction_disorder")) * 100), "evidence": f"Directional disorder in the selected radar zone is {score(number(metrics.get('direction_disorder')) * 100)}%."},
            {"name": "Counter-flow", "score": score(number(metrics.get("counter_flow")) * 100), "evidence": f"Opposing movement signal is {score(number(metrics.get('counter_flow')) * 100)}%."},
        ])
    elif flow:
        zone_flow = (flow.get("zones") or {}).get(risk_zone, {})
        net = number(zone_flow.get("net"))
        drivers.append({"name": "Inflow / outflow imbalance", "score": score(fabs(net) * 10), "evidence": f"Observed net flow for {risk_zone.replace('_', ' ').title()} is {net:+.1f} people per minute."})
        drivers.append({"name": "Movement consistency", "score": score(number(zone_flow.get("average_movement")) * 2), "evidence": "Flow Intelligence movement statistics are available; radar-specific signals are not yet completed."})
    return sorted(drivers, key=lambda item: item["score"], reverse=True)[:5]


def build_explanation(event, lineage, current, predictions, radar=None, plan=None, flow=None):
    future = predictions.get("30") or predictions.get("15") or {}
    risk_zone = (future.get("risk_zone") or current.get("highest_zone") or "ZONE_A")
    risk_score = _risk_score(future, radar)
    current["risk_level"] = (radar or {}).get("level") or level_from_score(risk_score)
    current["instability_score"] = score((radar or {}).get("overall_instability", risk_score))
    drivers = _drivers(current, future, radar, flow)
    recommendation = ((plan or {}).get("recommended") or {}).get("scenario") or ((plan or {}).get("recommended") or {}).get("name") or "Monitor and reassess"
    if recommendation == "INCREASE_EXIT_CAPACITY":
        recommendation = "Increase exit flow"
    recommendation = str(recommendation).replace("_", " ").title()
    confidence = score(number(future.get("confidence")) * 100)
    if not confidence:
        confidence = None
    zone_label = risk_zone.replace("_", " ").title()
    summary = f"{zone_label} requires attention. The current monitored crowd is {int(number(current.get('total_people')))} people, with {int(number((current.get('zone_counts') or {}).get(risk_zone)))} in {zone_label}."
    if drivers:
        summary += f" {drivers[0]['evidence']}"
    summary += f" The selected forecast indicates {future.get('risk_level', level_from_score(risk_score))} risk in {zone_label}."
    voice = summary + f" Recommended response: {recommendation}."
    forecast = {}
    for horizon in (15, 30, 60):
        item = predictions.get(str(horizon)) or {}
        item_score = _risk_score(item, radar if horizon == 0 else None)
        forecast[str(horizon)] = {"risk_level": item.get("risk_level") or level_from_score(item_score), "instability_score": item_score, "confidence": score(number(item.get("confidence")) * 100) or None, "risk_zone": item.get("risk_zone") or risk_zone}
    return {"event_id": event.id, "event_name": event.event_name, "source_lineage": lineage, "current": current, "forecast": forecast, "explanation": {"risk_zone": risk_zone, "confidence": confidence, "drivers": drivers, "summary": summary, "voice_summary": voice, "recommended_response": recommendation}, "alert_recommended": any(item["risk_level"] in {"HIGH", "CRITICAL"} and (item["confidence"] or 0) >= 60 for item in forecast.values())}
