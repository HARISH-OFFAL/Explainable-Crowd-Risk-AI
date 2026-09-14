"""Deterministic, approval-gated safety communication for CrowdGuard."""

from __future__ import annotations

from datetime import datetime, timezone

ZONES = ("ZONE_A", "ZONE_B", "ZONE_C")
COMMUNICATION_LLM_ENABLED = False


def _label(value):
    return str(value or "").replace("_", " ").title()


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _snapshot(monitoring):
    import json
    try:
        return json.loads(monitoring.final_snapshot_json or "{}")
    except (TypeError, ValueError):
        return {}


def build_context(monitoring, flow=None, forecast=None, response_plan=None, radar=None):
    snapshot = _snapshot(monitoring)
    counts = snapshot.get("zone_counts") or {}
    people = snapshot.get("total_people")
    zone = snapshot.get("most_crowded_zone") or (max(counts, key=counts.get) if any(counts.values()) else None)
    plan = response_plan or {}
    recommended = plan.get("recommended") or {}
    route_nodes = (recommended.get("route") or {}).get("nodes") or []
    safe_zone = next((item for item in route_nodes if item in ZONES and item != zone), None)
    flow = flow or {}
    radar = radar or {}
    instability = radar.get("overall_instability")
    forecast = forecast or {}
    forecast_item = forecast.get("30") or forecast.get(30) or {}
    risk = str(plan.get("risk_level") or snapshot.get("risk_level") or "INFORMATION").upper()
    action = str(recommended.get("scenario") or "MONITOR_ONLY").upper()
    if action == "COMBINED_RESPONSE":
        action = "REDIRECT_FLOW" if safe_zone else "GENERAL_SAFETY_ADVISORY"
    return {
        "event_id": monitoring.event_id, "monitoring_session_id": monitoring.id,
        "source_video": monitoring.source_name, "timestamp_sec": snapshot.get("source_timestamp_sec"),
        "current_people": people, "zone_counts": counts, "target_zone": zone,
        "density": snapshot.get("density_state"), "risk": risk,
        "instability": instability, "compression": radar.get("compression_score"),
        "counter_flow": flow.get("counter_flow"), "inflow": flow.get("inflow"),
        "outflow": flow.get("outflow"), "forecast": forecast_item,
        "recommended_action": action, "recommended_response": recommended,
        "safe_zone": safe_zone, "route_available": bool(safe_zone),
        "main_driver": "crowd concentration" if zone else "monitoring state",
    }


def priority(context):
    risk = context.get("risk", "INFORMATION")
    instability = _num(context.get("instability")) or 0
    action = context.get("recommended_action")
    forecast_status = str((context.get("forecast") or {}).get("risk_level") or (context.get("forecast") or {}).get("status") or "").upper()
    if risk in {"CRITICAL", "URGENT"} or instability >= 75:
        return "URGENT"
    if risk == "HIGH" or instability >= 50 or forecast_status in {"HIGH", "CRITICAL"}:
        return "HIGH"
    if action not in {None, "MONITOR_ONLY"} or risk in {"MEDIUM", "WATCH"}:
        return "ADVISORY"
    return "INFORMATION"


def needs_communication(context):
    return context.get("recommended_action") not in {None, "MONITOR_ONLY"} or priority(context) != "INFORMATION"


def _messages(context, language="en"):
    zone, safe, action = context.get("target_zone"), context.get("safe_zone"), context.get("recommended_action")
    z, s = _label(zone), _label(safe)
    route = f" Follow staff guidance toward {s}." if safe else " Alternate route information is unavailable; follow staff guidance."
    public_route = f" Please use the alternate route through {s} and follow staff guidance." if safe else " Please follow staff guidance in the area."
    if language == "ta":
        route_ta = f" {s} வழியாக பணியாளர்களின் வழிகாட்டுதலைப் பின்பற்றவும்." if safe else " பணியாளர்களின் வழிகாட்டுதலைப் பின்பற்றவும்."
        return {
            "ORGANIZER": f"{z or 'தேர்ந்தெடுக்கப்பட்ட பகுதி'} பகுதியில் கண்காணிப்பு கவனம் தேவை. CrowdGuard பரிந்துரை: {_label(action)}.",
            "SECURITY": f"{z or 'கவலை பகுதி'} அணுகுமுறைக்கு பணியாளர்களை வழிநடத்தவும்." + route_ta,
            "PUBLIC": f"கவனிக்கவும். {z or 'இந்த பகுதி'} பகுதியில் கூட்டம் அதிகமாக உள்ளது." + (f" {s} வழியாக செல்லவும்." if safe else " பணியாளர்களின் வழிகாட்டுதலைப் பின்பற்றவும்."),
            "EMERGENCY": f"Priority: {priority(context)}. {z or 'கவலை பகுதி'} பகுதியை கண்காணிக்கவும். பரிந்துரை: {_label(action)}.",
        }
    return {
        "ORGANIZER": f"{z or 'The selected area'} requires attention. Current crowd: {context.get('current_people') if context.get('current_people') is not None else 'N/A'}. Response Commander recommends {_label(action)}.",
        "SECURITY": f"Move staff toward the {z or 'concerned area'} approach and reduce incoming flow." + route,
            "PUBLIC": f"Attention please. The {z or 'concerned'} area is currently crowded." + public_route,
        "EMERGENCY": f"Priority {priority(context)}. Monitor {z or 'the concerned area'}. Recommended response: {_label(action)} at the recorded monitoring timestamp.",
    }


def build_plan(context):
    level = priority(context); messages = _messages(context)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(), "priority": level,
        "intent": context.get("recommended_action"), "requires_approval": True,
        "source_state": context, "recommended_response": context.get("recommended_response") or {},
        "audiences": [{"audience": audience, "intent": context.get("recommended_action"), "language": "en", "message": message, "delivery_channels": ["DASHBOARD", "EMAIL"] if audience != "PUBLIC" else ["DASHBOARD", "PA_SIMULATION"]} for audience, message in messages.items()],
        "translations": [{"audience": audience, "intent": context.get("recommended_action"), "language": "ta", "message": message} for audience, message in _messages(context, "ta").items()],
        "validation": {"route_available": context.get("route_available"), "safe_zone_required": context.get("recommended_action") == "REDIRECT_FLOW", "message_safety": True},
        "effectiveness_status": "NOT_MEASURED",
    }
