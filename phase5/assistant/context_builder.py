def _compact(value):
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value

def build_context(event, monitoring, flow, time_result, radar=None, plan=None, lineage=None, history=None):
    snapshot = (flow or {}).get("phase2_snapshot") or {}
    zones = snapshot.get("zone_counts") or {}
    predictions = (time_result or {}).get("predictions") or {}
    flow_zones = (flow or {}).get("zones") or {}
    risk_zone = max(zones, key=zones.get) if zones else None
    context = {"event": {"id": event.id, "name": event.event_name, "location": event.location}, "monitoring": {"session_id": monitoring.id, "current_people": snapshot.get("total_people"), "zone_counts": zones, "most_crowded_zone": risk_zone, "source_timestamp": (flow or {}).get("snapshot_provenance", {}).get("source_timestamp_sec")}, "flow": {"direction": (flow or {}).get("dominant_direction"), "state": (flow or {}).get("flow_state"), "counter_flow": (flow or {}).get("counter_flow"), "zones": flow_zones}, "forecast": {str(h): predictions.get(str(h)) for h in (15, 30, 60)}, "lineage": lineage or {}}
    if radar:
        context["instability"] = {"analysis_id": radar.get("id"), "score": radar.get("overall_instability"), "state": radar.get("level"), "highest_zone": radar.get("highest_instability_zone"), "zone_metrics": radar.get("zone_metrics")}
    if plan:
        recommended = plan.get("recommended") or {}
        context["response"] = {"recommended_action": recommended.get("scenario") or recommended.get("name"), "reason": plan.get("reasons"), "target_zone": plan.get("risk_zone"), "route": plan.get("recommended_route")}
    if history:
        context["history"] = history[-30:]
    return _compact(context)
