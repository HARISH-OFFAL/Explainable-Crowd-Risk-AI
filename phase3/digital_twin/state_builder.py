"""Builds an immutable simulation snapshot from Phase 3A output."""

from copy import deepcopy

ZONES = ("ZONE_A", "ZONE_B", "ZONE_C")


def build_current_state(flow_result, event_id=None, monitoring_session_id=None, flow_analysis_id=None):
    source_zones = flow_result.get("zones") or {}
    provenance = deepcopy(flow_result.get("snapshot_provenance") or {})
    if set(source_zones) != set(ZONES):
        raise ValueError("Complete zone snapshot is unavailable for this session")
    snapshot = {"event_id": event_id, "monitoring_session_id": monitoring_session_id or flow_result.get("session_id"), "flow_analysis_id": flow_analysis_id or flow_result.get("flow_analysis_id"), "source": flow_result.get("source"), "flow_state": flow_result.get("flow_state", "UNKNOWN"), "source_provenance": provenance, "zones": {}}
    for zone in ZONES:
        metrics = source_zones.get(zone) or {}
        required = {"people", "inflow", "outflow", "net", "average_movement", "state"}
        if not required.issubset(metrics):
            raise ValueError(f"Complete measured metrics are unavailable for {zone}")
        people = max(0.0, float(metrics["people"]))
        # No measured per-zone capacity exists in Phase 2/3. This is explicit
        # simulation capacity, not a claim about the physical venue.
        capacity = max(50.0, people * 1.8)
        snapshot["zones"][zone] = {
            "zone_id": zone,
            "current_people": people,
            "measurement": "Phase 3A completed-session final snapshot",
            "capacity": capacity,
            "density_index": min(100.0, people / capacity * 100),
            "inflow_rate": max(0.0, float(metrics["inflow"])),
            "outflow_rate": max(0.0, float(metrics["outflow"])),
            "net_flow": float(metrics["net"]),
            "relative_speed": max(0.0, float(metrics["average_movement"])),
            "flow_state": metrics["state"],
            "bottleneck_status": flow_result.get("flow_state") == "BOTTLENECK" and zone == flow_result.get("most_accumulating_zone"),
            "connected_zones": {"ZONE_A": ["ZONE_B"], "ZONE_B": ["ZONE_C"], "ZONE_C": []}[zone],
            "connected_exits": {"ZONE_A": [], "ZONE_B": ["EXIT_1"], "ZONE_C": ["EXIT_2"]}[zone],
        }
    snapshot["total_people"] = sum(snapshot["zones"][zone]["current_people"] for zone in ZONES)
    # Static/short recordings can have no usable motion-flow rates. Preserve
    # the measured snapshot, but expose a bounded estimated entry pressure so
    # entry-control scenarios still produce a meaningful projection.
    measured_inflow = sum(snapshot["zones"][zone]["inflow_rate"] for zone in ZONES)
    if measured_inflow <= 0 and not flow_result.get("external_entry_rate"):
        snapshot["external_entry_rate"] = max(1.0, min(30.0, snapshot["zones"]["ZONE_A"]["current_people"] * 0.5))
    else:
        snapshot["external_entry_rate"] = max(0.0, float(flow_result.get("external_entry_rate", 0)))
    snapshot["external_exit_rate"] = max(0.0, float(flow_result.get("external_exit_rate", 0)))
    return deepcopy(snapshot)
