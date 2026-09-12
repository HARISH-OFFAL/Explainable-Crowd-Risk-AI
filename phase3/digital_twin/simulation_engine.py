"""Deterministic, conservation-aware Digital Twin scenario simulation."""

from copy import deepcopy

CONFIG = {
    "step_seconds": 1,
    "default_horizon": 30,
    "allowed_horizons": (15, 30, 60),
    "speed_density_coefficient": 0.65,
    "score_weights": {"congestion_reduction": 0.45, "overload_penalty": 0.25, "flow_balance": 0.30},
}
ZONES = ("ZONE_A", "ZONE_B", "ZONE_C")


def _clamp(value, low, high):
    return max(low, min(high, value))


def _state_for(people, base):
    density = _clamp(people / max(base["capacity"], 1) * 100, 0, 100)
    net = base["inflow_rate"] - base["outflow_rate"]
    speed = max(base["relative_speed"], 0)
    if density >= 80 and net > 0 and speed < base["relative_speed"] * 0.65:
        state = "BOTTLENECK"
    elif net > 0.1:
        state = "Accumulating"
    elif net < -0.1:
        state = "Dispersing"
    else:
        state = "Balanced"
    return density, state


def simulate(snapshot, action=None, horizon=None):
    """Run one scenario using simultaneous updates and rates in persons/minute.

    Zone inflow/outflow values are measured pressure rates. Explicit
    ``external_entry_rate`` and ``external_exit_rate`` fields are the only
    population boundary terms; they are converted to persons/second here.
    Redirect moves a percentage of eligible source-zone inflow pressure, not a
    percentage of the entire population.
    """
    source = deepcopy(snapshot)
    horizon = int(horizon or CONFIG["default_horizon"])
    if horizon not in CONFIG["allowed_horizons"]:
        raise ValueError("Simulation horizon must be 15, 30, or 60 seconds")
    action = deepcopy(action or {"type": "NO_ACTION"})
    action_type = action.get("type", "NO_ACTION")
    restrict = _clamp(float(action.get("restrict_percent", 0)), 0, 100) / 100
    redirect_percent = _clamp(float(action.get("redirect_percent", 0)), 0, 100) / 100
    redirect_from = action.get("from_zone", "ZONE_B")
    redirect_to = action.get("to_zone", "ZONE_C")
    if redirect_percent and (redirect_from, redirect_to) != ("ZONE_B", "ZONE_C"):
        raise ValueError("Redirect currently supports Zone B to Zone C only")
    people = {zone: max(0.0, float(source["zones"][zone]["current_people"])) for zone in ZONES}
    capacities = {zone: max(0.0, float(source["zones"][zone]["capacity"])) for zone in ZONES}
    external_entry_rate = max(0.0, float(source.get("external_entry_rate", 0)))
    external_exit_rate = max(0.0, float(source.get("external_exit_rate", 0)))
    timeline = [{"timestamp": 0, "zones": {zone: round(people[zone], 6) for zone in ZONES}, "total_people": round(sum(people.values()), 6)}]
    transfer_total = 0.0
    blocked_redirect_total = 0.0

    for second in range(1, horizon + 1, CONFIG["step_seconds"]):
        old = people.copy()
        delta = {zone: 0.0 for zone in ZONES}
        for zone in ZONES:
            base = source["zones"][zone]
            inflow = max(0.0, float(base.get("inflow_rate", 0)))
            outflow = max(0.0, float(base.get("outflow_rate", 0)))
            # Restricting the entry gate changes the admission flow itself;
            # it must not be added as an extra population source later.
            if action_type in {"RESTRICT_ENTRY", "COMBINED_RESPONSE"} and zone == "ZONE_A":
                inflow *= 1 - restrict
            delta[zone] += (inflow - outflow) / 60.0 * CONFIG["step_seconds"]
        entry = external_entry_rate / 60.0 * CONFIG["step_seconds"]
        exit_flow = external_exit_rate / 60.0 * CONFIG["step_seconds"]
        delta["ZONE_A"] += entry * (1 - restrict)
        delta["ZONE_C"] -= min(old["ZONE_C"], exit_flow)
        if action_type in {"INCREASE_EXIT", "INCREASE_EXIT_CAPACITY", "COMBINED_RESPONSE"}:
            additional = max(0.0, float(action.get("additional_capacity", 0))) / 60.0 * CONFIG["step_seconds"]
            exit_zone = action.get("exit_zone", "ZONE_C")
            if exit_zone in ZONES:
                delta[exit_zone] -= min(max(0.0, old[exit_zone] + delta[exit_zone]), additional)
        if action_type in {"REDIRECT", "REDIRECT_FLOW", "COMBINED_RESPONSE"} and (redirect_percent or action.get("redirect_people") is not None):
            if action.get("redirect_people") is not None:
                requested = max(0.0, float(action.get("redirect_people", 0))) if second == 1 else 0.0
            else:
                measured_inflow = max(0.0, float(source["zones"][redirect_from].get("inflow_rate", 0)))
                # Recorded clips often have no calibrated inflow rate. Keep
                # Redirect useful in that case by distributing the requested
                # percentage of the source-zone population across the horizon.
                pressure = measured_inflow / 60.0 * redirect_percent if measured_inflow else old[redirect_from] * redirect_percent / max(horizon, 1)
                requested = pressure * CONFIG["step_seconds"]
            available_source = max(0.0, old[redirect_from] + delta[redirect_from])
            available_destination = max(0.0, capacities[redirect_to] - (old[redirect_to] + delta[redirect_to]))
            effective = min(requested, available_source, available_destination)
            blocked_redirect_total += requested - effective
            delta[redirect_from] -= effective
            delta[redirect_to] += effective
            transfer_total += effective
        # Double buffer: all deltas above use ``old`` and are applied together.
        people = {zone: _clamp(old[zone] + delta[zone], 0, capacities[zone]) for zone in ZONES}
        timeline.append({"timestamp": second, "zones": {zone: round(people[zone], 6) for zone in ZONES}, "total_people": round(sum(people.values()), 6)})

    zone_results = {}
    for zone in ZONES:
        base = source["zones"][zone]
        density, state = _state_for(people[zone], base)
        speed_factor = max(0.25, 1 - CONFIG["speed_density_coefficient"] * density / 100)
        inflow_rate = round(max(0.0, float(base.get("inflow_rate", 0))), 2)
        outflow_rate = round(max(0.0, float(base.get("outflow_rate", 0))), 2)
        zone_results[zone] = {"people": round(people[zone], 2), "density_index": round(density, 2), "inflow_rate": inflow_rate, "outflow_rate": outflow_rate, "net_flow": round(inflow_rate - outflow_rate, 2), "relative_speed": round(max(0.0, float(base.get("relative_speed", 0))) * speed_factor, 2), "flow_state": state, "capacity": capacities[zone]}
    return {"action": action, "horizon_seconds": horizon, "zones": zone_results, "total_people": round(sum(people.values()), 2), "timeline": timeline, "transfer_summary": {"effective_redirect_people": round(transfer_total, 6), "blocked_redirect_people": round(blocked_redirect_total, 6), "external_entry_rate": external_entry_rate, "external_exit_rate": external_exit_rate}, "simulation_label": "Estimated Scenario"}
