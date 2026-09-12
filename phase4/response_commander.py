"""Deterministic Phase 4 response planning built from completed Phase 3 data."""
from copy import deepcopy
import networkx as nx
from .schemas import normalize_state, normalize_prediction, VenueLayout
from phase3.digital_twin.simulation_engine import simulate as simulate_digital_twin

ZONES = ("ZONE_A", "ZONE_B", "ZONE_C")
EXITS = ("EXIT_1", "EXIT_2", "EXIT_3")

def number(value, default=0.0):
    try: return float(value)
    except (TypeError, ValueError): return default

RISK_THRESHOLDS = {"HIGH": 65.0, "MEDIUM": 40.0, "LOW": 0.0}


def venue_graph(layout=None, blocked_exits=(), closed_gates=()):
    if layout is not None:
        layout = VenueLayout.model_validate(layout).model_dump() if not isinstance(layout, VenueLayout) else layout.model_dump()
        graph = nx.DiGraph()
        for node in layout["nodes"]:
            graph.add_node(node["id"], **node)
        blocked = set(blocked_exits) | set(closed_gates)
        for edge in layout["edges"]:
            edge_data = dict(edge)
            edge_data["blocked"] = edge.get("blocked", False) or edge["source"] in blocked or edge["target"] in blocked
            graph.add_edge(edge["source"], edge["target"], **edge_data)
        return graph
    return nx.DiGraph()

def route(graph, source, exit_id, emergency=False):
    graph = graph.copy()
    graph.remove_edges_from([(a, b) for a, b, data in graph.edges(data=True) if data.get("blocked")])
    def cost(_, __, data):
        distance = number(data.get("distance", data.get("distance_weight", 1)), 1)
        congestion = number(data.get("congestion", 0), 0)
        return distance + congestion * 10 + (0 if emergency or not data.get("emergency_reserved") else 8)
    try:
        path = nx.dijkstra_path(graph, source, exit_id, weight=cost)
    except (nx.NetworkXNoPath, nx.NodeNotFound): return None
    edges = list(zip(path, path[1:]))
    return {"nodes": path, "edges": [{"source": a, "target": b} for a, b in edges], "exit": exit_id, "cost": round(sum(cost(a, b, graph[a][b]) for a, b in edges), 2), "warnings": ["Emergency corridor reserved for response teams"] if any(graph[a][b].get("emergency_reserved") for a, b in edges) else []}

def analyze_state(current, future=None, event_id=None, monitoring_session_id=None, flow_analysis_id=None, time_machine_session_id=None, prediction_horizon=30, layout=None, instability_signal=None):
    current = normalize_state(current)
    future = normalize_prediction(future)
    zones = current.get("zones") or {}; future_zones = (future or {}).get("zone_counts") or {}
    scores = {}
    zone_details = {}
    for zone in ZONES:
        item = zones.get(zone, {}); current_people = number(item.get("current_people", 0)); predicted_people = number(future_zones.get(zone, current_people)); net = number(item.get("net_flow", 0)); inflow = number(item.get("inflow_rate", 0)); outflow = number(item.get("outflow_rate", 0))
        growth = min(40.0, max(0.0, (predicted_people - current_people) / max(current_people, 1.0) * 40.0))
        accumulation = min(25.0, max(0.0, net) * 2.0)
        bottleneck = 20.0 if bool(item.get("bottleneck_status")) else 0.0
        imbalance = min(15.0, abs(net) / max(inflow + outflow, 1.0) * 15.0) if inflow + outflow else 0.0
        score = round(min(100.0, growth + accumulation + bottleneck + imbalance), 1)
        scores[zone] = score; zone_details[zone] = {"current_people": current_people, "predicted_people": predicted_people, "inflow": inflow, "outflow": outflow, "net_flow": net, "bottleneck": bool(item.get("bottleneck_status")), "components": {"projected_occupancy": round(growth, 1), "accumulation": round(accumulation, 1), "bottleneck": round(bottleneck, 1), "flow_imbalance": round(imbalance, 1)}, "risk_score": score, "risk_level": "HIGH" if score >= RISK_THRESHOLDS["HIGH"] else "MEDIUM" if score >= RISK_THRESHOLDS["MEDIUM"] else "LOW"}
    risk_zone = max(ZONES, key=lambda z: (scores[z], -ZONES.index(z))); selected = zone_details[risk_zone]; reasons = []
    if selected["predicted_people"] > selected["current_people"]: reasons.append(f"Projected {risk_zone.replace('ZONE_', 'Zone ')} occupancy is increasing")
    if selected["net_flow"] > 0: reasons.append(f"{risk_zone.replace('ZONE_', 'Zone ')} has positive net accumulation")
    if selected["bottleneck"]: reasons.append("A bottleneck is reported in the affected zone")
    if not reasons: reasons.append("Highest validated zone-risk score")
    graph = venue_graph(layout); routes = []
    if graph and risk_zone in graph:
        exits = [node for node, data in graph.nodes(data=True) if data.get("node_type") in {"EXIT", "EMERGENCY_EXIT"} and str(data.get("status", "OPEN")).upper() not in {"BLOCKED", "CLOSED"}]
        routes = sorted([item for item in (route(graph, risk_zone, exit_id) for exit_id in exits) if item], key=lambda item: item["cost"])
    return {"event_id": event_id, "monitoring_session_id": monitoring_session_id, "flow_analysis_id": flow_analysis_id, "time_machine_session_id": time_machine_session_id, "prediction_horizon": prediction_horizon, "risk_source": "Crowd Time Machine" if future_zones else "Flow Analysis", "source_state": deepcopy(current), "risk_zone": risk_zone, "risk_level": selected["risk_level"], "reasons": reasons, "zone_risk_scores": scores, "zone_risk_details": zone_details, "routes": routes, "recommended_exit": routes[0]["exit"] if routes else None, "recommended_route": routes[0] if routes else None, "instability_signal": deepcopy(instability_signal) if instability_signal else {}}

def simulate_plan(state, action, horizon=30, what_if=None):
    """Run every response from the same immutable Digital Twin baseline."""
    source = deepcopy(state["source_state"])
    baseline = {zone: number(source["zones"][zone].get("current_people", 0)) for zone in ZONES}
    for zone in ZONES:
        source["zones"][zone]["capacity"] = number(source["zones"][zone].get("capacity"), 0) or max(50.0, baseline[zone] * 1.8)
    capacities = {zone: max(number(source["zones"][zone].get("capacity"), 1), 1) for zone in ZONES}
    target = state.get("risk_zone", "ZONE_B")
    measured_outflow = sum(number(source["zones"][zone].get("outflow_rate", 0)) for zone in ZONES)
    actions = {
        "NO_ACTION": {"type": "NO_ACTION"},
        "REDIRECT_FLOW": {"type": "REDIRECT", "redirect_percent": 25, "from_zone": "ZONE_B", "to_zone": "ZONE_C"},
        "RESTRICT_ENTRY": {"type": "RESTRICT_ENTRY", "restrict_percent": 40},
        "INCREASE_EXIT_CAPACITY": {"type": "INCREASE_EXIT", "additional_capacity": 18, "exit_zone": target},
        "COMBINED_RESPONSE": {"type": "COMBINED_RESPONSE", "restrict_percent": 40, "redirect_percent": 25, "from_zone": "ZONE_B", "to_zone": "ZONE_C", "additional_capacity": 18, "exit_zone": target},
    }
    action_config = deepcopy(actions.get(action, {"type": "NO_ACTION"}))
    if action in {"INCREASE_EXIT_CAPACITY", "COMBINED_RESPONSE"} and measured_outflow <= 0:
        return {"scenario": action, "horizon_seconds": horizon, "baseline": baseline, "after": baseline, "timeline": [], "risk_before": "STABLE", "risk_after": "UNSUPPORTED", "response_score": 0.0, "valid": False, "unsupported_reason": "No configured zone outflow or exit capacity is available."}
    if what_if and what_if.get("zone") in ZONES:
        source["zones"][what_if["zone"]]["current_people"] *= max(0, float(what_if.get("multiplier", 1)))
    twin = simulate_digital_twin(source, action_config, horizon)
    after = {zone: number((twin.get("zones", {}).get(zone) or {}).get("people", baseline[zone])) for zone in ZONES}
    signal = state.get("instability_signal") or {}; radar_zones = signal.get("zone_metrics") or {}
    before_scores = {zone: number((radar_zones.get(zone) or {}).get("instability_score", 0)) for zone in ZONES}
    def estimated_scores(people):
        result = {}
        for zone in ZONES:
            ratio = people[zone] / max(baseline[zone], 1)
            base_score = before_scores[zone]
            occupancy_pressure = min(100, people[zone] / capacities[zone] * 100)
            movement_score = base_score * (.55 + .45 * ratio)
            result[zone] = round(max(0, min(100, movement_score * .65 + occupancy_pressure * .35)), 1)
        return result
    after_scores = estimated_scores(after); before_instability = sum(before_scores.values()) / 3; after_instability = sum(after_scores.values()) / 3
    risk = lambda score: "SEVERE" if score >= 81 else "HIGH" if score >= 66 else "UNSTABLE" if score >= 46 else "WATCH" if score >= 26 else "STABLE"
    baseline_peak = max(before_scores.values(), default=0); after_peak = max(after_scores.values(), default=0)
    balance_before = max(baseline.values()) - min(baseline.values()); balance_after = max(after.values()) - min(after.values())
    risk_reduction = max(0, baseline_peak - after_peak) / max(baseline_peak, 1) * 100
    instability_reduction = max(0, before_instability - after_instability) / max(before_instability, 1) * 100
    balance_improvement = max(0, balance_before - balance_after) / max(balance_before, 1) * 100
    hotspot = any(after_scores[zone] >= 66 and after_scores[zone] > before_scores[zone] + 5 for zone in ZONES)
    complexity = {"NO_ACTION": 0, "REDIRECT_FLOW": 5, "RESTRICT_ENTRY": 5, "INCREASE_EXIT_CAPACITY": 5, "COMBINED_RESPONSE": 12}.get(action, 8)
    score = round(max(0, min(100, risk_reduction * .30 + instability_reduction * .25 + risk_reduction * .15 + balance_improvement * .15 + (0 if hotspot else 15) - complexity)), 1)
    timeline = [{"timestamp": item["timestamp"], "zones": item["zones"], "zone_instability": estimated_scores(item["zones"])} for item in twin.get("timeline", []) if item["timestamp"] in {0, 10, 20, 30}]
    return {"scenario": action, "horizon_seconds": horizon, "baseline": baseline, "after": {zone: round(after[zone], 2) for zone in ZONES}, "timeline": timeline, "zone_instability": after_scores, "instability_before": round(before_instability, 1), "instability_after": round(after_instability, 1), "risk_before": risk(baseline_peak), "risk_after": risk(after_peak), "response_score": score, "valid": all(value >= 0 for value in after.values()), "new_hotspot": hotspot, "explanation": {"what": action.replace("_", " ").title(), "why": "; ".join(state.get("reasons", [])), "expected_effect": "Estimated result from the existing Digital Twin using the measured baseline."}}

def build_plan(state, horizon=30):
    simulations = [simulate_plan(state, action, horizon) for action in ("NO_ACTION", "REDIRECT_FLOW", "INCREASE_EXIT_CAPACITY", "RESTRICT_ENTRY", "COMBINED_RESPONSE")]
    supported = [item for item in simulations if item.get("valid")]
    recommended = max(supported or simulations, key=lambda item: item["response_score"])
    recommended["route"] = deepcopy(state.get("recommended_route"))
    result = deepcopy(state); result.update(simulations=simulations, recommended=recommended, status="RECOMMENDED"); return result


def _risk_value(level_name):
    return {"LOW": 20.0, "MEDIUM": 55.0, "HIGH": 85.0}.get(str(level_name).upper(), 0.0)


def impact_analysis(plan, scenario):
    """Compare one candidate response against the immutable observed baseline."""
    simulation = next((item for item in plan.get("simulations", []) if item.get("scenario") == scenario), None)
    if simulation is None:
        simulation = simulate_plan(plan, scenario, int(plan.get("prediction_horizon", 30)))
    signal = plan.get("instability_signal") or {}
    baseline = {zone: number(simulation.get("baseline", {}).get(zone, 0)) for zone in ZONES}
    after_people = {zone: number(simulation.get("after", {}).get(zone, baseline[zone])) for zone in ZONES}
    radar_zones = signal.get("zone_metrics") or {}
    before_scores = {zone: number((radar_zones.get(zone) or {}).get("instability_score", 0)) for zone in ZONES}
    after_scores = {zone: number((simulation.get("zone_instability") or {}).get(zone, before_scores[zone] * after_people[zone] / max(baseline[zone], 1))) for zone in ZONES}
    before_highest = signal.get("highest_instability_zone") or plan.get("risk_zone")
    after_highest = max(ZONES, key=lambda zone: (after_scores[zone], -ZONES.index(zone)))
    def metric(name, source):
        return max([number((radar_zones.get(zone) or {}).get(name, 0)) for zone in source], default=0.0)
    before = {"risk": plan.get("risk_level", "LOW"), "instability": number(signal.get("overall_instability", max(before_scores.values(), default=0))), "stability": number(signal.get("overall_stability", 100 - max(before_scores.values(), default=0))), "compression": metric("compression", ZONES), "counter_flow": metric("counter_flow", ZONES), "stop_go": metric("stop_go", ZONES), "motion_disorder": metric("direction_disorder", ZONES), "zone_people": baseline, "highest_zone": before_highest, "net_accumulation": {zone: number((plan.get("source_state", {}).get("zones", {}).get(zone) or {}).get("net_flow", 0)) for zone in ZONES}}
    after_instability = round(number(simulation.get("instability_after", sum(after_scores.values()) / 3)), 1)
    changed_zone = {zone: round(after_scores[zone] - before_scores[zone], 1) for zone in ZONES}
    scale = after_instability / max(before["instability"], 1)
    after = {"risk": simulation.get("risk_after", plan.get("risk_level", "LOW")), "instability": after_instability, "stability": round(100 - after_instability, 1), "compression": round(before["compression"] * scale, 3), "counter_flow": round(before["counter_flow"] * scale, 3), "stop_go": round(before["stop_go"] * scale, 3), "motion_disorder": round(before["motion_disorder"] * scale, 3), "zone_people": after_people, "highest_zone": after_highest, "net_accumulation": before["net_accumulation"]}
    reductions = {"risk": max(0, _risk_value(before["risk"]) - _risk_value(after["risk"])) / 65 * 100, "instability": max(0, before["instability"] - after["instability"]) / max(before["instability"], 1) * 100, "compression": max(0, before["compression"] - after["compression"]) * 100, "flow": max(0, before["counter_flow"] - after["counter_flow"]) * 100, "balance": max(0, max(before_scores.values(), default=0) - max(after_scores.values(), default=0))}
    new_hotspot = bool(simulation.get("new_hotspot")) or (after_highest != before_highest and after_scores[after_highest] > before_scores.get(after_highest, 0) + 3)
    complexity = {"NO_ACTION": 0, "REDIRECT_FLOW": 2, "INCREASE_EXIT_CAPACITY": 3, "RESTRICT_ENTRY": 3, "COMBINED_RESPONSE": 5}.get(scenario, 3)
    impact_score = round(number(simulation.get("response_score", reductions["risk"] * .2 + reductions["instability"] * .35 + reductions["compression"] * .15 + reductions["flow"] * .1 + reductions["balance"] * .2 - (18 if new_hotspot else 0) - complexity)), 1)
    worsened = [zone for zone in ZONES if changed_zone[zone] > 0]
    return {"scenario": scenario, "before": before, "after": after, "deltas": {"instability": round(after["instability"] - before["instability"], 1), "compression": round((after["compression"] - before["compression"]) * 100, 1), "counter_flow": round((after["counter_flow"] - before["counter_flow"]) * 100, 1), "zones": {zone: round(after["zone_people"][zone] - before["zone_people"][zone], 1) for zone in ZONES}, "zone_instability": changed_zone}, "impact_score": impact_score, "impact_breakdown": reductions, "new_hotspot": new_hotspot, "worsened_zones": worsened, "effect": "POSITIVE" if impact_score >= 20 else "NEUTRAL" if abs(reductions["instability"]) < 1 else "NEGATIVE", "simulation": simulation}


def command_feed(plan):
    signal = plan.get("instability_signal") or {}; snapshots = signal.get("snapshots") or []
    events = [{"timestamp": round(number(signal.get("timestamp", 0)), 1), "type": "SYSTEM", "severity": "INFO", "title": "Radar analysis loaded", "message": f"{len(snapshots)} time-indexed crowd snapshots available.", "source": "instability_radar"}]
    previous = None; previous_zone_levels = {}; previous_compression = 0.0; previous_counter = 0.0; previous_stop = False; previous_instability = None; last_trend = None; propagation_seen = False; first_snapshot = snapshots[0] if snapshots else None
    if first_snapshot:
        events.append({"timestamp": round(number(first_snapshot.get("timestamp", 0)), 1), "type": "OBSERVATION", "severity": "INFO", "title": "Crowd state sampled", "message": f"{first_snapshot.get('active_tracks', 0)} active tracks occupy {first_snapshot.get('occupied_cells', 0)} radar cells; instability is {round(number(first_snapshot.get('overall_instability', 0)))}.", "source": "instability_radar"})
    for snapshot in snapshots:
        current = snapshot.get("highest_instability_zone")
        if previous and current != previous:
            events.append({"timestamp": round(number(snapshot.get("timestamp", 0)), 1), "type": "OBSERVATION", "severity": "INFO", "title": "Highest zone changed", "message": f"{previous.replace('ZONE_', 'Zone ')} → {current.replace('ZONE_', 'Zone ')}.", "source": "instability_radar"})
        metrics = snapshot.get("zone_metrics") or {}; instability = number(snapshot.get("overall_instability", 0))
        if previous_instability is not None and abs(instability - previous_instability) >= 1.5:
            trend = "increasing" if instability > previous_instability else "decreasing"
            if trend != last_trend:
                events.append({"timestamp": round(number(snapshot.get("timestamp", 0)), 1), "type": "PREDICTION", "severity": "INFO", "title": f"Instability trend {trend}", "message": f"Overall instability moved {previous_instability:.1f} → {instability:.1f}; the current field is {trend}.", "source": "instability_radar"})
                last_trend = trend
        previous_instability = instability
        for zone in ZONES:
            level_name = (metrics.get(zone) or {}).get("level")
            if level_name and previous_zone_levels.get(zone) and level_name != previous_zone_levels[zone]:
                events.append({"timestamp": round(number(snapshot.get("timestamp", 0)), 1), "type": "WARNING" if level_name in {"UNSTABLE", "HIGH INSTABILITY", "SEVERE"} else "OBSERVATION", "severity": "WARNING", "title": "Zone state changed", "message": f"{zone.replace('ZONE_', 'Zone ')} changed from {previous_zone_levels[zone]} to {level_name}.", "source": "instability_radar"})
            previous_zone_levels[zone] = level_name
        compression = max([number((metrics.get(zone) or {}).get("compression", 0)) for zone in ZONES], default=0); counter = max([number((metrics.get(zone) or {}).get("counter_flow", 0)) for zone in ZONES], default=0); stop = max([number((metrics.get(zone) or {}).get("stop_go", 0)) for zone in ZONES], default=0) >= .1
        if previous_compression < .15 <= compression: events.append({"timestamp": round(number(snapshot.get("timestamp", 0)), 1), "type": "WARNING", "severity": "WARNING", "title": "Compression region active", "message": f"Compression proxy reached {round(compression * 100)}% in the current field.", "source": "instability_radar"})
        if previous_counter < .1 <= counter: events.append({"timestamp": round(number(snapshot.get("timestamp", 0)), 1), "type": "CRITICAL", "severity": "CRITICAL", "title": "Counter-flow detected", "message": f"Opposing streams reached {round(counter * 100)}% in the current radar region.", "source": "instability_radar"})
        if not previous_stop and stop: events.append({"timestamp": round(number(snapshot.get("timestamp", 0)), 1), "type": "OBSERVATION", "severity": "INFO", "title": "Stop-go transition detected", "message": "A local stop-go transition entered the radar field.", "source": "instability_radar"})
        previous_compression, previous_counter, previous_stop = compression, counter, stop
        if not propagation_seen and snapshot.get("propagation", {}).get("from") and snapshot.get("propagation", {}).get("to"):
            events.append({"timestamp": round(number(snapshot.get("timestamp", 0)), 1), "type": "PREDICTION", "severity": "INFO", "title": "Instability propagation detected", "message": f"The hotspot is moving with vector {snapshot['propagation'].get('vector', {})}.", "source": "instability_radar"})
            propagation_seen = True
        previous = current
    events.append({"timestamp": round(number(signal.get("timestamp", 0)), 1), "type": "RECOMMENDATION", "severity": "INFO", "title": "Response recommendation ready", "message": f"{str((plan.get('recommended') or {}).get('scenario', 'NO_ACTION')).replace('_', ' ').title()} is the current candidate recommendation.", "source": "response_commander"})
    simulation = plan.get("simulation")
    if simulation:
        events.extend([{ "timestamp": round(number(signal.get("timestamp", 0)), 1), "type": "SIMULATION", "severity": "INFO", "title": "Simulation completed", "message": f"{str(simulation.get('scenario', 'NO_ACTION')).replace('_', ' ').title()} scenario completed from the immutable baseline.", "source": "response_simulator" }, {"timestamp": round(number(signal.get("timestamp", 0)), 1), "type": "INFO", "severity": "INFO", "title": "Simulated result", "message": f"Risk {simulation.get('risk_before')} → {simulation.get('risk_after')}; population changes are simulated only.", "source": "response_simulator"}])
    if plan.get("status") == "APPROVED":
        events.append({"timestamp": round(number(signal.get("timestamp", 0)), 1), "type": "APPROVAL", "severity": "INFO", "title": "Response plan approved", "message": "Approved inside the decision-support system; no physical infrastructure was controlled.", "source": "operator"})
    unique = []; seen = set()
    for event in sorted(events, key=lambda item: item["timestamp"]):
        key = (event["type"], event["title"], event["message"])
        if key not in seen: seen.add(key); unique.append(event)
    return unique

def communications(state):
    zone = state["risk_zone"].replace("ZONE_", "Zone "); exit_name = (state.get("recommended_exit") or "EXIT_3").replace("EXIT_", "Exit ")
    return {"public": f"Visitors in {zone}, please proceed calmly toward {exit_name} using the selected route.", "staff": f"Staff near {zone} should redirect incoming movement toward {exit_name} and keep the emergency corridor clear.", "emergency": "Maintain the emergency corridor for medical and security access."}
