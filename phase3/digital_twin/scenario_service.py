from .simulation_engine import CONFIG, simulate


def score_scenario(baseline, scenario):
    baseline_total = float(baseline.get("total_people", 0))
    scenario_total = float(scenario.get("total_people", 0))
    baseline_peak = max(item["density_index"] for item in baseline["zones"].values())
    scenario_peak = max(item["density_index"] for item in scenario["zones"].values())
    baseline_accumulation = sum(max(0, item["net_flow"]) for item in baseline["zones"].values())
    scenario_accumulation = sum(max(0, item["net_flow"]) for item in scenario["zones"].values())
    overload = sum(max(0, item["density_index"] - 100) for item in scenario["zones"].values())
    reduction = max(0, baseline_peak - scenario_peak) / max(baseline_peak, 1)
    # Peak density can remain unchanged when the affected zone is not the
    # current peak. Include total projected population so useful actions on a
    # lower-density entry zone are still scored.
    population_reduction = max(0, baseline_total - scenario_total) / max(baseline_total, 1)
    balance = max(0, baseline_accumulation - scenario_accumulation) / max(baseline_accumulation, 1)
    score = 100 * (CONFIG["score_weights"]["congestion_reduction"] * max(reduction, population_reduction) + CONFIG["score_weights"]["flow_balance"] * balance - CONFIG["score_weights"]["overload_penalty"] * overload / 100)
    return round(max(0, score), 2)


def build_scenarios(snapshot, horizon):
    actions = [
        {"type": "NO_ACTION", "label": "No Action"},
        {"type": "RESTRICT_ENTRY", "label": "Restrict Entry 25%", "restrict_percent": 25},
        {"type": "INCREASE_EXIT", "label": "Increase Exit Capacity", "additional_capacity": 12},
        {"type": "REDIRECT", "label": "Redirect 20% eligible flow from Zone B to Zone C", "redirect_percent": 20, "from_zone": "ZONE_B", "to_zone": "ZONE_C"},
    ]
    results = []
    baseline = simulate(snapshot, actions[0], horizon)
    for action in actions:
        result = simulate(snapshot, action, horizon)
        result["score"] = score_scenario(baseline, result)
        results.append(result)
    best = max(results[1:], key=lambda item: item["score"], default=None)
    return {"baseline": baseline, "scenarios": results, "best": best}
