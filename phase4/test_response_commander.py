from .response_commander import analyze_state, build_plan, simulate_plan, venue_graph, route

def source():
    return {"zones": {"ZONE_A": {"current_people": 30, "net_flow": 0}, "ZONE_B": {"current_people": 70, "net_flow": 3}, "ZONE_C": {"current_people": 25, "net_flow": 0}}}

def test_blocked_exit_is_excluded():
    graph = venue_graph({"nodes": [{"id": "ZONE_B", "name": "Zone B", "node_type": "ZONE", "x_normalized": .5, "y_normalized": .2}, {"id": "EXIT_N", "name": "North Exit", "node_type": "EXIT", "x_normalized": .5, "y_normalized": .8}], "edges": [{"source": "ZONE_B", "target": "EXIT_N", "distance_weight": 1, "blocked": True}]})
    assert route(graph, "ZONE_B", "EXIT_N") is None

def test_no_venue_does_not_invent_exits():
    result = analyze_state(source(), {"zone_counts": {"ZONE_A": 30, "ZONE_B": 70, "ZONE_C": 25}}, layout={"nodes": [], "edges": []})
    assert result["routes"] == []

def test_no_action_is_deterministic_and_conserved():
    state = analyze_state(source(), {"zone_counts": {"ZONE_A": 30, "ZONE_B": 70, "ZONE_C": 25}})
    first = simulate_plan(state, "NO_ACTION"); second = simulate_plan(state, "NO_ACTION")
    assert first == second
    assert sum(first["baseline"].values()) == sum(first["after"].values())

def test_commander_selects_highest_score():
    plan = build_plan(analyze_state(source(), {"zone_counts": {"ZONE_A": 30, "ZONE_B": 70, "ZONE_C": 25}}))
    assert plan["recommended"]["response_score"] == max(item["response_score"] for item in plan["simulations"])

def test_numeric_strings_are_normalized():
    result = analyze_state({"zones": {"ZONE_A": {"current_people": "30", "net_flow": "+2", "inflow_rate": "4", "outflow_rate": "2"}, "ZONE_B": {"current_people": "10"}, "ZONE_C": {"current_people": "5"}}}, {"zone_counts": {"ZONE_A": "40", "ZONE_B": "10", "ZONE_C": "5"}})
    assert result["zone_risk_details"]["ZONE_A"]["current_people"] == 30.0
