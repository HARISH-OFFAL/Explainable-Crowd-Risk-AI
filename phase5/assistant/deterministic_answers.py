from .question_router import route_question

ZONES = ("ZONE_A", "ZONE_B", "ZONE_C")
def _name(zone): return str(zone or "the selected zone").replace("ZONE_", "Zone ")
def _num(value): return "unavailable" if value is None else str(round(value, 2) if isinstance(value, float) else value)

CONCEPTS = {"compression": "Compression Proxy represents increasing local crowd concentration combined with converging movement and reduced spacing or mobility. It is a proxy signal, not a direct measurement of physical pressure.", "density": "Density describes how many people occupy an area. Compression Proxy combines concentration with movement convergence and reduced mobility, so the two are related but not identical.", "counter-flow": "Counter-flow means people are moving in opposing directions in the same area, which can make movement less orderly.", "counter flow": "Counter-flow means people are moving in opposing directions in the same area, which can make movement less orderly.", "instability": "Crowd instability is a combined signal of concentration, movement disorder, counter-flow, stop-go behaviour and speed changes. It is an analytics score, not a direct physical danger measurement."}

def answer_question(question, context, conversation=None, language="auto"):
    route = route_question(question, conversation); intent = route["intent"]; monitoring = context.get("monitoring", {}); zones = monitoring.get("zone_counts") or {}; forecasts = context.get("forecast") or {}; zone = route.get("zone") or monitoring.get("most_crowded_zone"); evidence = {"monitoring_session_id": monitoring.get("session_id")}; followups = ["What happens next?", "Why is this zone risky?", "What should security do?"]
    text = question.lower()
    if str(language).lower() == "auto" and ("tamil" in text or "தமிழ்" in text or "தமிழில்" in text):
        language = "ta"
    if intent == "CONCEPT_EXPLANATION":
        key = next((key for key in CONCEPTS if key in text), "instability"); answer = CONCEPTS[key]; used = ["CrowdGuard concept definitions"]
    elif intent in {"CURRENT_STATE", "ZONE_ANALYSIS"}:
        if intent == "ZONE_ANALYSIS" and zone and zone in zones: answer = f"{_name(zone)} currently has {_num(zones.get(zone))} people."
        else: answer = f"The current monitored crowd is {_num(monitoring.get('current_people'))} people. The most crowded available zone is {_name(monitoring.get('most_crowded_zone'))}."
        used = ["monitoring"]; evidence["current_people"] = monitoring.get("current_people"); evidence["zone_counts"] = zones
    elif intent == "COMPARISON":
        chosen = [item for item in ZONES if item in text] or ["ZONE_A", "ZONE_C"]; answer = f"{_name(chosen[0])} has {_num(zones.get(chosen[0]))} people and {_name(chosen[1])} has {_num(zones.get(chosen[1]))} people."; used = ["monitoring"]; evidence["comparison"] = {item: zones.get(item) for item in chosen}
    elif intent == "FORECAST":
        horizon = route.get("horizon") or 30; item = forecasts.get(str(horizon)); answer = f"The +{horizon} second forecast is {_name(item.get('risk_zone'))} at {item.get('risk_level', 'unavailable')} risk, with {_num(item.get('total_people'))} people." if item else "A forecast is not available for this monitoring session."; used = [f"{horizon}-second forecast"]; evidence["forecast"] = item
    elif intent in {"WHY_RISK", "INSTABILITY"}:
        radar = context.get("instability") or {}; item = forecasts.get("30") or {}; target = zone or radar.get("highest_zone") or item.get("risk_zone"); metrics = (radar.get("zone_metrics") or {}).get(target, {}); answer = f"{_name(target)} requires attention based on the available analytics. It has {_num(zones.get(target))} current people" + (f", a +30 second forecast of {_num(item.get('zone_counts', {}).get(target))} people" if item.get("zone_counts") else "") + (f", and an instability score of {_num(metrics.get('instability_score'))}." if metrics.get("instability_score") is not None else "."); used = ["monitoring", "forecast"] + (["instability radar"] if radar else []); evidence.update({"zone": target, "zone_metrics": metrics, "forecast_30": item})
    elif intent == "FLOW":
        flow = context.get("flow") or {}; answer = f"Flow Intelligence reports {flow.get('state', 'unavailable')} movement, dominant direction {flow.get('direction', 'unavailable')}, and counter-flow={flow.get('counter_flow', 'unavailable')}."; used = ["flow analysis"]; evidence["flow"] = flow
    elif intent == "RESPONSE":
        response = context.get("response") or {}; action = response.get("recommended_action"); answer = f"The authoritative Response Commander recommendation is {_name(action)}." if action else "A Response Commander recommendation is not available for this monitoring session."; used = ["response commander"] if action else []; evidence["response"] = response
    elif intent == "HISTORY":
        answer = "The available detailed timeline is not sufficient to make a reliable claim for that interval."; used = ["available monitoring history"]
    elif intent == "PROJECT_EXPLANATION":
        answer = "CrowdGuard combines person detection, Flow Intelligence, Instability Radar, Crowd Time Machine forecasts and Response Commander recommendations into an explainable safety workflow."; used = ["CrowdGuard architecture"]
    else:
        answer = "That is outside the available CrowdGuard project context. Ask me about the current crowd, zones, forecasts, flow, instability or response recommendations."; used = []
    if str(language).lower() in {"ta", "tamil"}:
        answer = "CrowdGuard தரவின்படி: " + answer
    return {"answer": answer, "intent": intent, "language": "ta" if str(language).lower() in {"ta", "tamil"} else "en", "mode": "PROJECT_DATA" if used and intent not in {"CONCEPT_EXPLANATION", "PROJECT_EXPLANATION"} else "PROJECT_EXPLANATION", "evidence": evidence, "used_sources": used, "suggested_followups": followups, "route": route}
