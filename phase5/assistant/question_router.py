import re

INTENTS = ("CURRENT_STATE", "ZONE_ANALYSIS", "FORECAST", "WHY_RISK", "FLOW", "INSTABILITY", "RESPONSE", "COMPARISON", "HISTORY", "STORYBOARD", "COMMUNICATION", "CONCEPT_EXPLANATION", "PROJECT_EXPLANATION", "GENERAL_CONVERSATION", "UNKNOWN")

def route_question(message, conversation=None):
    text = (message or "").strip().lower()
    history = conversation or []
    last = history[-1] if history else {}
    if any(word in text for word in ("tamil", "தமிழ்", "தமிழில்", "translate")) and last.get("intent"):
        return {"intent": last.get("intent", "CURRENT_STATE"), "zone": last.get("zone"), "horizon": last.get("horizon")}
    if re.search(r"\b(why|how come|reason|எதற்கு|ஏன்)\b", text) and len(text.split()) <= 5 and "zone" not in text:
        return {"intent": last.get("intent", "WHY_RISK"), "zone": last.get("zone"), "horizon": last.get("horizon")}
    zone = next((f"ZONE_{letter}" for letter in "ABC" if re.search(rf"\b(zone\s*)?{letter.lower()}\b", text)), None)
    horizon = next((int(value) for value in (60, 30, 15) if re.search(rf"\b\+?{value}\s*(sec|seconds|s)?\b", text)), None)
    if any(word in text for word in ("storyboard", "timeline", "between", "from ")) and re.search(r"\d", text): intent = "HISTORY"
    elif any(word in text for word in ("compare", "versus", " vs ", "difference between zone")): intent = "COMPARISON"
    elif any(word in text for word in ("forecast", "predict", "next", "happen", "future", "seconds", "30 sec", "15 sec", "60 sec")): intent = "FORECAST"
    elif any(word in text for word in ("how many", "count", "happening now", "current", "right now", "most crowded")): intent = "ZONE_ANALYSIS" if zone or "zone" in text or "crowded" in text else "CURRENT_STATE"
    elif any(word in text for word in ("recommend", "should", "security", "organizer", "do we", "action")): intent = "RESPONSE"
    elif any(word in text for word in ("compression", "density", "counter-flow", "counter flow", "instability", "stop-go", "motion disorder")): intent = "CONCEPT_EXPLANATION" if text.startswith(("what is", "explain", "difference", "meaning")) else "INSTABILITY"
    elif any(word in text for word in ("flow", "inflow", "outflow", "movement", "direction")): intent = "FLOW"
    elif any(word in text for word in ("communication", "message", "announce", "public address")): intent = "COMMUNICATION"
    elif any(word in text for word in ("crowdguard", "digital twin", "bytetrack", "spatio-temporal", "heatmap", "response commander", "time machine", "storyboard", "fastapi", "project")): intent = "PROJECT_EXPLANATION"
    elif any(word in text for word in ("is it safer", "getting better", "worse", "risk")): intent = "WHY_RISK" if zone else "INSTABILITY"
    else: intent = "GENERAL_CONVERSATION"
    return {"intent": intent, "zone": zone, "horizon": horizon}
