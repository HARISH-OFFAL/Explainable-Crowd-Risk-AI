def calculate_pre_event_risk(
    expected_crowd_size: int,
    venue_capacity: int,
    entry_gates: int,
    exit_gates: int,
    emergency_exits: int,
    event_duration_minutes: int,
) -> dict:

    risk_score = 0
    reasons = []

    # 1. Crowd vs venue capacity
    crowd_ratio = expected_crowd_size / venue_capacity

    if crowd_ratio >= 1.0:
        risk_score += 4
        reasons.append("Expected crowd exceeds venue capacity")
    elif crowd_ratio >= 0.8:
        risk_score += 3
        reasons.append("Expected crowd is high compared to venue capacity")
    elif crowd_ratio >= 0.6:
        risk_score += 1

    # 2. Entry gates
    if entry_gates == 0:
        risk_score += 3
        reasons.append("No entry gates available")
    elif entry_gates == 1:
        risk_score += 2
        reasons.append("Only one entry gate is available")

    # 3. Exit gates
    if exit_gates == 0:
        risk_score += 4
        reasons.append("No exit gates available")
    elif exit_gates == 1:
        risk_score += 2
        reasons.append("Only one exit gate is available")

    # 4. Emergency exits
    if emergency_exits == 0:
        risk_score += 4
        reasons.append("No emergency exits available")
    elif emergency_exits == 1:
        risk_score += 2
        reasons.append("Only one emergency exit is available")

    # 5. Event duration
    if event_duration_minutes >= 360:
        risk_score += 2
        reasons.append("Event duration is long")
    elif event_duration_minutes >= 240:
        risk_score += 1

    # Final risk level
    if risk_score >= 8:
        risk_level = "HIGH"
    elif risk_score >= 4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "reasons": reasons,
    }