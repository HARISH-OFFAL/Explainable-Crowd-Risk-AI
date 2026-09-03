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
    recommendations = []

    # 1. Crowd vs venue capacity
    crowd_ratio = expected_crowd_size / venue_capacity

    if crowd_ratio >= 1.0:
        risk_score += 4
        reasons.append("Expected crowd exceeds venue capacity")
        recommendations.append(
            "Reduce expected crowd or use a larger venue"
        )

    elif crowd_ratio >= 0.8:
        risk_score += 3
        reasons.append("Expected crowd is high compared to venue capacity")
        recommendations.append(
            "Increase crowd monitoring and control at entry points"
        )

    elif crowd_ratio >= 0.6:
        risk_score += 1
        recommendations.append(
            "Monitor crowd density during peak entry periods"
        )

    # 2. Entry gates
    if entry_gates == 0:
        risk_score += 3
        reasons.append("No entry gates available")
        recommendations.append(
            "Provide controlled and clearly marked entry gates"
        )

    elif entry_gates == 1:
        risk_score += 2
        reasons.append("Only one entry gate is available")
        recommendations.append(
            "Consider opening additional entry gates to reduce congestion"
        )

    # 3. Exit gates
    if exit_gates == 0:
        risk_score += 4
        reasons.append("No exit gates available")
        recommendations.append(
            "Provide sufficient exit gates before allowing the event to proceed"
        )

    elif exit_gates == 1:
        risk_score += 2
        reasons.append("Only one exit gate is available")
        recommendations.append(
            "Add more exit gates to improve crowd dispersal"
        )

    # 4. Emergency exits
    if emergency_exits == 0:
        risk_score += 4
        reasons.append("No emergency exits available")
        recommendations.append(
            "Create and clearly mark emergency evacuation exits"
        )

    elif emergency_exits == 1:
        risk_score += 2
        reasons.append("Only one emergency exit is available")
        recommendations.append(
            "Provide additional emergency exits where possible"
        )

    # 5. Event duration
    if event_duration_minutes >= 360:
        risk_score += 2
        reasons.append("Event duration is long")
        recommendations.append(
            "Arrange medical support, security shifts, and regular crowd monitoring"
        )

    elif event_duration_minutes >= 240:
        risk_score += 1
        recommendations.append(
            "Plan periodic crowd checks and staff rotation"
        )

    # Final risk level
    if risk_score >= 8:
        risk_level = "HIGH"
        recommendations.append(
            "Increase security personnel and prepare emergency evacuation routes"
        )

    elif risk_score >= 4:
        risk_level = "MEDIUM"
        recommendations.append(
            "Strengthen monitoring and review crowd management arrangements"
        )

    else:
        risk_level = "LOW"
        recommendations.append(
            "Maintain standard safety monitoring and emergency readiness"
        )

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "reasons": reasons,
        "recommendations": recommendations,
    }