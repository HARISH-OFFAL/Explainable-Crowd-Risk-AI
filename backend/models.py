from sqlalchemy import Integer, String, Text, Float
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    event_name: Mapped[str] = mapped_column(String(200), nullable=False)
    location: Mapped[str] = mapped_column(String(300), nullable=False)
    event_datetime: Mapped[str] = mapped_column(String(50), nullable=False)

    expected_crowd_size: Mapped[int] = mapped_column(Integer, nullable=False)
    venue_capacity: Mapped[int] = mapped_column(Integer, nullable=False)

    entry_gates: Mapped[int] = mapped_column(Integer, nullable=False)
    exit_gates: Mapped[int] = mapped_column(Integer, nullable=False)
    emergency_exits: Mapped[int] = mapped_column(Integer, nullable=False)

    event_duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False)


class EventDocument(Base):
    __tablename__ = "event_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    event_id: Mapped[int] = mapped_column(Integer, nullable=False)

    document_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    document_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="Pending"
    )

    remarks: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True
    )


class MonitoringSession(Base):
    """A lightweight Phase 2 run associated with one registered event."""

    __tablename__ = "monitoring_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    event_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(30), nullable=False)
    source_name: Mapped[str] = mapped_column(String(255), nullable=False)
    started_at: Mapped[str] = mapped_column(String(50), nullable=False)
    ended_at: Mapped[str | None] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="READY")
    # Compact final Phase 2 packet shared by Flow Intelligence and Digital Twin.
    final_snapshot_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class Phase2RiskRecord(Base):
    """Periodic two-second Phase 2 summaries; raw frames are never stored."""

    __tablename__ = "phase2_risk_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    event_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    monitoring_session_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    timestamp: Mapped[str] = mapped_column(String(50), nullable=False)

    zone_a_people: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    zone_b_people: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    zone_c_people: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    zone_a_risk_score: Mapped[float | None] = mapped_column(nullable=True)
    zone_a_risk_level: Mapped[str | None] = mapped_column(String(30), nullable=True)
    zone_b_risk_score: Mapped[float | None] = mapped_column(nullable=True)
    zone_b_risk_level: Mapped[str | None] = mapped_column(String(30), nullable=True)
    zone_c_risk_score: Mapped[float | None] = mapped_column(nullable=True)
    zone_c_risk_level: Mapped[str | None] = mapped_column(String(30), nullable=True)

    future_risk_score: Mapped[float | None] = mapped_column(nullable=True)
    future_risk_level: Mapped[str | None] = mapped_column(String(30), nullable=True)
    xai_summary: Mapped[str | None] = mapped_column(String(1000), nullable=True)


class FlowAnalysisSession(Base):
    """Persistent identity for one Phase 3A analysis of one monitoring run."""

    __tablename__ = "flow_analysis_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    event_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    monitoring_session_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True, index=True)
    source_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="QUEUED")
    cache_key: Mapped[str] = mapped_column(String(255), nullable=False)
    snapshot_timestamp_sec: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[str] = mapped_column(String(50), nullable=False)
    completed_at: Mapped[str | None] = mapped_column(String(50), nullable=True)


class TimeMachineSession(Base):
    """Dedicated Phase 3 future-state analysis session."""

    __tablename__ = "time_machine_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    event_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    video_path: Mapped[str] = mapped_column(String(500), nullable=False)
    video_name: Mapped[str] = mapped_column(String(255), nullable=False)
    fps: Mapped[float | None] = mapped_column(nullable=True)
    duration: Mapped[float | None] = mapped_column(nullable=True)
    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="PENDING")
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    created_at: Mapped[str] = mapped_column(String(50), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(50), nullable=False)


class TimeMachineSnapshot(Base):
    """Saved current/predicted zone state for one selected horizon."""

    __tablename__ = "time_machine_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    source_timestamp: Mapped[float] = mapped_column(nullable=False)
    horizon_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    total_people: Mapped[int] = mapped_column(Integer, nullable=False)
    zone_a_people: Mapped[int] = mapped_column(Integer, nullable=False)
    zone_b_people: Mapped[int] = mapped_column(Integer, nullable=False)
    zone_c_people: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_zone: Mapped[str | None] = mapped_column(String(30), nullable=True)
    risk_level: Mapped[str | None] = mapped_column(String(30), nullable=True)
    confidence: Mapped[float | None] = mapped_column(nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class ResponsePlan(Base):
    __tablename__ = "response_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    event_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    monitoring_session_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    flow_analysis_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    time_machine_session_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    prediction_horizon: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    risk_zone: Mapped[str] = mapped_column(String(30), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(30), nullable=False)
    response_score: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="DRAFT")
    plan_json: Mapped[str] = mapped_column(Text, nullable=False)
    approved_at: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[str] = mapped_column(String(50), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(50), nullable=False)


class VenueLayout(Base):
    __tablename__ = "venue_layouts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    event_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(50), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(50), nullable=False)


class InstabilitySnapshot(Base):
    __tablename__ = "instability_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    event_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    monitoring_session_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    flow_analysis_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    timestamp: Mapped[str] = mapped_column(String(50), nullable=False)
    overall_instability: Mapped[float] = mapped_column(Float, nullable=False)
    overall_stability: Mapped[float] = mapped_column(Float, nullable=False)
    zone_a_instability: Mapped[float] = mapped_column(Float, nullable=False)
    zone_b_instability: Mapped[float] = mapped_column(Float, nullable=False)
    zone_c_instability: Mapped[float] = mapped_column(Float, nullable=False)
    compression_score: Mapped[float] = mapped_column(Float, nullable=False)
    counter_flow_score: Mapped[float] = mapped_column(Float, nullable=False)
    stop_go_score: Mapped[float] = mapped_column(Float, nullable=False)
    direction_disorder_score: Mapped[float] = mapped_column(Float, nullable=False)
    speed_drop_score: Mapped[float] = mapped_column(Float, nullable=False)
    highest_instability_zone: Mapped[str] = mapped_column(String(30), nullable=False)
    propagation_from: Mapped[str | None] = mapped_column(String(30), nullable=True)
    propagation_to: Mapped[str | None] = mapped_column(String(30), nullable=True)
    propagation_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(50), nullable=False)
