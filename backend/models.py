from sqlalchemy import Integer, String
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
