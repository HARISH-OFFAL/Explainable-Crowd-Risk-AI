import os
import shutil
import json
import threading
import re
import smtplib
from email.message import EmailMessage
from email.utils import formatdate
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path
import cv2
import bcrypt

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from .database import Base, SessionLocal, engine, get_db
from .models import Event, User, EventDocument, MonitoringSession, Phase2RiskRecord, FlowAnalysisSession, TimeMachineSession, TimeMachineSnapshot, ResponsePlan, VenueLayout, InstabilitySnapshot, StoryboardSession, CommunicationPlan, CommunicationMessage, CommunicationObservation, DashboardSafetyAlert
from .schemas import (
    EventCreate,
    EventUpdate,
    EventDocumentCreate,
    DocumentStatusUpdate,
    MonitoringSessionCreate,
    Phase2RiskUpdate,
)
from .risk_engine import calculate_pre_event_risk
from .phase2_web_engine import Phase2WebEngine, normalized_crowd_scale, warm_detector
from phase3.flow_service import FLOW_CONFIG, analyze_recorded_video, load_cached_flow, save_cached_flow
from phase3.digital_twin.state_builder import build_current_state
from phase3.digital_twin.scenario_service import build_scenarios
from phase3.digital_twin.simulation_engine import CONFIG, simulate
from phase3.time_machine_service import analyze_time_machine, load_cached_time_machine, save_cached_time_machine, HORIZONS
from phase4.response_commander import analyze_state, build_plan, simulate_plan, communications, venue_graph, impact_analysis, command_feed
from phase4.schemas import VenueLayout as VenueLayoutSchema
from phase4.instability_radar import analyze as analyze_instability_radar
from phase5.explainability_engine import build_explanation
from phase5.assistant.context_builder import build_context
from phase5.assistant.conversation_service import chat as crowdguard_chat
from phase6.storyboard_service import generate_storyboard, ALGORITHM_VERSION as STORYBOARD_ALGORITHM_VERSION
from .communication_engine import build_context as build_communication_context, build_plan as build_communication_plan
from .safety_brief_pdf import build_safety_brief_pdf, safety_brief_filename


Base.metadata.create_all(bind=engine)
# Existing SQLite databases are intentionally preserved. Add only the new
# nullable snapshot column when upgrading an older checkout.
with engine.begin() as connection:
    columns = {row[1] for row in connection.execute(text("PRAGMA table_info(monitoring_sessions)"))}
    if "final_snapshot_json" not in columns:
        connection.execute(text("ALTER TABLE monitoring_sessions ADD COLUMN final_snapshot_json TEXT"))
    response_columns = {row[1] for row in connection.execute(text("PRAGMA table_info(response_plans)"))}
    if response_columns and "prediction_horizon" not in response_columns:
        connection.execute(text("ALTER TABLE response_plans ADD COLUMN prediction_horizon INTEGER NOT NULL DEFAULT 30"))
    phase2_columns = {row[1] for row in connection.execute(text("PRAGMA table_info(phase2_risk_records)"))}
    if phase2_columns and "source_timestamp_sec" not in phase2_columns:
        connection.execute(text("ALTER TABLE phase2_risk_records ADD COLUMN source_timestamp_sec FLOAT"))


app = FastAPI(
    title="Explainable Crowd Risk AI",
    description="Crowd safety and early risk prediction system",
    version="0.1.0",
)


@app.on_event("startup")
def warm_crowd_detector():
    """Load and warm YOLO before the first monitoring click."""
    warm_detector()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE2_VIDEOS = PROJECT_ROOT / "phase2" / "videos"
WEB_MONITOR_ENGINES = {}
WEB_MONITOR_PERSISTED_WINDOWS = {}
EMAIL_ALERT_LOCKS = {}
PHASE3_ANALYSES = {}
TIME_MACHINE_CACHE = {}
ZONES = ("ZONE_A", "ZONE_B", "ZONE_C")


def _safe_user(user):
    return {"id": user.id, "full_name": user.full_name, "email": user.email, "role": "EVENT_ORGANIZER" if user.role == "ORGANIZER" else user.role, "organization": user.organization}


@app.post("/auth/register")
def register_user(payload: dict, db: Session = Depends(get_db)):
    full_name = str(payload.get("full_name") or "").strip()
    email = str(payload.get("email") or "").strip().lower()
    password = str(payload.get("password") or "")
    role = str(payload.get("role") or "ORGANIZER").strip().upper()
    organization = str(payload.get("organization") or "").strip() or None
    if not full_name:
        raise HTTPException(status_code=422, detail="Please enter your full name.")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    if len(password) < 8 or not re.search(r"[A-Z]", password) or not re.search(r"[a-z]", password) or not re.search(r"\d", password):
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters with uppercase, lowercase, and a number.")
    if role != "ORGANIZER":
        raise HTTPException(status_code=403, detail="Authority accounts are provisioned by the system administrator.")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    now = utc_now()
    user = User(full_name=full_name, email=email, password_hash=bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(), role=role, organization=organization, is_active=True, created_at=now)
    db.add(user); db.commit(); db.refresh(user)
    return {"message": "Account created successfully", "user": _safe_user(user)}


@app.post("/auth/login")
def login_user(payload: dict, db: Session = Depends(get_db)):
    email = str(payload.get("email") or "").strip().lower()
    password = str(payload.get("password") or "")
    requested_role = str(payload.get("role") or "").strip().upper()
    user = db.query(User).filter(User.email == email).first()
    if user and user.is_active and bcrypt.checkpw(password.encode(), user.password_hash.encode()):
        if requested_role in {"EVENT_ORGANIZER", "ORGANIZER"} and user.role != "ORGANIZER":
            raise HTTPException(status_code=401, detail="Account exists, but it is not registered as an Event Organizer account.")
        if requested_role == "AUTHORITY" and user.role != "AUTHORITY":
            raise HTTPException(status_code=401, detail="Account exists, but it is not registered as an Authority account.")
        return {"message": "Login successful", "user": _safe_user(user), "token": str(uuid4())}
    raise HTTPException(status_code=401, detail="Invalid email or password.")


def save_phase2_snapshot(session_id: int, analytics: dict, db: Session):
    """Save one compact, detector-derived packet for downstream Phase 3 pages."""
    if not analytics or not analytics.get("processed_frames"):
        return
    session = db.query(MonitoringSession).filter(MonitoringSession.id == session_id).first()
    if session is None:
        return
    zones = analytics.get("zone_counts") or {}
    packet = {
        "session_id": session_id,
        "event": analytics.get("event") or {"event_name": session.source_name},
        "source": analytics.get("source"),
        "source_frame_index": analytics.get("source_frame_index"),
        "source_timestamp_sec": analytics.get("source_timestamp_sec"),
        "total_people": int(analytics.get("people_count", analytics.get("detected_people_count", 0))),
        "detected_people_count": int(analytics.get("detected_people_count", analytics.get("accepted_person_detections", 0))),
        "zone_counts": {zone: int(zones.get(zone, 0)) for zone in ("ZONE_A", "ZONE_B", "ZONE_C")},
        "crowd_scale_counts": {zone: int(zones.get(zone, 0)) for zone in ("ZONE_A", "ZONE_B", "ZONE_C")},
        "raw_zone_counts": {zone: int((analytics.get("raw_zone_counts") or {}).get(zone, 0)) for zone in ("ZONE_A", "ZONE_B", "ZONE_C")},
        "density_state": analytics.get("density_state"),
        "density_value": analytics.get("density_value"),
        "zone_crowd_ranking": analytics.get("zone_crowd_ranking") or {},
        "zone_crowd_levels": analytics.get("zone_crowd_levels") or {},
        "most_crowded_zone": analytics.get("most_crowded_zone"),
    }
    session.final_snapshot_json = json.dumps(packet)
    source_frame_index = analytics.get("source_frame_index")
    latest = db.query(Phase2RiskRecord).filter(Phase2RiskRecord.monitoring_session_id == session_id).order_by(Phase2RiskRecord.id.desc()).first()
    latest_meta = {}
    if latest and latest.xai_summary:
        try:
            latest_meta = json.loads(latest.xai_summary)
        except (TypeError, ValueError):
            latest_meta = {}
    if not latest or latest_meta.get("source_frame_index") != source_frame_index:
        zone_levels = analytics.get("zone_crowd_levels") or {}
        db.add(Phase2RiskRecord(
            event_id=session.event_id,
            monitoring_session_id=session_id,
            timestamp=utc_now(),
            source_timestamp_sec=analytics.get("source_timestamp_sec"),
            zone_a_people=int(zones.get("ZONE_A", 0)),
            zone_b_people=int(zones.get("ZONE_B", 0)),
            zone_c_people=int(zones.get("ZONE_C", 0)),
            zone_a_risk_level=zone_levels.get("ZONE_A"),
            zone_b_risk_level=zone_levels.get("ZONE_B"),
            zone_c_risk_level=zone_levels.get("ZONE_C"),
            xai_summary=json.dumps({
                "source_frame_index": source_frame_index,
                "density_state": analytics.get("density_state"),
                "density_value": analytics.get("density_value"),
                "people_count": analytics.get("people_count"),
            }),
        ))
    db.commit()


def load_phase2_snapshot(session):
    if not session.final_snapshot_json:
        return None
    try:
        packet = json.loads(session.final_snapshot_json)
    except (TypeError, ValueError):
        return None
    if packet.get("session_id") != session.id:
        return None
    # Upgrade snapshots written before the 100+ operating-scale switch while
    # preserving their detector-derived zone proportions.
    if "crowd_scale_counts" not in packet:
        packet["crowd_scale_counts"] = normalized_crowd_scale(packet.get("zone_counts") or {})
        packet["zone_counts"] = packet["crowd_scale_counts"]
        packet["total_people"] = sum(packet["crowd_scale_counts"].values())
    return packet


def apply_phase2_snapshot(flow_result, snapshot):
    """Keep Phase 3 movement analytics, but make occupancy session-consistent."""
    if not snapshot:
        return flow_result
    result = dict(flow_result)
    counts = snapshot.get("zone_counts") or {}
    result["session_id"] = snapshot["session_id"]
    result["snapshot_people"] = snapshot.get("total_people", sum(counts.values()))
    result["phase2_snapshot"] = snapshot
    provenance = dict(result.get("snapshot_provenance") or {})
    provenance.update({"session_id": snapshot["session_id"], "source_frame_index": snapshot.get("source_frame_index"), "source_timestamp_sec": snapshot.get("source_timestamp_sec"), "source": "Phase 2 final detector snapshot"})
    result["snapshot_provenance"] = provenance
    for zone in ("ZONE_A", "ZONE_B", "ZONE_C"):
        result.setdefault("zones", {}).setdefault(zone, {})["people"] = int(counts.get(zone, 0))
    result["total_people"] = int(snapshot.get("total_people", sum(counts.values())))
    result["density_state"] = snapshot.get("density_state")
    return result


# ---------------------------------------------------------
# CORS CONFIGURATION
# Allows React frontend to communicate with FastAPI backend
# ---------------------------------------------------------

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Explainable Crowd Risk AI API is running"
    }


# =========================================================
# EVENT ENDPOINTS
# =========================================================

@app.post("/events")
def create_event(
    event: EventCreate,
    db: Session = Depends(get_db)
):
    risk_result = calculate_pre_event_risk(
        expected_crowd_size=event.expected_crowd_size,
        venue_capacity=event.venue_capacity,
        entry_gates=event.entry_gates,
        exit_gates=event.exit_gates,
        emergency_exits=event.emergency_exits,
        event_duration_minutes=event.event_duration_minutes,
    )

    new_event = Event(
        event_name=event.event_name,
        location=event.location,
        event_datetime=event.event_datetime,
        expected_crowd_size=event.expected_crowd_size,
        venue_capacity=event.venue_capacity,
        entry_gates=event.entry_gates,
        exit_gates=event.exit_gates,
        emergency_exits=event.emergency_exits,
        event_duration_minutes=event.event_duration_minutes,
    )

    db.add(new_event)
    db.commit()
    db.refresh(new_event)

    return {
        "message": "Event registered successfully",
        "event_id": new_event.id,
        "event_name": new_event.event_name,
        "pre_event_risk": {
            "risk_level": risk_result["risk_level"],
            "risk_score": risk_result["risk_score"],
            "reasons": risk_result["reasons"],
            "recommendations": risk_result["recommendations"],
        },
    }


@app.get("/events")
def get_events(
    db: Session = Depends(get_db)
):
    events = db.query(Event).all()

    return [
        {
            "id": event.id,
            "event_name": event.event_name,
            "location": event.location,
            "event_datetime": event.event_datetime,
            "expected_crowd_size": event.expected_crowd_size,
            "venue_capacity": event.venue_capacity,
            "entry_gates": event.entry_gates,
            "exit_gates": event.exit_gates,
            "emergency_exits": event.emergency_exits,
            "event_duration_minutes": event.event_duration_minutes,
        }
        for event in events
    ]


@app.get("/events/{event_id}")
def get_event(
    event_id: int,
    db: Session = Depends(get_db)
):
    event = (
        db.query(Event)
        .filter(Event.id == event_id)
        .first()
    )

    if event is None:
        raise HTTPException(
            status_code=404,
            detail="Event not found"
        )

    risk_result = calculate_pre_event_risk(
        expected_crowd_size=event.expected_crowd_size,
        venue_capacity=event.venue_capacity,
        entry_gates=event.entry_gates,
        exit_gates=event.exit_gates,
        emergency_exits=event.emergency_exits,
        event_duration_minutes=event.event_duration_minutes,
    )

    return {
        "id": event.id,
        "event_name": event.event_name,
        "location": event.location,
        "event_datetime": event.event_datetime,
        "expected_crowd_size": event.expected_crowd_size,
        "venue_capacity": event.venue_capacity,
        "entry_gates": event.entry_gates,
        "exit_gates": event.exit_gates,
        "emergency_exits": event.emergency_exits,
        "event_duration_minutes": event.event_duration_minutes,
        "pre_event_risk": {
            "risk_level": risk_result["risk_level"],
            "risk_score": risk_result["risk_score"],
            "reasons": risk_result["reasons"],
            "recommendations": risk_result["recommendations"],
        },
    }


@app.put("/events/{event_id}")
def update_event(
    event_id: int,
    updated_event: EventUpdate,
    db: Session = Depends(get_db)
):
    event = (
        db.query(Event)
        .filter(Event.id == event_id)
        .first()
    )

    if event is None:
        raise HTTPException(
            status_code=404,
            detail="Event not found"
        )

    event.event_name = updated_event.event_name
    event.location = updated_event.location
    event.event_datetime = updated_event.event_datetime

    event.expected_crowd_size = (
        updated_event.expected_crowd_size
    )

    event.venue_capacity = (
        updated_event.venue_capacity
    )

    event.entry_gates = updated_event.entry_gates
    event.exit_gates = updated_event.exit_gates
    event.emergency_exits = updated_event.emergency_exits

    event.event_duration_minutes = (
        updated_event.event_duration_minutes
    )

    db.commit()
    db.refresh(event)

    risk_result = calculate_pre_event_risk(
        expected_crowd_size=event.expected_crowd_size,
        venue_capacity=event.venue_capacity,
        entry_gates=event.entry_gates,
        exit_gates=event.exit_gates,
        emergency_exits=event.emergency_exits,
        event_duration_minutes=event.event_duration_minutes,
    )

    return {
        "message": "Event updated successfully",
        "event_id": event.id,
        "event_name": event.event_name,
        "pre_event_risk": {
            "risk_level": risk_result["risk_level"],
            "risk_score": risk_result["risk_score"],
            "reasons": risk_result["reasons"],
            "recommendations": risk_result["recommendations"],
        },
    }


@app.delete("/events/{event_id}")
def delete_event(
    event_id: int,
    db: Session = Depends(get_db)
):
    event = (
        db.query(Event)
        .filter(Event.id == event_id)
        .first()
    )

    if event is None:
        raise HTTPException(
            status_code=404,
            detail="Event not found"
        )

    db.delete(event)
    db.commit()

    return {
        "message": "Event deleted successfully",
        "event_id": event_id
    }


# =========================================================
# PHASE 2 MONITORING INTEGRATION
# =========================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def serialize_session(session):
    return {
        "id": session.id,
        "event_id": session.event_id,
        "source_type": session.source_type,
        "source_name": session.source_name,
        "started_at": session.started_at,
        "ended_at": session.ended_at,
        "status": session.status,
    }


@app.post("/events/{event_id}/monitoring-sessions")
def create_monitoring_session(
    event_id: int,
    session_data: MonitoringSessionCreate,
    db: Session = Depends(get_db),
):
    event = db.query(Event).filter(Event.id == event_id).first()
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")

    source_type = session_data.source_type.lower().strip()
    if source_type not in {"recorded", "live"}:
        raise HTTPException(status_code=400, detail="source_type must be recorded or live")

    session = MonitoringSession(
        event_id=event_id,
        source_type=source_type,
        source_name=session_data.source_name.strip(),
        started_at=utc_now(),
        status="READY",
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return {
        "message": "Monitoring session prepared",
        "event": {
            "id": event.id,
            "event_name": event.event_name,
            "location": event.location,
        },
        "session": serialize_session(session),
    }


@app.get("/events/{event_id}/monitoring-sessions")
def get_monitoring_sessions(event_id: int, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    sessions = (
        db.query(MonitoringSession)
        .filter(MonitoringSession.event_id == event_id)
        .order_by(MonitoringSession.id.desc())
        .all()
    )
    return {"event_id": event_id, "sessions": [serialize_session(item) for item in sessions]}


@app.post("/monitoring-sessions/{session_id}/risk-update")
def create_phase2_risk_update(
    session_id: int,
    update: Phase2RiskUpdate,
    db: Session = Depends(get_db),
):
    session = db.query(MonitoringSession).filter(MonitoringSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Monitoring session not found")
    if session.status == "STOPPED":
        raise HTTPException(status_code=409, detail="Monitoring session is stopped")

    record = Phase2RiskRecord(
        event_id=session.event_id,
        monitoring_session_id=session.id,
        timestamp=update.timestamp or utc_now(),
        zone_a_people=update.zone_a_people,
        zone_b_people=update.zone_b_people,
        zone_c_people=update.zone_c_people,
        zone_a_risk_score=update.zone_a_risk_score,
        zone_a_risk_level=update.zone_a_risk_level,
        zone_b_risk_score=update.zone_b_risk_score,
        zone_b_risk_level=update.zone_b_risk_level,
        zone_c_risk_score=update.zone_c_risk_score,
        zone_c_risk_level=update.zone_c_risk_level,
        future_risk_score=update.future_risk_score,
        future_risk_level=update.future_risk_level,
        xai_summary=update.xai_summary,
    )
    session.status = "RUNNING"
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"message": "Phase 2 risk update stored", "event_id": session.event_id, "session_id": session.id, "record_id": record.id}


@app.post("/monitoring-sessions/{session_id}/start")
def start_monitoring_session(session_id: int, db: Session = Depends(get_db)):
    """Compatibility alias: start the browser monitoring engine, never a desktop popup."""
    return start_web_monitoring_session(session_id, db)


def start_web_monitoring_session(session_id: int, db: Session):
    session = db.query(MonitoringSession).filter(MonitoringSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Monitoring session not found")
    engine = WEB_MONITOR_ENGINES.get(session_id)
    if engine is not None and engine.running:
        return {"message": "Browser monitoring is already running", "session": serialize_session(session)}
    event = db.query(Event).filter(Event.id == session.event_id).first()
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    risk = calculate_pre_event_risk(event.expected_crowd_size, event.venue_capacity, event.entry_gates, event.exit_gates, event.emergency_exits, event.event_duration_minutes)
    context = {
        "id": event.id,
        "event_name": event.event_name,
        "location": event.location,
        "event_datetime": event.event_datetime,
        "expected_crowd_size": event.expected_crowd_size,
        "venue_capacity": event.venue_capacity,
        "pre_event_risk": {"risk_level": risk["risk_level"], "risk_score": risk["risk_score"]},
    }
    try:
        engine = Phase2WebEngine(context, Path(session.source_name).name, session.source_type)
        engine.start()
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    WEB_MONITOR_ENGINES[session_id] = engine
    session.status = "RUNNING"
    db.commit()
    db.refresh(session)
    return {"message": "Browser Phase 2 monitoring started", "session": serialize_session(session)}


def persist_web_window(session_id: int, db: Session, analytics: dict):
    """Persist each completed two-second web inference window once."""
    sequence = int(analytics.get("window_sequence") or 0)
    if sequence <= WEB_MONITOR_PERSISTED_WINDOWS.get(session_id, 0):
        return
    current = analytics.get("current_risk") or {}
    zones = analytics.get("zones") or {}
    if current.get("risk_score") is None:
        return
    session = db.query(MonitoringSession).filter(MonitoringSession.id == session_id).first()
    if session is None:
        return
    record = Phase2RiskRecord(
        event_id=session.event_id,
        monitoring_session_id=session.id,
        timestamp=utc_now(),
        source_timestamp_sec=analytics.get("source_timestamp_sec"),
        zone_a_people=(zones.get("ZONE_A") or {}).get("people", 0),
        zone_b_people=(zones.get("ZONE_B") or {}).get("people", 0),
        zone_c_people=(zones.get("ZONE_C") or {}).get("people", 0),
        zone_a_risk_score=(zones.get("ZONE_A") or {}).get("risk_score"),
        zone_a_risk_level=(zones.get("ZONE_A") or {}).get("level"),
        zone_b_risk_score=(zones.get("ZONE_B") or {}).get("risk_score"),
        zone_b_risk_level=(zones.get("ZONE_B") or {}).get("level"),
        zone_c_risk_score=(zones.get("ZONE_C") or {}).get("risk_score"),
        zone_c_risk_level=(zones.get("ZONE_C") or {}).get("level"),
        future_risk_score=(analytics.get("forecast") or {}).get("probability"),
        future_risk_level=(analytics.get("forecast") or {}).get("status"),
        xai_summary=json.dumps(current.get("explanation") or []),
    )
    db.add(record)
    db.commit()
    WEB_MONITOR_PERSISTED_WINDOWS[session_id] = sequence


@app.post("/monitoring-sessions/{session_id}/web-start")
def web_start_monitoring_session(session_id: int, db: Session = Depends(get_db)):
    return start_web_monitoring_session(session_id, db)


@app.get("/monitoring-sessions/{session_id}/stream.mjpg")
def stream_monitoring_video(session_id: int):
    engine = WEB_MONITOR_ENGINES.get(session_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="Browser monitoring engine is not running")
    return StreamingResponse(engine.stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/monitoring-videos/{video_name}")
def get_monitoring_video(video_name: str):
    """Serve the original recording for smooth browser-native playback.

    AI inference runs independently in Phase2WebEngine; this endpoint must never
    proxy the annotated MJPEG stream used for diagnostics.
    """
    safe_name = Path(video_name).name
    video_path = PHASE2_VIDEOS / safe_name
    if video_path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"} or not video_path.is_file():
        raise HTTPException(status_code=404, detail="Recorded video not found")
    return FileResponse(video_path, media_type="video/mp4", filename=safe_name)


@app.post("/monitoring-videos/upload")
def upload_monitoring_video(file: UploadFile = File(...)):
    """Store a user recording in the same directory and pipeline as demos."""
    allowed = {".mp4", ".avi", ".mov", ".mkv"}
    original_name = Path(file.filename or "").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported video format")
    PHASE2_VIDEOS.mkdir(parents=True, exist_ok=True)
    stored_name = f"upload_{uuid4().hex}{suffix}"
    destination = PHASE2_VIDEOS / stored_name
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"source_name": stored_name, "original_name": original_name}


@app.get("/monitoring-sessions/{session_id}/analytics")
def get_live_monitoring_analytics(session_id: int, db: Session = Depends(get_db)):
    engine = WEB_MONITOR_ENGINES.get(session_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="Browser monitoring engine is not running")
    analytics = engine.analytics()
    save_phase2_snapshot(session_id, analytics, db)
    persist_web_window(session_id, db, analytics)
    return analytics


@app.post("/monitoring-sessions/{session_id}/camera-frame")
def process_camera_frame(session_id: int, payload: dict, db: Session = Depends(get_db)):
    engine = WEB_MONITOR_ENGINES.get(session_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="Browser monitoring engine is not running")
    image_base64 = payload.get("image_base64")
    source_timestamp_sec = payload.get("source_timestamp_sec")
    if not image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")
    try:
        analytics = engine.process_live_base64(image_base64, source_timestamp_sec)
        save_phase2_snapshot(session_id, analytics, db)
        persist_web_window(session_id, db, analytics)
        return analytics
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/monitoring-sessions/{session_id}/email-alert")
def send_monitoring_email_alert(
    session_id: int,
    payload: dict,
    db: Session = Depends(get_db),
):
    recipient = str(payload.get("recipient_email") or "").strip()
    operator_note = str(payload.get("operator_note") or "").strip()
    attach_snapshot = bool(payload.get("attach_snapshot", True))

    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", recipient):
        raise HTTPException(status_code=400, detail="Invalid recipient email")

    if len(operator_note) > 2000:
        raise HTTPException(status_code=400, detail="Operator note is too long")

    lock = EMAIL_ALERT_LOCKS.setdefault(session_id, threading.Lock())
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Alert is already being sent")

    try:
        session = (
            db.query(MonitoringSession)
            .filter(MonitoringSession.id == session_id)
            .first()
        )

        if session is None:
            raise HTTPException(status_code=404, detail="Monitoring session not found")

        if not session.final_snapshot_json:
            raise HTTPException(status_code=409, detail="No monitoring snapshot is available")

        try:
            snapshot = json.loads(session.final_snapshot_json)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail="Monitoring snapshot is invalid",
            ) from exc

        smtp_host = os.getenv("SMTP_HOST", "").strip()
        smtp_port = os.getenv("SMTP_PORT", "").strip()
        smtp_username = os.getenv("SMTP_USERNAME", "").strip()
        smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
        smtp_from_email = os.getenv("SMTP_FROM_EMAIL", "").strip()
        smtp_from_name = os.getenv(
            "SMTP_FROM_NAME",
            "CrowdGuard Safety Intelligence",
        ).strip()

        missing = []
        if not smtp_host:
            missing.append("SMTP_HOST")
        if not smtp_port:
            missing.append("SMTP_PORT")
        if not smtp_username:
            missing.append("SMTP_USERNAME")
        if not smtp_password:
            missing.append("SMTP_PASSWORD")
        if not smtp_from_email:
            missing.append("SMTP_FROM_EMAIL")

        if missing:
            print("EMAIL ALERT CONFIG ERROR: Missing:", ", ".join(missing))
            raise HTTPException(
                status_code=500,
                detail="Email service is not configured correctly",
            )

        try:
            smtp_port_int = int(smtp_port)
        except ValueError as exc:
            raise HTTPException(
                status_code=500,
                detail="Email service port configuration is invalid",
            ) from exc

        source = snapshot.get("source") or {}
        zones = snapshot.get("zone_counts") or {}

        zone_a = int(zones.get("ZONE_A", 0) or 0)
        zone_b = int(zones.get("ZONE_B", 0) or 0)
        zone_c = int(zones.get("ZONE_C", 0) or 0)

        current_people = int(
            snapshot.get("total_people", zone_a + zone_b + zone_c) or 0
        )

        event_data = snapshot.get("event") or {}
        event = db.query(Event).filter(Event.id == session.event_id).first()
        event_name = (
            event_data.get("event_name")
            or session.source_name
            or "Unknown Event"
        )

        source_name = (
            source.get("name")
            or session.source_name
            or "Unknown Source"
        )

        video_timestamp = float(snapshot.get("source_timestamp_sec", 0) or 0)

        zone_counts_for_max = {
            "ZONE_A": zone_a,
            "ZONE_B": zone_b,
            "ZONE_C": zone_c,
        }

        highest_zone = (
            snapshot.get("most_crowded_zone")
            or max(zone_counts_for_max, key=zone_counts_for_max.get)
        )

        density_state = snapshot.get("density_state") or "Not available"
        density_value = snapshot.get("density_value")

        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        body = "\n".join([
            "CrowdGuard Safety Intelligence",
            "",
            "MONITORING ALERT",
            "",
            f"Event: {event_name}",
            f"Monitoring session: #{session.id}",
            f"Source: {source_name}",
            f"Video timestamp: {video_timestamp:.1f} sec",
            "",
            "CURRENT CROWD STATE",
            "",
            f"Current people: {current_people}",
            f"Zone A: {zone_a}",
            f"Zone B: {zone_b}",
            f"Zone C: {zone_c}",
            f"Highest crowd zone: {highest_zone}",
            f"Density status: {density_state}",
            f"Density value: {density_value if density_value is not None else 'Not available'}",
            "",
            f"Operator note: {operator_note or 'None'}",
            "",
            f"Generated: {generated}",
            "",
            (
                "This alert was generated from the latest synchronized "
                "CrowdGuard monitoring analytics snapshot."
            ),
        ])

        if payload.get("communication_message"):
            body = "\n".join(["CrowdGuard Safety Communication", "", f"Event: {event_name}", f"Monitoring session: #{session.id}", f"Video timestamp: {video_timestamp:.1f} sec", f"Priority: {payload.get('communication_priority') or 'ADVISORY'}", "", "APPROVED MESSAGE", "", str(payload["communication_message"]), "", f"Generated: {generated}"])
        message = EmailMessage()
        message["From"] = f"{smtp_from_name} <{smtp_from_email}>"
        message["To"] = recipient
        message["Subject"] = payload.get("communication_subject") or f"CrowdGuard Monitoring Alert - {event_name} - Session #{session.id}"
        message["Date"] = formatdate(localtime=False)
        message.set_content(body)

        if attach_snapshot:
            communication_plan_id = payload.get("communication_plan_id")
            plan = None
            approved_message = payload.get("communication_message")
            audience = payload.get("communication_audience") or "Not available"
            language = payload.get("communication_language") or "en"
            if communication_plan_id is not None:
                plan = db.query(CommunicationPlan).filter(
                    CommunicationPlan.id == int(communication_plan_id),
                    CommunicationPlan.event_id == session.event_id,
                    CommunicationPlan.monitoring_session_id == session.id,
                ).first()
                approved_plan_state = plan is not None and (
                    plan.status in {"APPROVED", "SENT"}
                    or (plan.status == "SENDING" and plan.approved_at)
                )
                if not approved_plan_state:
                    raise HTTPException(status_code=409, detail="An approved communication plan is required for the safety brief")
                selected = db.query(CommunicationMessage).filter(
                    CommunicationMessage.communication_plan_id == plan.id,
                    CommunicationMessage.audience == audience,
                    CommunicationMessage.language == language,
                ).first()
                if selected is None or not (selected.approved_message or selected.status in {"APPROVED", "SENT"}):
                    raise HTTPException(status_code=409, detail="The selected approved communication is not available")
                approved_message = selected.approved_message or selected.message
                plan_data = communication_payload(plan, db)
                brief_filename = safety_brief_filename(session.event_id, plan.id)
            else:
                plan_data = {"priority": payload.get("communication_priority"), "intent": "MONITOR_ONLY", "status": "NOT APPLICABLE", "source_state": {**snapshot, "monitoring_session_id": session.id, "current_people": current_people, "target_zone": highest_zone, "density": density_state, "timestamp_sec": video_timestamp}}
                brief_filename = safety_brief_filename(session.event_id, "NA")
            pdf_bytes = build_safety_brief_pdf(event, plan_data, approved_message, audience=audience, language=language, generated_at=datetime.now(timezone.utc).isoformat())
            message.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=brief_filename)

        print("\n========== EMAIL ALERT ==========")
        print("Session:", session.id)
        print("Recipient:", recipient)
        print("SMTP host:", smtp_host)
        print("SMTP port:", smtp_port_int)
        print("SMTP username configured:", bool(smtp_username))
        print("SMTP password configured:", bool(smtp_password))
        print("Snapshot attached:", attach_snapshot)
        print("=================================\n")

        print("SMTP: connecting...")

        with smtplib.SMTP(
            smtp_host,
            smtp_port_int,
            timeout=20,
        ) as smtp:
            smtp.ehlo()

            print("SMTP: connected")
            print("SMTP: starting TLS...")

            smtp.starttls()
            smtp.ehlo()

            print("SMTP: TLS success")
            print("SMTP: login starting...")

            smtp.login(
                smtp_username,
                smtp_password,
            )

            print("SMTP: login success")

            smtp.send_message(message)

            print("SMTP: message sent successfully")

        return {
            "message": f"Alert sent successfully to {recipient}",
            "recipient": recipient,
            "session_id": session.id,
        }

    except HTTPException:
        raise

    except smtplib.SMTPAuthenticationError as exc:
        print("\n========== EMAIL ALERT ERROR ==========")
        print("ERROR TYPE: SMTPAuthenticationError")
        print("ERROR MESSAGE:", str(exc))
        print("Possible cause: invalid/revoked Gmail App Password")
        print("=======================================\n")

        raise HTTPException(
            status_code=500,
            detail="Unable to authenticate with email service",
        ) from exc

    except smtplib.SMTPException as exc:
        print("\n========== EMAIL ALERT ERROR ==========")
        print("ERROR TYPE:", type(exc).__name__)
        print("ERROR MESSAGE:", str(exc))
        print("=======================================\n")

        raise HTTPException(
            status_code=500,
            detail="Unable to send monitoring alert",
        ) from exc

    except Exception as exc:
        print("\n========== EMAIL ALERT ERROR ==========")
        print("ERROR TYPE:", type(exc).__name__)
        print("ERROR MESSAGE:", str(exc))
        print("SMTP_HOST configured:", bool(os.getenv("SMTP_HOST")))
        print("SMTP_PORT configured:", bool(os.getenv("SMTP_PORT")))
        print("SMTP_USERNAME configured:", bool(os.getenv("SMTP_USERNAME")))
        print("SMTP_PASSWORD configured:", bool(os.getenv("SMTP_PASSWORD")))
        print("SMTP_FROM_EMAIL configured:", bool(os.getenv("SMTP_FROM_EMAIL")))
        print("=======================================\n")

        raise HTTPException(
            status_code=500,
            detail="Unable to send monitoring alert",
        ) from exc

    finally:
        lock.release()


@app.get("/monitoring-sessions/{session_id}/risk-updates")
def get_phase2_risk_updates(session_id: int, db: Session = Depends(get_db)):
    session = db.query(MonitoringSession).filter(MonitoringSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Monitoring session not found")
    records = (
        db.query(Phase2RiskRecord)
        .filter(Phase2RiskRecord.monitoring_session_id == session_id)
        .order_by(Phase2RiskRecord.id.asc())
        .all()
    )
    return {
        "event_id": session.event_id,
        "session_id": session.id,
        "records": [
            {column.name: getattr(record, column.name) for column in Phase2RiskRecord.__table__.columns}
            for record in records
        ],
    }


@app.post("/monitoring-sessions/{session_id}/stop")
def stop_monitoring_session(session_id: int, db: Session = Depends(get_db)):
    session = db.query(MonitoringSession).filter(MonitoringSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Monitoring session not found")
    engine = WEB_MONITOR_ENGINES.pop(session_id, None)
    WEB_MONITOR_PERSISTED_WINDOWS.pop(session_id, None)
    if engine is not None:
        save_phase2_snapshot(session_id, engine.analytics(), db)
        engine.stop()
    session.status = "STOPPED"
    session.ended_at = utc_now()
    db.commit()
    db.refresh(session)
    return {"message": "Monitoring session stopped", "session": serialize_session(session)}


@app.post("/monitoring-sessions/{session_id}/flow-analysis")
def start_flow_analysis(session_id: int, db: Session = Depends(get_db)):
    """Start bounded recorded-session Phase 3 analysis without changing Phase 2."""
    session = db.query(MonitoringSession).filter(MonitoringSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Monitoring session not found")
    if session.source_type != "recorded":
        raise HTTPException(status_code=400, detail="Flow analysis currently supports completed recorded sessions")
    flow_record = db.query(FlowAnalysisSession).filter(FlowAnalysisSession.monitoring_session_id == session.id).first()
    if flow_record is None:
        flow_record = FlowAnalysisSession(event_id=session.event_id, monitoring_session_id=session.id, source_name=session.source_name, status="QUEUED", cache_key=f"session-{session.id}", created_at=utc_now())
        db.add(flow_record)
        db.commit()
        db.refresh(flow_record)
    existing = PHASE3_ANALYSES.get(session_id)
    if existing and existing.get("status") in {"QUEUED", "RUNNING", "COMPLETE"}:
        existing_result = existing.get("result") or {}
        if existing.get("status") != "COMPLETE" or existing_result.get("analysis_version") == FLOW_CONFIG["analysis_version"]:
            return existing
        # A code/configuration change invalidates the in-memory result too.
        # This prevents an old spider-web trajectory result being returned
        # until the whole API process is manually restarted.
        PHASE3_ANALYSES.pop(session_id, None)
    # A source filename is not a session identity. Never reuse another
    # completed run just because it used the same video file.
    cache_key = f"session-{session.id}"
    cached = load_cached_flow(session.source_name, cache_key=cache_key, expected_version=FLOW_CONFIG["analysis_version"])
    if cached:
        cached = apply_phase2_snapshot(cached, load_phase2_snapshot(session))
        flow_record.status = "COMPLETE"
        flow_record.snapshot_timestamp_sec = (cached.get("snapshot_provenance") or {}).get("source_timestamp_sec")
        flow_record.completed_at = utc_now()
        db.commit()
        PHASE3_ANALYSES[session_id] = {"status": "COMPLETE", "progress": 100, "event_id": session.event_id, "monitoring_session_id": session_id, "flow_analysis_id": flow_record.id, "result": cached, "cached": True}
        return PHASE3_ANALYSES[session_id]
    PHASE3_ANALYSES[session_id] = {"status": "QUEUED", "progress": 0, "event_id": session.event_id, "monitoring_session_id": session_id, "flow_analysis_id": flow_record.id}

    def worker():
        PHASE3_ANALYSES[session_id].update(status="RUNNING", progress=1)
        worker_db = SessionLocal()
        try:
            running_record = worker_db.query(FlowAnalysisSession).filter(FlowAnalysisSession.id == flow_record.id).first()
            if running_record:
                running_record.status = "RUNNING"
                worker_db.commit()
            result = analyze_recorded_video(session.source_name, lambda value: PHASE3_ANALYSES[session_id].update(progress=value), session_id=session.id)
            saved_session = worker_db.query(MonitoringSession).filter(MonitoringSession.id == session.id).first()
            result = apply_phase2_snapshot(result, load_phase2_snapshot(saved_session) if saved_session else None)
            save_cached_flow(session.source_name, result, cache_key=cache_key)
            record = worker_db.query(FlowAnalysisSession).filter(FlowAnalysisSession.id == flow_record.id).first()
            if record:
                record.status = "COMPLETE"
                record.snapshot_timestamp_sec = (result.get("snapshot_provenance") or {}).get("source_timestamp_sec")
                record.completed_at = utc_now()
                worker_db.commit()
            PHASE3_ANALYSES[session_id] = {"status": "COMPLETE", "progress": 100, "event_id": session.event_id, "monitoring_session_id": session_id, "flow_analysis_id": flow_record.id, "result": result}
        except Exception as error:
            record = worker_db.query(FlowAnalysisSession).filter(FlowAnalysisSession.id == flow_record.id).first()
            if record:
                record.status = "ERROR"
                worker_db.commit()
            PHASE3_ANALYSES[session_id] = {"status": "ERROR", "progress": 0, "event_id": session.event_id, "monitoring_session_id": session_id, "flow_analysis_id": flow_record.id, "error": str(error)}
        finally:
            worker_db.close()

    threading.Thread(target=worker, daemon=True).start()
    return PHASE3_ANALYSES[session_id]


@app.get("/monitoring-sessions/{session_id}/flow-analysis")
def get_flow_analysis(session_id: int, db: Session = Depends(get_db)):
    session = db.query(MonitoringSession).filter(MonitoringSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Monitoring session not found")
    flow_record = db.query(FlowAnalysisSession).filter(FlowAnalysisSession.monitoring_session_id == session.id).first()
    return PHASE3_ANALYSES.get(session_id, {"status": flow_record.status if flow_record else "NOT_STARTED", "progress": 100 if flow_record and flow_record.status == "COMPLETE" else 0, "event_id": session.event_id, "monitoring_session_id": session_id, "flow_analysis_id": flow_record.id if flow_record else None})


@app.get("/flow-analyses/{flow_analysis_id}")
def get_flow_analysis_by_id(flow_analysis_id: int, event_id: int | None = None, db: Session = Depends(get_db)):
    """Return one exact Phase 3 record by FlowAnalysisSession primary key."""
    flow_record = db.query(FlowAnalysisSession).filter(FlowAnalysisSession.id == flow_analysis_id).first()
    if flow_record is None:
        raise HTTPException(status_code=404, detail=f"Flow Analysis #{flow_analysis_id} was not found")
    session = db.query(MonitoringSession).filter(MonitoringSession.id == flow_record.monitoring_session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="The monitoring session for this Flow Analysis was not found")
    if event_id is not None and flow_record.event_id != event_id:
        raise HTTPException(status_code=404, detail="Flow Analysis does not belong to this event")
    analysis = PHASE3_ANALYSES.get(session.id)
    if analysis is None and flow_record.status == "COMPLETE":
        cached = load_cached_flow(session.source_name, cache_key=flow_record.cache_key, expected_version=FLOW_CONFIG["analysis_version"])
        if cached:
            cached = apply_phase2_snapshot(cached, load_phase2_snapshot(session))
            analysis = {"status": "COMPLETE", "progress": 100, "event_id": session.event_id, "monitoring_session_id": session.id, "flow_analysis_id": flow_record.id, "result": cached, "cached": True}
    if analysis is None:
        analysis = {"status": flow_record.status, "progress": 100 if flow_record.status == "COMPLETE" else 0, "event_id": session.event_id, "monitoring_session_id": session.id, "flow_analysis_id": flow_record.id}
    return {**analysis, "flow_analysis_id": flow_record.id, "monitoring_session_id": session.id, "event_id": session.event_id, "source_name": session.source_name}


def phase3_result_for_session(session):
    analysis = PHASE3_ANALYSES.get(session.id)
    if analysis and analysis.get("status") == "COMPLETE":
        return analysis.get("result")
    return load_cached_flow(session.source_name, cache_key=f"session-{session.id}", expected_version=FLOW_CONFIG["analysis_version"])


@app.get("/monitoring-sessions/{session_id}/digital-twin/state")
def get_digital_twin_state(session_id: int, db: Session = Depends(get_db)):
    session = db.query(MonitoringSession).filter(MonitoringSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Monitoring session not found")
    result = phase3_result_for_session(session)
    if not result:
        raise HTTPException(status_code=409, detail="Complete Phase 3 Flow Intelligence before opening the Digital Twin")
    result = apply_phase2_snapshot(result, load_phase2_snapshot(session))
    flow_record = db.query(FlowAnalysisSession).filter(FlowAnalysisSession.monitoring_session_id == session.id).first()
    if flow_record is None:
        flow_record = FlowAnalysisSession(event_id=session.event_id, monitoring_session_id=session.id, source_name=session.source_name, status="COMPLETE", cache_key=f"session-{session.id}", snapshot_timestamp_sec=(result.get("snapshot_provenance") or {}).get("source_timestamp_sec"), created_at=utc_now(), completed_at=utc_now())
        db.add(flow_record)
        db.commit()
        db.refresh(flow_record)
    try:
        snapshot = build_current_state(result, session.event_id, session.id, flow_record.id if flow_record else None)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"event_id": session.event_id, "monitoring_session_id": session.id, "flow_analysis_id": flow_record.id if flow_record else None, "source": session.source_name, "configuration": {"simulation_capacity": True, "allowed_horizons": CONFIG["allowed_horizons"]}, "current_state": snapshot, "source_provenance": snapshot["source_provenance"]}


@app.post("/monitoring-sessions/{session_id}/digital-twin/simulate")
def run_digital_twin_simulation(session_id: int, payload: dict, db: Session = Depends(get_db)):
    session = db.query(MonitoringSession).filter(MonitoringSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Monitoring session not found")
    result = phase3_result_for_session(session)
    if not result:
        raise HTTPException(status_code=409, detail="Complete Phase 3 Flow Intelligence before running a simulation")
    try:
        result = apply_phase2_snapshot(result, load_phase2_snapshot(session))
        flow_record = db.query(FlowAnalysisSession).filter(FlowAnalysisSession.monitoring_session_id == session.id).first()
        snapshot = build_current_state(result, session.event_id, session.id, flow_record.id if flow_record else None)
        scenarios = build_scenarios(snapshot, payload.get("horizon_seconds", CONFIG["default_horizon"]))
        selected_type = payload.get("action", {}).get("type") if isinstance(payload.get("action"), dict) else None
        selected = next((item for item in scenarios["scenarios"] if item["action"].get("type") == selected_type), scenarios["scenarios"][0])
        return {"event_id": session.event_id, "monitoring_session_id": session.id, "flow_analysis_id": flow_record.id if flow_record else None, "current_state": snapshot, "baseline": scenarios["baseline"], "selected": selected, "scenarios": scenarios["scenarios"], "best": scenarios["best"], "simulation_label": "ESTIMATED EFFECT - decision-support simulation only"}
        return {"session_id": session.id, "current_state": snapshot, "baseline": scenarios["baseline"], "selected": selected, "scenarios": scenarios["scenarios"], "best": scenarios["best"], "simulation_label": "ESTIMATED EFFECT — decision-support simulation only"}
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


# =========================================================
# CROWD TIME MACHINE ENDPOINTS
# =========================================================

def serialize_time_machine(session):
    return {"id": session.id, "event_id": session.event_id, "video_name": session.video_name, "fps": session.fps, "duration": session.duration, "frame_count": session.frame_count, "status": session.status, "progress": session.progress, "error": session.error}


@app.post("/time-machine/sessions")
def create_time_machine_session(payload: dict, db: Session = Depends(get_db)):
    video_name = Path(str(payload.get("video_name") or "")).name
    video_path = PHASE2_VIDEOS / video_name
    if video_path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"} or not video_path.is_file():
        raise HTTPException(status_code=404, detail="Selected video was not found")
    event_id = payload.get("event_id")
    if event_id is not None and db.query(Event).filter(Event.id == event_id).first() is None:
        raise HTTPException(status_code=404, detail="Event not found")
    now = utc_now()
    session = TimeMachineSession(event_id=event_id, video_path=str(video_path), video_name=video_name, status="PENDING", progress=0, created_at=now, updated_at=now)
    db.add(session); db.commit(); db.refresh(session)
    return {"session": serialize_time_machine(session)}


@app.post("/time-machine/sessions/{session_id}/analyze")
def start_time_machine_analysis(session_id: int, db: Session = Depends(get_db)):
    session = db.query(TimeMachineSession).filter(TimeMachineSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Time Machine session not found")
    if session.status == "COMPLETED" and session.result_json:
        return {"session": serialize_time_machine(session), "cached": True}
    if TIME_MACHINE_CACHE.get(session_id, {}).get("status") in {"ANALYZING", "COMPLETED"}:
        return {"session": serialize_time_machine(session)}
    cached_result = load_cached_time_machine(session.video_name)
    if cached_result:
        session.status = "COMPLETED"; session.progress = 100; session.fps = cached_result["fps"]; session.duration = cached_result["duration"]; session.frame_count = cached_result["frame_count"]; session.result_json = json.dumps(cached_result); session.updated_at = utc_now(); db.commit(); db.refresh(session)
        if not db.query(TimeMachineSnapshot).filter(TimeMachineSnapshot.session_id == session_id).first():
            for value in HORIZONS:
                prediction = cached_result["predictions"][str(value)]; counts = prediction["zone_counts"]
                db.add(TimeMachineSnapshot(session_id=session_id, source_timestamp=cached_result["source_timestamp"], horizon_seconds=value, total_people=prediction["total_people"], zone_a_people=counts["ZONE_A"], zone_b_people=counts["ZONE_B"], zone_c_people=counts["ZONE_C"], risk_zone=prediction["risk_zone"], risk_level=prediction["risk_level"], confidence=prediction["confidence"], metadata_json=json.dumps(prediction)))
            db.commit()
        return {"session": serialize_time_machine(session), "cached": True}
    session.status = "ANALYZING"; session.progress = 1; session.updated_at = utc_now(); db.commit()
    def worker():
        worker_db = SessionLocal()
        try:
            last_progress = -1
            def progress(value):
                nonlocal last_progress
                value = int(value)
                if value == last_progress:
                    return
                last_progress = value
                item = worker_db.query(TimeMachineSession).filter(TimeMachineSession.id == session_id).first()
                if item:
                    item.progress = value; item.updated_at = utc_now(); worker_db.commit()
            result = analyze_time_machine(session.video_name, progress)
            save_cached_time_machine(session.video_name, result)
            item = worker_db.query(TimeMachineSession).filter(TimeMachineSession.id == session_id).first()
            if item:
                item.status = "COMPLETED"; item.progress = 100; item.fps = result["fps"]; item.duration = result["duration"]; item.frame_count = result["frame_count"]; item.result_json = json.dumps(result); item.updated_at = utc_now(); worker_db.commit()
                for value in HORIZONS:
                    prediction = result["predictions"][str(value)]
                    counts = prediction["zone_counts"]
                    worker_db.add(TimeMachineSnapshot(session_id=session_id, source_timestamp=result["source_timestamp"], horizon_seconds=value, total_people=prediction["total_people"], zone_a_people=counts["ZONE_A"], zone_b_people=counts["ZONE_B"], zone_c_people=counts["ZONE_C"], risk_zone=prediction["risk_zone"], risk_level=prediction["risk_level"], confidence=prediction["confidence"], metadata_json=json.dumps(prediction)))
                worker_db.commit()
        except Exception as error:
            item = worker_db.query(TimeMachineSession).filter(TimeMachineSession.id == session_id).first()
            if item:
                item.status = "FAILED"; item.error = str(error); item.updated_at = utc_now(); worker_db.commit()
        finally:
            worker_db.close(); TIME_MACHINE_CACHE.pop(session_id, None)
    TIME_MACHINE_CACHE[session_id] = {"status": "ANALYZING"}
    threading.Thread(target=worker, daemon=True).start()
    return {"session": serialize_time_machine(session)}


def get_time_machine_or_404(session_id, db):
    session = db.query(TimeMachineSession).filter(TimeMachineSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Time Machine session not found")
    if session.status != "COMPLETED" or not session.result_json:
        raise HTTPException(status_code=409, detail="Time Machine analysis is not complete")
    return session, json.loads(session.result_json)


@app.get("/time-machine/sessions/{session_id}/status")
def time_machine_status(session_id: int, db: Session = Depends(get_db)):
    session = db.query(TimeMachineSession).filter(TimeMachineSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Time Machine session not found")
    return {"session": serialize_time_machine(session)}


@app.get("/time-machine/sessions/{session_id}/current-state")
def time_machine_current_state(session_id: int, db: Session = Depends(get_db)):
    session, result = get_time_machine_or_404(session_id, db)
    return {"session": serialize_time_machine(session), "source_timestamp": result["source_timestamp"], "current": result["current"], "metadata": {key: result[key] for key in ("fps", "width", "height", "duration", "frame_count")}}


@app.get("/time-machine/sessions/{session_id}/prediction")
def time_machine_prediction(session_id: int, horizon: int = 30, db: Session = Depends(get_db)):
    if horizon not in HORIZONS[1:]:
        raise HTTPException(status_code=400, detail="Prediction horizon must be 15, 30, or 60 seconds")
    session, result = get_time_machine_or_404(session_id, db)
    return {"session": serialize_time_machine(session), "source_timestamp": result["source_timestamp"], "current": result["current"], "prediction": result["predictions"][str(horizon)], "predictions": result["predictions"], "tracks": result.get("tracks", []), "metadata": {key: result[key] for key in ("fps", "width", "height", "duration", "frame_count")}}


@app.post("/time-machine/sessions/{session_id}/interventions")
def time_machine_interventions(session_id: int, db: Session = Depends(get_db)):
    session, result = get_time_machine_or_404(session_id, db)
    baseline = result["predictions"]["30"]["zone_counts"]
    actions = [{"name": "No Action", "counts": baseline}, {"name": "Reduce risk-zone inflow", "counts": dict(baseline)}, {"name": "Redirect B → C", "counts": dict(baseline)}, {"name": "Increase exit flow", "counts": dict(baseline)}]
    risk_zone = max(ZONES := ("ZONE_A", "ZONE_B", "ZONE_C"), key=lambda zone: baseline[zone])
    actions[1]["counts"][risk_zone] = max(0, actions[1]["counts"][risk_zone] - max(1, round(actions[1]["counts"][risk_zone] * .15)))
    actions[2]["counts"]["ZONE_B"] = max(0, actions[2]["counts"]["ZONE_B"] - max(1, round(actions[2]["counts"]["ZONE_B"] * .15))); actions[2]["counts"]["ZONE_C"] += max(1, round(baseline["ZONE_B"] * .15))
    actions[3]["counts"]["ZONE_C"] = max(0, actions[3]["counts"]["ZONE_C"] - max(1, round(actions[3]["counts"]["ZONE_C"] * .15)))
    for item in actions: item["total_people"] = sum(item["counts"].values()); item["risk"] = "HIGH" if max(item["counts"].values()) >= 70 else "MEDIUM" if max(item["counts"].values()) >= 40 else "LOW"
    best = min(actions[1:], key=lambda item: (max(item["counts"].values()), item["total_people"]))
    return {"session": serialize_time_machine(session), "baseline": baseline, "scenarios": actions, "best": best, "risk_zone": risk_zone}


@app.get("/events/{event_id}/time-machine/sessions")
def list_time_machine_sessions(event_id: int, db: Session = Depends(get_db)):
    if db.query(Event).filter(Event.id == event_id).first() is None:
        raise HTTPException(status_code=404, detail="Event not found")
    sessions = db.query(TimeMachineSession).filter(TimeMachineSession.event_id == event_id).order_by(TimeMachineSession.id.desc()).all()
    return {"event_id": event_id, "sessions": [serialize_time_machine(item) for item in sessions]}


# =========================================================
# PHASE 4 RESPONSE COMMANDER ENDPOINTS
# =========================================================

@app.post("/events/{event_id}/instability-radar/analyze")
def analyze_instability(event_id: int, payload: dict, db: Session = Depends(get_db)):
    monitoring_id = payload.get("monitoring_session_id"); flow_id = payload.get("flow_analysis_id")
    monitoring = db.query(MonitoringSession).filter(MonitoringSession.id == monitoring_id, MonitoringSession.event_id == event_id).first()
    flow_record = db.query(FlowAnalysisSession).filter(FlowAnalysisSession.id == flow_id, FlowAnalysisSession.event_id == event_id, FlowAnalysisSession.monitoring_session_id == monitoring_id).first()
    if monitoring is None or flow_record is None: raise HTTPException(status_code=404, detail="Instability Radar requires matching monitoring and Flow Analysis sessions")
    result = phase3_result_for_session(monitoring)
    if not result: raise HTTPException(status_code=409, detail="Flow Analysis result is unavailable")
    result = apply_phase2_snapshot(result, load_phase2_snapshot(monitoring))
    try: radar = analyze_instability_radar(result)
    except ValueError as error: raise HTTPException(status_code=409, detail=str(error)) from error
    radar.update({"event_id": event_id, "monitoring_session_id": monitoring_id, "flow_analysis_id": flow_id, "source": "Flow Intelligence trajectories", "source_name": monitoring.source_name, "frame_width": (result.get("frame") or {}).get("width", 1280), "frame_height": (result.get("frame") or {}).get("height", 720)})
    now = utc_now(); record = InstabilitySnapshot(event_id=event_id, monitoring_session_id=monitoring_id, flow_analysis_id=flow_id, timestamp=now, overall_instability=radar["overall_instability"], overall_stability=radar["overall_stability"], zone_a_instability=radar["zone_metrics"]["ZONE_A"]["instability_score"], zone_b_instability=radar["zone_metrics"]["ZONE_B"]["instability_score"], zone_c_instability=radar["zone_metrics"]["ZONE_C"]["instability_score"], compression_score=max(radar["zone_metrics"][zone]["compression"] for zone in ZONES), counter_flow_score=max(radar["zone_metrics"][zone]["counter_flow"] for zone in ZONES), stop_go_score=max(radar["zone_metrics"][zone]["stop_go"] for zone in ZONES), direction_disorder_score=max(radar["zone_metrics"][zone]["direction_disorder"] for zone in ZONES), speed_drop_score=max(radar["zone_metrics"][zone]["speed_drop"] for zone in ZONES), highest_instability_zone=radar["highest_instability_zone"], propagation_from=radar["propagation"]["from"], propagation_to=radar["propagation"]["to"], propagation_confidence=radar["propagation"]["confidence"], snapshot_json=json.dumps(radar), created_at=now); db.add(record); db.commit(); db.refresh(record)
    radar["analysis_id"] = record.id; return radar


@app.get("/events/{event_id}/instability-radar/{analysis_id}")
def get_instability(event_id: int, analysis_id: int, db: Session = Depends(get_db)):
    record = db.query(InstabilitySnapshot).filter(InstabilitySnapshot.id == analysis_id, InstabilitySnapshot.event_id == event_id).first()
    if record is None: raise HTTPException(status_code=404, detail="Instability Radar analysis not found")
    data = json.loads(record.snapshot_json); data["analysis_id"] = record.id; return data


@app.get("/events/{event_id}/instability-radar/{analysis_id}/snapshot")
def get_instability_snapshot(event_id: int, analysis_id: int, db: Session = Depends(get_db)):
    return get_instability(event_id, analysis_id, db)


@app.get("/events/{event_id}/instability-radar/{analysis_id}/forecast")
def get_instability_forecast(event_id: int, analysis_id: int, horizon: int = 0, db: Session = Depends(get_db)):
    if horizon not in {0, 10, 20, 30}: raise HTTPException(status_code=400, detail="Radar forecast horizon must be 0, 10, 20, or 30 seconds")
    data = get_instability(event_id, analysis_id, db)
    return next(item for item in data.get("forecast", []) if item["horizon_seconds"] == horizon)


# =========================================================
# PHASE 6 AI CROWD STORYBOARD ENDPOINTS
# =========================================================

def _storyboard_payload(record):
    data = json.loads(record.result_json)
    data["storyboard_id"] = record.id
    data["status"] = record.status
    data["created_at"] = record.created_at
    return data


@app.post("/events/{event_id}/storyboard/generate")
def generate_crowd_storyboard(event_id: int, payload: dict, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    monitoring_id = payload.get("monitoring_session_id")
    monitoring = db.query(MonitoringSession).filter(MonitoringSession.id == monitoring_id, MonitoringSession.event_id == event_id).first()
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    if monitoring is None:
        raise HTTPException(status_code=404, detail="Monitoring session does not belong to this event")
    flow_record = db.query(FlowAnalysisSession).filter(FlowAnalysisSession.monitoring_session_id == monitoring.id, FlowAnalysisSession.event_id == event_id).first()
    if payload.get("flow_analysis_id") is not None:
        flow_record = db.query(FlowAnalysisSession).filter(FlowAnalysisSession.id == payload["flow_analysis_id"], FlowAnalysisSession.monitoring_session_id == monitoring.id, FlowAnalysisSession.event_id == event_id).first()
    flow_result = phase3_result_for_session(monitoring)
    if not flow_result:
        raise HTTPException(status_code=409, detail="Complete Flow Intelligence before generating a storyboard")
    phase2_records = db.query(Phase2RiskRecord).filter(Phase2RiskRecord.monitoring_session_id == monitoring.id, Phase2RiskRecord.event_id == event_id).order_by(Phase2RiskRecord.id.asc()).all()
    phase2 = []
    for item in phase2_records:
        metadata = {}
        try:
            metadata = json.loads(item.xai_summary or "{}")
        except (TypeError, ValueError):
            pass
        phase2.append({"timestamp": item.timestamp, "source_timestamp_sec": item.source_timestamp_sec, "zone_a_people": item.zone_a_people, "zone_b_people": item.zone_b_people, "zone_c_people": item.zone_c_people, "total_people": metadata.get("people_count"), "density_state": metadata.get("density_state"), "density_value": metadata.get("density_value"), "zone_a_risk_level": item.zone_a_risk_level, "zone_b_risk_level": item.zone_b_risk_level, "zone_c_risk_level": item.zone_c_risk_level, "future_risk_level": item.future_risk_level})
    radar = None
    radar_id = payload.get("instability_analysis_id")
    if radar_id is not None:
        radar_record = db.query(InstabilitySnapshot).filter(InstabilitySnapshot.id == radar_id, InstabilitySnapshot.event_id == event_id, InstabilitySnapshot.monitoring_session_id == monitoring.id).first()
        if radar_record is None:
            raise HTTPException(status_code=404, detail="Instability Radar source is not part of this monitoring lineage")
        radar = json.loads(radar_record.snapshot_json)
    time_result = None
    time_id = payload.get("time_machine_session_id")
    if time_id is not None:
        time_record = db.query(TimeMachineSession).filter(TimeMachineSession.id == time_id, TimeMachineSession.event_id == event_id).first()
        if time_record is None or time_record.status != "COMPLETED":
            raise HTTPException(status_code=404, detail="Time Machine source is not part of this event lineage")
        try:
            time_result = json.loads(time_record.result_json or "{}")
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=409, detail="Time Machine result is invalid") from error
    lineage = {"flow_analysis_id": flow_record.id if flow_record else None, "time_machine_session_id": time_id, "instability_analysis_id": radar_id, "response_plan_id": payload.get("response_plan_id")}
    result = generate_storyboard(event, monitoring, flow_result, phase2, radar, time_result, lineage)
    for card in result.get("cards", []):
        timestamp = card["timestamp_sec"]
        card["frame_reference"]["image_url"] = f"/events/{event_id}/storyboards/frame?monitoring_session_id={monitoring.id}&timestamp={timestamp}"
    for interval in result.get("detailed_timeline", {}).get("intervals", []):
        timestamp = interval.get("representative_timestamp_sec")
        if timestamp is not None:
            interval["image_url"] = f"/events/{event_id}/storyboards/frame?monitoring_session_id={monitoring.id}&timestamp={timestamp}"
    now = utc_now()
    record = StoryboardSession(event_id=event_id, monitoring_session_id=monitoring.id, flow_analysis_id=flow_record.id if flow_record else None, time_machine_session_id=time_id, instability_analysis_id=radar_id, response_plan_id=payload.get("response_plan_id"), algorithm_version=STORYBOARD_ALGORITHM_VERSION, status="COMPLETE", result_json=json.dumps(result), created_at=now)
    db.add(record)
    db.commit()
    db.refresh(record)
    result["storyboard_id"] = record.id
    return result


@app.get("/events/{event_id}/storyboards/frame")
def get_storyboard_frame(event_id: int, monitoring_session_id: int, timestamp: float = 0.0, db: Session = Depends(get_db)):
    """Return the actual source-video frame nearest to a storyboard event."""
    monitoring = db.query(MonitoringSession).filter(MonitoringSession.id == monitoring_session_id, MonitoringSession.event_id == event_id).first()
    if monitoring is None:
        raise HTTPException(status_code=404, detail="Monitoring session not found")
    video_path = PHASE2_VIDEOS / Path(monitoring.source_name).name
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Storyboard source video not found")
    cache_dir = PROJECT_ROOT / "phase6" / "cache" / "frames"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{monitoring.id}-{max(0, int(round(timestamp * 1000)))}.jpg"
    if not cache_path.is_file():
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise HTTPException(status_code=409, detail="Unable to read storyboard source video")
            fps = max(float(capture.get(cv2.CAP_PROP_FPS) or 30.0), 1.0)
            frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1), 1)
            requested_frame = max(0, int(round(max(0.0, float(timestamp)) * fps)))
            safe_frame = min(requested_frame, frame_count - 1)
            capture.set(cv2.CAP_PROP_POS_FRAMES, safe_frame)
            ok, frame = capture.read()
            if not ok:
                raise HTTPException(status_code=404, detail="Storyboard frame is unavailable")
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            if not ok:
                raise HTTPException(status_code=500, detail="Unable to encode storyboard frame")
            cache_path.write_bytes(encoded.tobytes())
        finally:
            capture.release()
    return FileResponse(cache_path, media_type="image/jpeg")


@app.get("/events/{event_id}/storyboards/{storyboard_id}")
def get_crowd_storyboard(event_id: int, storyboard_id: int, db: Session = Depends(get_db)):
    record = db.query(StoryboardSession).filter(StoryboardSession.id == storyboard_id, StoryboardSession.event_id == event_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail="Storyboard not found")
    return _storyboard_payload(record)

def venue_layout_payload(record):
    data = json.loads(record.config_json)
    return {"id": record.id, "event_id": record.event_id, "created_at": record.created_at, "updated_at": record.updated_at, **data}


@app.get("/events/{event_id}/venue-layout")
def get_venue_layout(event_id: int, db: Session = Depends(get_db)):
    if db.query(Event).filter(Event.id == event_id).first() is None:
        raise HTTPException(status_code=404, detail="Event not found")
    record = db.query(VenueLayout).filter(VenueLayout.event_id == event_id).first()
    return venue_layout_payload(record) if record else {"event_id": event_id, "configured": False, "nodes": [], "edges": []}


@app.post("/events/{event_id}/venue-layout")
def create_venue_layout(event_id: int, payload: dict, db: Session = Depends(get_db)):
    if db.query(Event).filter(Event.id == event_id).first() is None:
        raise HTTPException(status_code=404, detail="Event not found")
    try: config = VenueLayoutSchema.model_validate(payload).model_dump()
    except ValueError as error: raise HTTPException(status_code=422, detail=str(error)) from error
    if db.query(VenueLayout).filter(VenueLayout.event_id == event_id).first():
        raise HTTPException(status_code=409, detail="Venue layout already exists; use PUT to update it")
    now = utc_now(); record = VenueLayout(event_id=event_id, name=config["name"], config_json=json.dumps(config), created_at=now, updated_at=now); db.add(record); db.commit(); db.refresh(record)
    return venue_layout_payload(record)


@app.put("/events/{event_id}/venue-layout")
def update_venue_layout(event_id: int, payload: dict, db: Session = Depends(get_db)):
    record = db.query(VenueLayout).filter(VenueLayout.event_id == event_id).first()
    if record is None: raise HTTPException(status_code=404, detail="Venue layout is not configured for this event")
    try: config = VenueLayoutSchema.model_validate(payload).model_dump()
    except ValueError as error: raise HTTPException(status_code=422, detail=str(error)) from error
    record.name = config["name"]; record.config_json = json.dumps(config); record.updated_at = utc_now(); db.commit(); db.refresh(record)
    return venue_layout_payload(record)


@app.get("/events/{event_id}/response-commander/source-state")
def response_source_state(event_id: int, monitoring_session_id: int, flow_analysis_id: int, time_machine_session_id: int, db: Session = Depends(get_db)):
    return _build_response_state(event_id, monitoring_session_id, flow_analysis_id, time_machine_session_id, db)


def _build_response_state(event_id, monitoring_id, flow_id, time_machine_id, db):
    monitoring = db.query(MonitoringSession).filter(MonitoringSession.id == monitoring_id, MonitoringSession.event_id == event_id).first()
    flow_record = db.query(FlowAnalysisSession).filter(FlowAnalysisSession.id == flow_id, FlowAnalysisSession.event_id == event_id, FlowAnalysisSession.monitoring_session_id == monitoring_id).first()
    time_session = db.query(TimeMachineSession).filter(TimeMachineSession.id == time_machine_id, TimeMachineSession.event_id == event_id).first()
    if monitoring is None or flow_record is None or time_session is None: raise HTTPException(status_code=404, detail="Selected source sessions do not share the requested event lineage")
    if flow_record.status != "COMPLETE": raise HTTPException(status_code=409, detail="Selected Flow Analysis is not complete")
    flow_result = phase3_result_for_session(monitoring)
    if not flow_result: raise HTTPException(status_code=409, detail="Selected Flow Analysis result is unavailable")
    try: time_result = json.loads(time_session.result_json or "{}")
    except ValueError as error: raise HTTPException(status_code=409, detail="Selected Time Machine result is invalid") from error
    if time_session.status != "COMPLETED" or not time_result: raise HTTPException(status_code=409, detail="Selected Time Machine analysis is not complete")
    return {"event_id": event_id, "monitoring_session_id": monitoring_id, "flow_analysis_id": flow_id, "time_machine_session_id": time_machine_id, "current": build_current_state(apply_phase2_snapshot(flow_result, load_phase2_snapshot(monitoring)), event_id, monitoring_id, flow_id), "predictions": time_result.get("predictions", {})}


@app.get("/events/{event_id}/explainable-ai")
def explainable_ai(event_id: int, monitoring_session_id: int, flow_analysis_id: int, time_machine_session_id: int, instability_analysis_id: int | None = None, response_plan_id: int | None = None, horizon: int = 30, db: Session = Depends(get_db)):
    """Normalize selected Phase 2-4 outputs for the Phase 5 assistant.

    Every source is explicit and must belong to this event and monitoring run;
    Phase 5 never silently chooses a latest record from another lineage.
    """
    if horizon not in {15, 30, 60}:
        raise HTTPException(status_code=400, detail="Phase 5 horizon must be 15, 30, or 60 seconds")
    event = db.query(Event).filter(Event.id == event_id).first()
    monitoring = db.query(MonitoringSession).filter(MonitoringSession.id == monitoring_session_id, MonitoringSession.event_id == event_id).first()
    flow_record = db.query(FlowAnalysisSession).filter(FlowAnalysisSession.id == flow_analysis_id, FlowAnalysisSession.event_id == event_id, FlowAnalysisSession.monitoring_session_id == monitoring_session_id).first()
    time_session = db.query(TimeMachineSession).filter(TimeMachineSession.id == time_machine_session_id, TimeMachineSession.event_id == event_id).first()
    if event is None or monitoring is None or flow_record is None or time_session is None:
        raise HTTPException(status_code=404, detail="Selected Phase 2/3 sources do not share this event lineage")
    if flow_record.status != "COMPLETE":
        raise HTTPException(status_code=409, detail="Complete Flow Intelligence to enable movement explanation.")
    if time_session.status != "COMPLETED" or not time_session.result_json:
        raise HTTPException(status_code=409, detail="Complete Crowd Time Machine to enable forecast explanation.")
    flow_result = phase3_result_for_session(monitoring)
    if not flow_result:
        raise HTTPException(status_code=409, detail="Flow Intelligence result is unavailable")
    flow_result = apply_phase2_snapshot(flow_result, load_phase2_snapshot(monitoring))
    try:
        time_result = json.loads(time_session.result_json)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=409, detail="Time Machine result is invalid") from error
    radar = None
    if instability_analysis_id is not None:
        record = db.query(InstabilitySnapshot).filter(InstabilitySnapshot.id == instability_analysis_id, InstabilitySnapshot.event_id == event_id, InstabilitySnapshot.monitoring_session_id == monitoring_session_id, InstabilitySnapshot.flow_analysis_id == flow_analysis_id).first()
        if record is None:
            raise HTTPException(status_code=404, detail="Selected Instability Radar source is not part of this lineage")
        radar = json.loads(record.snapshot_json)
    plan = None
    if response_plan_id is not None:
        record = db.query(ResponsePlan).filter(ResponsePlan.id == response_plan_id, ResponsePlan.event_id == event_id, ResponsePlan.monitoring_session_id == monitoring_session_id, ResponsePlan.flow_analysis_id == flow_analysis_id, ResponsePlan.time_machine_session_id == time_machine_session_id).first()
        if record is None:
            raise HTTPException(status_code=404, detail="Selected Response Commander plan is not part of this lineage")
        plan = json.loads(record.plan_json)
    snapshot = flow_result.get("phase2_snapshot") or {}
    zone_counts = {zone: int(snapshot.get("zone_counts", {}).get(zone, 0)) for zone in ("ZONE_A", "ZONE_B", "ZONE_C")}
    current = {"total_people": int(snapshot.get("total_people", sum(zone_counts.values()))), "zone_counts": zone_counts, "highest_zone": max(zone_counts, key=zone_counts.get) if any(zone_counts.values()) else None, "risk_level": None, "instability_score": None}
    lineage = {"monitoring_session_id": monitoring_session_id, "flow_analysis_id": flow_analysis_id, "time_machine_session_id": time_machine_session_id, "instability_analysis_id": instability_analysis_id, "response_plan_id": response_plan_id}
    explanation = build_explanation(event, lineage, current, time_result.get("predictions", {}), radar=radar, plan=plan, flow=flow_result)
    provenance = flow_result.get("snapshot_provenance") or {}
    explanation["video_source"] = monitoring.source_name
    explanation["video_timestamp_sec"] = provenance.get("source_timestamp_sec")
    explanation["source_basis"] = "Video-derived Monitoring snapshot + Flow Intelligence + Time Machine analysis"
    return explanation


@app.post("/events/{event_id}/assistant/chat")
def assistant_chat(event_id: int, payload: dict, db: Session = Depends(get_db)):
    """Answer from the exact selected analytics lineage; never reruns detection."""
    message = str(payload.get("message") or "").strip()
    if not message or len(message) > 2000:
        raise HTTPException(status_code=400, detail="A message between 1 and 2000 characters is required")
    monitoring_id = payload.get("monitoring_session_id")
    flow_id = payload.get("flow_analysis_id")
    time_id = payload.get("time_machine_session_id")
    if not all(isinstance(value, int) for value in (monitoring_id, flow_id, time_id)):
        raise HTTPException(status_code=400, detail="Explicit monitoring_session_id, flow_analysis_id, and time_machine_session_id are required")
    event = db.query(Event).filter(Event.id == event_id).first()
    monitoring = db.query(MonitoringSession).filter(MonitoringSession.id == monitoring_id, MonitoringSession.event_id == event_id).first()
    flow_record = db.query(FlowAnalysisSession).filter(FlowAnalysisSession.id == flow_id, FlowAnalysisSession.event_id == event_id, FlowAnalysisSession.monitoring_session_id == monitoring_id).first()
    time_session = db.query(TimeMachineSession).filter(TimeMachineSession.id == time_id, TimeMachineSession.event_id == event_id).first()
    if event is None or monitoring is None or flow_record is None or time_session is None:
        raise HTTPException(status_code=404, detail="Selected assistant sources do not share this event lineage")
    if flow_record.status != "COMPLETE" or time_session.status != "COMPLETED" or not time_session.result_json:
        raise HTTPException(status_code=409, detail="Complete Flow Intelligence and Crowd Time Machine analyses are required")
    flow_result = phase3_result_for_session(monitoring)
    if not flow_result:
        raise HTTPException(status_code=409, detail="Flow Intelligence result is unavailable")
    flow_result = apply_phase2_snapshot(flow_result, load_phase2_snapshot(monitoring))
    try:
        time_result = json.loads(time_session.result_json)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=409, detail="Time Machine result is invalid") from error
    instability_id = payload.get("instability_analysis_id")
    radar = None
    if not isinstance(instability_id, int):
        radar_record = db.query(InstabilitySnapshot).filter(InstabilitySnapshot.event_id == event_id, InstabilitySnapshot.monitoring_session_id == monitoring_id, InstabilitySnapshot.flow_analysis_id == flow_id).order_by(InstabilitySnapshot.id.desc()).first()
        if radar_record:
            instability_id = radar_record.id
    if isinstance(instability_id, int):
        radar_record = db.query(InstabilitySnapshot).filter(InstabilitySnapshot.id == instability_id, InstabilitySnapshot.event_id == event_id, InstabilitySnapshot.monitoring_session_id == monitoring_id, InstabilitySnapshot.flow_analysis_id == flow_id).first()
        if radar_record:
            radar = json.loads(radar_record.snapshot_json)
            radar["id"] = radar_record.id
    plan = None
    response_id = payload.get("response_plan_id")
    if not isinstance(response_id, int):
        plan_record = db.query(ResponsePlan).filter(ResponsePlan.event_id == event_id, ResponsePlan.monitoring_session_id == monitoring_id, ResponsePlan.flow_analysis_id == flow_id, ResponsePlan.time_machine_session_id == time_id).order_by(ResponsePlan.id.desc()).first()
        if plan_record:
            response_id = plan_record.id
    if isinstance(response_id, int):
        plan_record = db.query(ResponsePlan).filter(ResponsePlan.id == response_id, ResponsePlan.event_id == event_id, ResponsePlan.monitoring_session_id == monitoring_id, ResponsePlan.flow_analysis_id == flow_id, ResponsePlan.time_machine_session_id == time_id).first()
        if plan_record:
            plan = json.loads(plan_record.plan_json)
            plan["id"] = plan_record.id
    lineage = {"monitoring_session_id": monitoring_id, "flow_analysis_id": flow_id, "time_machine_session_id": time_id, "instability_analysis_id": instability_id, "response_plan_id": response_id}
    records = db.query(Phase2RiskRecord).filter(Phase2RiskRecord.monitoring_session_id == monitoring_id, Phase2RiskRecord.event_id == event_id).order_by(Phase2RiskRecord.id.desc()).limit(30).all()
    history = [{"timestamp": item.source_timestamp_sec, "zone_counts": {"ZONE_A": item.zone_a_people, "ZONE_B": item.zone_b_people, "ZONE_C": item.zone_c_people}, "future_risk": item.future_risk_level} for item in reversed(records)]
    context = build_context(event, monitoring, flow_result, time_result, radar=radar, plan=plan, lineage=lineage, history=history)
    conversation = payload.get("conversation") if isinstance(payload.get("conversation"), list) else []
    safe_conversation = [{"role": str(item.get("role", ""))[:20], "content": str(item.get("content", ""))[:500], "intent": item.get("intent"), "zone": item.get("zone"), "horizon": item.get("horizon")} for item in conversation[-10:] if isinstance(item, dict)]
    result = crowdguard_chat(message, context, safe_conversation, str(payload.get("language") or "auto"))
    result["evidence"]["lineage"] = lineage
    return result

def response_plan_payload(plan):
    data = json.loads(plan.plan_json)
    data.update({"id": plan.id, "status": plan.status, "created_at": plan.created_at, "updated_at": plan.updated_at, "approved_at": plan.approved_at})
    return data


@app.post("/events/{event_id}/response-commander/plans")
def create_response_plan(event_id: int, payload: dict, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    monitoring_id = payload.get("monitoring_session_id")
    flow_id = payload.get("flow_analysis_id")
    time_machine_id = payload.get("time_machine_session_id")
    if not all(isinstance(value, int) for value in (monitoring_id, flow_id, time_machine_id)):
        raise HTTPException(status_code=400, detail="Explicit monitoring_session_id, flow_analysis_id, and time_machine_session_id are required")
    monitoring = db.query(MonitoringSession).filter(MonitoringSession.id == monitoring_id, MonitoringSession.event_id == event_id).first()
    flow_record = db.query(FlowAnalysisSession).filter(FlowAnalysisSession.id == flow_id, FlowAnalysisSession.event_id == event_id, FlowAnalysisSession.monitoring_session_id == monitoring_id).first()
    time_session = db.query(TimeMachineSession).filter(TimeMachineSession.id == time_machine_id, TimeMachineSession.event_id == event_id).first()
    if monitoring is None or flow_record is None or time_session is None:
        raise HTTPException(status_code=404, detail="Invalid Phase 2/3 source lineage for this event")
    if flow_record.status != "COMPLETE":
        raise HTTPException(status_code=409, detail="Complete Flow Intelligence analysis is required")
    if time_session.status != "COMPLETED" or not time_session.result_json:
        raise HTTPException(status_code=409, detail="Complete Crowd Time Machine analysis is required")
    flow_result = phase3_result_for_session(monitoring)
    if not flow_result:
        raise HTTPException(status_code=409, detail="Phase 3 source result is unavailable")
    # STOPPED is a valid terminal state for recorded playback. It is usable
    # when Phase 2 persisted a final snapshot or Phase 3 preserved the same
    # completed-session snapshot. Playback status alone is not lineage data.
    final_snapshot = None
    if monitoring.final_snapshot_json:
        try:
            final_snapshot = json.loads(monitoring.final_snapshot_json)
        except (TypeError, ValueError):
            final_snapshot = None
    has_final_state = bool((final_snapshot or {}).get("zone_counts")) or bool((flow_result.get("phase2_snapshot") or {}).get("zone_counts"))
    if monitoring.status == "STOPPED" and not has_final_state:
        raise HTTPException(status_code=409, detail="Monitoring session is STOPPED and has no usable final crowd analytics snapshot")
    flow_result = apply_phase2_snapshot(flow_result, load_phase2_snapshot(monitoring))
    try:
        state = build_current_state(flow_result, event_id, monitoring_id, flow_id)
        time_result = json.loads(time_session.result_json)
        future = time_result.get("predictions", {}).get(str(payload.get("horizon_seconds", 30)))
        horizon = int(payload.get("horizon_seconds", 30)); layout_record = db.query(VenueLayout).filter(VenueLayout.event_id == event_id).first(); layout = json.loads(layout_record.config_json) if layout_record else None
        try:
            instability_signal = analyze_instability_radar(flow_result)
        except ValueError:
            instability_signal = {"available": False, "reason": "Insufficient stable movement history for instability analysis."}
        plan = build_plan(analyze_state(state, future, event_id, monitoring_id, flow_id, time_machine_id, horizon, layout, instability_signal), horizon)
        plan["instability_signal"] = instability_signal
        plan["venue_layout"] = layout or {"configured": False, "nodes": [], "edges": []}
    except (TypeError, ValueError, KeyError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    plan["communications"] = communications(plan)
    now = utc_now()
    record = ResponsePlan(event_id=event_id, monitoring_session_id=monitoring_id, flow_analysis_id=flow_id, time_machine_session_id=time_machine_id, prediction_horizon=horizon, risk_zone=plan["risk_zone"], risk_level=plan["risk_level"], response_score=plan["recommended"]["response_score"], status="RECOMMENDED", plan_json=json.dumps(plan), created_at=now, updated_at=now)
    db.add(record); db.commit(); db.refresh(record)
    return response_plan_payload(record)


@app.get("/events/{event_id}/response-commander/plans/{plan_id}")
def get_response_plan(event_id: int, plan_id: int, db: Session = Depends(get_db)):
    record = db.query(ResponsePlan).filter(ResponsePlan.id == plan_id, ResponsePlan.event_id == event_id).first()
    if record is None: raise HTTPException(status_code=404, detail="Plan not found")
    return response_plan_payload(record)


@app.post("/events/{event_id}/response-commander/plans/{plan_id}/simulate")
def simulate_response_plan(event_id: int, plan_id: int, payload: dict | None = None, db: Session = Depends(get_db)):
    record = db.query(ResponsePlan).filter(ResponsePlan.id == plan_id, ResponsePlan.event_id == event_id).first()
    if record is None: raise HTTPException(status_code=404, detail="Plan not found")
    data = response_plan_payload(record); selected = data["recommended"]
    if payload and payload.get("scenario"):
        selected = simulate_plan(data, payload["scenario"], int(payload.get("horizon_seconds", 30)), payload.get("what_if"))
    data["simulation"] = selected; data["status"] = "SIMULATED"
    record.status = "SIMULATED"; record.updated_at = utc_now(); record.plan_json = json.dumps(data); db.commit()
    return data


@app.post("/events/{event_id}/response-commander/plans/{plan_id}/approve")
def approve_response_plan(event_id: int, plan_id: int, db: Session = Depends(get_db)):
    record = db.query(ResponsePlan).filter(ResponsePlan.id == plan_id, ResponsePlan.event_id == event_id).first()
    if record is None: raise HTTPException(status_code=404, detail="Plan not found")
    data = response_plan_payload(record)
    if "simulation" not in data: raise HTTPException(status_code=409, detail="Simulate the response before approval")
    record.status = "APPROVED"; record.approved_at = utc_now(); record.updated_at = utc_now(); data["status"] = "APPROVED"; data["approved_at"] = record.approved_at; record.plan_json = json.dumps(data); db.commit()
    return {**data, "approval_note": "Approved inside the application for simulated decision support only; no physical infrastructure was controlled."}


@app.post("/events/{event_id}/response-commander/plans/{plan_id}/what-if")
def response_plan_what_if(event_id: int, plan_id: int, payload: dict, db: Session = Depends(get_db)):
    record = db.query(ResponsePlan).filter(ResponsePlan.id == plan_id, ResponsePlan.event_id == event_id).first()
    if record is None: raise HTTPException(status_code=404, detail="Plan not found")
    data = response_plan_payload(record); data["what_if"] = {"request": payload, "result": simulate_plan(data, data["recommended"]["scenario"], int(payload.get("horizon_seconds", 30)), payload)}
    return data["what_if"]


@app.get("/events/{event_id}/response-commander/plans/{plan_id}/impact")
def response_plan_impact(event_id: int, plan_id: int, scenario: str = "NO_ACTION", db: Session = Depends(get_db)):
    record = db.query(ResponsePlan).filter(ResponsePlan.id == plan_id, ResponsePlan.event_id == event_id).first()
    if record is None: raise HTTPException(status_code=404, detail="Plan not found")
    data = response_plan_payload(record)
    if scenario not in {item.get("scenario") for item in data.get("simulations", [])}: raise HTTPException(status_code=400, detail="Unsupported response scenario")
    return impact_analysis(data, scenario)


@app.get("/events/{event_id}/response-commander/plans/{plan_id}/command-feed")
def response_plan_command_feed(event_id: int, plan_id: int, db: Session = Depends(get_db)):
    record = db.query(ResponsePlan).filter(ResponsePlan.id == plan_id, ResponsePlan.event_id == event_id).first()
    if record is None: raise HTTPException(status_code=404, detail="Plan not found")
    return {"events": command_feed(response_plan_payload(record))}


@app.get("/events/{event_id}/response-commander/plans/{plan_id}/communications")
def response_plan_communications(event_id: int, plan_id: int, db: Session = Depends(get_db)):
    record = db.query(ResponsePlan).filter(ResponsePlan.id == plan_id, ResponsePlan.event_id == event_id).first()
    if record is None: raise HTTPException(status_code=404, detail="Plan not found")
    return response_plan_payload(record).get("communications", {})


# =========================================================
# DOCUMENT ENDPOINTS
# =========================================================

@app.post("/documents")
def create_document(
    document: EventDocumentCreate,
    db: Session = Depends(get_db)
):
    event = (
        db.query(Event)
        .filter(Event.id == document.event_id)
        .first()
    )

    if event is None:
        raise HTTPException(
            status_code=404,
            detail="Event not found"
        )

    new_document = EventDocument(
        event_id=document.event_id,
        document_type=document.document_type,
        document_name=document.document_name,
        status=document.status,
        remarks=document.remarks,
    )

    db.add(new_document)
    db.commit()
    db.refresh(new_document)

    return {
        "message": "Document added successfully",
        "document_id": new_document.id,
        "event_id": new_document.event_id,
        "document_type": new_document.document_type,
        "document_name": new_document.document_name,
        "status": new_document.status,
        "remarks": new_document.remarks,
    }


@app.get("/documents")
def get_documents(
    db: Session = Depends(get_db)
):
    documents = db.query(EventDocument).all()

    return [
        {
            "id": document.id,
            "event_id": document.event_id,
            "document_type": document.document_type,
            "document_name": document.document_name,
            "status": document.status,
            "remarks": document.remarks,
        }
        for document in documents
    ]


@app.get("/events/{event_id}/documents")
def get_event_documents(
    event_id: int,
    db: Session = Depends(get_db)
):
    event = (
        db.query(Event)
        .filter(Event.id == event_id)
        .first()
    )

    if event is None:
        raise HTTPException(
            status_code=404,
            detail="Event not found"
        )

    documents = (
        db.query(EventDocument)
        .filter(
            EventDocument.event_id == event_id
        )
        .all()
    )

    return {
        "event_id": event.id,
        "event_name": event.event_name,
        "documents": [
            {
                "id": document.id,
                "document_type": document.document_type,
                "document_name": document.document_name,
                "status": document.status,
                "remarks": document.remarks,
            }
            for document in documents
        ],
    }


# =========================================================
# NEW: OPEN / VIEW UPLOADED DOCUMENT
# =========================================================

@app.get("/documents/{document_id}/file")
def view_document_file(
    document_id: int,
    db: Session = Depends(get_db)
):
    document = (
        db.query(EventDocument)
        .filter(
            EventDocument.id == document_id
        )
        .first()
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    safe_filename = os.path.basename(
        document.document_name
    )

    file_path = os.path.join(
        "uploads",
        safe_filename
    )

    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=404,
            detail="Uploaded file not found"
        )

    return FileResponse(
        path=file_path,
        filename=safe_filename,
        content_disposition_type="inline",
    )


@app.put("/documents/{document_id}/status")
def update_document_status(
    document_id: int,
    status_update: DocumentStatusUpdate,
    db: Session = Depends(get_db)
):
    document = (
        db.query(EventDocument)
        .filter(
            EventDocument.id == document_id
        )
        .first()
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    allowed_statuses = [
        "Pending",
        "Under Review",
        "Approved",
        "Rejected",
        "Additional Information Required",
    ]

    if status_update.status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid status",
                "allowed_statuses": allowed_statuses,
            }
        )

    document.status = status_update.status
    document.remarks = status_update.remarks

    db.commit()
    db.refresh(document)

    return {
        "message": "Document status updated successfully",
        "document_id": document.id,
        "event_id": document.event_id,
        "document_type": document.document_type,
        "document_name": document.document_name,
        "status": document.status,
        "remarks": document.remarks,
    }


@app.delete("/documents/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    document = (
        db.query(EventDocument)
        .filter(
            EventDocument.id == document_id
        )
        .first()
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    db.delete(document)
    db.commit()

    return {
        "message": "Document deleted successfully",
        "document_id": document_id
    }


@app.post("/documents/upload")
def upload_document(
    event_id: int = Form(...),
    document_type: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    event = (
        db.query(Event)
        .filter(Event.id == event_id)
        .first()
    )

    if event is None:
        raise HTTPException(
            status_code=404,
            detail="Event not found"
        )

    upload_folder = "uploads"
    os.makedirs(
        upload_folder,
        exist_ok=True
    )

    safe_filename = os.path.basename(
        file.filename
    )

    file_path = os.path.join(
        upload_folder,
        safe_filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    new_document = EventDocument(
        event_id=event_id,
        document_type=document_type,
        document_name=safe_filename,
        status="Pending",
        remarks="Document uploaded successfully",
    )

    db.add(new_document)
    db.commit()
    db.refresh(new_document)

    return {
        "message": "Document uploaded successfully",
        "document_id": new_document.id,
        "event_id": new_document.event_id,
        "document_type": new_document.document_type,
        "document_name": new_document.document_name,
        "status": new_document.status,
        "file_path": file_path,
    }


# =========================================================
# ADAPTIVE SAFETY COMMUNICATION CENTER
# =========================================================

def communication_payload(plan, db):
    data = json.loads(plan.source_snapshot_json)
    messages = db.query(CommunicationMessage).filter(CommunicationMessage.communication_plan_id == plan.id).order_by(CommunicationMessage.id.asc()).all()
    sent = [item for item in messages if item.status == "SENT" and item.sent_at and item.channel]
    valid_status = plan.status if plan.status != "SENT" or sent else ("APPROVED" if plan.approved_at else "PENDING_APPROVAL")
    channel_labels = {"EMAIL": "SENT VIA EMAIL", "DASHBOARD": "SENT TO DASHBOARD", "PA_SIMULATION": "PA SIMULATED"}
    data.update({"communication_plan_id": plan.id, "status": valid_status, "delivery_status": channel_labels.get(sent[-1].channel) if sent else ("FAILED" if plan.status == "FAILED" else valid_status), "approved_by": plan.approved_by, "approved_at": plan.approved_at, "history_count": len(sent), "messages": [{"id": item.id, "audience": item.audience, "language": item.language, "message": item.message, "channel": item.channel, "status": item.status, "status_label": channel_labels.get(item.channel, item.status), "approved_message": item.approved_message, "sent_at": item.sent_at} for item in messages]})
    return data


def _communication_sources(event_id, payload, db):
    monitoring_id = payload.get("monitoring_session_id")
    if not isinstance(monitoring_id, int):
        raise HTTPException(status_code=400, detail="Exact monitoring_session_id is required")
    monitoring = db.query(MonitoringSession).filter(MonitoringSession.id == monitoring_id, MonitoringSession.event_id == event_id).first()
    if monitoring is None:
        raise HTTPException(status_code=404, detail="Monitoring session is not part of this event lineage")
    flow_id = payload.get("flow_analysis_id"); flow_record = None; flow_result = None
    if flow_id is not None:
        flow_record = db.query(FlowAnalysisSession).filter(FlowAnalysisSession.id == flow_id, FlowAnalysisSession.event_id == event_id, FlowAnalysisSession.monitoring_session_id == monitoring_id).first()
        if flow_record is None or flow_record.status != "COMPLETE": raise HTTPException(status_code=404, detail="Flow source is not part of this monitoring lineage")
        flow_result = phase3_result_for_session(monitoring) or {}
    time_id = payload.get("time_machine_session_id"); forecast = {}; time_record = None
    if time_id is not None:
        time_record = db.query(TimeMachineSession).filter(TimeMachineSession.id == time_id, TimeMachineSession.event_id == event_id).first()
        if time_record is None or time_record.status != "COMPLETED": raise HTTPException(status_code=404, detail="Time Machine source is not part of this event lineage")
        forecast = json.loads(time_record.result_json or "{}").get("predictions", {})
    response_id = payload.get("response_plan_id"); response_plan = None
    if response_id is not None:
        response_query = db.query(ResponsePlan).filter(ResponsePlan.id == response_id, ResponsePlan.event_id == event_id, ResponsePlan.monitoring_session_id == monitoring_id)
        if flow_id is not None: response_query = response_query.filter(ResponsePlan.flow_analysis_id == flow_id)
        if time_id is not None: response_query = response_query.filter(ResponsePlan.time_machine_session_id == time_id)
        response_record = response_query.first()
        if response_record is None: raise HTTPException(status_code=404, detail="Response Commander plan is not part of this monitoring lineage")
        response_plan = json.loads(response_record.plan_json or "{}")
    return monitoring, flow_record, time_record, flow_result, forecast, response_plan


@app.post("/events/{event_id}/communication/plans")
def create_communication_plan(event_id: int, payload: dict, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if event is None: raise HTTPException(status_code=404, detail="Event not found")
    monitoring, flow_record, time_record, flow_result, forecast, response_plan = _communication_sources(event_id, payload, db)
    radar = response_plan.get("instability_signal") if response_plan else None
    context = build_communication_context(monitoring, flow_result, forecast, response_plan, radar)
    generated = build_communication_plan(context)
    generated["communication_needed"] = bool(generated["priority"] != "INFORMATION" or context.get("recommended_action") != "MONITOR_ONLY")
    generated["lineage"] = {"event_id": event_id, "monitoring_session_id": monitoring.id, "flow_analysis_id": flow_record.id if flow_record else None, "time_machine_session_id": time_record.id if time_record else None, "response_plan_id": payload.get("response_plan_id")}
    now = utc_now(); record = CommunicationPlan(event_id=event_id, monitoring_session_id=monitoring.id, flow_analysis_id=flow_record.id if flow_record else None, time_machine_session_id=time_record.id if time_record else None, response_plan_id=payload.get("response_plan_id"), priority=generated["priority"], intent=generated["intent"], target_zone=context.get("target_zone"), safe_zone=context.get("safe_zone"), status="PENDING_APPROVAL", source_snapshot_json=json.dumps(generated), created_at=now)
    db.add(record); db.commit(); db.refresh(record)
    for item in generated["audiences"] + generated["translations"]:
        db.add(CommunicationMessage(communication_plan_id=record.id, audience=item["audience"], language=item["language"], message=item["message"], status="DRAFT"))
    db.commit()
    return communication_payload(record, db)


@app.get("/events/{event_id}/communication/plans/{plan_id}")
def get_communication_plan(event_id: int, plan_id: int, db: Session = Depends(get_db)):
    record = db.query(CommunicationPlan).filter(CommunicationPlan.id == plan_id, CommunicationPlan.event_id == event_id).first()
    if record is None: raise HTTPException(status_code=404, detail="Communication plan not found")
    return communication_payload(record, db)


@app.post("/events/{event_id}/communication/plans/{plan_id}/approve")
def approve_communication_plan(event_id: int, plan_id: int, payload: dict | None = None, db: Session = Depends(get_db)):
    record = db.query(CommunicationPlan).filter(CommunicationPlan.id == plan_id, CommunicationPlan.event_id == event_id).first()
    if record is None: raise HTTPException(status_code=404, detail="Communication plan not found")
    record.status = "APPROVED"; record.approved_by = (payload or {}).get("approved_by") or "organizer"; record.approved_at = utc_now()
    for item in db.query(CommunicationMessage).filter(CommunicationMessage.communication_plan_id == record.id).all(): item.status = "APPROVED"; item.approved_message = item.message
    db.commit(); return communication_payload(record, db)


@app.post("/events/{event_id}/communication/plans/{plan_id}/reject")
def reject_communication_plan(event_id: int, plan_id: int, db: Session = Depends(get_db)):
    record = db.query(CommunicationPlan).filter(CommunicationPlan.id == plan_id, CommunicationPlan.event_id == event_id).first()
    if record is None: raise HTTPException(status_code=404, detail="Communication plan not found")
    record.status = "CANCELLED"; db.commit(); return communication_payload(record, db)


@app.post("/events/{event_id}/communication/plans/{plan_id}/send")
def send_communication(event_id: int, plan_id: int, payload: dict, db: Session = Depends(get_db)):
    record = db.query(CommunicationPlan).filter(CommunicationPlan.id == plan_id, CommunicationPlan.event_id == event_id).first()
    if record is None: raise HTTPException(status_code=404, detail="Communication plan not found")
    if record.status not in {"APPROVED", "SENT"}: raise HTTPException(status_code=409, detail="Operator approval is required before delivery")
    audience = payload.get("audience"); channel = payload.get("channel", "DASHBOARD"); language = payload.get("language", "en")
    message = db.query(CommunicationMessage).filter(CommunicationMessage.communication_plan_id == plan_id, CommunicationMessage.audience == audience, CommunicationMessage.language == language).first()
    if message is None: raise HTTPException(status_code=404, detail="Audience message not found")
    if channel not in {"DASHBOARD", "PA_SIMULATION", "EMAIL"}: raise HTTPException(status_code=400, detail="Unsupported delivery channel")
    prior = db.query(CommunicationObservation).filter(CommunicationObservation.communication_plan_id == plan_id, CommunicationObservation.observation_type == f"DELIVERY_{channel}").all()
    recipient_key = payload.get("recipient") or payload.get("recipient_email")
    duplicate = next((item for item in prior if (lambda meta: meta.get("audience") == audience and meta.get("language") == language and (channel != "EMAIL" or meta.get("recipient") == recipient_key))(json.loads(item.metrics_json or "{}"))), None)
    if duplicate: raise HTTPException(status_code=409, detail=f"Duplicate delivery blocked for {channel}")
    if channel == "PA_SIMULATION" and payload.get("speech_started") is not True: raise HTTPException(status_code=409, detail="PA simulation was not confirmed by the browser voice engine")
    record.status = "SENDING"; message.channel = channel; message.status = "SENDING"; db.commit()
    try:
        if channel == "EMAIL":
            recipient = str(payload.get("recipient") or payload.get("recipient_email") or "").strip()
            if not recipient: raise HTTPException(status_code=400, detail="A recipient email is required for EMAIL delivery")
            send_monitoring_email_alert(record.monitoring_session_id, {"recipient_email": recipient, "communication_message": message.approved_message or message.message, "communication_priority": record.priority, "communication_subject": "CrowdGuard Safety Communication", "communication_plan_id": record.id, "communication_audience": audience, "communication_language": language}, db)
        if channel == "DASHBOARD":
            db.add(DashboardSafetyAlert(event_id=record.event_id, monitoring_session_id=record.monitoring_session_id, communication_plan_id=record.id, communication_message_id=message.id, audience=audience, priority=record.priority, intent=record.intent, title=f"{audience.title()} Safety Guidance", message=message.message, target_zone=record.target_zone, recommended_action=record.intent, status="ACTIVE", created_at=utc_now(), source_timestamp_sec=json.loads(record.source_snapshot_json).get("source_state", {}).get("timestamp_sec")))
        db.add(CommunicationObservation(communication_plan_id=record.id, observation_type=f"DELIVERY_{channel}", timestamp_sec=None, metrics_json=json.dumps({"audience": audience, "language": language, "channel": channel, "recipient": payload.get("recipient") or payload.get("recipient_email"), "message_id": message.id, "result": "SUCCESS"})))
        message.status = "SENT"; message.sent_at = utc_now(); record.status = "APPROVED"; db.commit()
    except HTTPException:
        message.status = "FAILED"; record.status = "APPROVED"; db.commit(); raise
    except Exception as error:
        message.status = "FAILED"; record.status = "APPROVED"; db.commit(); raise HTTPException(status_code=502, detail=f"{channel} delivery failed: {error}") from error
    return {"status": "APPROVED", "channel": channel, "status_label": {"EMAIL": "SENT VIA EMAIL", "DASHBOARD": "SENT TO DASHBOARD", "PA_SIMULATION": "PA SIMULATED"}[channel], "simulation": channel == "PA_SIMULATION", "message": message.message, "plan": communication_payload(record, db)}


@app.get("/events/{event_id}/communication/plans/{plan_id}/download")
def download_communication_plan_pdf(event_id: int, plan_id: int, audience: str = "PUBLIC", language: str = "en", db: Session = Depends(get_db)):
    record = db.query(CommunicationPlan).filter(CommunicationPlan.id == plan_id, CommunicationPlan.event_id == event_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail="Communication plan not found")
    if record.status not in {"APPROVED", "SENT"}:
        raise HTTPException(status_code=409, detail="Operator approval is required before downloading the safety brief")
    selected = db.query(CommunicationMessage).filter(CommunicationMessage.communication_plan_id == plan_id, CommunicationMessage.audience == audience, CommunicationMessage.language == language).first()
    if selected is None or not (selected.approved_message or selected.status in {"APPROVED", "SENT"}):
        raise HTTPException(status_code=404, detail="Approved audience message not found")
    event = db.query(Event).filter(Event.id == event_id).first()
    plan_data = communication_payload(record, db)
    content = build_safety_brief_pdf(event, plan_data, selected.approved_message or selected.message, audience=audience, language=language)
    filename = safety_brief_filename(event_id, plan_id)
    return Response(content=content, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/events/{event_id}/communication/history")
def communication_history(event_id: int, monitoring_session_id: int | None = None, db: Session = Depends(get_db)):
    query = db.query(CommunicationPlan).filter(CommunicationPlan.event_id == event_id).order_by(CommunicationPlan.id.desc())
    if monitoring_session_id is not None: query = query.filter(CommunicationPlan.monitoring_session_id == monitoring_session_id)
    return {"plans": [communication_payload(item, db) for item in query.limit(30).all()]}


@app.get("/events/{event_id}/communication/plans/{plan_id}/effectiveness")
def communication_effectiveness(event_id: int, plan_id: int, db: Session = Depends(get_db)):
    record = db.query(CommunicationPlan).filter(CommunicationPlan.id == plan_id, CommunicationPlan.event_id == event_id).first()
    if record is None: raise HTTPException(status_code=404, detail="Communication plan not found")
    observations = db.query(CommunicationObservation).filter(CommunicationObservation.communication_plan_id == plan_id).order_by(CommunicationObservation.id.asc()).all()
    return {"plan_id": plan_id, "status": "INSUFFICIENT FOLLOW-UP DATA" if len(observations) < 2 else "OBSERVED CHANGE", "observations": [{"type": item.observation_type, "timestamp_sec": item.timestamp_sec, "metrics": json.loads(item.metrics_json)} for item in observations], "causality_note": "Observed changes are reported after communication; they are not claimed as caused by it."}


def _dashboard_alert_payload(alert):
    return {"id": alert.id, "event_id": alert.event_id, "monitoring_session_id": alert.monitoring_session_id, "communication_plan_id": alert.communication_plan_id, "audience": alert.audience, "priority": alert.priority, "intent": alert.intent, "title": alert.title, "message": alert.message, "target_zone": alert.target_zone, "recommended_action": alert.recommended_action, "status": alert.status, "created_at": alert.created_at, "read_at": alert.read_at, "dismissed_at": alert.dismissed_at, "source_timestamp_sec": alert.source_timestamp_sec}


@app.get("/events/{event_id}/dashboard-safety-alerts")
def get_dashboard_safety_alerts(event_id: int, active_only: bool = True, db: Session = Depends(get_db)):
    query = db.query(DashboardSafetyAlert).filter(DashboardSafetyAlert.event_id == event_id).order_by(DashboardSafetyAlert.id.desc())
    if active_only: query = query.filter(DashboardSafetyAlert.status == "ACTIVE")
    return {"alerts": [_dashboard_alert_payload(item) for item in query.limit(5).all()], "unread_count": db.query(DashboardSafetyAlert).filter(DashboardSafetyAlert.event_id == event_id, DashboardSafetyAlert.status == "ACTIVE").count()}


@app.get("/dashboard-safety-alerts")
def get_all_dashboard_safety_alerts(active_only: bool = True, db: Session = Depends(get_db)):
    query = db.query(DashboardSafetyAlert).order_by(DashboardSafetyAlert.id.desc())
    if active_only: query = query.filter(DashboardSafetyAlert.status == "ACTIVE")
    return {"alerts": [_dashboard_alert_payload(item) for item in query.limit(5).all()], "unread_count": db.query(DashboardSafetyAlert).filter(DashboardSafetyAlert.status == "ACTIVE").count()}


@app.patch("/dashboard-alerts/{alert_id}/read")
def mark_dashboard_alert_read(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(DashboardSafetyAlert).filter(DashboardSafetyAlert.id == alert_id).first()
    if alert is None: raise HTTPException(status_code=404, detail="Dashboard alert not found")
    alert.status = "READ"; alert.read_at = utc_now(); db.commit(); return _dashboard_alert_payload(alert)


@app.patch("/dashboard-alerts/{alert_id}/dismiss")
def dismiss_dashboard_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(DashboardSafetyAlert).filter(DashboardSafetyAlert.id == alert_id).first()
    if alert is None: raise HTTPException(status_code=404, detail="Dashboard alert not found")
    alert.status = "DISMISSED"; alert.dismissed_at = utc_now(); db.commit(); return _dashboard_alert_payload(alert)
