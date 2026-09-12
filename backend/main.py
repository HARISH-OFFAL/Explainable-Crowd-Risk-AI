import os
import shutil
import json
import threading
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from .database import Base, SessionLocal, engine, get_db
from .models import Event, EventDocument, MonitoringSession, Phase2RiskRecord, FlowAnalysisSession, TimeMachineSession, TimeMachineSnapshot, ResponsePlan, VenueLayout, InstabilitySnapshot
from .schemas import (
    EventCreate,
    EventUpdate,
    EventDocumentCreate,
    DocumentStatusUpdate,
    MonitoringSessionCreate,
    Phase2RiskUpdate,
)
from .risk_engine import calculate_pre_event_risk
from .phase2_web_engine import Phase2WebEngine, normalized_crowd_scale, load_detector
from phase3.flow_service import FLOW_CONFIG, analyze_recorded_video, load_cached_flow, save_cached_flow
from phase3.digital_twin.state_builder import build_current_state
from phase3.digital_twin.scenario_service import build_scenarios
from phase3.digital_twin.simulation_engine import CONFIG, simulate
from phase3.time_machine_service import analyze_time_machine, HORIZONS
from phase4.response_commander import analyze_state, build_plan, simulate_plan, communications, venue_graph, impact_analysis, command_feed
from phase4.schemas import VenueLayout as VenueLayoutSchema
from phase4.instability_radar import analyze as analyze_instability_radar


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


app = FastAPI(
    title="Explainable Crowd Risk AI",
    description="Crowd safety and early risk prediction system",
    version="0.1.0",
)


@app.on_event("startup")
def warm_crowd_detector():
    """Load YOLO once at server startup so the first monitoring click is fast."""
    load_detector()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE2_VIDEOS = PROJECT_ROOT / "phase2" / "videos"
WEB_MONITOR_ENGINES = {}
WEB_MONITOR_PERSISTED_WINDOWS = {}
PHASE3_ANALYSES = {}
TIME_MACHINE_CACHE = {}
ZONES = ("ZONE_A", "ZONE_B", "ZONE_C")


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
    }
    session.final_snapshot_json = json.dumps(packet)
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
