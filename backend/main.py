import os
import shutil
import json
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
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import Event, EventDocument, MonitoringSession, Phase2RiskRecord
from .schemas import (
    EventCreate,
    EventUpdate,
    EventDocumentCreate,
    DocumentStatusUpdate,
    MonitoringSessionCreate,
    Phase2RiskUpdate,
)
from .risk_engine import calculate_pre_event_risk
from .phase2_web_engine import Phase2WebEngine


Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="Explainable Crowd Risk AI",
    description="Crowd safety and early risk prediction system",
    version="0.1.0",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE2_VIDEOS = PROJECT_ROOT / "phase2" / "videos"
WEB_MONITOR_ENGINES = {}
WEB_MONITOR_PERSISTED_WINDOWS = {}


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
    context = {"id": event.id, "event_name": event.event_name, "location": event.location, "pre_event_risk": {"risk_level": risk["risk_level"], "risk_score": risk["risk_score"]}}
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


@app.get("/monitoring-sessions/{session_id}/analytics")
def get_live_monitoring_analytics(session_id: int, db: Session = Depends(get_db)):
    engine = WEB_MONITOR_ENGINES.get(session_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="Browser monitoring engine is not running")
    analytics = engine.analytics()
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
        engine.stop()
    session.status = "STOPPED"
    session.ended_at = utc_now()
    db.commit()
    db.refresh(session)
    return {"message": "Monitoring session stopped", "session": serialize_session(session)}


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
