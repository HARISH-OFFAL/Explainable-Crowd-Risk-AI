import os
import shutil

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import Event, EventDocument
from .schemas import (
    EventCreate,
    EventUpdate,
    EventDocumentCreate,
    DocumentStatusUpdate,
)
from .risk_engine import calculate_pre_event_risk


Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="Explainable Crowd Risk AI",
    description="Crowd safety and early risk prediction system",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "message": "Explainable Crowd Risk AI API is running"
    }


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
def get_events(db: Session = Depends(get_db)):
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
    event = db.query(Event).filter(
        Event.id == event_id
    ).first()

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
    event = db.query(Event).filter(
        Event.id == event_id
    ).first()

    if event is None:
        raise HTTPException(
            status_code=404,
            detail="Event not found"
        )

    event.event_name = updated_event.event_name
    event.location = updated_event.location
    event.event_datetime = updated_event.event_datetime
    event.expected_crowd_size = updated_event.expected_crowd_size
    event.venue_capacity = updated_event.venue_capacity
    event.entry_gates = updated_event.entry_gates
    event.exit_gates = updated_event.exit_gates
    event.emergency_exits = updated_event.emergency_exits
    event.event_duration_minutes = updated_event.event_duration_minutes

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


@app.post("/documents")
def create_document(
    document: EventDocumentCreate,
    db: Session = Depends(get_db)
):
    event = db.query(Event).filter(
        Event.id == document.event_id
    ).first()

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
def get_documents(db: Session = Depends(get_db)):
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
    event = db.query(Event).filter(
        Event.id == event_id
    ).first()

    if event is None:
        raise HTTPException(
            status_code=404,
            detail="Event not found"
        )

    documents = db.query(EventDocument).filter(
        EventDocument.event_id == event_id
    ).all()

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


@app.put("/documents/{document_id}/status")
def update_document_status(
    document_id: int,
    status_update: DocumentStatusUpdate,
    db: Session = Depends(get_db)
):
    document = db.query(EventDocument).filter(
        EventDocument.id == document_id
    ).first()

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


@app.post("/documents/upload")
def upload_document(
    event_id: int = Form(...),
    document_type: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    event = db.query(Event).filter(
        Event.id == event_id
    ).first()

    if event is None:
        raise HTTPException(
            status_code=404,
            detail="Event not found"
        )

    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(
        upload_folder,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    new_document = EventDocument(
        event_id=event_id,
        document_type=document_type,
        document_name=file.filename,
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