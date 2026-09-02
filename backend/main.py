from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import Event
from .schemas import EventCreate


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