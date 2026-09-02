from pydantic import BaseModel, Field


class EventCreate(BaseModel):
    event_name: str = Field(min_length=1, max_length=200)
    location: str = Field(min_length=1, max_length=300)
    event_datetime: str

    expected_crowd_size: int = Field(gt=0)
    venue_capacity: int = Field(gt=0)

    entry_gates: int = Field(ge=0)
    exit_gates: int = Field(ge=0)
    emergency_exits: int = Field(ge=0)

    event_duration_minutes: int = Field(gt=0)