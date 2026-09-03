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


class EventUpdate(BaseModel):
    event_name: str = Field(min_length=1, max_length=200)
    location: str = Field(min_length=1, max_length=300)
    event_datetime: str

    expected_crowd_size: int = Field(gt=0)
    venue_capacity: int = Field(gt=0)

    entry_gates: int = Field(ge=0)
    exit_gates: int = Field(ge=0)
    emergency_exits: int = Field(ge=0)

    event_duration_minutes: int = Field(gt=0)


class EventDocumentCreate(BaseModel):
    event_id: int = Field(gt=0)

    document_type: str = Field(
        min_length=1,
        max_length=100
    )

    document_name: str = Field(
        min_length=1,
        max_length=255
    )

    status: str = Field(
        default="Pending",
        min_length=1,
        max_length=50
    )

    remarks: str | None = Field(
        default=None,
        max_length=500
    )


class DocumentStatusUpdate(BaseModel):
    status: str = Field(
        min_length=1,
        max_length=50
    )

    remarks: str | None = Field(
        default=None,
        max_length=500
    )