"""Typed Phase 4 boundary models for Phase 2/3 analytics."""
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


def number(value: Any, field: str = "numeric field") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, received boolean")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError as error:
            raise ValueError(f"{field} must be numeric, received {value!r}") from error
    raise ValueError(f"{field} must be numeric, received {type(value).__name__}")


class ZoneState(BaseModel):
    model_config = ConfigDict(extra="allow")
    current_people: float = Field(default=0, ge=0)
    capacity: float | None = Field(default=None, ge=0)
    inflow_rate: float = Field(default=0, ge=0)
    outflow_rate: float = Field(default=0, ge=0)
    net_flow: float = 0
    relative_speed: float = Field(default=0, ge=0)

    @field_validator("current_people", "capacity", "inflow_rate", "outflow_rate", "net_flow", "relative_speed", mode="before")
    @classmethod
    def coerce_numeric(cls, value, info):
        return None if value is None and info.field_name == "capacity" else number(value, f"zones.*.{info.field_name}")


class CurrentResponseState(BaseModel):
    model_config = ConfigDict(extra="allow")
    event_id: int | None = None
    monitoring_session_id: int | None = None
    flow_analysis_id: int | None = None
    total_people: float = Field(default=0, ge=0)
    zones: dict[str, ZoneState]

    @field_validator("total_people", mode="before")
    @classmethod
    def coerce_total(cls, value):
        return number(value, "total_people")


class PredictionCounts(BaseModel):
    model_config = ConfigDict(extra="allow")
    zone_counts: dict[str, float]
    total_people: float = Field(default=0, ge=0)

    @field_validator("total_people", mode="before")
    @classmethod
    def coerce_total(cls, value):
        return number(value, "prediction.total_people")

    @field_validator("zone_counts", mode="before")
    @classmethod
    def coerce_counts(cls, value):
        if not isinstance(value, dict):
            raise ValueError("prediction.zone_counts must be an object")
        return {zone: number(count, f"prediction.zone_counts.{zone}") for zone, count in value.items()}


def normalize_state(value: dict) -> dict:
    """Validate and return a plain numeric state for arithmetic."""
    return CurrentResponseState.model_validate(value).model_dump()


def normalize_prediction(value: dict | None) -> dict:
    if not value:
        return {"zone_counts": {}}
    return PredictionCounts.model_validate(value).model_dump()


class VenueNode(BaseModel):
    id: str
    name: str
    node_type: str
    x_normalized: float = Field(ge=0, le=1)
    y_normalized: float = Field(ge=0, le=1)
    capacity: float | None = Field(default=None, ge=0)
    status: str = "OPEN"


class VenueEdge(BaseModel):
    source: str
    target: str
    distance_weight: float = Field(default=1, gt=0)
    capacity: float | None = Field(default=None, ge=0)
    blocked: bool = False
    emergency_reserved: bool = False


class VenueLayout(BaseModel):
    name: str = "Event venue layout"
    nodes: list[VenueNode] = Field(default_factory=list)
    edges: list[VenueEdge] = Field(default_factory=list)
