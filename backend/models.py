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