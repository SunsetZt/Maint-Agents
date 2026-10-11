"""Pure domain types and invariants for the industrial operations platform."""

from datetime import UTC, datetime


def as_utc(value: datetime) -> datetime:
    """Treat naive database timestamps as UTC; convert existing timezones to UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
