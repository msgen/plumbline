"""Injected time source. Nothing else may read the system clock."""
from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FixedClock:
    """Replay/test clock; advance explicitly."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("clock time must be timezone-aware")
        self._now = start.astimezone(UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta


def to_ny(ts: datetime) -> datetime:
    return ts.astimezone(NY)


def is_regular_session(ts: datetime) -> bool:
    """True for weekday timestamps inside 09:30-16:00 New York (holidays not handled)."""
    local = to_ny(ts)
    return local.weekday() < 5 and REGULAR_OPEN <= local.time() < REGULAR_CLOSE


def clamp_sip_end(end: datetime, now: datetime, minutes: int = 15) -> datetime:
    """Free-tier SIP queries must end at least `minutes` in the past."""
    return min(end, now - timedelta(minutes=minutes))
