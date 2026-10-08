"""Injected time source. Nothing else may read the system clock."""
from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
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


def is_early_close(day: date) -> bool:
    """NYSE 13:00 closes: the day after Thanksgiving, Christmas Eve, and July 3.

    Rule-based for weekdays only. Days the market is shut have no bars, so they never matter.
    """
    if day.weekday() >= 5:
        return False
    if day.month == 11 and day.weekday() == 4:  # Friday after the fourth Thursday
        return 23 <= day.day <= 29
    if day.month == 12 and day.day == 24:
        return True
    if day.month == 7 and day.day == 3:  # only if July 4 is also a weekday
        return date(day.year, 7, 4).weekday() < 5
    return False


def regular_close_minute(day: date) -> int:
    """Minute of the New York day (from midnight) at which the regular session ends."""
    return 13 * 60 if is_early_close(day) else 16 * 60


def month_starts(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Split [start, end) into calendar-month windows (UTC)."""
    out, cur = [], start
    while cur < end:
        nxt = datetime(cur.year + (cur.month == 12), cur.month % 12 + 1, 1, tzinfo=UTC)
        out.append((cur, min(nxt, end)))
        cur = nxt
    return out
