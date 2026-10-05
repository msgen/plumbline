"""Sliding-window rate limiter with injectable time and sleep."""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        max_calls: int,
        period: float = 60.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_calls, self.period = max_calls, period
        self._mono, self._sleep = monotonic, sleep
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        now = self._mono()
        while self._calls and now - self._calls[0] >= self.period:
            self._calls.popleft()
        if len(self._calls) >= self.max_calls:
            self._sleep(self.period - (now - self._calls[0]))
            now = self._mono()
            while self._calls and now - self._calls[0] >= self.period:
                self._calls.popleft()
        self._calls.append(now)
