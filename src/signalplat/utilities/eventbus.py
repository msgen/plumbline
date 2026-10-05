"""In-process publish/subscribe. The only upward path (signals -> dashboard, notifier)."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        self._subs[topic].append(handler)

    def publish(self, topic: str, event: Any) -> None:
        for handler in list(self._subs[topic]):
            handler(event)
