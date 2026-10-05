"""Tiny JSON-over-HTTP client with retry. Transport is injectable for recorded-HTTP tests."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

Transport = Callable[[str, dict[str, str]], tuple[int, bytes]]


def urllib_transport(url: str, headers: dict[str, str]) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers=headers)  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class HttpError(RuntimeError):
    def __init__(self, status: int, url: str, body: bytes) -> None:
        super().__init__(f"HTTP {status} for {url}: {body[:200]!r}")
        self.status = status


class JsonHttp:
    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        transport: Transport = urllib_transport,
        retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Accept": "application/json", **(headers or {})}
        self._transport, self._retries, self._sleep = transport, retries, sleep

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        for attempt in range(self._retries + 1):
            status, body = self._transport(url, self.headers)
            if status == 200:
                return json.loads(body)
            retryable = status == 429 or status >= 500
            if not retryable or attempt == self._retries:
                raise HttpError(status, url, body)
            self._sleep(2.0**attempt)
        raise AssertionError("unreachable")
