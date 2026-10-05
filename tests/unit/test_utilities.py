from datetime import UTC, datetime, timedelta

import pytest

from signalplat.utilities.clock import (
    FixedClock,
    clamp_sip_end,
    is_regular_session,
    to_ny,
)
from signalplat.utilities.config import load_config
from signalplat.utilities.eventbus import EventBus
from signalplat.utilities.ids import hash_config
from signalplat.utilities.ratelimit import RateLimiter


def test_regular_session_follows_new_york_dst():
    # 13:30 UTC is 09:30 NY in summer (EDT) but only 08:30 NY in winter (EST)
    assert is_regular_session(datetime(2026, 7, 6, 13, 30, tzinfo=UTC))
    assert not is_regular_session(datetime(2026, 1, 5, 13, 30, tzinfo=UTC))
    assert is_regular_session(datetime(2026, 1, 5, 14, 30, tzinfo=UTC))
    assert not is_regular_session(datetime(2026, 7, 4, 15, 0, tzinfo=UTC))  # Saturday
    assert to_ny(datetime(2026, 7, 6, 13, 30, tzinfo=UTC)).hour == 9


def test_clamp_sip_end():
    now = datetime(2026, 7, 6, 15, 0, tzinfo=UTC)
    assert clamp_sip_end(now, now) == now - timedelta(minutes=15)
    early = now - timedelta(hours=2)
    assert clamp_sip_end(early, now) == early


def test_fixed_clock_requires_tz_and_advances():
    with pytest.raises(ValueError):
        FixedClock(datetime(2026, 1, 1))  # noqa: DTZ001
    c = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
    c.advance(timedelta(seconds=30))
    assert c.now() == datetime(2026, 1, 1, 0, 0, 30, tzinfo=UTC)


def test_rate_limiter_sleeps_when_window_full():
    t = [0.0]
    slept = []
    rl = RateLimiter(2, 60.0, monotonic=lambda: t[0],
                     sleep=lambda s: (slept.append(s), t.__setitem__(0, t[0] + s)))
    rl.acquire()
    rl.acquire()
    assert slept == []
    rl.acquire()
    assert slept == [60.0]


def test_eventbus_and_hash_and_config(tmp_path):
    bus, got = EventBus(), []
    bus.subscribe("signal", got.append)
    bus.publish("signal", 1)
    bus.publish("other", 2)
    assert got == [1]
    assert hash_config({"a": 1, "b": 2}) == hash_config({"b": 2, "a": 1})
    p = tmp_path / "c.yaml"
    p.write_text("x: 1\n")
    assert load_config(p) == {"x": 1}
    p.write_text("- 1\n")
    with pytest.raises(TypeError):
        load_config(p)


def test_repo_configs_load():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for f in (root / "config").glob("*.yaml"):
        load_config(f)
