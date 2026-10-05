from datetime import UTC, datetime

import pandas as pd
import pytest

from signalplat.utilities.pit import FramePointInTimeView, poison_after

AS_OF = datetime(2026, 7, 6, 14, 0, tzinfo=UTC)


def _bars():
    ts = pd.date_range("2026-07-06 13:30", periods=60, freq="min", tz="UTC")
    return pd.DataFrame({"available_at": ts, "close": range(60), "volume": 100})


def test_view_hides_future_records():
    v = FramePointInTimeView({"bars": _bars()}, AS_OF)
    assert v.get("bars")["available_at"].max() <= AS_OF


def test_poisoning_future_data_does_not_change_the_view():
    clean = FramePointInTimeView({"bars": _bars()}, AS_OF).get("bars")
    poisoned = FramePointInTimeView({"bars": poison_after(_bars(), AS_OF)}, AS_OF).get("bars")
    pd.testing.assert_frame_equal(clean, poisoned)


def test_naive_as_of_and_missing_available_at_rejected():
    with pytest.raises(ValueError):
        FramePointInTimeView({"bars": _bars()}, datetime(2026, 7, 6, 14, 0))
    with pytest.raises(ValueError):
        FramePointInTimeView({"bars": pd.DataFrame({"x": [1]})}, AS_OF)
