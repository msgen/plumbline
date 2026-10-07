"""DTOs and Protocol interfaces only, no logic."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

import pandas as pd

# Per symbol-day facts about stored minute bars (the table behind the E0 audit).
MINUTE_SUMMARY_COLUMNS = [
    "symbol", "day", "rows", "timestamps", "reg_minutes", "reg_first", "reg_last",
    "vol_all", "vol_regular", "outside",
]


class Feed(StrEnum):
    SIP = "sip"
    IEX = "iex"


@dataclass(frozen=True)
class Candidate:
    symbol: str
    strategy: str  # "A_breakout" | "B_vwap_reclaim" | "C_catalyst_gap"
    as_of: datetime  # UTC, when the setup became knowable
    ref_price: float
    stop: float
    targets: tuple[float, ...]


@dataclass(frozen=True)
class NewsItem:
    id: str
    created_at: datetime
    available_at: datetime
    headline: str
    summary: str
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class ClassifierSpec:
    name: str
    version: str
    classes: tuple[str, ...]


@dataclass(frozen=True)
class ClassifierResult:
    item_id: str
    model_version: str
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketContext:
    as_of: datetime
    spy_vs_vwap: float | None = None
    vix: float | None = None


class PointInTimeView(Protocol):
    """Read-only view of records with available_at <= as_of."""

    as_of: datetime

    def get(self, name: str) -> pd.DataFrame: ...


class BarsAccessor(Protocol):
    def minute_bars(
        self, symbols: Sequence[str], start: datetime, end: datetime, feed: Feed
    ) -> pd.DataFrame: ...


class CatalystClassifier(Protocol):
    def classify(
        self, items: Sequence[NewsItem], spec: ClassifierSpec
    ) -> list[ClassifierResult]: ...


class SetupEngine(Protocol):
    def detect(
        self, view: PointInTimeView, ctx: MarketContext, as_of: datetime
    ) -> list[Candidate]: ...


class ReferenceAccessor(Protocol):
    def assets(self) -> pd.DataFrame:
        """Active and inactive US equities: symbol, name, exchange, status, tradable."""
        ...


class NewsAccessor(Protocol):
    def news(self, symbols: Sequence[str], start: datetime, end: datetime) -> list[NewsItem]: ...


class FilingsAccessor(Protocol):
    def filings(self, symbols: Sequence[str], start: datetime, end: datetime) -> pd.DataFrame:
        """symbol, cik, accession, form, items, accepted_at, available_at, primary_document."""
        ...
