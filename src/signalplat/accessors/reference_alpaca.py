"""Alpaca asset list (active and inactive) as a contract frame."""
from __future__ import annotations

import pandas as pd

from signalplat.utilities.http import JsonHttp

ASSET_COLUMNS = ["symbol", "name", "exchange", "status", "tradable", "asset_class"]


class AlpacaReference:
    def __init__(self, http: JsonHttp) -> None:
        self._http = http

    def assets(self) -> pd.DataFrame:
        rows: list[dict] = []
        for status in ("active", "inactive"):
            rows += self._http.get("/v2/assets", {"status": status, "asset_class": "us_equity"})
        if not rows:
            return pd.DataFrame(columns=ASSET_COLUMNS)
        df = pd.DataFrame(rows)
        df["asset_class"] = df.get("class", "us_equity")
        for col in ASSET_COLUMNS:
            if col not in df:
                df[col] = None
        return df[ASSET_COLUMNS].drop_duplicates("symbol").reset_index(drop=True)
