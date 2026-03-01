from __future__ import annotations

from typing import List

import pandas as pd

from pattern_engine.config import Config
from pattern_engine.pivots import collect_pivots


class DoublePatternDetector:
    def detect(self, ohlc: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        piv = collect_pivots(ohlc, cfg.pivot_window, cfg.level_lookback_bars)
        if len(piv) < 3:
            return pd.DataFrame(
                columns=[
                    "pattern",
                    "first_date",
                    "second_date",
                    "neckline_date",
                    "first_price",
                    "second_price",
                    "neckline_price",
                ]
            )

        out: List[dict] = []
        tol = 0.02
        close = ohlc["Close"]
        for i in range(len(piv) - 2):
            a, b, c = piv.iloc[i], piv.iloc[i + 1], piv.iloc[i + 2]
            if a["kind"] == "H" and b["kind"] == "L" and c["kind"] == "H":
                if abs(float(a["price"]) - float(c["price"])) / max(float(a["price"]), 1e-12) <= tol:
                    confirmed = bool((close.loc[close.index > c["date"]] < float(b["price"])).any())
                    out.append(
                        {
                            "pattern": "DoubleTop",
                            "first_date": a["date"],
                            "second_date": c["date"],
                            "neckline_date": b["date"],
                            "first_price": float(a["price"]),
                            "second_price": float(c["price"]),
                            "neckline_price": float(b["price"]),
                            "confirmed": confirmed,
                        }
                    )
            if a["kind"] == "L" and b["kind"] == "H" and c["kind"] == "L":
                if abs(float(a["price"]) - float(c["price"])) / max(float(a["price"]), 1e-12) <= tol:
                    confirmed = bool((close.loc[close.index > c["date"]] > float(b["price"])).any())
                    out.append(
                        {
                            "pattern": "DoubleBottom",
                            "first_date": a["date"],
                            "second_date": c["date"],
                            "neckline_date": b["date"],
                            "first_price": float(a["price"]),
                            "second_price": float(c["price"]),
                            "neckline_price": float(b["price"]),
                            "confirmed": confirmed,
                        }
                    )
        return pd.DataFrame(out)
