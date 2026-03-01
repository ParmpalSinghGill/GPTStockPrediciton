from __future__ import annotations

from typing import List

import pandas as pd

from pattern_engine.config import Config
from pattern_engine.pivots import collect_pivots


class HeadShouldersDetector:
    def detect(self, ohlc: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        piv = collect_pivots(ohlc, cfg.pivot_window, cfg.level_lookback_bars)
        if len(piv) < 5:
            return pd.DataFrame(
                columns=[
                    "pattern",
                    "left_shoulder_date",
                    "head_date",
                    "right_shoulder_date",
                    "neckline_price",
                    "confirmed",
                ]
            )
        out: List[dict] = []
        close = ohlc["Close"]
        for i in range(len(piv) - 4):
            a, b, c, d, e = piv.iloc[i : i + 5].itertuples(index=False)
            if a.kind == "H" and b.kind == "L" and c.kind == "H" and d.kind == "L" and e.kind == "H":
                ls, hd, rs = float(a.price), float(c.price), float(e.price)
                if hd > max(ls, rs) * 1.03 and abs(ls - rs) / max((ls + rs) / 2, 1e-12) <= 0.03:
                    neckline = (float(b.price) + float(d.price)) / 2.0
                    confirmed = bool((close.loc[close.index > e.date] < neckline).any())
                    out.append(
                        {
                            "pattern": "HeadAndShoulders",
                            "left_shoulder_date": a.date,
                            "head_date": c.date,
                            "right_shoulder_date": e.date,
                            "left_shoulder_price": ls,
                            "head_price": hd,
                            "right_shoulder_price": rs,
                            "neckline_price": neckline,
                            "confirmed": confirmed,
                        }
                    )
            if a.kind == "L" and b.kind == "H" and c.kind == "L" and d.kind == "H" and e.kind == "L":
                ls, hd, rs = float(a.price), float(c.price), float(e.price)
                if hd < min(ls, rs) * 0.97 and abs(ls - rs) / max((ls + rs) / 2, 1e-12) <= 0.03:
                    neckline = (float(b.price) + float(d.price)) / 2.0
                    confirmed = bool((close.loc[close.index > e.date] > neckline).any())
                    out.append(
                        {
                            "pattern": "InverseHeadAndShoulders",
                            "left_shoulder_date": a.date,
                            "head_date": c.date,
                            "right_shoulder_date": e.date,
                            "left_shoulder_price": ls,
                            "head_price": hd,
                            "right_shoulder_price": rs,
                            "neckline_price": neckline,
                            "confirmed": confirmed,
                        }
                    )
        return pd.DataFrame(out)
