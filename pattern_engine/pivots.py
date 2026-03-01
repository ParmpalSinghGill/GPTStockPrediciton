from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def collect_pivots(ohlc: pd.DataFrame, w: int, lookback_bars: int) -> pd.DataFrame:
    d = ohlc.tail(lookback_bars).copy()
    highs = d["High"].to_numpy(dtype=float)
    lows = d["Low"].to_numpy(dtype=float)
    idx = d.index
    offset = len(ohlc) - len(d)
    rows: List[dict] = []
    if len(d) < 2 * w + 1:
        return pd.DataFrame(columns=["idx", "date", "kind", "price"])
    for i in range(w, len(d) - w):
        if lows[i] == np.min(lows[i - w : i + w + 1]):
            rows.append({"idx": offset + i, "date": idx[i], "kind": "L", "price": float(lows[i])})
        if highs[i] == np.max(highs[i - w : i + w + 1]):
            rows.append({"idx": offset + i, "date": idx[i], "kind": "H", "price": float(highs[i])})
    if not rows:
        return pd.DataFrame(columns=["idx", "date", "kind", "price"])
    return pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
