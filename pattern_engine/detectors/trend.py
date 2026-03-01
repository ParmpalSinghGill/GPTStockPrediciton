from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from pattern_engine.config import Config


class TrendDetector:
    """Fits short-term and long-term linear trend lines on close prices."""

    def detect(self, ohlc: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        close = ohlc["Close"].astype(float).dropna()
        if close.empty:
            return self._empty()

        rows = []
        short = self._best_fit(close, int(cfg.short_trend_min_bars), int(cfg.short_trend_max_bars), "short_term")
        if short is not None:
            rows.append(short)
        long = self._best_fit(close, int(cfg.long_trend_min_bars), int(cfg.long_trend_max_bars), "long_term")
        if long is not None:
            rows.append(long)

        if not rows:
            return self._empty()
        return pd.DataFrame(rows)

    def _best_fit(
        self,
        close: pd.Series,
        min_bars: int,
        max_bars: int,
        trend_horizon: str,
    ) -> dict | None:
        n = len(close)
        if n < 2 or min_bars < 2:
            return None

        lo = max(2, min_bars)
        hi = min(max_bars, n)
        if hi < lo:
            return None

        best: Tuple[float, int, float, float] | None = None
        close_np = close.to_numpy(dtype=float)

        for w in range(lo, hi + 1):
            y = close_np[n - w :]
            x = np.arange(w, dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            y_hat = intercept + slope * x
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 if ss_tot <= 1e-12 else max(0.0, 1.0 - (ss_res / ss_tot))

            if best is None or r2 > best[0]:
                best = (r2, w, float(slope), float(intercept))

        if best is None:
            return None

        r2, window, slope, intercept = best
        start_idx = n - window
        end_idx = n - 1
        start_price = float(intercept)
        end_price = float(intercept + slope * (window - 1))
        mean_price = float(np.mean(close_np[start_idx : end_idx + 1]))
        slope_pct_per_bar = 0.0 if abs(mean_price) <= 1e-12 else (slope / mean_price) * 100.0

        direction = "sideways"
        if slope_pct_per_bar > 0.01:
            direction = "up"
        elif slope_pct_per_bar < -0.01:
            direction = "down"

        return {
            "trend_horizon": trend_horizon,
            "window_bars": int(window),
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "start_date": close.index[start_idx],
            "end_date": close.index[end_idx],
            "start_price": float(start_price),
            "end_price": float(end_price),
            "slope": float(slope),
            "slope_pct_per_bar": float(slope_pct_per_bar),
            "r2": float(r2),
            "direction": direction,
        }

    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "trend_horizon",
                "window_bars",
                "start_idx",
                "end_idx",
                "start_date",
                "end_date",
                "start_price",
                "end_price",
                "slope",
                "slope_pct_per_bar",
                "r2",
                "direction",
            ]
        )
