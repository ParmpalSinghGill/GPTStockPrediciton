from __future__ import annotations

from typing import List

import pandas as pd

from pattern_engine.config import Config
from pattern_engine.pivots import collect_pivots


class RectanglePatternDetector:
    def detect(self, ohlc: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        piv = collect_pivots(ohlc, cfg.pivot_window, cfg.level_lookback_bars)
        if len(piv) < 6:
            return pd.DataFrame(
                columns=[
                    "start_idx",
                    "end_idx",
                    "start_date",
                    "end_date",
                    "upper_level",
                    "lower_level",
                    "top_touches",
                    "bottom_touches",
                    "outside_bars",
                ]
            )

        close_last = float(ohlc["Close"].iloc[-1])
        tol_abs = close_last * float(cfg.level_tolerance_pct)
        tol_pct = float(cfg.level_tolerance_pct)
        min_bars = int(cfg.rectangle_min_bars)
        max_bars = int(cfg.rectangle_max_bars)
        min_touches = int(cfg.rectangle_min_touches)
        max_outside = int(cfg.rectangle_max_outside_bars)

        def _cluster_levels(pts: pd.DataFrame) -> List[dict]:
            arr = pts.sort_values("idx")
            clusters: List[dict] = []
            for _, r in arr.iterrows():
                p = float(r["price"])
                x = int(r["idx"])
                placed = False
                for c in clusters:
                    if abs(p - float(c["level"])) <= tol_abs:
                        n = int(c["touches"])
                        c["level"] = (float(c["level"]) * n + p) / (n + 1)
                        c["touches"] = n + 1
                        c["idxs"].append(x)
                        placed = True
                        break
                if not placed:
                    clusters.append({"level": p, "touches": 1, "idxs": [x]})
            return clusters

        highs = _cluster_levels(piv[piv["kind"] == "H"])
        lows = _cluster_levels(piv[piv["kind"] == "L"])
        if not highs or not lows:
            return pd.DataFrame(
                columns=[
                    "start_idx",
                    "end_idx",
                    "start_date",
                    "end_date",
                    "upper_level",
                    "lower_level",
                    "top_touches",
                    "bottom_touches",
                    "outside_bars",
                ]
            )

        out: List[dict] = []
        close = ohlc["Close"].astype(float).reset_index(drop=True)
        high = ohlc["High"].astype(float).reset_index(drop=True)
        low = ohlc["Low"].astype(float).reset_index(drop=True)

        for h in highs:
            if int(h["touches"]) < min_touches:
                continue
            for l in lows:
                if int(l["touches"]) < min_touches:
                    continue
                upper = float(h["level"])
                lower = float(l["level"])
                if upper <= lower:
                    continue

                mid = (upper + lower) / 2.0
                if mid <= 0:
                    continue
                height_pct = (upper - lower) / mid
                if height_pct < 0.01 or height_pct > 0.35:
                    continue

                start_idx = min(min(h["idxs"]), min(l["idxs"]))
                end_idx = max(max(h["idxs"]), max(l["idxs"]))
                span = end_idx - start_idx + 1
                if span < min_bars or span > max_bars:
                    continue
                if end_idx >= len(ohlc):
                    continue

                seg_close = close.iloc[start_idx : end_idx + 1]
                outside = int(
                    ((seg_close > upper * (1.0 + tol_pct)) | (seg_close < lower * (1.0 - tol_pct))).sum()
                )
                if outside > max_outside:
                    continue

                seg_high = high.iloc[start_idx : end_idx + 1]
                seg_low = low.iloc[start_idx : end_idx + 1]
                top_touches = int((seg_high >= upper * (1.0 - tol_pct)).sum())
                bottom_touches = int((seg_low <= lower * (1.0 + tol_pct)).sum())
                if top_touches < min_touches or bottom_touches < min_touches:
                    continue

                out.append(
                    {
                        "start_idx": int(start_idx),
                        "end_idx": int(end_idx),
                        "start_date": ohlc.index[int(start_idx)],
                        "end_date": ohlc.index[int(end_idx)],
                        "upper_level": float(upper),
                        "lower_level": float(lower),
                        "top_touches": int(top_touches),
                        "bottom_touches": int(bottom_touches),
                        "outside_bars": int(outside),
                    }
                )

        if not out:
            return pd.DataFrame(
                columns=[
                    "start_idx",
                    "end_idx",
                    "start_date",
                    "end_date",
                    "upper_level",
                    "lower_level",
                    "top_touches",
                    "bottom_touches",
                    "outside_bars",
                ]
            )

        df = pd.DataFrame(out).sort_values(
            ["end_idx", "top_touches", "bottom_touches", "outside_bars"],
            ascending=[False, False, False, True],
        )

        kept: List[pd.Series] = []
        for _, r in df.iterrows():
            overlap = False
            for k in kept:
                inter = max(
                    0,
                    min(int(r["end_idx"]), int(k["end_idx"]))
                    - max(int(r["start_idx"]), int(k["start_idx"])),
                )
                union = max(int(r["end_idx"]), int(k["end_idx"])) - min(
                    int(r["start_idx"]), int(k["start_idx"])
                )
                if union > 0 and (inter / union) > 0.75:
                    overlap = True
                    break
            if not overlap:
                kept.append(r)

        return pd.DataFrame(kept).reset_index(drop=True)
