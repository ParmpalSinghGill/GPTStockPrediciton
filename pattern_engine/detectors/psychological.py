from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from pattern_engine.config import Config
from pattern_engine.pivots import collect_pivots


class PsychologicalLineDetector:
    def detect_round_levels(self, ohlc: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        d = ohlc.tail(cfg.level_lookback_bars).copy()
        close = d["Close"].astype(float)
        low = d["Low"].astype(float)
        high = d["High"].astype(float)
        last = float(close.iloc[-1])

        if cfg.psychological_step > 0:
            step = float(cfg.psychological_step)
        else:
            m = float(close.median())
            if m < 50:
                step = 5.0
            elif m < 200:
                step = 10.0
            elif m < 1000:
                step = 50.0
            elif m < 5000:
                step = 100.0
            else:
                step = 500.0

        lo = float(low.min())
        hi = float(high.max())
        start = np.floor(lo / step) * step
        end = np.ceil(hi / step) * step
        levels = np.arange(start, end + step, step)

        rows = []
        tol = cfg.level_tolerance_pct
        for lv in levels:
            touches = int(((low <= lv * (1.0 + tol)) & (high >= lv * (1.0 - tol))).sum())
            if touches >= cfg.min_psy_touches:
                kind = "support" if lv <= last else "resistance"
                rows.append({"kind": kind, "level": float(lv), "touches": touches, "source": "psychological"})

        if not rows:
            return pd.DataFrame(columns=["kind", "level", "touches", "source"])

        out = pd.DataFrame(rows).sort_values(["touches", "level"], ascending=[False, True])
        sup = out[out["kind"] == "support"].head(3)
        res = out[out["kind"] == "resistance"].head(3)
        return pd.concat([sup, res], ignore_index=True)

    def detect_horizontal_sr_levels(self, ohlc: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        piv = collect_pivots(ohlc, cfg.pivot_window, cfg.level_lookback_bars)
        if piv.empty:
            return pd.DataFrame(columns=["kind", "level", "touches", "source"])

        close_last = float(ohlc["Close"].iloc[-1])
        tol_abs = close_last * float(cfg.level_tolerance_pct)
        rows: List[dict] = []

        for kind_sym, kind_out in [("L", "support"), ("H", "resistance")]:
            pts = piv[piv["kind"] == kind_sym].sort_values("idx")
            if pts.empty:
                continue
            clusters: List[dict] = []
            for _, r in pts.iterrows():
                p = float(r["price"])
                placed = False
                for c in clusters:
                    if abs(p - float(c["level"])) <= tol_abs:
                        n = int(c["touches"])
                        c["level"] = (float(c["level"]) * n + p) / (n + 1)
                        c["touches"] = n + 1
                        placed = True
                        break
                if not placed:
                    clusters.append({"level": p, "touches": 1})
            for c in clusters:
                if int(c["touches"]) >= 2:
                    rows.append(
                        {
                            "kind": kind_out,
                            "level": float(c["level"]),
                            "touches": int(c["touches"]),
                            "source": "horizontal_sr",
                        }
                    )

        if not rows:
            return pd.DataFrame(columns=["kind", "level", "touches", "source"])

        out = pd.DataFrame(rows).sort_values(["touches", "level"], ascending=[False, True])
        sup = out[out["kind"] == "support"].head(4)
        res = out[out["kind"] == "resistance"].head(4)
        return pd.concat([sup, res], ignore_index=True)

    def detect_sloping_lines(self, ohlc: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        piv = collect_pivots(ohlc, cfg.pivot_window, cfg.level_lookback_bars)
        if len(piv) < 4:
            return pd.DataFrame(
                columns=[
                    "line_type",
                    "x1_idx",
                    "x2_idx",
                    "x1_date",
                    "x2_date",
                    "y1",
                    "y2",
                    "slope",
                    "touches",
                    "cross_bars",
                    "cross_streak",
                ]
            )

        close = ohlc["Close"].astype(float)
        tol = float(cfg.level_tolerance_pct)
        min_touches = int(cfg.min_trend_touches)
        max_cross = int(cfg.max_cross_bars)
        max_streak = int(cfg.max_cross_streak)
        candidates: List[dict] = []

        def eval_line(df_pts: pd.DataFrame, line_type: str) -> None:
            arr = df_pts.sort_values("idx").reset_index(drop=True)
            for i in range(len(arr) - 1):
                for j in range(i + 1, len(arr)):
                    x1, y1 = int(arr.at[i, "idx"]), float(arr.at[i, "price"])
                    x2, y2 = int(arr.at[j, "idx"]), float(arr.at[j, "price"])
                    if x2 <= x1:
                        continue
                    slope = (y2 - y1) / (x2 - x1)
                    if line_type == "up_support" and slope <= 0:
                        continue
                    if line_type == "down_resistance" and slope >= 0:
                        continue

                    touches = 0
                    last_touch_idx = x2
                    for _, r in arr.iterrows():
                        x = int(r["idx"])
                        if x < x1:
                            continue
                        y_line = y1 + slope * (x - x1)
                        if abs(float(r["price"]) - y_line) / max(abs(y_line), 1e-12) <= tol:
                            touches += 1
                            last_touch_idx = max(last_touch_idx, x)
                    if touches < min_touches:
                        continue

                    max_age = max(1, int(cfg.trend_max_age_bars))
                    if last_touch_idx < (len(ohlc) - max_age):
                        continue

                    cross_bars = 0
                    streak = 0
                    max_seen_streak = 0
                    for x in range(x1, min(last_touch_idx + 1, len(ohlc))):
                        y_line = y1 + slope * (x - x1)
                        if line_type == "up_support":
                            crossed = float(close.iloc[x]) < y_line * (1.0 - tol)
                        else:
                            crossed = float(close.iloc[x]) > y_line * (1.0 + tol)
                        if crossed:
                            cross_bars += 1
                            streak += 1
                        else:
                            streak = 0
                        max_seen_streak = max(max_seen_streak, streak)
                    if cross_bars > max_cross or max_seen_streak > max_streak:
                        continue

                    candidates.append(
                        {
                            "line_type": line_type,
                            "x1_idx": x1,
                            "x2_idx": last_touch_idx,
                            "x1_date": ohlc.index[x1],
                            "x2_date": ohlc.index[last_touch_idx],
                            "y1": float(y1),
                            "y2": float(y1 + slope * (last_touch_idx - x1)),
                            "slope": float(slope),
                            "touches": int(touches),
                            "cross_bars": int(cross_bars),
                            "cross_streak": int(max_seen_streak),
                        }
                    )

        eval_line(piv[piv["kind"] == "L"], "up_support")
        eval_line(piv[piv["kind"] == "H"], "down_resistance")

        if not candidates:
            return pd.DataFrame(
                columns=[
                    "line_type",
                    "x1_idx",
                    "x2_idx",
                    "x1_date",
                    "x2_date",
                    "y1",
                    "y2",
                    "slope",
                    "touches",
                    "cross_bars",
                    "cross_streak",
                ]
            )

        df = pd.DataFrame(candidates).sort_values(
            ["x2_idx", "touches", "cross_bars"], ascending=[False, False, True]
        )

        out_rows: List[pd.Series] = []
        for lt in ["up_support", "down_resistance"]:
            part = df[df["line_type"] == lt]
            kept: List[pd.Series] = []
            for _, r in part.iterrows():
                overlap = False
                for k in kept:
                    inter = max(0, min(int(r["x2_idx"]), int(k["x2_idx"])) - max(int(r["x1_idx"]), int(k["x1_idx"])))
                    union = max(int(r["x2_idx"]), int(k["x2_idx"])) - min(int(r["x1_idx"]), int(k["x1_idx"]))
                    if union > 0 and (inter / union) > 0.75:
                        overlap = True
                        break
                if not overlap:
                    kept.append(r)
                if len(kept) >= 3:
                    break
            out_rows.extend(kept)

        return pd.DataFrame(out_rows).reset_index(drop=True)
