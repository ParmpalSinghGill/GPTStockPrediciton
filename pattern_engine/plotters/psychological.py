from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from pattern_engine.config import Config
from pattern_engine.plotters.common import plot_candles, style_x


class PsychologicalPlotter:
    def plot(
        self,
        symbol: str,
        ohlc: pd.DataFrame,
        lv: pd.DataFrame,
        trend_lines: pd.DataFrame,
        trends: pd.DataFrame,
        cfg: Config,
        out_path: Path,
    ) -> None:
        data = ohlc.tail(cfg.lookback_bars).copy()
        base_idx = len(ohlc) - len(data)
        idx_map = {d: i for i, d in enumerate(data.index)}
        fig, ax = plt.subplots(figsize=(14, 7))
        plot_candles(ax, data)
        for _, r in lv.iterrows():
            is_sup = r["kind"] == "support"
            source = str(r.get("source", ""))
            c = "#2c7fb8" if is_sup else "#b2182b"
            ls = ":" if "psychological" in source else ("--" if is_sup else "-.")
            ax.axhline(float(r["level"]), color=c, linestyle=ls, linewidth=1.2, alpha=0.9)

        if not trend_lines.empty:
            for _, r in trend_lines.iterrows():
                d1 = pd.to_datetime(r["x1_date"])
                d2 = pd.to_datetime(r["x2_date"])
                if d1 not in idx_map or d2 not in idx_map:
                    continue
                i1, i2 = idx_map[d1], idx_map[d2]
                y1, y2 = float(r["y1"]), float(r["y2"])
                i_end = len(data) - 1 + max(8, len(data) // 12)
                if i2 == i1:
                    y_end = y2
                else:
                    slope = (y2 - y1) / (i2 - i1)
                    y_end = y1 + slope * (i_end - i1)
                c = "#1f78b4" if r["line_type"] == "up_support" else "#e31a1c"
                ax.plot([i1, i_end], [y1, y_end], color=c, linewidth=2.0, alpha=0.95)

        if not trends.empty:
            for _, r in trends.iterrows():
                start_abs = int(r["start_idx"])
                end_abs = int(r["end_idx"])
                if end_abs < base_idx:
                    continue
                slope = float(r["slope"])
                start_price = float(r["start_price"])
                vis_start_abs = max(base_idx, start_abs)
                vis_end_abs = min(len(ohlc) - 1, end_abs)
                x0 = vis_start_abs - base_idx
                x1 = vis_end_abs - base_idx
                y0 = start_price + slope * (vis_start_abs - start_abs)
                y1 = start_price + slope * (vis_end_abs - start_abs)
                if str(r["trend_horizon"]) == "long_term":
                    c, lw = "#6a3d9a", 2.8
                else:
                    c, lw = "#ff7f00", 2.2
                ax.plot([x0, x1], [y0, y1], color=c, linewidth=lw, alpha=0.95)
        ax.set_xlim(-1, len(data) - 1 + max(8, len(data) // 12))
        style_x(ax, data)
        ax.set_title(f"{symbol} S/R + Psychological + Short/Long Trends")
        ax.grid(alpha=0.15)
        ax.legend(
            handles=[
                Line2D([0], [0], color="#2c7fb8", linestyle=":", label="Psychological Horizontal"),
                Line2D([0], [0], color="#444", linestyle="--", label="Horizontal Support/Resistance"),
                Line2D([0], [0], color="#1f78b4", linewidth=2, label="Up Sloping Support"),
                Line2D([0], [0], color="#e31a1c", linewidth=2, label="Down Sloping Resistance"),
                Line2D([0], [0], color="#ff7f00", linewidth=2.2, label="Short-Term Trend (1-2M)"),
                Line2D([0], [0], color="#6a3d9a", linewidth=2.8, label="Long-Term Trend (1-2Y)"),
            ],
            loc="best",
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
