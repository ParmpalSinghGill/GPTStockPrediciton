from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from pattern_engine.config import Config
from pattern_engine.plotters.common import plot_candles, style_x


class RectanglePatternPlotter:
    def plot(self, symbol: str, ohlc: pd.DataFrame, rects: pd.DataFrame, cfg: Config, out_path: Path) -> None:
        data = ohlc.tail(cfg.lookback_bars).copy()
        offset = len(ohlc) - len(data)
        fig, ax = plt.subplots(figsize=(14, 7))
        plot_candles(ax, data)

        if not rects.empty:
            for _, r in rects.iterrows():
                s = int(r["start_idx"]) - offset
                e = int(r["end_idx"]) - offset
                if e < 0 or s >= len(data):
                    continue
                s = max(0, s)
                e = min(len(data) - 1, e)
                if e <= s:
                    continue
                lower = float(r["lower_level"])
                upper = float(r["upper_level"])
                w = (e - s) + 1
                h = upper - lower
                ax.add_patch(
                    Rectangle(
                        (s - 0.5, lower),
                        w,
                        h,
                        facecolor="none",
                        edgecolor="#6a3d9a",
                        linewidth=2.0,
                        linestyle="--",
                        alpha=0.95,
                    )
                )

        style_x(ax, data)
        ax.set_title(f"{symbol} Rectangle Patterns")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
