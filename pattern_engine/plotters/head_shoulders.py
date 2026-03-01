from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pattern_engine.config import Config
from pattern_engine.plotters.common import plot_candles, style_x


class HeadShouldersPlotter:
    def plot(self, symbol: str, ohlc: pd.DataFrame, hs: pd.DataFrame, cfg: Config, out_path: Path) -> None:
        data = ohlc.tail(cfg.lookback_bars).copy()
        idx_map = {d: i for i, d in enumerate(data.index)}
        fig, ax = plt.subplots(figsize=(14, 7))
        plot_candles(ax, data)

        if not hs.empty:
            for _, r in hs.iterrows():
                d1 = pd.to_datetime(r["left_shoulder_date"])
                d2 = pd.to_datetime(r["head_date"])
                d3 = pd.to_datetime(r["right_shoulder_date"])
                if d1 in idx_map and d2 in idx_map and d3 in idx_map:
                    c = "#2166ac" if r["pattern"] == "InverseHeadAndShoulders" else "#b2182b"
                    i1, i2, i3 = idx_map[d1], idx_map[d2], idx_map[d3]
                    ax.plot(
                        [i1, i2, i3],
                        [r["left_shoulder_price"], r["head_price"], r["right_shoulder_price"]],
                        color=c,
                        linewidth=2,
                    )
                    ax.axhline(float(r["neckline_price"]), color="#111", linestyle="--", linewidth=1)

        style_x(ax, data)
        ax.set_title(f"{symbol} Head & Shoulders")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
