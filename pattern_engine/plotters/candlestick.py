from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from pattern_engine.config import Config
from pattern_engine.plotters.common import plot_candles, style_x


class CandlestickPlotter:
    def plot(self, symbol: str, ohlc: pd.DataFrame, pats: pd.DataFrame, cfg: Config, out_path: Path) -> None:
        data = ohlc.tail(cfg.lookback_bars).copy()
        pats = pats.loc[pats.index.intersection(data.index)]
        idx_map = {d: i for i, d in enumerate(data.index)}
        fig, ax = plt.subplots(figsize=(14, 7))
        plot_candles(ax, data)

        bullish_tags = {
            "BullishEngulfing", "Hammer", "InvertedHammer", "MorningStar", "PiercingLine", "BullishHarami", "TweezerBottom", "MarubozuBullish"
        }
        bearish_tags = {
            "BearishEngulfing", "ShootingStar", "EveningStar", "DarkCloudCover", "BearishHarami", "HangingMan", "TweezerTop", "MarubozuBearish"
        }
        neutral_style = {"marker": "o", "color": "#444", "size": 12}
        bullish_style = {"marker": "^", "color": "#1a9850", "size": 30}
        bearish_style = {"marker": "v", "color": "#b2182b", "size": 30}
        legend_handles: dict[str, Line2D] = {}

        for dt, r in pats.iterrows():
            i = idx_map[dt]
            p = str(r["pattern"])
            if p in bullish_tags:
                style = bullish_style
                y = float(data["Low"].iloc[i]) * 0.995
            elif p in bearish_tags:
                style = bearish_style
                y = float(data["High"].iloc[i]) * 1.005
            else:
                style = neutral_style
                y = float(data["Close"].iloc[i])
            ax.scatter(i, y, marker=style["marker"], color=style["color"], s=style["size"])
            if p not in legend_handles:
                legend_handles[p] = Line2D(
                    [0],
                    [0],
                    marker=style["marker"],
                    color="none",
                    markerfacecolor=style["color"],
                    markeredgecolor=style["color"],
                    markersize=6,
                    linewidth=0,
                    label=p,
                )

        style_x(ax, data)
        ax.set_title(f"{symbol} Candlestick Patterns")
        ax.grid(alpha=0.15)
        if legend_handles:
            ax.legend(
                handles=[legend_handles[k] for k in sorted(legend_handles)],
                title="Patterns",
                loc="upper left",
                fontsize=8,
                title_fontsize=9,
                ncol=2,
                frameon=True,
            )
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
