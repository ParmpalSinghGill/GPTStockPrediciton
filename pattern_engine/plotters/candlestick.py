from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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

        for dt, r in pats.iterrows():
            i = idx_map[dt]
            p = str(r["pattern"])
            if p in bullish_tags:
                ax.scatter(i, float(data["Low"].iloc[i]) * 0.995, marker="^", color="#1a9850", s=30)
            elif p in bearish_tags:
                ax.scatter(i, float(data["High"].iloc[i]) * 1.005, marker="v", color="#b2182b", s=30)
            else:
                ax.scatter(i, float(data["Close"].iloc[i]), marker="o", color="#444", s=12)

        style_x(ax, data)
        ax.set_title(f"{symbol} Candlestick Patterns")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
