from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pattern_engine.config import Config
from pattern_engine.plotters.common import plot_candles, style_x


class DoublePatternPlotter:
    def plot(self, symbol: str, ohlc: pd.DataFrame, dbl: pd.DataFrame, cfg: Config, out_path: Path) -> None:
        data = ohlc.tail(cfg.lookback_bars).copy()
        idx_map = {d: i for i, d in enumerate(data.index)}
        fig, ax = plt.subplots(figsize=(14, 7))
        plot_candles(ax, data)

        if not dbl.empty:
            for _, r in dbl.iterrows():
                d1 = pd.to_datetime(r["first_date"])
                d2 = pd.to_datetime(r["second_date"])
                dn = pd.to_datetime(r["neckline_date"])
                if d1 in idx_map and d2 in idx_map and dn in idx_map:
                    c = "#2166ac" if r["pattern"] == "DoubleBottom" else "#b2182b"
                    i1, i2, inx = idx_map[d1], idx_map[d2], idx_map[dn]
                    ax.plot([i1, inx, i2], [r["first_price"], r["neckline_price"], r["second_price"]], color=c, linewidth=2)

        style_x(ax, data)
        ax.set_title(f"{symbol} Double Top/Bottom")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
