from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def plot_candles(ax, data: pd.DataFrame) -> None:
    w = 0.58
    for i, (_, r) in enumerate(data.iterrows()):
        up = r["Close"] >= r["Open"]
        c = "#1b9e77" if up else "#d95f02"
        low_body = min(r["Open"], r["Close"])
        h = abs(r["Close"] - r["Open"])
        if h < 1e-9:
            h = 1e-9
        ax.vlines(i, r["Low"], r["High"], color=c, linewidth=0.9, alpha=0.9)
        ax.add_patch(Rectangle((i - w / 2.0, low_body), w, h, facecolor=c, edgecolor=c, linewidth=0.8, alpha=0.95))


def style_x(ax, data: pd.DataFrame) -> None:
    x = np.arange(len(data), dtype=float)
    step = max(1, len(data) // 12)
    ticks = x[::step]
    labels = [str(data.index[int(t)].date()) for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
