#!/usr/bin/env python3
"""Generate one plot per detected rectangle pattern in a single folder.

Output naming:
- <SYMBOL>_1.png
- <SYMBOL>_2.png
- ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from pattern_engine.config import Config
from pattern_engine.data import load_symbol_frame, prepare_ohlc, resolve_symbols
from pattern_engine.detectors import RectanglePatternDetector
from pattern_engine.plotters.common import plot_candles, style_x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot each rectangle pattern as a separate image.")
    p.add_argument("--data-path", default="Data/AllSTOCKS.pk")
    p.add_argument("--symbols", default="", help="Comma-separated symbols.")
    p.add_argument("--symbols-file", default="Data/nifty50_symbols.txt")
    p.add_argument("--output-dir", default="pattern_plots/rectangle_each")
    p.add_argument(
        "--padding-bars",
        type=int,
        default=20,
        help="Candle padding on both left/right sides of detected rectangle window.",
    )
    p.add_argument("--pivot-window", type=int, default=5)
    p.add_argument("--level-lookback-bars", type=int, default=900)
    p.add_argument("--level-tolerance-pct", type=float, default=0.01)
    p.add_argument("--rectangle-min-bars", type=int, default=20)
    p.add_argument("--rectangle-max-bars", type=int, default=220)
    p.add_argument("--rectangle-min-touches", type=int, default=2)
    p.add_argument("--rectangle-max-outside-bars", type=int, default=2)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        lookback_bars=260,
        level_lookback_bars=int(args.level_lookback_bars),
        pivot_window=int(args.pivot_window),
        level_tolerance_pct=float(args.level_tolerance_pct),
        min_psy_touches=2,
        psychological_step=0.0,
        min_trend_touches=3,
        max_cross_bars=2,
        max_cross_streak=2,
        trend_max_age_bars=220,
        rectangle_min_bars=int(args.rectangle_min_bars),
        rectangle_max_bars=int(args.rectangle_max_bars),
        rectangle_min_touches=int(args.rectangle_min_touches),
        rectangle_max_outside_bars=int(args.rectangle_max_outside_bars),
        short_trend_min_bars=21,
        short_trend_max_bars=42,
        long_trend_min_bars=252,
        long_trend_max_bars=504,
    )


def plot_rectangle_event(
    symbol: str,
    ohlc: pd.DataFrame,
    rect_row: pd.Series,
    out_path: Path,
    padding_bars: int,
) -> None:
    start_idx = int(rect_row["start_idx"])
    end_idx = int(rect_row["end_idx"])
    left = max(0, start_idx - int(padding_bars))
    right = min(len(ohlc) - 1, end_idx + int(padding_bars))

    window = ohlc.iloc[left : right + 1].copy()
    rect_start_local = start_idx - left
    rect_end_local = end_idx - left

    lower = float(rect_row["lower_level"])
    upper = float(rect_row["upper_level"])

    fig, ax = plt.subplots(figsize=(14, 7))
    plot_candles(ax, window)

    rect_width = (rect_end_local - rect_start_local) + 1
    rect_height = upper - lower
    ax.add_patch(
        Rectangle(
            (rect_start_local - 0.5, lower),
            rect_width,
            rect_height,
            facecolor="none",
            edgecolor="#6a3d9a",
            linewidth=2.2,
            linestyle="--",
            alpha=0.95,
        )
    )

    # Visual guides for padded boundaries around the rectangle.
    ax.axvline(x=rect_start_local - 0.5, color="#1f78b4", linestyle=":", linewidth=1.1, alpha=0.75)
    ax.axvline(x=rect_end_local + 0.5, color="#1f78b4", linestyle=":", linewidth=1.1, alpha=0.75)

    style_x(ax, window)
    ax.set_title(
        f"{symbol} Rectangle | {pd.to_datetime(rect_row['start_date']).date()} to "
        f"{pd.to_datetime(rect_row['end_date']).date()} | pad={padding_bars} bars"
    )
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    data_path = Path(args.data_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    padding_bars = int(args.padding_bars)

    symbols = resolve_symbols(args.symbols, args.symbols_file)
    detector = RectanglePatternDetector()

    total_images = 0
    total_rectangles = 0
    print(f"Generating rectangle plots for {len(symbols)} symbols into: {out_dir}")
    for i, symbol in enumerate(symbols, start=1):
        try:
            ohlc = prepare_ohlc(load_symbol_frame(data_path, symbol))
            rects = detector.detect(ohlc, cfg)
            n = int(len(rects))
            total_rectangles += n
            if n == 0:
                print(f"[{i}/{len(symbols)}] {symbol}: no rectangles")
                continue

            for j, (_, r) in enumerate(rects.iterrows(), start=1):
                out_path = out_dir / f"{symbol}_{j}.png"
                plot_rectangle_event(symbol, ohlc, r, out_path, padding_bars=padding_bars)
                total_images += 1

            print(f"[{i}/{len(symbols)}] {symbol}: rectangles={n}, images_saved={n}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(symbols)}] {symbol}: error -> {exc}")

    print(f"Done. total_rectangles={total_rectangles}, total_images={total_images}, folder={out_dir}")


if __name__ == "__main__":
    main()

