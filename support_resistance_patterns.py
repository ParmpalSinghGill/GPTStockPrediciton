#!/usr/bin/env python3
"""Generate per-pattern stock charts with modular detectors/plotters.

Outputs:
- pattern_plots/psychological_lines/<SYMBOL>_psychological.png
- pattern_plots/head_shoulders/<SYMBOL>_head_shoulders.png
- pattern_plots/double_patterns/<SYMBOL>_double_patterns.png
- pattern_plots/candlestick_patterns/<SYMBOL>_candlestick_patterns.png
- pattern_plots/rectangle_patterns/<SYMBOL>_rectangle_patterns.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from pattern_engine.config import Config
from pattern_engine.data import load_symbol_frame, prepare_ohlc, resolve_symbols
from pattern_engine.detectors import (
    CandlestickPatternDetector,
    DoublePatternDetector,
    HeadShouldersDetector,
    PsychologicalLineDetector,
    RectanglePatternDetector,
    TrendDetector,
)
from pattern_engine.plotters import (
    CandlestickPlotter,
    DoublePatternPlotter,
    HeadShouldersPlotter,
    PsychologicalPlotter,
    RectanglePatternPlotter,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate stock pattern charts in flat folders.")
    p.add_argument("--data-path", default="Data/AllSTOCKS.pk")
    p.add_argument("--symbols", default="", help="Comma-separated symbols.")
    p.add_argument(
        "--symbols-file",
        default="Data/nifty50_symbols.txt",
        help="File with comma-separated symbols; used when --symbols is empty.",
    )
    p.add_argument("--output-root", default="pattern_plots")
    p.add_argument("--lookback-bars", type=int, default=260)
    p.add_argument("--level-lookback-bars", type=int, default=900)
    p.add_argument("--pivot-window", type=int, default=5)
    p.add_argument("--level-tolerance-pct", type=float, default=0.01)
    p.add_argument("--min-psy-touches", type=int, default=2)
    p.add_argument("--psychological-step", type=float, default=0.0)
    p.add_argument("--min-trend-touches", type=int, default=3)
    p.add_argument("--max-cross-bars", type=int, default=2)
    p.add_argument("--max-cross-streak", type=int, default=2)
    p.add_argument(
        "--trend-max-age-bars",
        type=int,
        default=220,
        help="Keep only sloping lines whose latest touch is within these recent bars.",
    )
    p.add_argument("--rectangle-min-bars", type=int, default=20)
    p.add_argument("--rectangle-max-bars", type=int, default=220)
    p.add_argument("--rectangle-min-touches", type=int, default=2)
    p.add_argument("--rectangle-max-outside-bars", type=int, default=2)
    p.add_argument("--short-trend-min-bars", type=int, default=21, help="Short trend min bars (~1 month).")
    p.add_argument("--short-trend-max-bars", type=int, default=42, help="Short trend max bars (~2 months).")
    p.add_argument("--long-trend-min-bars", type=int, default=252, help="Long trend min bars (~1 year).")
    p.add_argument("--long-trend-max-bars", type=int, default=504, help="Long trend max bars (~2 years).")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        lookback_bars=int(args.lookback_bars),
        level_lookback_bars=int(args.level_lookback_bars),
        pivot_window=int(args.pivot_window),
        level_tolerance_pct=float(args.level_tolerance_pct),
        min_psy_touches=int(args.min_psy_touches),
        psychological_step=float(args.psychological_step),
        min_trend_touches=int(args.min_trend_touches),
        max_cross_bars=int(args.max_cross_bars),
        max_cross_streak=int(args.max_cross_streak),
        trend_max_age_bars=int(args.trend_max_age_bars),
        rectangle_min_bars=int(args.rectangle_min_bars),
        rectangle_max_bars=int(args.rectangle_max_bars),
        rectangle_min_touches=int(args.rectangle_min_touches),
        rectangle_max_outside_bars=int(args.rectangle_max_outside_bars),
        short_trend_min_bars=int(args.short_trend_min_bars),
        short_trend_max_bars=int(args.short_trend_max_bars),
        long_trend_min_bars=int(args.long_trend_min_bars),
        long_trend_max_bars=int(args.long_trend_max_bars),
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    data_path = Path(args.data_path)
    symbols = resolve_symbols(args.symbols, args.symbols_file)
    out_root = Path(args.output_root)

    dir_psy = out_root / "psychological_lines"
    dir_hs = out_root / "head_shoulders"
    dir_dbl = out_root / "double_patterns"
    dir_cdl = out_root / "candlestick_patterns"
    dir_rect = out_root / "rectangle_patterns"
    for d in [dir_psy, dir_hs, dir_dbl, dir_cdl, dir_rect]:
        d.mkdir(parents=True, exist_ok=True)

    psycho_detector = PsychologicalLineDetector()
    hs_detector = HeadShouldersDetector()
    double_detector = DoublePatternDetector()
    candle_detector = CandlestickPatternDetector()
    rectangle_detector = RectanglePatternDetector()
    trend_detector = TrendDetector()

    psycho_plotter = PsychologicalPlotter()
    hs_plotter = HeadShouldersPlotter()
    double_plotter = DoublePatternPlotter()
    candle_plotter = CandlestickPlotter()
    rectangle_plotter = RectanglePatternPlotter()

    rows: List[dict] = []
    for s in symbols:
        try:
            ohlc = prepare_ohlc(load_symbol_frame(data_path, s))

            psy_round = psycho_detector.detect_round_levels(ohlc, cfg)
            sr_h = psycho_detector.detect_horizontal_sr_levels(ohlc, cfg)
            psy = pd.concat([psy_round, sr_h], ignore_index=True)
            trend_lines = psycho_detector.detect_sloping_lines(ohlc, cfg)

            hs = hs_detector.detect(ohlc, cfg)
            dbl = double_detector.detect(ohlc, cfg)
            cdl = candle_detector.detect(ohlc)
            rect = rectangle_detector.detect(ohlc, cfg)
            trends = trend_detector.detect(ohlc, cfg)

            psycho_plotter.plot(s, ohlc, psy, trend_lines, trends, cfg, dir_psy / f"{s}_psychological.png")
            hs_plotter.plot(s, ohlc, hs, cfg, dir_hs / f"{s}_head_shoulders.png")
            double_plotter.plot(s, ohlc, dbl, cfg, dir_dbl / f"{s}_double_patterns.png")
            candle_plotter.plot(s, ohlc, cdl, cfg, dir_cdl / f"{s}_candlestick_patterns.png")
            rectangle_plotter.plot(s, ohlc, rect, cfg, dir_rect / f"{s}_rectangle_patterns.png")

            short_tr = trends[trends["trend_horizon"] == "short_term"]
            long_tr = trends[trends["trend_horizon"] == "long_term"]
            short_dir = str(short_tr.iloc[0]["direction"]) if not short_tr.empty else "n/a"
            long_dir = str(long_tr.iloc[0]["direction"]) if not long_tr.empty else "n/a"
            short_bars = int(short_tr.iloc[0]["window_bars"]) if not short_tr.empty else 0
            long_bars = int(long_tr.iloc[0]["window_bars"]) if not long_tr.empty else 0

            rows.append(
                {
                    "symbol": s,
                    "psychological_levels": int(len(psy_round)),
                    "horizontal_sr_levels": int(len(sr_h)),
                    "sloping_psychological_lines": int(len(trend_lines)),
                    "head_shoulders": int(len(hs)),
                    "double_patterns": int(len(dbl)),
                    "candlestick_patterns": int(len(cdl)),
                    "rectangle_patterns": int(len(rect)),
                    "short_term_trend": short_dir,
                    "short_term_window_bars": short_bars,
                    "long_term_trend": long_dir,
                    "long_term_window_bars": long_bars,
                    "date_min": str(ohlc.index.min().date()),
                    "date_max": str(ohlc.index.max().date()),
                }
            )

            print(
                f"[ok] {s}: psych={len(psy_round)} sr={len(sr_h)} trend={len(trend_lines)} hs={len(hs)} "
                f"double={len(dbl)} candle={len(cdl)} rect={len(rect)} short={short_dir} long={long_dir}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {s}: {exc}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "pattern_summary.csv", index=False)
    with open(out_root / "pattern_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)


if __name__ == "__main__":
    main()
