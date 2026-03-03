#!/usr/bin/env python3
"""Backtest pattern breakout/breakdown trades confirmed by candlestick patterns.

Workflow:
1) Detect major patterns: rectangle, triangle, head-and-shoulders.
2) Wait for breakout/breakdown from pattern boundary.
3) Require same-direction candlestick confirmation on breakout candle.
4) Enter on next candle open, set stop-loss + target from pattern geometry.
5) Track equity from initial capital and log complete trade context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pattern_engine.config import Config
from pattern_engine.data import load_symbol_frame, prepare_ohlc, resolve_symbols
from pattern_engine.detectors import (
    CandlestickPatternDetector,
    HeadShouldersDetector,
    PsychologicalLineDetector,
    RectanglePatternDetector,
    TrendDetector,
)


BULLISH_CANDLE_PATTERNS = {
    "BullishEngulfing",
    "Hammer",
    "InvertedHammer",
    "MorningStar",
    "PiercingLine",
    "BullishHarami",
    "TweezerBottom",
    "MarubozuBullish",
    "DragonflyDoji",
}

BEARISH_CANDLE_PATTERNS = {
    "BearishEngulfing",
    "ShootingStar",
    "EveningStar",
    "DarkCloudCover",
    "BearishHarami",
    "HangingMan",
    "TweezerTop",
    "MarubozuBearish",
    "GravestoneDoji",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pattern breakout backtest with candlestick confirmation.")
    p.add_argument("--data-path", default="Data/AllSTOCKS.pk")
    p.add_argument("--symbols", default="", help="Comma-separated symbols.")
    p.add_argument("--symbols-file", default="Data/nifty50_symbols.txt")
    p.add_argument("--output-dir", default="backtest_reports")
    p.add_argument("--initial-capital", type=float, default=10000.0)
    p.add_argument("--transaction-cost", type=float, default=0.001, help="Cost per side (e.g. 0.001 = 10 bps).")
    p.add_argument("--breakout-buffer-pct", type=float, default=0.002, help="Buffer for breakout/breakdown validation.")
    p.add_argument("--max-pattern-age-bars", type=int, default=120)
    p.add_argument("--max-hold-bars", type=int, default=40)
    p.add_argument("--pivot-window", type=int, default=5)
    p.add_argument("--level-lookback-bars", type=int, default=900)
    p.add_argument("--level-tolerance-pct", type=float, default=0.01)
    p.add_argument("--rectangle-min-bars", type=int, default=20)
    p.add_argument("--rectangle-max-bars", type=int, default=220)
    p.add_argument("--rectangle-min-touches", type=int, default=2)
    p.add_argument("--rectangle-max-outside-bars", type=int, default=2)
    p.add_argument("--min-trend-touches", type=int, default=3)
    p.add_argument("--max-cross-bars", type=int, default=2)
    p.add_argument("--max-cross-streak", type=int, default=2)
    p.add_argument("--trend-max-age-bars", type=int, default=220)
    p.add_argument("--short-trend-min-bars", type=int, default=21)
    p.add_argument("--short-trend-max-bars", type=int, default=42)
    p.add_argument("--long-trend-min-bars", type=int, default=252)
    p.add_argument("--long-trend-max-bars", type=int, default=504)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        lookback_bars=260,
        level_lookback_bars=int(args.level_lookback_bars),
        pivot_window=int(args.pivot_window),
        level_tolerance_pct=float(args.level_tolerance_pct),
        min_psy_touches=2,
        psychological_step=0.0,
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


def _line_value(y1: float, x1: int, slope: float, x: int) -> float:
    return float(y1 + slope * (x - x1))


def detect_triangles(ohlc: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    psycho = PsychologicalLineDetector()
    lines = psycho.detect_sloping_lines(ohlc, cfg)
    if lines.empty:
        return pd.DataFrame(
            columns=[
                "pattern_id",
                "start_idx",
                "end_idx",
                "start_date",
                "end_date",
                "support_x1_idx",
                "support_y1",
                "support_slope",
                "res_x1_idx",
                "res_y1",
                "res_slope",
                "gap_start",
                "gap_end",
            ]
        )

    ups = lines[lines["line_type"] == "up_support"].reset_index(drop=True)
    dns = lines[lines["line_type"] == "down_resistance"].reset_index(drop=True)
    rows: List[dict] = []
    pid = 0
    for _, su in ups.iterrows():
        for _, re in dns.iterrows():
            start_idx = max(int(su["x1_idx"]), int(re["x1_idx"]))
            end_idx = min(int(su["x2_idx"]), int(re["x2_idx"]))
            span = end_idx - start_idx + 1
            if span < max(12, int(cfg.rectangle_min_bars)):
                continue
            if span > int(cfg.rectangle_max_bars):
                continue

            su_y1 = float(su["y1"])
            su_x1 = int(su["x1_idx"])
            su_slope = float(su["slope"])
            re_y1 = float(re["y1"])
            re_x1 = int(re["x1_idx"])
            re_slope = float(re["slope"])

            upper_start = _line_value(re_y1, re_x1, re_slope, start_idx)
            upper_end = _line_value(re_y1, re_x1, re_slope, end_idx)
            lower_start = _line_value(su_y1, su_x1, su_slope, start_idx)
            lower_end = _line_value(su_y1, su_x1, su_slope, end_idx)
            if upper_start <= lower_start or upper_end <= lower_end:
                continue

            gap_start = upper_start - lower_start
            gap_end = upper_end - lower_end
            if gap_end <= 0:
                continue
            if gap_start < (gap_end * 1.2):
                continue

            denom = su_slope - re_slope
            if abs(denom) < 1e-12:
                continue
            apex = (re_y1 - su_y1 + su_slope * su_x1 - re_slope * re_x1) / denom
            if apex <= end_idx:
                continue
            if (apex - end_idx) > int(cfg.rectangle_max_bars):
                continue

            pid += 1
            rows.append(
                {
                    "pattern_id": f"triangle_{pid}",
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "start_date": ohlc.index[int(start_idx)],
                    "end_date": ohlc.index[int(end_idx)],
                    "support_x1_idx": su_x1,
                    "support_y1": su_y1,
                    "support_slope": su_slope,
                    "res_x1_idx": re_x1,
                    "res_y1": re_y1,
                    "res_slope": re_slope,
                    "gap_start": float(gap_start),
                    "gap_end": float(gap_end),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "pattern_id",
                "start_idx",
                "end_idx",
                "start_date",
                "end_date",
                "support_x1_idx",
                "support_y1",
                "support_slope",
                "res_x1_idx",
                "res_y1",
                "res_slope",
                "gap_start",
                "gap_end",
            ]
        )
    out = pd.DataFrame(rows).sort_values(["end_idx", "gap_start"], ascending=[False, False]).reset_index(drop=True)
    return out.head(15).copy()


def _candle_direction(pattern_name: str) -> str:
    if pattern_name in BULLISH_CANDLE_PATTERNS:
        return "long"
    if pattern_name in BEARISH_CANDLE_PATTERNS:
        return "short"
    return "neutral"


def _trend_snapshot(ohlc_up_to_t: pd.DataFrame, cfg: Config, trend_detector: TrendDetector) -> Dict[str, object]:
    trends = trend_detector.detect(ohlc_up_to_t, cfg)
    out = {
        "short_trend": "n/a",
        "short_window_bars": 0,
        "short_r2": np.nan,
        "long_trend": "n/a",
        "long_window_bars": 0,
        "long_r2": np.nan,
    }
    if trends.empty:
        return out

    s = trends[trends["trend_horizon"] == "short_term"]
    l = trends[trends["trend_horizon"] == "long_term"]
    if not s.empty:
        out["short_trend"] = str(s.iloc[0]["direction"])
        out["short_window_bars"] = int(s.iloc[0]["window_bars"])
        out["short_r2"] = float(s.iloc[0]["r2"])
    if not l.empty:
        out["long_trend"] = str(l.iloc[0]["direction"])
        out["long_window_bars"] = int(l.iloc[0]["window_bars"])
        out["long_r2"] = float(l.iloc[0]["r2"])
    return out


def backtest_symbol(
    symbol: str,
    ohlc: pd.DataFrame,
    cfg: Config,
    initial_capital: float,
    transaction_cost: float,
    breakout_buffer_pct: float,
    max_pattern_age_bars: int,
    max_hold_bars: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    rect_detector = RectanglePatternDetector()
    hs_detector = HeadShouldersDetector()
    cdl_detector = CandlestickPatternDetector()
    trend_detector = TrendDetector()

    rects = rect_detector.detect(ohlc, cfg)
    hs = hs_detector.detect(ohlc, cfg)
    triangles = detect_triangles(ohlc, cfg)
    cdl = cdl_detector.detect(ohlc)

    index_pos = {d: i for i, d in enumerate(ohlc.index)}
    cdl_by_idx: Dict[int, List[str]] = {}
    for dt, row in cdl.iterrows():
        i = index_pos.get(dt)
        if i is None:
            continue
        cdl_by_idx.setdefault(i, []).append(str(row["pattern"]))

    pattern_rows: List[dict] = []
    for i, r in rects.iterrows():
        pattern_rows.append(
            {
                "pattern_id": f"rectangle_{i+1}",
                "pattern_type": "Rectangle",
                "start_idx": int(r["start_idx"]),
                "end_idx": int(r["end_idx"]),
                "start_date": pd.to_datetime(r["start_date"]),
                "end_date": pd.to_datetime(r["end_date"]),
                "upper_level": float(r["upper_level"]),
                "lower_level": float(r["lower_level"]),
                "neckline": np.nan,
                "head_price": np.nan,
                "right_shoulder_price": np.nan,
                "support_x1_idx": np.nan,
                "support_y1": np.nan,
                "support_slope": np.nan,
                "res_x1_idx": np.nan,
                "res_y1": np.nan,
                "res_slope": np.nan,
            }
        )

    for _, r in triangles.iterrows():
        pattern_rows.append(
            {
                "pattern_id": str(r["pattern_id"]),
                "pattern_type": "Triangle",
                "start_idx": int(r["start_idx"]),
                "end_idx": int(r["end_idx"]),
                "start_date": pd.to_datetime(r["start_date"]),
                "end_date": pd.to_datetime(r["end_date"]),
                "upper_level": np.nan,
                "lower_level": np.nan,
                "neckline": np.nan,
                "head_price": np.nan,
                "right_shoulder_price": np.nan,
                "support_x1_idx": int(r["support_x1_idx"]),
                "support_y1": float(r["support_y1"]),
                "support_slope": float(r["support_slope"]),
                "res_x1_idx": int(r["res_x1_idx"]),
                "res_y1": float(r["res_y1"]),
                "res_slope": float(r["res_slope"]),
            }
        )

    for i, r in hs.iterrows():
        start_date = pd.to_datetime(r["left_shoulder_date"])
        end_date = pd.to_datetime(r["right_shoulder_date"])
        if (start_date not in index_pos) or (end_date not in index_pos):
            continue
        pattern_rows.append(
            {
                "pattern_id": f"hs_{i+1}",
                "pattern_type": str(r["pattern"]),
                "start_idx": int(index_pos[start_date]),
                "end_idx": int(index_pos[end_date]),
                "start_date": start_date,
                "end_date": end_date,
                "upper_level": np.nan,
                "lower_level": np.nan,
                "neckline": float(r["neckline_price"]),
                "head_price": float(r["head_price"]),
                "right_shoulder_price": float(r["right_shoulder_price"]),
                "support_x1_idx": np.nan,
                "support_y1": np.nan,
                "support_slope": np.nan,
                "res_x1_idx": np.nan,
                "res_y1": np.nan,
                "res_slope": np.nan,
            }
        )

    patterns = pd.DataFrame(pattern_rows).sort_values("end_idx").reset_index(drop=True)
    if patterns.empty:
        empty = pd.DataFrame()
        metrics = {
            "symbol": symbol,
            "initial_capital": float(initial_capital),
            "final_capital": float(initial_capital),
            "total_return_pct": 0.0,
            "num_trades": 0,
            "win_rate_pct": np.nan,
            "max_drawdown_pct": np.nan,
            "pattern_events_considered": 0,
        }
        return empty, pd.DataFrame(), patterns, metrics

    capital = float(initial_capital)
    trades: List[dict] = []
    used_patterns: set[str] = set()
    equity_points = [{"date": str(ohlc.index[0].date()), "equity": capital}]

    i = 1
    while i < len(ohlc) - 1:
        close_i = float(ohlc["Close"].iloc[i])
        open_next = float(ohlc["Open"].iloc[i + 1])

        candidates: List[dict] = []
        today_candles = cdl_by_idx.get(i, [])
        if today_candles:
            for _, p in patterns.iterrows():
                pid = str(p["pattern_id"])
                if pid in used_patterns:
                    continue
                end_idx = int(p["end_idx"])
                if i <= end_idx:
                    continue
                if (i - end_idx) > int(max_pattern_age_bars):
                    continue

                pattern_type = str(p["pattern_type"])
                direction = ""
                support_used = np.nan
                resistance_used = np.nan
                height = np.nan
                breakout_level = np.nan

                if pattern_type == "Rectangle":
                    upper = float(p["upper_level"])
                    lower = float(p["lower_level"])
                    if close_i > upper * (1.0 + breakout_buffer_pct):
                        direction = "long"
                        support_used = lower
                        resistance_used = upper
                        breakout_level = upper
                        height = upper - lower
                    elif close_i < lower * (1.0 - breakout_buffer_pct):
                        direction = "short"
                        support_used = lower
                        resistance_used = upper
                        breakout_level = lower
                        height = upper - lower

                elif pattern_type == "Triangle":
                    su_y1 = float(p["support_y1"])
                    su_x1 = int(p["support_x1_idx"])
                    su_slope = float(p["support_slope"])
                    re_y1 = float(p["res_y1"])
                    re_x1 = int(p["res_x1_idx"])
                    re_slope = float(p["res_slope"])
                    support_t = _line_value(su_y1, su_x1, su_slope, i)
                    resist_t = _line_value(re_y1, re_x1, re_slope, i)
                    if close_i > resist_t * (1.0 + breakout_buffer_pct):
                        direction = "long"
                        support_used = support_t
                        resistance_used = resist_t
                        breakout_level = resist_t
                        height = max(0.0, resist_t - support_t)
                    elif close_i < support_t * (1.0 - breakout_buffer_pct):
                        direction = "short"
                        support_used = support_t
                        resistance_used = resist_t
                        breakout_level = support_t
                        height = max(0.0, resist_t - support_t)

                elif pattern_type == "HeadAndShoulders":
                    neckline = float(p["neckline"])
                    rs = float(p["right_shoulder_price"])
                    hd = float(p["head_price"])
                    if close_i < neckline * (1.0 - breakout_buffer_pct):
                        direction = "short"
                        support_used = neckline
                        resistance_used = rs
                        breakout_level = neckline
                        height = max(0.0, abs(hd - neckline))

                elif pattern_type == "InverseHeadAndShoulders":
                    neckline = float(p["neckline"])
                    rs = float(p["right_shoulder_price"])
                    hd = float(p["head_price"])
                    if close_i > neckline * (1.0 + breakout_buffer_pct):
                        direction = "long"
                        support_used = rs
                        resistance_used = neckline
                        breakout_level = neckline
                        height = max(0.0, abs(neckline - hd))

                if direction == "":
                    continue

                matched_candle = ""
                for cpat in today_candles:
                    if _candle_direction(cpat) == direction:
                        matched_candle = cpat
                        break
                if matched_candle == "":
                    continue

                strength = abs(close_i - breakout_level) / max(abs(breakout_level), 1e-12)
                candidates.append(
                    {
                        "pattern": p,
                        "direction": direction,
                        "candlestick_pattern": matched_candle,
                        "support_used": float(support_used),
                        "resistance_used": float(resistance_used),
                        "height": float(height),
                        "breakout_level": float(breakout_level),
                        "breakout_strength": float(strength),
                    }
                )

        if not candidates:
            i += 1
            continue

        candidates = sorted(candidates, key=lambda x: x["breakout_strength"], reverse=True)
        chosen = candidates[0]
        p = chosen["pattern"]
        direction = chosen["direction"]
        entry_idx = i + 1
        entry_price = open_next
        support_used = float(chosen["support_used"])
        resistance_used = float(chosen["resistance_used"])
        pattern_height = max(float(chosen["height"]), entry_price * 0.01)

        if direction == "long":
            stop_loss = support_used
            if not np.isfinite(stop_loss) or stop_loss >= entry_price:
                stop_loss = entry_price * 0.98
            target = max(entry_price + pattern_height, entry_price + (entry_price - stop_loss))
        else:
            stop_loss = resistance_used
            if not np.isfinite(stop_loss) or stop_loss <= entry_price:
                stop_loss = entry_price * 1.02
            target = min(entry_price - pattern_height, entry_price - (stop_loss - entry_price))

        qty = 0.0 if entry_price <= 0 else (capital / entry_price)
        if qty <= 0:
            i += 1
            continue

        exit_idx = min(len(ohlc) - 1, entry_idx + int(max_hold_bars))
        exit_price = float(ohlc["Close"].iloc[exit_idx])
        exit_reason = "time_exit"

        j = entry_idx
        while j <= min(len(ohlc) - 1, entry_idx + int(max_hold_bars)):
            lo = float(ohlc["Low"].iloc[j])
            hi = float(ohlc["High"].iloc[j])
            if direction == "long":
                stop_hit = lo <= stop_loss
                target_hit = hi >= target
                if stop_hit and target_hit:
                    exit_idx = j
                    exit_price = stop_loss
                    exit_reason = "stop_loss_and_target_same_bar_stop_assumed"
                    break
                if stop_hit:
                    exit_idx = j
                    exit_price = stop_loss
                    exit_reason = "stop_loss"
                    break
                if target_hit:
                    exit_idx = j
                    exit_price = target
                    exit_reason = "target_hit"
                    break
            else:
                stop_hit = hi >= stop_loss
                target_hit = lo <= target
                if stop_hit and target_hit:
                    exit_idx = j
                    exit_price = stop_loss
                    exit_reason = "stop_loss_and_target_same_bar_stop_assumed"
                    break
                if stop_hit:
                    exit_idx = j
                    exit_price = stop_loss
                    exit_reason = "stop_loss"
                    break
                if target_hit:
                    exit_idx = j
                    exit_price = target
                    exit_reason = "target_hit"
                    break
            j += 1

        if direction == "long":
            gross_pnl = (exit_price - entry_price) * qty
        else:
            gross_pnl = (entry_price - exit_price) * qty

        cost = transaction_cost * qty * (entry_price + exit_price)
        net_pnl = gross_pnl - cost
        ret_pct = 0.0 if capital <= 0 else (net_pnl / capital) * 100.0
        capital_before = capital
        capital = capital + net_pnl

        trend_info = _trend_snapshot(ohlc.iloc[: i + 1], cfg, trend_detector)
        trades.append(
            {
                "symbol": symbol,
                "pattern_id": str(p["pattern_id"]),
                "pattern_type": str(p["pattern_type"]),
                "breakout_direction": direction,
                "pattern_start_candle": str(pd.to_datetime(p["start_date"]).date()),
                "pattern_end_candle": str(pd.to_datetime(p["end_date"]).date()),
                "breakout_candle": str(ohlc.index[i].date()),
                "candlestick_pattern": str(chosen["candlestick_pattern"]),
                "candlestick_direction": direction,
                "support_used": float(support_used),
                "resistance_used": float(resistance_used),
                "breakout_level": float(chosen["breakout_level"]),
                "pattern_height": float(pattern_height),
                "entry_date": str(ohlc.index[entry_idx].date()),
                "entry_idx": int(entry_idx),
                "entry_price": float(entry_price),
                "stop_loss": float(stop_loss),
                "target_price": float(target),
                "exit_date": str(ohlc.index[exit_idx].date()),
                "exit_idx": int(exit_idx),
                "exit_price": float(exit_price),
                "exit_reason": exit_reason,
                "holding_bars": int(exit_idx - entry_idx + 1),
                "quantity": float(qty),
                "gross_pnl": float(gross_pnl),
                "transaction_cost": float(cost),
                "net_pnl": float(net_pnl),
                "trade_return_pct_on_capital": float(ret_pct),
                "capital_before": float(capital_before),
                "capital_after": float(capital),
                "short_term_trend": str(trend_info["short_trend"]),
                "short_term_window_bars": int(trend_info["short_window_bars"]),
                "short_term_r2": float(trend_info["short_r2"]) if np.isfinite(trend_info["short_r2"]) else np.nan,
                "long_term_trend": str(trend_info["long_trend"]),
                "long_term_window_bars": int(trend_info["long_window_bars"]),
                "long_term_r2": float(trend_info["long_r2"]) if np.isfinite(trend_info["long_r2"]) else np.nan,
            }
        )

        used_patterns.add(str(p["pattern_id"]))
        equity_points.append({"date": str(ohlc.index[exit_idx].date()), "equity": float(capital)})
        i = exit_idx + 1

    trades_df = pd.DataFrame(trades)
    events_df = patterns.copy()
    equity_df = pd.DataFrame(equity_points)

    if trades_df.empty:
        metrics = {
            "symbol": symbol,
            "initial_capital": float(initial_capital),
            "final_capital": float(initial_capital),
            "total_return_pct": 0.0,
            "num_trades": 0,
            "win_rate_pct": np.nan,
            "max_drawdown_pct": np.nan,
            "pattern_events_considered": int(len(events_df)),
        }
        return trades_df, equity_df, events_df, metrics

    win_rate = float((trades_df["net_pnl"] > 0).mean() * 100.0)
    peak = equity_df["equity"].cummax()
    dd = (equity_df["equity"] / peak) - 1.0
    metrics = {
        "symbol": symbol,
        "initial_capital": float(initial_capital),
        "final_capital": float(capital),
        "total_return_pct": float((capital / initial_capital - 1.0) * 100.0),
        "num_trades": int(len(trades_df)),
        "win_rate_pct": win_rate,
        "max_drawdown_pct": float(dd.min() * 100.0),
        "pattern_events_considered": int(len(events_df)),
    }
    return trades_df, equity_df, events_df, metrics


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    symbols = resolve_symbols(args.symbols, args.symbols_file)
    data_path = Path(args.data_path)
    summary_rows: List[dict] = []

    for s in symbols:
        try:
            ohlc = prepare_ohlc(load_symbol_frame(data_path, s))
            trades, equity, events, metrics = backtest_symbol(
                symbol=s,
                ohlc=ohlc,
                cfg=cfg,
                initial_capital=float(args.initial_capital),
                transaction_cost=float(args.transaction_cost),
                breakout_buffer_pct=float(args.breakout_buffer_pct),
                max_pattern_age_bars=int(args.max_pattern_age_bars),
                max_hold_bars=int(args.max_hold_bars),
            )

            sym_dir = out_root / s
            sym_dir.mkdir(parents=True, exist_ok=True)
            trades.to_csv(sym_dir / "trades.csv", index=False)
            equity.to_csv(sym_dir / "equity_curve.csv", index=False)
            events.to_csv(sym_dir / "pattern_events.csv", index=False)
            with open(sym_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            summary_rows.append(metrics)
            print(
                f"[ok] {s}: trades={metrics['num_trades']} final_capital={metrics['final_capital']:.2f} "
                f"return={metrics['total_return_pct']:.2f}%"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {s}: {exc}")

    if summary_rows:
        summary = pd.DataFrame(summary_rows).sort_values("total_return_pct", ascending=False)
        summary.to_csv(out_root / "summary.csv", index=False)
        with open(out_root / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(orient="records"), f, indent=2)


if __name__ == "__main__":
    main()
