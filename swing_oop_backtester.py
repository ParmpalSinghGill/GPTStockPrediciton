#!/usr/bin/env python3
"""Confluence-based OOP swing-trading backtester.

Design separation:
1) Pattern/Candlestick feature preparation
2) Entry strategy logic (LONG/SHORT classes)
3) Execution engine + risk management
4) Reporting/export
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pattern_engine.detectors import CandlestickPatternDetector


CHART_PATTERN_COLS = [
    "rectangle_breakout",
    "rectangle_breakdown",
    "triangle_breakout",
    "triangle_breakdown",
    "double_bottom_breakout",
    "double_top_breakdown",
    "head_and_shoulders_breakdown",
    "inverse_head_and_shoulders_breakout",
]

CANDLE_PATTERN_COLS = [
    "bullish_candlestick_pattern",
    "bearish_candlestick_pattern",
]

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
    parser = argparse.ArgumentParser(description="Confluence-based swing backtest.")
    parser.add_argument("--data-dir", default="Data", help="Folder containing CSV files.")
    parser.add_argument(
        "--data-path",
        default="Data/AllSTOCKS.pk",
        help="Fallback pickle path if no CSV found for a symbol.",
    )
    parser.add_argument("--symbols", default="", help="Comma-separated symbols. Example: RELIANCE,TCS")
    parser.add_argument("--symbols-file", default="Data/nifty50_symbols.txt", help="Symbols file (comma-separated).")
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01, help="Capital fraction risked per trade.")
    parser.add_argument("--entry-slippage", type=float, default=0.0005, help="Entry slippage fraction.")
    parser.add_argument("--exit-slippage", type=float, default=0.0005, help="Exit slippage fraction.")
    parser.add_argument("--output-file", default="swing_backtest_results.xlsx")
    return parser.parse_args()


def resolve_symbols(symbols: str, symbols_file: str) -> List[str]:
    if symbols.strip():
        return [s.strip().upper() for s in symbols.split(",") if s.strip()]
    p = Path(symbols_file)
    if p.exists():
        txt = p.read_text(encoding="utf-8").strip()
        if txt:
            return [s.strip().upper() for s in txt.split(",") if s.strip()]
    return []


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("_", "")


def _pick_numeric_col(df: pd.DataFrame, names: List[str]) -> Optional[pd.Series]:
    target = {_norm(n) for n in names}
    for c in df.columns:
        if _norm(str(c)) in target:
            return pd.to_numeric(df[c], errors="coerce")
    return None


def _pick_existing_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    target = {_norm(n) for n in names}
    for c in df.columns:
        if _norm(str(c)) in target:
            return str(c)
    return None


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return frame with canonical OHLC column names while retaining extra columns."""
    o = _pick_numeric_col(df, ["Open"])
    h = _pick_numeric_col(df, ["High"])
    l = _pick_numeric_col(df, ["Low"])
    c = _pick_numeric_col(df, ["Adj Close", "AdjClose", "Close"])
    v = _pick_numeric_col(df, ["Volume"])
    if c is None:
        raise ValueError("No close price column found.")
    if o is None:
        o = c.shift(1).fillna(c)
    if h is None:
        h = pd.concat([o, c], axis=1).max(axis=1)
    if l is None:
        l = pd.concat([o, c], axis=1).min(axis=1)
    if v is None:
        v = pd.Series(np.nan, index=df.index, dtype=float)

    out = df.copy()
    out["Open"] = o
    out["High"] = h
    out["Low"] = l
    out["Close"] = c
    out["Volume"] = v

    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out


def load_symbol_frame(symbol: str, data_dir: Path, pickle_path: Path) -> pd.DataFrame:
    candidates = [data_dir / f"{symbol}.csv", data_dir / f"{symbol.lower()}.csv"]
    wildcard = sorted(data_dir.glob(f"*{symbol}*.csv"))
    candidates.extend([p for p in wildcard if p not in candidates])

    for csv_path in candidates:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        date_col = _pick_existing_col(df, ["Date", "Datetime", "Timestamp"])
        if date_col is not None:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.set_index(date_col)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{csv_path} has no datetime column.")
        return normalize_frame(df)

    if pickle_path.exists():
        raw = pd.read_pickle(pickle_path)
        if symbol not in raw:
            raise KeyError(f"Symbol '{symbol}' not found in CSV files or fallback pickle.")
        item = raw[symbol]
        if isinstance(item, pd.DataFrame):
            sym_df = item.copy()
        elif isinstance(item, dict) and {"data", "columns", "index"}.issubset(item.keys()):
            sym_df = pd.DataFrame(item["data"], columns=item["columns"], index=item["index"])
        else:
            raise ValueError(f"Unsupported pickle payload for symbol '{symbol}'.")
        return normalize_frame(sym_df)

    raise FileNotFoundError(f"No CSV found for {symbol} in {data_dir} and pickle not found at {pickle_path}.")


def _to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(float) != 0
    txt = s.astype(str).str.strip().str.lower()
    true_values = {"1", "true", "t", "yes", "y", "bull", "bear"}
    return txt.isin(true_values)


class PatternFeatureBuilder:
    """Ensures required chart/candlestick confluence flags exist on input frame."""

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # If candlestick boolean flags are missing, derive from existing detector labels.
        if "bullish_candlestick_pattern" not in out.columns or "bearish_candlestick_pattern" not in out.columns:
            cdet = CandlestickPatternDetector()
            patterns = cdet.detect(out[["Open", "High", "Low", "Close"]])
            out["bullish_candlestick_pattern"] = False
            out["bearish_candlestick_pattern"] = False
            out["candlestick_name"] = ""
            if not patterns.empty:
                out.loc[patterns.index, "candlestick_name"] = patterns["pattern"].astype(str)
                out.loc[patterns.index, "bullish_candlestick_pattern"] = (
                    patterns["pattern"].astype(str).isin(BULLISH_CANDLE_PATTERNS).values
                )
                out.loc[patterns.index, "bearish_candlestick_pattern"] = (
                    patterns["pattern"].astype(str).isin(BEARISH_CANDLE_PATTERNS).values
                )
        else:
            if "candlestick_name" not in out.columns:
                out["candlestick_name"] = ""

        # Normalize all required booleans. Missing chart pattern flags are set False.
        missing_chart_cols: List[str] = []
        for col in CHART_PATTERN_COLS + CANDLE_PATTERN_COLS:
            if col not in out.columns:
                out[col] = False
                if col in CHART_PATTERN_COLS:
                    missing_chart_cols.append(col)
            out[col] = _to_bool_series(out[col])

        if missing_chart_cols:
            print(
                "[warn] Missing chart-pattern columns in input data: "
                + ", ".join(missing_chart_cols)
                + ". Defaulted to False. "
                "If your existing pattern detection functions are external, run them before backtest and add these columns."
            )

        return out


class BaseEntryStrategy(ABC):
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, index: int) -> Optional[Dict[str, float | str | None]]:
        """Returns standardized signal object or None."""


class LongEntryStrategy(BaseEntryStrategy):
    breakout_cols = [
        "rectangle_breakout",
        "triangle_breakout",
        "double_bottom_breakout",
        "inverse_head_and_shoulders_breakout",
    ]

    def generate_signal(self, df: pd.DataFrame, index: int) -> Optional[Dict[str, float | str | None]]:
        if index < 2 or (index + 1) >= len(df):
            return None
        has_breakout = any(bool(df[col].iloc[index]) for col in self.breakout_cols)
        has_bullish_candle = bool(df["bullish_candlestick_pattern"].iloc[index])
        if not (has_breakout and has_bullish_candle):
            return None

        entry_price = float(df["Open"].iloc[index + 1]) * 1.001
        stop_loss = float(df["Low"].iloc[index - 2 : index + 1].min())
        risk = entry_price - stop_loss
        if risk <= 0:
            return None
        target = entry_price + (2.0 * risk)

        trigger_patterns = [c for c in self.breakout_cols if bool(df[c].iloc[index])]
        return {
            "signal": "LONG",
            "entry_price": float(entry_price),
            "stop_loss": float(stop_loss),
            "target": float(target),
            "trigger_pattern": " + ".join(trigger_patterns),
            "entry_reason": "Chart breakout confluence + bullish candlestick on signal candle",
        }


class ShortEntryStrategy(BaseEntryStrategy):
    breakdown_cols = [
        "rectangle_breakdown",
        "triangle_breakdown",
        "double_top_breakdown",
        "head_and_shoulders_breakdown",
    ]

    def generate_signal(self, df: pd.DataFrame, index: int) -> Optional[Dict[str, float | str | None]]:
        if index < 2 or (index + 1) >= len(df):
            return None
        has_breakdown = any(bool(df[col].iloc[index]) for col in self.breakdown_cols)
        has_bearish_candle = bool(df["bearish_candlestick_pattern"].iloc[index])
        if not (has_breakdown and has_bearish_candle):
            return None

        entry_price = float(df["Open"].iloc[index + 1]) * 0.999
        stop_loss = float(df["High"].iloc[index - 2 : index + 1].max())
        risk = stop_loss - entry_price
        if risk <= 0:
            return None
        target = entry_price - (2.0 * risk)

        trigger_patterns = [c for c in self.breakdown_cols if bool(df[c].iloc[index])]
        return {
            "signal": "SHORT",
            "entry_price": float(entry_price),
            "stop_loss": float(stop_loss),
            "target": float(target),
            "trigger_pattern": " + ".join(trigger_patterns),
            "entry_reason": "Chart breakdown confluence + bearish candlestick on signal candle",
        }


class Backtester:
    """Strategy-agnostic execution engine."""

    def __init__(
        self,
        strategy_long: BaseEntryStrategy,
        strategy_short: BaseEntryStrategy,
        initial_capital: float,
        risk_per_trade: float,
        entry_slippage: float = 0.0,
        exit_slippage: float = 0.0,
    ) -> None:
        self.strategy_long = strategy_long
        self.strategy_short = strategy_short
        self.initial_capital = float(initial_capital)
        self.risk_per_trade = float(risk_per_trade)
        self.entry_slippage = float(entry_slippage)
        self.exit_slippage = float(exit_slippage)
        self.capital = float(initial_capital)

    def calculate_position_size(self, risk_per_unit: float) -> float:
        if risk_per_unit <= 0:
            return 0.0
        return (self.capital * self.risk_per_trade) / risk_per_unit

    def update_capital(self, pnl: float) -> float:
        self.capital += float(pnl)
        return self.capital

    def execute_trade(self, trade: Dict[str, object], candle: pd.Series, idx: int) -> Optional[Dict[str, object]]:
        trade_type = str(trade["Trade type"])
        entry = float(trade["Entry price"])
        stop = float(trade["Stop loss"])
        target = float(trade["Target"])
        qty = float(trade["Quantity"])

        low = float(candle["Low"])
        high = float(candle["High"])
        exit_reason: Optional[str] = None
        raw_exit: Optional[float] = None

        if trade_type == "Long":
            stop_hit = low <= stop
            tgt_hit = high >= target
            if stop_hit and tgt_hit:
                exit_reason = "Stop Loss Hit"
                raw_exit = stop
            elif stop_hit:
                exit_reason = "Stop Loss Hit"
                raw_exit = stop
            elif tgt_hit:
                exit_reason = "Target Hit"
                raw_exit = target
        else:
            stop_hit = high >= stop
            tgt_hit = low <= target
            if stop_hit and tgt_hit:
                exit_reason = "Stop Loss Hit"
                raw_exit = stop
            elif stop_hit:
                exit_reason = "Stop Loss Hit"
                raw_exit = stop
            elif tgt_hit:
                exit_reason = "Target Hit"
                raw_exit = target

        if exit_reason is None or raw_exit is None:
            return None

        if trade_type == "Long":
            exit_price = raw_exit * (1.0 - self.exit_slippage)
            pnl = (exit_price - entry) * qty
        else:
            exit_price = raw_exit * (1.0 + self.exit_slippage)
            pnl = (entry - exit_price) * qty

        capital_before = float(self.capital)
        capital_after = self.update_capital(pnl)
        holding_days = int(idx - int(trade["Entry index"]) + 1)
        profit_pct = 0.0 if capital_before <= 0 else (pnl / capital_before) * 100.0

        trade["Exit date"] = pd.to_datetime(candle.name)
        trade["Exit price"] = float(exit_price)
        trade["Result"] = exit_reason
        trade["Profit/Loss"] = float(pnl)
        trade["Profit %"] = float(profit_pct)
        trade["Holding period"] = holding_days
        trade["Capital after trade"] = float(capital_after)
        return trade

    def run_for_stock(self, df: pd.DataFrame, stock_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        self.capital = float(self.initial_capital)
        equity_points: List[float] = [self.capital]

        trades: List[Dict[str, object]] = []
        open_trade: Optional[Dict[str, object]] = None
        print(f"[{stock_name}] Starting confluence backtest on {len(df)} candles...")

        for i in range(len(df)):
            if i % 250 == 0 and i > 0:
                print(f"[{stock_name}] Progress {i}/{len(df)} | Capital={self.capital:.2f}")

            candle = df.iloc[i]

            if open_trade is not None and i >= int(open_trade["Entry index"]):
                closed = self.execute_trade(open_trade, candle, i)
                if closed is not None:
                    trades.append(closed)
                    equity_points.append(self.capital)
                    open_trade = None

            if open_trade is not None:
                continue

            if i < 2 or (i + 1) >= len(df):
                continue

            long_signal = self.strategy_long.generate_signal(df, i)
            short_signal = self.strategy_short.generate_signal(df, i)

            selected: Optional[Dict[str, float | str | None]] = None
            if long_signal is not None and short_signal is None:
                selected = long_signal
            elif short_signal is not None and long_signal is None:
                selected = short_signal
            elif long_signal is not None and short_signal is not None:
                long_risk = float(long_signal["entry_price"]) - float(long_signal["stop_loss"])
                short_risk = float(short_signal["stop_loss"]) - float(short_signal["entry_price"])
                selected = long_signal if long_risk >= short_risk else short_signal

            if selected is None:
                continue

            side = "Long" if str(selected["signal"]) == "LONG" else "Short"
            entry_idx = i + 1
            raw_entry = float(selected["entry_price"])
            stop = float(selected["stop_loss"])
            target = float(selected["target"])

            if side == "Long":
                entry_price = raw_entry * (1.0 + self.entry_slippage)
                risk_per_unit = entry_price - stop
            else:
                entry_price = raw_entry * (1.0 - self.entry_slippage)
                risk_per_unit = stop - entry_price
            if risk_per_unit <= 0:
                continue

            qty = self.calculate_position_size(risk_per_unit)
            if qty <= 0:
                continue

            open_trade = {
                "Stock name": stock_name,
                "Trade type": side,
                "Pattern that triggered trade": str(selected.get("trigger_pattern", "")),
                "Entry reason": str(selected.get("entry_reason", "")),
                "Signal date": pd.to_datetime(df.index[i]),
                "Entry date": pd.to_datetime(df.index[entry_idx]),
                "Entry index": int(entry_idx),
                "Entry price": float(entry_price),
                "Stop loss": float(stop),
                "Target": float(target),
                "Quantity": float(qty),
                "Exit date": pd.NaT,
                "Exit price": np.nan,
                "Result": "",
                "Profit/Loss": np.nan,
                "Profit %": np.nan,
                "Holding period": np.nan,
                "Capital after trade": np.nan,
            }

        if open_trade is not None:
            last = df.iloc[-1]
            side = str(open_trade["Trade type"])
            entry = float(open_trade["Entry price"])
            qty = float(open_trade["Quantity"])
            raw_exit = float(last["Close"])
            if side == "Long":
                exit_price = raw_exit * (1.0 - self.exit_slippage)
                pnl = (exit_price - entry) * qty
            else:
                exit_price = raw_exit * (1.0 + self.exit_slippage)
                pnl = (entry - exit_price) * qty
            capital_before = float(self.capital)
            capital_after = self.update_capital(pnl)
            holding_days = int((len(df) - 1) - int(open_trade["Entry index"]) + 1)
            profit_pct = 0.0 if capital_before <= 0 else (pnl / capital_before) * 100.0

            open_trade["Exit date"] = pd.to_datetime(df.index[-1])
            open_trade["Exit price"] = float(exit_price)
            open_trade["Result"] = "End of Data Exit"
            open_trade["Profit/Loss"] = float(pnl)
            open_trade["Profit %"] = float(profit_pct)
            open_trade["Holding period"] = holding_days
            open_trade["Capital after trade"] = float(capital_after)
            trades.append(open_trade)
            equity_points.append(self.capital)

        print(f"[{stock_name}] Completed. Trades={len(trades)} Final capital={self.capital:.2f}")

        cols = [
            "Stock name",
            "Trade type",
            "Pattern that triggered trade",
            "Entry reason",
            "Signal date",
            "Entry date",
            "Entry price",
            "Stop loss",
            "Target",
            "Exit date",
            "Exit price",
            "Result",
            "Profit/Loss",
            "Profit %",
            "Holding period",
            "Capital after trade",
        ]
        if not trades:
            return pd.DataFrame(columns=cols), pd.Series(equity_points, dtype=float)

        trades_df = pd.DataFrame(trades).drop(columns=["Entry index", "Quantity"])
        trades_df = trades_df[cols]
        return trades_df, pd.Series(equity_points, dtype=float)

    def calculate_metrics(self, trades_df: pd.DataFrame, equity_series: pd.Series) -> Dict[str, float]:
        if trades_df.empty:
            return {
                "Total trades": 0,
                "Win rate": np.nan,
                "Final capital": float(self.initial_capital),
                "Max drawdown": np.nan,
                "Profit factor": np.nan,
                "Average holding period": np.nan,
                "Long trades": 0,
                "Long PnL": 0.0,
                "Long win rate": np.nan,
                "Short trades": 0,
                "Short PnL": 0.0,
                "Short win rate": np.nan,
            }

        wins = trades_df["Profit/Loss"] > 0
        total_trades = int(len(trades_df))
        win_rate = float(wins.mean() * 100.0)
        final_capital = float(trades_df["Capital after trade"].iloc[-1])

        peak = equity_series.cummax()
        dd = (equity_series / peak) - 1.0
        max_drawdown = float(dd.min() * 100.0)

        gross_profit = float(trades_df.loc[trades_df["Profit/Loss"] > 0, "Profit/Loss"].sum())
        gross_loss = float(-trades_df.loc[trades_df["Profit/Loss"] < 0, "Profit/Loss"].sum())
        profit_factor = np.nan if gross_loss == 0 else float(gross_profit / gross_loss)

        avg_holding = float(pd.to_numeric(trades_df["Holding period"], errors="coerce").mean())

        long_df = trades_df[trades_df["Trade type"] == "Long"]
        short_df = trades_df[trades_df["Trade type"] == "Short"]
        long_trades = int(len(long_df))
        short_trades = int(len(short_df))
        long_win = float(long_df["Profit/Loss"].gt(0).mean() * 100.0) if long_trades else np.nan
        short_win = float(short_df["Profit/Loss"].gt(0).mean() * 100.0) if short_trades else np.nan

        return {
            "Total trades": total_trades,
            "Win rate": win_rate,
            "Final capital": final_capital,
            "Max drawdown": max_drawdown,
            "Profit factor": profit_factor,
            "Average holding period": avg_holding,
            "Long trades": long_trades,
            "Long PnL": float(long_df["Profit/Loss"].sum()) if long_trades else 0.0,
            "Long win rate": long_win,
            "Short trades": short_trades,
            "Short PnL": float(short_df["Profit/Loss"].sum()) if short_trades else 0.0,
            "Short win rate": short_win,
        }


def export_results_excel(
    output_file: Path,
    per_stock_trades: Dict[str, pd.DataFrame],
    per_stock_metrics: Dict[str, Dict[str, float]],
) -> None:
    summary_rows: List[Dict[str, object]] = []
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for symbol, trades_df in per_stock_trades.items():
            trades_df.to_excel(writer, sheet_name=symbol[:31], index=False)
            row = {"Stock": symbol}
            row.update(per_stock_metrics[symbol])
            summary_rows.append(row)

        summary = pd.DataFrame(summary_rows)
        if not summary.empty:
            all_trades = pd.concat(
                [df for df in per_stock_trades.values() if not df.empty],
                ignore_index=True,
            ) if any(not df.empty for df in per_stock_trades.values()) else pd.DataFrame()

            totals = {
                "Stock": "OVERALL",
                "Total trades": int(summary["Total trades"].sum()),
                "Win rate": float(all_trades["Profit/Loss"].gt(0).mean() * 100.0) if not all_trades.empty else np.nan,
                "Final capital": float(summary["Final capital"].sum()),
                "Max drawdown": float(summary["Max drawdown"].min()) if summary["Max drawdown"].notna().any() else np.nan,
                "Profit factor": np.nan,
                "Average holding period": float(all_trades["Holding period"].mean()) if not all_trades.empty else np.nan,
                "Long trades": int((all_trades["Trade type"] == "Long").sum()) if not all_trades.empty else 0,
                "Long PnL": float(all_trades.loc[all_trades["Trade type"] == "Long", "Profit/Loss"].sum())
                if not all_trades.empty else 0.0,
                "Long win rate": float(
                    all_trades.loc[all_trades["Trade type"] == "Long", "Profit/Loss"].gt(0).mean() * 100.0
                ) if ((not all_trades.empty) and (all_trades["Trade type"].eq("Long").any())) else np.nan,
                "Short trades": int((all_trades["Trade type"] == "Short").sum()) if not all_trades.empty else 0,
                "Short PnL": float(all_trades.loc[all_trades["Trade type"] == "Short", "Profit/Loss"].sum())
                if not all_trades.empty else 0.0,
                "Short win rate": float(
                    all_trades.loc[all_trades["Trade type"] == "Short", "Profit/Loss"].gt(0).mean() * 100.0
                ) if ((not all_trades.empty) and (all_trades["Trade type"].eq("Short").any())) else np.nan,
            }

            if not all_trades.empty:
                gp = float(all_trades.loc[all_trades["Profit/Loss"] > 0, "Profit/Loss"].sum())
                gl = float(-all_trades.loc[all_trades["Profit/Loss"] < 0, "Profit/Loss"].sum())
                totals["Profit factor"] = np.nan if gl == 0 else float(gp / gl)

            summary = pd.concat([summary, pd.DataFrame([totals])], ignore_index=True)

        summary.to_excel(writer, sheet_name="Summary", index=False)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    pickle_path = Path(args.data_path)
    output_file = Path(args.output_file)

    symbols = resolve_symbols(args.symbols, args.symbols_file)
    if not symbols:
        symbols = sorted({p.stem.upper() for p in data_dir.glob("*.csv")})
    if not symbols:
        raise ValueError("No symbols provided and none discovered from CSV files.")

    feature_builder = PatternFeatureBuilder()
    long_strategy = LongEntryStrategy()
    short_strategy = ShortEntryStrategy()

    per_stock_trades: Dict[str, pd.DataFrame] = {}
    per_stock_metrics: Dict[str, Dict[str, float]] = {}

    print(f"Running confluence swing backtest for {len(symbols)} symbols...")
    for pos, symbol in enumerate(symbols, start=1):
        print(f"\n[{pos}/{len(symbols)}] Loading {symbol}...")
        try:
            df = load_symbol_frame(symbol, data_dir=data_dir, pickle_path=pickle_path)
            df = feature_builder.build(df)

            bt = Backtester(
                strategy_long=long_strategy,
                strategy_short=short_strategy,
                initial_capital=float(args.initial_capital),
                risk_per_trade=float(args.risk_per_trade),
                entry_slippage=float(args.entry_slippage),
                exit_slippage=float(args.exit_slippage),
            )
            trades_df, equity = bt.run_for_stock(df, stock_name=symbol)
            metrics = bt.calculate_metrics(trades_df, equity)
            per_stock_trades[symbol] = trades_df
            per_stock_metrics[symbol] = metrics

            print(
                f"[{symbol}] trades={metrics['Total trades']} win_rate={metrics['Win rate']:.2f}% "
                f"long={metrics['Long trades']} short={metrics['Short trades']} final_capital={metrics['Final capital']:.2f}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{symbol}] ERROR: {exc}")
            per_stock_trades[symbol] = pd.DataFrame()
            per_stock_metrics[symbol] = {
                "Total trades": 0,
                "Win rate": np.nan,
                "Final capital": float(args.initial_capital),
                "Max drawdown": np.nan,
                "Profit factor": np.nan,
                "Average holding period": np.nan,
                "Long trades": 0,
                "Long PnL": 0.0,
                "Long win rate": np.nan,
                "Short trades": 0,
                "Short PnL": 0.0,
                "Short win rate": np.nan,
            }

    export_results_excel(output_file, per_stock_trades, per_stock_metrics)
    print(f"\nBacktest complete. Results saved to: {output_file}")


if __name__ == "__main__":
    main()
