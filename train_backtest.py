#!/usr/bin/env python3
"""Swing-trading model training and strict out-of-sample backtesting.

Key behavior:
1) Builds predictive models for swing horizons (default 5/10/21 trading days).
2) Uses time-series CV on pre-holdout data only to select model + threshold.
3) Backtests each swing strategy on strict holdouts:
   - last 2 years (never used in training)
   - last 1 year (never used in training)
4) Compares active strategy performance with passive alternatives:
   FD 12% annualized, NIFTY50, SENSEX, and stock buy-and-hold.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRADING_DAYS_PER_YEAR = 252


@dataclass
class StrategyConfig:
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 10 bps per side
    fd_annual_rate: float = 0.12
    threshold_grid: Tuple[float, ...] = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03)
    holding_periods: Tuple[int, ...] = (5, 10, 21)
    random_state: int = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train swing-trading models and backtest 1y/2y holdouts."
    )
    parser.add_argument(
        "--data-path", default="Data/AllSTOCKS.pk", help="Path to AllSTOCKS pickle."
    )
    parser.add_argument(
        "--symbol", default="RELIANCE", help="Stock symbol key in AllSTOCKS.pk."
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory where reports and artifacts will be written.",
    )
    parser.add_argument(
        "--initial-capital", type=float, default=100000.0, help="Backtest start capital."
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
        help="Transaction cost per buy/sell (e.g., 0.001 = 10 bps).",
    )
    parser.add_argument(
        "--fd-rate", type=float, default=0.12, help="Annualized fixed deposit return."
    )
    parser.add_argument(
        "--thresholds",
        default="0,0.005,0.01,0.015,0.02,0.03",
        help="Comma-separated predicted-return thresholds for entering swing trades.",
    )
    parser.add_argument(
        "--holding-periods",
        default="5,10,21",
        help="Comma-separated holding periods in trading days (e.g., 5,10,21).",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility."
    )
    return parser.parse_args()


def load_symbol_frame(data_path: Path, symbol: str) -> pd.DataFrame:
    raw = pd.read_pickle(data_path)
    if symbol not in raw:
        sample = sorted(raw.keys())[:20]
        raise KeyError(
            f"Symbol '{symbol}' not found in {data_path}. Sample available symbols: {sample}"
        )
    item = raw[symbol]
    if isinstance(item, pd.DataFrame):
        df = item.copy()
    elif isinstance(item, dict) and {"data", "columns", "index"}.issubset(item.keys()):
        df = pd.DataFrame(item["data"], columns=item["columns"], index=item["index"])
    else:
        raise ValueError(
            f"Unsupported data format for symbol '{symbol}': {type(item)}."
        )
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.dropna(how="all")


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan)


def get_price_series(df: pd.DataFrame) -> pd.Series:
    if "Adj Close" in df.columns and "Close" in df.columns:
        adj = df["Adj Close"].astype(float)
        close = df["Close"].astype(float)
        # Some symbols have stale/mostly-missing adjusted prices in recent periods.
        recent_n = min(252, len(df))
        adj_recent_cov = float(adj.tail(recent_n).notna().mean())
        close_recent_cov = float(close.tail(recent_n).notna().mean())
        if (adj_recent_cov >= 0.95) and (adj.notna().sum() >= close.notna().sum() * 0.95):
            return adj
        return close
    if "Adj Close" in df.columns:
        return df["Adj Close"].astype(float)
    if "Close" in df.columns:
        return df["Close"].astype(float)
    raise ValueError("Input symbol does not contain Close/Adj Close columns.")


def compute_rsi(price: pd.Series, period: int = 14) -> pd.Series:
    diff = price.diff()
    up = diff.clip(lower=0.0)
    down = (-diff).clip(lower=0.0)
    avg_gain = up.rolling(period).mean()
    avg_loss = down.rolling(period).mean()
    rs = _safe_div(avg_gain, avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    price = get_price_series(df)
    high = df["High"].astype(float) if "High" in df.columns else pd.Series(index=df.index, dtype=float)
    low = df["Low"].astype(float) if "Low" in df.columns else pd.Series(index=df.index, dtype=float)
    open_ = df["Open"].astype(float) if "Open" in df.columns else pd.Series(index=df.index, dtype=float)
    volume = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(index=df.index, dtype=float)

    ret_1 = price.pct_change(fill_method=None)
    feat = pd.DataFrame(index=df.index)
    feat["ret_1"] = ret_1
    feat["ret_2"] = price.pct_change(2, fill_method=None)
    feat["ret_3"] = price.pct_change(3, fill_method=None)
    feat["ret_5"] = price.pct_change(5, fill_method=None)
    feat["ret_10"] = price.pct_change(10, fill_method=None)
    feat["ret_20"] = price.pct_change(20, fill_method=None)

    feat["vol_5"] = ret_1.rolling(5).std()
    feat["vol_10"] = ret_1.rolling(10).std()
    feat["vol_20"] = ret_1.rolling(20).std()

    sma_5 = price.rolling(5).mean()
    sma_10 = price.rolling(10).mean()
    sma_20 = price.rolling(20).mean()
    sma_50 = price.rolling(50).mean()
    ema_12 = price.ewm(span=12, adjust=False).mean()
    ema_26 = price.ewm(span=26, adjust=False).mean()

    feat["sma_5_20"] = _safe_div(sma_5, sma_20) - 1.0
    feat["sma_10_50"] = _safe_div(sma_10, sma_50) - 1.0
    feat["price_vs_sma20"] = _safe_div(price, sma_20) - 1.0
    feat["price_vs_sma50"] = _safe_div(price, sma_50) - 1.0
    feat["macd"] = _safe_div(ema_12 - ema_26, price)
    feat["rsi_14"] = compute_rsi(price, 14)

    if volume.notna().any():
        feat["vol_chg_1"] = volume.pct_change(fill_method=None)
        feat["vol_chg_5"] = volume.pct_change(5, fill_method=None)
        feat["vol_ratio_5_20"] = _safe_div(volume.rolling(5).mean(), volume.rolling(20).mean()) - 1.0
    else:
        feat["vol_chg_1"] = np.nan
        feat["vol_chg_5"] = np.nan
        feat["vol_ratio_5_20"] = np.nan

    if high.notna().any() and low.notna().any():
        feat["hl_spread"] = _safe_div(high - low, price)
    else:
        feat["hl_spread"] = np.nan
    if open_.notna().any():
        feat["co_spread"] = _safe_div(price - open_, open_)
    else:
        feat["co_spread"] = np.nan

    feat["day_of_week"] = feat.index.dayofweek.astype(float)
    feat["month"] = feat.index.month.astype(float)
    return feat.replace([np.inf, -np.inf], np.nan)


def build_forward_return_target(price: pd.Series, hold_days: int) -> pd.Series:
    return (price.shift(-hold_days) / price) - 1.0


def annualized_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    vol = float(r.std(ddof=1))
    if vol == 0.0:
        return np.nan
    return float((r.mean() / vol) * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    drawdown = (equity / peak) - 1.0
    return float(drawdown.min())


def compute_return_metrics(
    returns: pd.Series, initial_capital: float, num_trades: float | None = None
) -> Dict[str, float]:
    r = returns.dropna().astype(float)
    if r.empty:
        return {
            "observations": 0,
            "total_return_pct": np.nan,
            "cagr_pct": np.nan,
            "annual_vol_pct": np.nan,
            "sharpe": np.nan,
            "max_drawdown_pct": np.nan,
            "win_rate_pct": np.nan,
            "final_capital": np.nan,
            "num_trades": np.nan if num_trades is None else float(num_trades),
        }

    equity = initial_capital * (1.0 + r).cumprod()
    total_return = (equity.iloc[-1] / initial_capital) - 1.0
    n = len(r)
    cagr = (1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / n) - 1.0
    ann_vol = float(r.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return {
        "observations": int(n),
        "total_return_pct": float(total_return * 100.0),
        "cagr_pct": float(cagr * 100.0),
        "annual_vol_pct": float(ann_vol * 100.0),
        "sharpe": annualized_sharpe(r),
        "max_drawdown_pct": float(max_drawdown(equity) * 100.0),
        "win_rate_pct": float((r > 0).mean() * 100.0),
        "final_capital": float(equity.iloc[-1]),
        "num_trades": np.nan if num_trades is None else float(num_trades),
    }


def simulate_swing_strategy(
    pred_score: pd.Series,
    daily_returns: pd.Series,
    hold_days: int,
    threshold: float,
    transaction_cost: float,
) -> pd.DataFrame:
    idx = pred_score.index.intersection(daily_returns.index)
    pred_score = pred_score.reindex(idx).astype(float)
    daily_returns = daily_returns.reindex(idx).astype(float).fillna(0.0)

    raw_signal = (pred_score > threshold).astype(int)
    tradable_signal = raw_signal.shift(1).fillna(0).astype(int)

    strategy_returns = np.zeros(len(idx), dtype=float)
    position = np.zeros(len(idx), dtype=float)
    turnover = np.zeros(len(idx), dtype=float)
    entry = np.zeros(len(idx), dtype=float)
    exit_ = np.zeros(len(idx), dtype=float)

    in_position = False
    days_left = 0
    for i in range(len(idx)):
        day_turnover = 0.0

        if (not in_position) and tradable_signal.iat[i] == 1:
            in_position = True
            days_left = hold_days
            day_turnover += 1.0
            entry[i] = 1.0

        pos_today = 1.0 if in_position else 0.0
        day_ret = pos_today * daily_returns.iat[i]

        if in_position:
            days_left -= 1
            if days_left == 0:
                in_position = False
                day_turnover += 1.0
                exit_[i] = 1.0

        day_ret -= day_turnover * transaction_cost

        strategy_returns[i] = day_ret
        position[i] = pos_today
        turnover[i] = day_turnover

    # Force final liquidation if backtest ends while still in position.
    if in_position and len(idx) > 0:
        strategy_returns[-1] -= transaction_cost
        turnover[-1] += 1.0
        exit_[-1] = 1.0

    return pd.DataFrame(
        {
            "pred_score": pred_score,
            "signal_raw": raw_signal,
            "signal_tradable": tradable_signal,
            "daily_return": daily_returns,
            "position": position,
            "turnover": turnover,
            "entry": entry,
            "exit": exit_,
            "strategy_return": strategy_returns,
        },
        index=idx,
    )


def extract_trade_log(sim_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    idx = sim_df.index
    open_idx = None

    for i in range(len(sim_df)):
        if sim_df["entry"].iat[i] == 1.0 and open_idx is None:
            open_idx = i
        if sim_df["exit"].iat[i] == 1.0 and open_idx is not None:
            trade_slice = sim_df.iloc[open_idx : i + 1]
            trade_ret = float((1.0 + trade_slice["strategy_return"]).prod() - 1.0)
            underlying_ret = float((1.0 + trade_slice["daily_return"]).prod() - 1.0)
            rows.append(
                {
                    "entry_date": str(idx[open_idx].date()),
                    "exit_date": str(idx[i].date()),
                    "holding_days": int(i - open_idx + 1),
                    "net_trade_return_pct": trade_ret * 100.0,
                    "underlying_return_pct": underlying_ret * 100.0,
                    "entry_pred_score": float(sim_df["pred_score"].iat[open_idx]),
                }
            )
            open_idx = None

    return pd.DataFrame(rows)


def build_models(random_state: int) -> Dict[str, object]:
    return {
        "ridge": Pipeline(
            steps=[("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=350,
            max_depth=8,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
            random_state=random_state,
        ),
    }


def model_selection_via_cv_swing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    daily_ret_train: pd.Series,
    config: StrategyConfig,
    hold_days: int,
) -> Tuple[pd.DataFrame, str, float, float]:
    models = build_models(config.random_state)
    tscv = TimeSeriesSplit(n_splits=5)
    rows: List[dict] = []
    model_best: Dict[str, Tuple[float, float, float]] = {}
    # model_best[name] = (best_mean_sharpe, best_threshold, best_mean_total_return)

    for model_name, model in models.items():
        fold_metrics: List[Tuple[float, float, float]] = []
        sharpe_by_threshold: Dict[float, List[float]] = {
            th: [] for th in config.threshold_grid
        }
        total_ret_by_threshold: Dict[float, List[float]] = {
            th: [] for th in config.threshold_grid
        }

        for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X_train), start=1):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            dret_va = daily_ret_train.iloc[va_idx]

            mdl = clone(model)
            mdl.fit(X_tr, y_tr)
            pred_va = pd.Series(mdl.predict(X_va), index=X_va.index, name="pred")

            mae = float(mean_absolute_error(y_va, pred_va))
            rmse = float(np.sqrt(mean_squared_error(y_va, pred_va)))
            dir_acc = float(((pred_va > 0) == (y_va > 0)).mean())
            fold_metrics.append((mae, rmse, dir_acc))

            for th in config.threshold_grid:
                sim_va = simulate_swing_strategy(
                    pred_score=pred_va,
                    daily_returns=dret_va,
                    hold_days=hold_days,
                    threshold=th,
                    transaction_cost=config.transaction_cost,
                )
                strat_ret = sim_va["strategy_return"]
                sharpe = annualized_sharpe(strat_ret)
                tot = float((1.0 + strat_ret).prod() - 1.0)
                sharpe_by_threshold[th].append(sharpe)
                total_ret_by_threshold[th].append(tot)

            rows.append(
                {
                    "hold_days": hold_days,
                    "model": model_name,
                    "fold": fold_idx,
                    "mae": mae,
                    "rmse": rmse,
                    "directional_accuracy": dir_acc,
                }
            )

        mean_mae = float(np.mean([m[0] for m in fold_metrics]))
        mean_rmse = float(np.mean([m[1] for m in fold_metrics]))
        mean_dir_acc = float(np.mean([m[2] for m in fold_metrics]))

        best_threshold = config.threshold_grid[0]
        best_sharpe = -np.inf
        best_total_ret = -np.inf

        for th in config.threshold_grid:
            sh_vals = pd.Series(sharpe_by_threshold[th], dtype=float).replace(
                [np.inf, -np.inf], np.nan
            )
            tr_vals = pd.Series(total_ret_by_threshold[th], dtype=float).replace(
                [np.inf, -np.inf], np.nan
            )
            mean_sharpe = float(sh_vals.mean(skipna=True))
            mean_total_ret = float(tr_vals.mean(skipna=True))

            rows.append(
                {
                    "hold_days": hold_days,
                    "model": model_name,
                    "fold": "mean",
                    "threshold": th,
                    "mean_cv_sharpe": mean_sharpe,
                    "mean_cv_total_return_pct": mean_total_ret * 100.0,
                    "mean_mae": mean_mae,
                    "mean_rmse": mean_rmse,
                    "mean_directional_accuracy": mean_dir_acc,
                }
            )

            if (mean_sharpe > best_sharpe) or (
                np.isclose(mean_sharpe, best_sharpe, equal_nan=False)
                and mean_total_ret > best_total_ret
            ):
                best_sharpe = mean_sharpe
                best_total_ret = mean_total_ret
                best_threshold = th

        model_best[model_name] = (best_sharpe, float(best_threshold), best_total_ret)

    cv_df = pd.DataFrame(rows)
    ranked = sorted(
        model_best.items(),
        key=lambda kv: (
            -np.nan_to_num(kv[1][0], nan=-1e9),
            -np.nan_to_num(kv[1][2], nan=-1e9),
            cv_df.loc[
                (cv_df["model"] == kv[0]) & (cv_df["fold"] == "mean"), "mean_rmse"
            ].min(),
        ),
    )
    best_model_name = ranked[0][0]
    best_threshold = ranked[0][1][1]
    best_cv_sharpe = ranked[0][1][0]
    return cv_df, best_model_name, best_threshold, best_cv_sharpe


def download_index_series(
    ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.Series:
    start_buffer = (start_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end_buffer = (end_date + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    df = yf.download(
        ticker,
        start=start_buffer,
        end=end_buffer,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise ValueError(f"Unable to download index data for {ticker}.")
    if "Close" not in df.columns:
        raise ValueError(f"Downloaded {ticker} data does not contain Close column.")
    close = df["Close"].copy()
    close.index = pd.to_datetime(close.index)
    return close.sort_index()


def build_baseline_returns(
    window_index: pd.DatetimeIndex,
    fd_annual_rate: float,
    nifty_close: pd.Series | None,
    sensex_close: pd.Series | None,
) -> Dict[str, pd.Series]:
    baselines: Dict[str, pd.Series] = {}
    if len(window_index) == 0:
        return baselines

    fd_daily_rate = (1.0 + fd_annual_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    baselines["fd_12pct"] = pd.Series(fd_daily_rate, index=window_index, dtype=float)

    if nifty_close is not None:
        baselines["nifty50"] = (
            nifty_close.reindex(window_index).ffill().pct_change().fillna(0.0)
        )
    if sensex_close is not None:
        baselines["sensex"] = (
            sensex_close.reindex(window_index).ffill().pct_change().fillna(0.0)
        )
    return baselines


def run_backtest_window_swing(
    pred_score: pd.Series,
    daily_returns: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    hold_days: int,
    threshold: float,
    config: StrategyConfig,
    nifty_close: pd.Series | None,
    sensex_close: pd.Series | None,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series], pd.DataFrame]:
    pred_win = pred_score.loc[(pred_score.index >= start_date) & (pred_score.index <= end_date)]
    if pred_win.empty:
        raise ValueError(f"No prediction rows in backtest window {start_date} to {end_date}.")

    sim = simulate_swing_strategy(
        pred_score=pred_win,
        daily_returns=daily_returns.reindex(pred_win.index),
        hold_days=hold_days,
        threshold=threshold,
        transaction_cost=config.transaction_cost,
    )
    sim["stock_buy_hold"] = sim["daily_return"]

    baseline_returns = build_baseline_returns(
        sim.index, config.fd_annual_rate, nifty_close, sensex_close
    )
    for k, v in baseline_returns.items():
        sim[k] = v.reindex(sim.index).fillna(0.0)

    rows: List[dict] = []
    strategy_metrics = compute_return_metrics(
        sim["strategy_return"],
        initial_capital=config.initial_capital,
        num_trades=float(sim["entry"].sum()),
    )
    strategy_metrics["name"] = "model_strategy"
    rows.append(strategy_metrics)

    stock_metrics = compute_return_metrics(
        sim["stock_buy_hold"], initial_capital=config.initial_capital
    )
    stock_metrics["name"] = "stock_buy_hold"
    rows.append(stock_metrics)

    for name in ["fd_12pct", "nifty50", "sensex"]:
        if name in sim.columns:
            base_metrics = compute_return_metrics(
                sim[name], initial_capital=config.initial_capital
            )
            base_metrics["name"] = name
            rows.append(base_metrics)

    metrics_df = pd.DataFrame(rows)
    metrics_df["beat_model_strategy"] = pd.Series(
        pd.NA, index=metrics_df.index, dtype="boolean"
    )
    metrics_df["strategy_beats_all_passive"] = pd.Series(
        pd.NA, index=metrics_df.index, dtype="boolean"
    )

    if "model_strategy" in metrics_df["name"].values:
        strategy_total = float(
            metrics_df.loc[metrics_df["name"] == "model_strategy", "total_return_pct"].iloc[
                0
            ]
        )
        mask_others = metrics_df["name"] != "model_strategy"
        metrics_df.loc[mask_others, "beat_model_strategy"] = metrics_df.loc[
            mask_others, "total_return_pct"
        ].apply(lambda x: bool(strategy_total > float(x)))

        passive_mask = metrics_df["name"].isin(["fd_12pct", "nifty50", "sensex"])
        if passive_mask.any():
            passive_totals = metrics_df.loc[passive_mask, "total_return_pct"].astype(float)
            beats_all_passive = bool((strategy_total > passive_totals).all())
            metrics_df.loc[
                metrics_df["name"] == "model_strategy", "strategy_beats_all_passive"
            ] = beats_all_passive

    curve_map = {
        "strategy": sim["strategy_return"],
        "stock_buy_hold": sim["stock_buy_hold"],
        "fd_12pct": sim["fd_12pct"] if "fd_12pct" in sim.columns else None,
        "nifty50": sim["nifty50"] if "nifty50" in sim.columns else None,
        "sensex": sim["sensex"] if "sensex" in sim.columns else None,
    }
    return metrics_df, curve_map, sim


def extract_feature_importance(model: object, feature_names: Iterable[str]) -> pd.DataFrame:
    feature_names = list(feature_names)
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = getattr(model, "feature_importances_")
    elif hasattr(model, "named_steps") and "model" in model.named_steps:
        inner_model = model.named_steps["model"]
        if hasattr(inner_model, "coef_"):
            imp = np.abs(np.asarray(inner_model.coef_).reshape(-1))
    if imp is None:
        return pd.DataFrame(columns=["feature", "importance"])
    imp_arr = np.asarray(imp).reshape(-1)
    if len(imp_arr) != len(feature_names):
        return pd.DataFrame(columns=["feature", "importance"])
    out = pd.DataFrame({"feature": feature_names, "importance": imp_arr})
    return out.sort_values("importance", ascending=False)


def plot_equity_curves(
    series_map: Dict[str, pd.Series | None],
    initial_capital: float,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(12, 6))
    for name, ret in series_map.items():
        if ret is None:
            continue
        r = ret.dropna()
        if r.empty:
            continue
        equity = initial_capital * (1.0 + r).cumprod()
        plt.plot(equity.index, equity.values, label=name)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity Value")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    symbol = args.symbol.upper()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir) / symbol
    output_dir.mkdir(parents=True, exist_ok=True)

    config = StrategyConfig(
        initial_capital=float(args.initial_capital),
        transaction_cost=float(args.transaction_cost),
        fd_annual_rate=float(args.fd_rate),
        threshold_grid=tuple(float(x.strip()) for x in args.thresholds.split(",") if x.strip()),
        holding_periods=tuple(int(x.strip()) for x in args.holding_periods.split(",") if x.strip()),
        random_state=int(args.random_state),
    )
    if not config.threshold_grid:
        raise ValueError("At least one threshold is required.")
    if not config.holding_periods:
        raise ValueError("At least one holding period is required.")

    df = load_symbol_frame(data_path, symbol)
    price = get_price_series(df)
    daily_ret = price.pct_change(fill_method=None)
    features = engineer_features(df)

    if len(features.dropna()) < 1200:
        raise ValueError(
            f"Not enough usable rows after feature engineering for {symbol}."
        )

    end_date = features.index.max()
    backtest_2y_start = end_date - pd.DateOffset(years=2)
    backtest_1y_start = end_date - pd.DateOffset(years=1)

    yf_cache_dir = output_dir / ".yf_cache"
    yf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YFINANCE_CACHE_DIR", str(yf_cache_dir))
    try:
        yf.set_tz_cache_location(str(yf_cache_dir))
    except Exception:  # noqa: BLE001
        pass

    nifty_close = None
    sensex_close = None
    index_fetch_error = None
    try:
        nifty_close = download_index_series(
            "^NSEI", start_date=features.index.min(), end_date=features.index.max()
        )
        sensex_close = download_index_series(
            "^BSESN", start_date=features.index.min(), end_date=features.index.max()
        )
    except Exception as exc:  # noqa: BLE001
        index_fetch_error = str(exc)

    all_cv_rows: List[pd.DataFrame] = []
    all_summary_rows: List[pd.DataFrame] = []
    strategy_registry: List[dict] = []
    per_holding_latest: Dict[int, dict] = {}

    for hold_days in config.holding_periods:
        target = build_forward_return_target(price, hold_days)
        data = features.copy()
        data["target_return"] = target
        data["daily_return"] = daily_ret
        data = data.dropna()

        train_mask = data.index < backtest_2y_start
        train_df = data.loc[train_mask].copy()
        holdout_df = data.loc[~train_mask].copy()
        if len(train_df) < 600:
            raise ValueError(
                f"Training rows too few ({len(train_df)}) for hold_days={hold_days}."
            )
        if len(holdout_df) < 300:
            raise ValueError(
                f"Holdout rows too few ({len(holdout_df)}) for hold_days={hold_days}."
            )

        X_train = train_df.drop(columns=["target_return", "daily_return"])
        y_train = train_df["target_return"]
        dret_train = train_df["daily_return"]

        X_all = data.drop(columns=["target_return", "daily_return"])
        y_all = data["target_return"]

        cv_df, best_model_name, best_threshold, best_cv_sharpe = model_selection_via_cv_swing(
            X_train=X_train,
            y_train=y_train,
            daily_ret_train=dret_train,
            config=config,
            hold_days=hold_days,
        )
        all_cv_rows.append(cv_df)

        model = build_models(config.random_state)[best_model_name]
        model.fit(X_train, y_train)

        pred_all = pd.Series(model.predict(X_all), index=X_all.index, name="predicted_return")
        pred_exit_price = price.reindex(pred_all.index) * (1.0 + pred_all)
        pred_df = pd.DataFrame(
            {
                "predicted_return": pred_all,
                "actual_forward_return": y_all,
                "daily_return": data["daily_return"],
                "close_price": price.reindex(pred_all.index),
                "predicted_exit_price": pred_exit_price,
            }
        ).dropna()

        metrics_2y, curves_2y, sim_2y = run_backtest_window_swing(
            pred_score=pred_df["predicted_return"],
            daily_returns=pred_df["daily_return"],
            start_date=backtest_2y_start,
            end_date=end_date,
            hold_days=hold_days,
            threshold=best_threshold,
            config=config,
            nifty_close=nifty_close,
            sensex_close=sensex_close,
        )
        metrics_2y["window"] = "2y_holdout"
        metrics_2y["hold_days"] = hold_days
        metrics_2y["best_model"] = best_model_name
        metrics_2y["best_threshold"] = best_threshold
        metrics_2y["cv_sharpe"] = best_cv_sharpe

        metrics_1y, curves_1y, sim_1y = run_backtest_window_swing(
            pred_score=pred_df["predicted_return"],
            daily_returns=pred_df["daily_return"],
            start_date=backtest_1y_start,
            end_date=end_date,
            hold_days=hold_days,
            threshold=best_threshold,
            config=config,
            nifty_close=nifty_close,
            sensex_close=sensex_close,
        )
        metrics_1y["window"] = "1y_holdout"
        metrics_1y["hold_days"] = hold_days
        metrics_1y["best_model"] = best_model_name
        metrics_1y["best_threshold"] = best_threshold
        metrics_1y["cv_sharpe"] = best_cv_sharpe

        summary_h = pd.concat([metrics_2y, metrics_1y], ignore_index=True)
        all_summary_rows.append(summary_h)

        cv_df.to_csv(output_dir / f"cv_results_h{hold_days}.csv", index=False)
        pred_df.to_csv(output_dir / f"predictions_h{hold_days}.csv", index_label="date")
        summary_h.to_csv(output_dir / f"backtest_summary_h{hold_days}.csv", index=False)

        feature_imp = extract_feature_importance(model, X_train.columns)
        feature_imp.to_csv(output_dir / f"feature_importance_h{hold_days}.csv", index=False)

        sim_2y.to_csv(output_dir / f"daily_backtest_2y_h{hold_days}.csv", index_label="date")
        sim_1y.to_csv(output_dir / f"daily_backtest_1y_h{hold_days}.csv", index_label="date")
        extract_trade_log(sim_2y).to_csv(
            output_dir / f"trade_log_2y_h{hold_days}.csv", index=False
        )
        extract_trade_log(sim_1y).to_csv(
            output_dir / f"trade_log_1y_h{hold_days}.csv", index=False
        )

        plot_equity_curves(
            series_map=curves_2y,
            initial_capital=config.initial_capital,
            title=f"{symbol} Swing {hold_days}D vs Baselines (2Y Holdout)",
            out_path=output_dir / f"equity_curves_2y_h{hold_days}.png",
        )
        plot_equity_curves(
            series_map=curves_1y,
            initial_capital=config.initial_capital,
            title=f"{symbol} Swing {hold_days}D vs Baselines (1Y Holdout)",
            out_path=output_dir / f"equity_curves_1y_h{hold_days}.png",
        )

        latest_pred = float(pred_df["predicted_return"].iloc[-1])
        per_holding_latest[hold_days] = {
            "last_feature_date": str(pred_df.index.max().date()),
            "last_close_price": float(pred_df["close_price"].iloc[-1]),
            "predicted_forward_return_pct": latest_pred * 100.0,
            "predicted_exit_price": float(pred_df["predicted_exit_price"].iloc[-1]),
            "entry_signal_today": bool(latest_pred > best_threshold),
        }

        strategy_registry.append(
            {
                "hold_days": hold_days,
                "best_model": best_model_name,
                "best_threshold": best_threshold,
                "cv_sharpe": best_cv_sharpe,
            }
        )

    cv_all = pd.concat(all_cv_rows, ignore_index=True)
    summary_all = pd.concat(all_summary_rows, ignore_index=True)
    summary_all = summary_all[
        [
            "window",
            "hold_days",
            "best_model",
            "best_threshold",
            "cv_sharpe",
            "name",
            "observations",
            "total_return_pct",
            "cagr_pct",
            "annual_vol_pct",
            "sharpe",
            "max_drawdown_pct",
            "win_rate_pct",
            "num_trades",
            "final_capital",
            "beat_model_strategy",
            "strategy_beats_all_passive",
        ]
    ]

    cv_all.to_csv(output_dir / "cv_results_all.csv", index=False)
    summary_all.to_csv(output_dir / "backtest_summary_all.csv", index=False)

    return_comp = (
        summary_all.pivot_table(
            index=["window", "hold_days"],
            columns="name",
            values="total_return_pct",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in ["model_strategy", "fd_12pct", "nifty50", "sensex", "stock_buy_hold"]:
        if col not in return_comp.columns:
            return_comp[col] = np.nan

    return_comp["strategy_gt_fd12"] = return_comp["model_strategy"] > return_comp["fd_12pct"]
    return_comp["strategy_gt_nifty50"] = return_comp["model_strategy"] > return_comp["nifty50"]
    return_comp["strategy_gt_sensex"] = return_comp["model_strategy"] > return_comp["sensex"]
    return_comp["strategy_gt_all_passive"] = (
        return_comp["strategy_gt_fd12"]
        & return_comp["strategy_gt_nifty50"]
        & return_comp["strategy_gt_sensex"]
    )
    return_comp.to_csv(output_dir / "return_comparison.csv", index=False)

    ranked = sorted(
        strategy_registry,
        key=lambda x: -np.nan_to_num(float(x["cv_sharpe"]), nan=-1e9),
    )
    selected = ranked[0]

    selected_rows = summary_all[
        (summary_all["hold_days"] == selected["hold_days"])
        & (summary_all["name"] == "model_strategy")
    ][["window", "total_return_pct", "strategy_beats_all_passive"]]

    selected_comp_rows = return_comp[return_comp["hold_days"] == selected["hold_days"]]
    selected_comparison = {
        row["window"]: {
            "model_strategy_return_pct": float(row["model_strategy"]),
            "fd_12pct_return_pct": float(row["fd_12pct"]),
            "nifty50_return_pct": float(row["nifty50"]) if not pd.isna(row["nifty50"]) else None,
            "sensex_return_pct": float(row["sensex"]) if not pd.isna(row["sensex"]) else None,
            "strategy_gt_fd12": bool(row["strategy_gt_fd12"]),
            "strategy_gt_nifty50": bool(row["strategy_gt_nifty50"]),
            "strategy_gt_sensex": bool(row["strategy_gt_sensex"]),
            "strategy_gt_all_passive": bool(row["strategy_gt_all_passive"]),
        }
        for _, row in selected_comp_rows.iterrows()
    }

    selected_outcomes = {
        row["window"]: {
            "strategy_total_return_pct": float(row["total_return_pct"]),
            "beats_fd_nifty_sensex": (
                None
                if pd.isna(row["strategy_beats_all_passive"])
                else bool(row["strategy_beats_all_passive"])
            ),
        }
        for _, row in selected_rows.iterrows()
    }

    analysis = {
        "symbol": symbol,
        "data_path": str(data_path),
        "date_range": {
            "min": str(df.index.min().date()),
            "max": str(df.index.max().date()),
            "feature_min": str(features.index.min().date()),
            "feature_max": str(features.index.max().date()),
        },
        "split": {
            "train_end_exclusive": str(backtest_2y_start.date()),
            "backtest_2y_start": str(backtest_2y_start.date()),
            "backtest_1y_start": str(backtest_1y_start.date()),
            "backtest_end": str(end_date.date()),
        },
        "config": {
            "holding_periods": list(config.holding_periods),
            "threshold_grid": list(config.threshold_grid),
            "transaction_cost": config.transaction_cost,
            "fd_annual_rate": config.fd_annual_rate,
            "initial_capital": config.initial_capital,
        },
        "selected_strategy_by_cv": selected,
        "selected_strategy_holdout_outcomes": selected_outcomes,
        "selected_strategy_return_comparison": selected_comparison,
        "latest_forecasts": per_holding_latest,
        "index_fetch_error": index_fetch_error,
        "artifacts": {
            "cv_results_all_csv": str(output_dir / "cv_results_all.csv"),
            "backtest_summary_all_csv": str(output_dir / "backtest_summary_all.csv"),
            "return_comparison_csv": str(output_dir / "return_comparison.csv"),
        },
    }
    with open(output_dir / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print("=== Run Complete (Swing Trading) ===")
    print(f"Symbol: {symbol}")
    print(f"Holding periods tested: {list(config.holding_periods)}")
    print(
        f"Selected by CV -> hold_days={selected['hold_days']}, "
        f"model={selected['best_model']}, threshold={selected['best_threshold']}, "
        f"cv_sharpe={selected['cv_sharpe']:.4f}"
    )
    if index_fetch_error:
        print(f"Index fetch warning: {index_fetch_error}")

    print("\nModel strategy rows:")
    model_rows = summary_all[summary_all["name"] == "model_strategy"][
        [
            "window",
            "hold_days",
            "total_return_pct",
            "cagr_pct",
            "num_trades",
            "strategy_beats_all_passive",
        ]
    ]
    print(model_rows.to_string(index=False))
    print("\nReturn comparison (all returns in %):")
    comp_cols = [
        "window",
        "hold_days",
        "model_strategy",
        "fd_12pct",
        "nifty50",
        "sensex",
        "stock_buy_hold",
        "strategy_gt_all_passive",
    ]
    print(return_comp[comp_cols].to_string(index=False))
    print(f"\nArtifacts saved in: {output_dir}")


if __name__ == "__main__":
    main()
