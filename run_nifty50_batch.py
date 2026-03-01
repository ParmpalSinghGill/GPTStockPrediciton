#!/usr/bin/env python3
"""Run swing backtests for all NIFTY50 symbols and rank top performers."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


NIFTY50_CSV_URL = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch run train_backtest.py on all NIFTY50 symbols."
    )
    parser.add_argument(
        "--data-path", default="Data/AllSTOCKS.pk", help="Path to AllSTOCKS pickle."
    )
    parser.add_argument(
        "--output-root",
        default="reports_nifty50",
        help="Directory where per-symbol outputs and ranking files are written.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel symbol runs.",
    )
    parser.add_argument(
        "--holding-periods",
        default="5,10,21",
        help="Holding periods in trading days passed to train_backtest.py.",
    )
    parser.add_argument(
        "--thresholds",
        default="0,0.005,0.01,0.015,0.02,0.03",
        help="Threshold grid passed to train_backtest.py.",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
        help="Transaction cost per buy/sell passed to train_backtest.py.",
    )
    parser.add_argument(
        "--fd-rate",
        type=float,
        default=0.12,
        help="FD annual rate passed to train_backtest.py.",
    )
    return parser.parse_args()


def normalize_symbol(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def load_nifty50_symbols() -> List[str]:
    nifty = pd.read_csv(NIFTY50_CSV_URL)
    syms = (
        nifty["Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    if len(syms) != 50:
        print(f"[warn] expected 50 constituents, got {len(syms)}")
    return syms


def map_symbols_to_dataset(nifty_symbols: List[str], data_path: Path) -> Tuple[List[str], List[str], pd.DataFrame]:
    raw = pd.read_pickle(data_path)
    keys = set(raw.keys())
    norm_to_key = {normalize_symbol(k): k for k in keys}

    mapped: List[str] = []
    missing: List[str] = []
    rows: List[dict] = []
    for sym in nifty_symbols:
        if sym in keys:
            mapped.append(sym)
            rows.append(
                {"nifty_symbol": sym, "dataset_symbol": sym, "match_type": "exact"}
            )
        else:
            n = normalize_symbol(sym)
            if n in norm_to_key:
                mapped_sym = norm_to_key[n]
                mapped.append(mapped_sym)
                rows.append(
                    {
                        "nifty_symbol": sym,
                        "dataset_symbol": mapped_sym,
                        "match_type": "normalized",
                    }
                )
            else:
                missing.append(sym)
    return mapped, missing, pd.DataFrame(rows)


def run_one_symbol(
    symbol: str,
    output_root: Path,
    holding_periods: str,
    thresholds: str,
    transaction_cost: float,
    fd_rate: float,
) -> dict:
    cmd = [
        ".venv/bin/python",
        "train_backtest.py",
        "--symbol",
        symbol,
        "--output-dir",
        str(output_root),
        "--holding-periods",
        holding_periods,
        "--thresholds",
        thresholds,
        "--transaction-cost",
        str(transaction_cost),
        "--fd-rate",
        str(fd_rate),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    sym_dir = output_root / symbol
    analysis_path = sym_dir / "analysis.json"
    rc_path = sym_dir / "return_comparison.csv"
    summary_path = sym_dir / "backtest_summary_all.csv"

    result = {
        "symbol": symbol,
        "returncode": int(proc.returncode),
        "elapsed_sec": float(elapsed),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-30:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-30:]),
        "analysis_path": str(analysis_path),
        "return_comparison_path": str(rc_path),
        "summary_path": str(summary_path),
    }

    if proc.returncode != 0:
        return result
    if not analysis_path.exists() or not rc_path.exists() or not summary_path.exists():
        result["returncode"] = 999
        result["stderr_tail"] = (
            result["stderr_tail"] + "\nMissing expected output artifacts."
        ).strip()
    return result


def build_rankings(
    symbols: List[str],
    output_root: Path,
    analysis_rows: List[dict],
) -> Dict[str, pd.DataFrame]:
    comp_rows: List[pd.DataFrame] = []
    best_rows: List[dict] = []

    for sym in symbols:
        comp_path = output_root / sym / "return_comparison.csv"
        analysis_path = output_root / sym / "analysis.json"
        if not comp_path.exists() or not analysis_path.exists():
            continue

        comp = pd.read_csv(comp_path)
        comp["symbol"] = sym
        comp_rows.append(comp)

        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)
        selected = analysis.get("selected_strategy_by_cv", {})
        sel_hold = int(selected.get("hold_days"))
        selected_comp = comp[comp["hold_days"] == sel_hold].copy()
        if selected_comp.empty:
            continue

        one_y = selected_comp[selected_comp["window"] == "1y_holdout"]
        two_y = selected_comp[selected_comp["window"] == "2y_holdout"]
        if one_y.empty or two_y.empty:
            continue
        r1 = one_y.iloc[0]
        r2 = two_y.iloc[0]
        best_rows.append(
            {
                "symbol": sym,
                "selected_hold_days": sel_hold,
                "selected_model": selected.get("best_model"),
                "cv_sharpe": selected.get("cv_sharpe"),
                "ret_1y_pct": float(r1["model_strategy"]),
                "ret_2y_pct": float(r2["model_strategy"]),
                "fd_1y_pct": float(r1["fd_12pct"]),
                "fd_2y_pct": float(r2["fd_12pct"]),
                "nifty_1y_pct": float(r1["nifty50"]),
                "nifty_2y_pct": float(r2["nifty50"]),
                "sensex_1y_pct": float(r1["sensex"]),
                "sensex_2y_pct": float(r2["sensex"]),
                "beat_all_1y": bool(r1["strategy_gt_all_passive"]),
                "beat_all_2y": bool(r2["strategy_gt_all_passive"]),
                "beat_all_both_windows": bool(
                    bool(r1["strategy_gt_all_passive"])
                    and bool(r2["strategy_gt_all_passive"])
                ),
                "avg_1y_2y_pct": float((r1["model_strategy"] + r2["model_strategy"]) / 2.0),
                "min_excess_vs_passive_pct": float(
                    min(
                        r1["model_strategy"] - r1["fd_12pct"],
                        r1["model_strategy"] - r1["nifty50"],
                        r1["model_strategy"] - r1["sensex"],
                        r2["model_strategy"] - r2["fd_12pct"],
                        r2["model_strategy"] - r2["nifty50"],
                        r2["model_strategy"] - r2["sensex"],
                    )
                ),
            }
        )

    all_comp = pd.concat(comp_rows, ignore_index=True) if comp_rows else pd.DataFrame()
    selected_df = pd.DataFrame(best_rows)
    if not selected_df.empty:
        selected_df = selected_df.sort_values(
            ["beat_all_both_windows", "avg_1y_2y_pct", "ret_1y_pct", "ret_2y_pct"],
            ascending=[False, False, False, False],
        )

    failed_df = pd.DataFrame(analysis_rows)
    if not failed_df.empty:
        failed_df = failed_df.sort_values(["returncode", "symbol"])

    return {
        "all_comparison": all_comp,
        "selected_strategy_ranking": selected_df,
        "run_status": failed_df,
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("[info] Loading NIFTY50 symbols from NSE CSV...")
    nifty_symbols = load_nifty50_symbols()
    mapped, missing, mapping_df = map_symbols_to_dataset(
        nifty_symbols, Path(args.data_path)
    )
    mapping_df.to_csv(output_root / "nifty50_symbol_mapping.csv", index=False)
    print(f"[info] Constituents: {len(nifty_symbols)} | mapped: {len(mapped)} | missing: {len(missing)}")
    if missing:
        print("[warn] Missing symbols:", missing)

    analysis_rows: List[dict] = []
    print(f"[info] Starting batch run with workers={args.workers} ...")
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        fut_map = {
            ex.submit(
                run_one_symbol,
                sym,
                output_root,
                args.holding_periods,
                args.thresholds,
                args.transaction_cost,
                args.fd_rate,
            ): sym
            for sym in mapped
        }
        done = 0
        total = len(fut_map)
        for fut in as_completed(fut_map):
            sym = fut_map[fut]
            try:
                out = fut.result()
            except Exception as exc:  # noqa: BLE001
                out = {
                    "symbol": sym,
                    "returncode": 998,
                    "elapsed_sec": 0.0,
                    "stdout_tail": "",
                    "stderr_tail": str(exc),
                    "analysis_path": "",
                    "return_comparison_path": "",
                    "summary_path": "",
                }
            analysis_rows.append(out)
            done += 1
            if out["returncode"] == 0:
                print(f"[{done}/{total}] OK   {sym:12s} {out['elapsed_sec']:.1f}s")
            else:
                print(f"[{done}/{total}] FAIL {sym:12s} code={out['returncode']}")

    ranking = build_rankings(mapped, output_root, analysis_rows)
    for name, df in ranking.items():
        df.to_csv(output_root / f"{name}.csv", index=False)

    selected_df = ranking["selected_strategy_ranking"]
    print("\n=== Top Performing (Selected strategy per stock) ===")
    if selected_df.empty:
        print("No successful symbol results to rank.")
    else:
        cols = [
            "symbol",
            "selected_hold_days",
            "ret_1y_pct",
            "ret_2y_pct",
            "fd_1y_pct",
            "fd_2y_pct",
            "nifty_1y_pct",
            "nifty_2y_pct",
            "sensex_1y_pct",
            "sensex_2y_pct",
            "beat_all_1y",
            "beat_all_2y",
            "beat_all_both_windows",
            "avg_1y_2y_pct",
        ]
        print(selected_df[cols].head(20).to_string(index=False))

    status = ranking["run_status"]
    fail_count = int((status["returncode"] != 0).sum()) if not status.empty else 0
    print(f"\n[done] Results in: {output_root}")
    print(f"[done] Total mapped symbols: {len(mapped)} | failures: {fail_count}")


if __name__ == "__main__":
    main()
