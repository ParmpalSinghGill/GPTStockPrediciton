# Swing Trading Model + Strict 1Y/2Y Holdout Backtesting

This project is configured for **active swing trading**, not long-term holding.

It tests fixed holding-period strategies such as:

- 5 trading days (~1 week)
- 10 trading days (~2 weeks)
- 21 trading days (~1 month)

For each holding period, the script:

1. Trains models to predict forward swing return.
2. Selects the best model + entry threshold by time-series CV.
3. Runs out-of-sample backtests on:
   - last 2 years (not used in training)
   - last 1 year (not used in training)
4. Compares strategy return against passive alternatives:
   - FD at 12%
   - NIFTY50
   - SENSEX
   - stock buy-and-hold

## Data

Input: `Data/AllSTOCKS.pk`

Expected format:

- `dict[symbol] -> {'data', 'columns', 'index'}`
- or DataFrame per symbol.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python train_backtest.py --symbol RELIANCE
```

Optional arguments:

- `--holding-periods "5,10,21"`
- `--thresholds "0,0.005,0.01,0.015,0.02,0.03"`
- `--transaction-cost 0.001`
- `--initial-capital 100000`
- `--fd-rate 0.12`

## Run NIFTY50 Batch

```bash
.venv/bin/python run_nifty50_batch.py --workers 4
```

## Pattern Detection + Plotting (Any Stock)

### Prepare NIFTY symbol list

Create `Data/nifty50_symbols.txt` with comma-separated symbols. Example:

```text
RELIANCE,TCS,INFY,HDFCBANK
```

### Run for selected symbols

```bash
.venv/bin/python support_resistance_patterns.py --symbols "RELIANCE,TCS,INFY"
```

### Run for full NIFTY list from file

```bash
.venv/bin/python support_resistance_patterns.py --symbols-file Data/nifty50_symbols.txt --output-root pattern_plots
```

Important options:

- `--lookback-bars 260`
- `--level-lookback-bars 900`
- `--pivot-window 5`
- `--level-tolerance-pct 0.01`
- `--min-psy-touches 2`
- `--psychological-step 0` (auto step)
- `--min-trend-touches 3`
- `--max-cross-bars 2`
- `--max-cross-streak 2`
- `--trend-max-age-bars 220`
- `--rectangle-min-bars 20`
- `--rectangle-max-bars 220`
- `--rectangle-min-touches 2`
- `--rectangle-max-outside-bars 2`

Outputs are generated in flat folders under `pattern_plots/`:

- `psychological_lines/<SYMBOL>_psychological.png`
- `head_shoulders/<SYMBOL>_head_shoulders.png`
- `double_patterns/<SYMBOL>_double_patterns.png`
- `candlestick_patterns/<SYMBOL>_candlestick_patterns.png`
- `rectangle_patterns/<SYMBOL>_rectangle_patterns.png`
- `pattern_summary.csv`
- `pattern_summary.json`

Main batch outputs are saved in `reports_nifty50/`:

- `selected_strategy_ranking.csv`
- `all_comparison.csv`
- `run_status.csv`

## Outputs

Artifacts are generated under:

`reports/<SYMBOL>/`

Main files:

- `analysis.json` - selected strategy and whether it beats FD/NIFTY/SENSEX
- `backtest_summary_all.csv` - all windows and holding periods
- `cv_results_all.csv` - CV metrics for all candidates

Per holding period `h` (e.g. `h=5`):

- `predictions_h<h>.csv`
- `backtest_summary_h<h>.csv`
- `cv_results_h<h>.csv`
- `feature_importance_h<h>.csv`
- `daily_backtest_2y_h<h>.csv`
- `daily_backtest_1y_h<h>.csv`
- `trade_log_2y_h<h>.csv`
- `trade_log_1y_h<h>.csv`
- `equity_curves_2y_h<h>.png`
- `equity_curves_1y_h<h>.png`
