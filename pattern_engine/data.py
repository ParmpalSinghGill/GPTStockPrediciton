from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def load_symbol_frame(data_path: Path, symbol: str) -> pd.DataFrame:
    raw = pd.read_pickle(data_path)
    if symbol not in raw:
        raise KeyError(f"Symbol '{symbol}' not found")
    item = raw[symbol]
    if isinstance(item, pd.DataFrame):
        df = item.copy()
    elif isinstance(item, dict) and {"data", "columns", "index"}.issubset(item.keys()):
        df = pd.DataFrame(item["data"], columns=item["columns"], index=item["index"])
    else:
        raise ValueError(f"Unsupported format for {symbol}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.dropna(how="all")


def _norm(c: str) -> str:
    return c.strip().lower().replace(" ", "").replace("_", "")


def _col(df: pd.DataFrame, names: List[str]) -> pd.Series | None:
    names_n = [_norm(n) for n in names]
    for c in df.columns:
        if _norm(str(c)) in names_n:
            return pd.to_numeric(df[c], errors="coerce")
    return None


def prepare_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    o = _col(df, ["Open"])
    h = _col(df, ["High"])
    l = _col(df, ["Low"])
    c = _col(df, ["Adj Close", "AdjClose", "Close"])
    v = _col(df, ["Volume"])
    if c is None:
        raise ValueError("No close price")
    if o is None:
        o = c.shift(1).fillna(c)
    if h is None:
        h = pd.concat([o, c], axis=1).max(axis=1)
    if l is None:
        l = pd.concat([o, c], axis=1).min(axis=1)
    if v is None:
        v = pd.Series(np.nan, index=df.index, dtype=float)
    out = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=df.index)
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def resolve_symbols(symbols: str, symbols_file: str) -> List[str]:
    if symbols.strip():
        return [s.strip().upper() for s in symbols.split(",") if s.strip()]
    f = Path(symbols_file)
    if f.exists():
        txt = f.read_text(encoding="utf-8").strip()
        if txt:
            return [s.strip().upper() for s in txt.split(",") if s.strip()]
    return ["RELIANCE"]
