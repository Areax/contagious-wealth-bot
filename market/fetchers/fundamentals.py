"""Debt, free cash flow, cash and P/E for every stock the Stock lookup page can
show (sector top-10 holdings plus the daily S&P 500 scan's stocks).

From Yahoo via yfinance: yearly and quarterly balance sheet / cash flow rows,
plus trailing and forward P/E. Each stock is cached for a week since these
numbers only change when a company reports.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from market.fetchers.paths import DATA_DIR

CACHE_PATH = DATA_DIR / "fundamentals.json"
TTL = 7 * 24 * 3600
YEARS, QUARTERS = 5, 6
DEBT_ROWS = ["Total Debt"]
CASH_ROWS = ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]
FCF_ROWS = ["Free Cash Flow"]


def _row(df: pd.DataFrame | None, names: list[str], col) -> float | None:
    if df is None or df.empty or col not in df.columns:
        return None
    for name in names:
        if name in df.index:
            v = df.at[name, col]
            if pd.notna(v):
                return float(v)
    return None


def _series(bs, cf, n, quarterly: bool) -> list[dict]:
    cols = sorted(set(bs.columns if bs is not None else []) | set(cf.columns if cf is not None else []))[-n:]
    out = []
    for col in cols:
        debt, cash, fcf = _row(bs, DEBT_ROWS, col), _row(bs, CASH_ROWS, col), _row(cf, FCF_ROWS, col)
        if debt is None and cash is None and fcf is None:
            continue  # Yahoo often lists an old column with nothing in it
        ts = pd.Timestamp(col)
        out.append({"period": f"Q{(ts.month - 1) // 3 + 1} {ts.year}" if quarterly else str(ts.year),
                    "end": ts.strftime("%Y-%m-%d"), "debt": debt, "cash": cash, "fcf": fcf})
    return out


def _num(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _one(sym: str) -> dict:
    t = yf.Ticker(sym)
    info = t.info or {}
    annual = _series(t.balance_sheet, t.cashflow, YEARS, False)
    quarterly = _series(t.quarterly_balance_sheet, t.quarterly_cashflow, QUARTERS, True)
    q_fcf = [q["fcf"] for q in quarterly[-4:] if q["fcf"] is not None]
    return {
        "fetched": time.time(),
        "pe": _num(info.get("trailingPE")), "forward_pe": _num(info.get("forwardPE")),
        "currency": info.get("financialCurrency") or info.get("currency"),
        "annual": annual, "quarterly": quarterly,
        "fcf_ttm": sum(q_fcf) if len(q_fcf) == 4 else None,
    }


def symbols() -> list[str]:
    syms = set()
    for name in ("sector_signals.json", "daily_scan.json"):
        path = DATA_DIR / name
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for sec in data.get("sectors", []):
            syms.update(s["symbol"] for s in sec.get("stocks", []) if not s.get("missing"))
        syms.update(data.get("stocks", {}))
    return sorted(syms)


def fetch() -> dict:
    try:
        cache = json.loads(CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        cache = {}
    now = time.time()
    for sym in symbols():
        entry = cache.get(sym)
        if entry and now - entry.get("fetched", 0) < TTL:
            continue
        try:
            cache[sym] = _one(sym)
        except Exception:
            pass  # keep the older entry, if any
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, separators=(",", ":")))
    return cache


if __name__ == "__main__":
    data = fetch()
    print(len(data), "stocks")
    for sym in ("SOLV", "AAPL"):
        if sym in data:
            d = data[sym]
            print(sym, d["pe"], d["forward_pe"], d["fcf_ttm"], d["annual"][-1] if d["annual"] else None)
