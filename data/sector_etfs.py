"""
The 11 SPDR Select Sector ETFs (the "XL" funds) and their top-10 holdings,
from yfinance's fund data (the same free Yahoo feed as everything else).
Holdings move slowly, so they are cached for a week.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yfinance as yf

SECTOR_ETFS = [
    ("XLK", "Technology"),
    ("XLC", "Communication Services"),
    ("XLY", "Consumer Discretionary"),
    ("XLP", "Consumer Staples"),
    ("XLV", "Health Care"),
    ("XLF", "Financials"),
    ("XLI", "Industrials"),
    ("XLE", "Energy"),
    ("XLB", "Materials"),
    ("XLRE", "Real Estate"),
    ("XLU", "Utilities"),
]

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "etf_holdings"
CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60


def get_top_holdings(etf: str, force_refresh: bool = False) -> list[dict]:
    """Returns [{symbol, name, weight}] for the fund's top 10 holdings,
    weight as a fraction (0.24 = 24% of the fund)."""
    path = CACHE_DIR / f"{etf}.json"
    if not force_refresh and path.exists() and time.time() - path.stat().st_mtime < CACHE_MAX_AGE_SECONDS:
        return json.loads(path.read_text())

    try:
        df = yf.Ticker(etf).funds_data.top_holdings
        records = [
            {"symbol": str(sym), "name": str(row["Name"]), "weight": float(row["Holding Percent"])}
            for sym, row in df.iterrows()
        ]
    except Exception as exc:
        if path.exists():
            return json.loads(path.read_text())
        raise RuntimeError(f"Could not fetch holdings for {etf} and no cache exists: {exc}") from exc

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2))
    except OSError:
        pass
    return records
