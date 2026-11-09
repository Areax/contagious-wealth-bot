"""S&P 500 1-day performance heatmap data — built from free Yahoo Finance data
(not scraped from Finviz, which blocks iframes and forbids automated scraping).
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yfinance as yf

from market.fetchers.sp500_list import get_constituents
from market.fetchers.paths import DATA_DIR

CACHE_PATH = DATA_DIR / "heatmap.json"
MCAP_CACHE_PATH = DATA_DIR / "market_caps.json"
MCAP_MAX_AGE_SECONDS = 3 * 24 * 3600  # market caps drift slowly; only used for box sizing


def _load_market_caps(tickers):
    if MCAP_CACHE_PATH.exists():
        age = time.time() - MCAP_CACHE_PATH.stat().st_mtime
        if age < MCAP_MAX_AGE_SECONDS:
            cached = json.loads(MCAP_CACHE_PATH.read_text())
            if all(t in cached for t in tickers):
                return cached

    caps = {}

    def _one(t):
        try:
            fi = yf.Ticker(t).fast_info
            return t, fi.get("marketCap")
        except Exception:
            return t, None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(_one, t) for t in tickers]
        for fut in as_completed(futures):
            t, cap = fut.result()
            caps[t] = cap

    MCAP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MCAP_CACHE_PATH.write_text(json.dumps(caps))
    return caps


def fetch():
    constituents = get_constituents()
    tickers = [c["ticker"] for c in constituents]

    hist = yf.download(tickers, period="5d", group_by="ticker", threads=True, progress=False)

    caps = _load_market_caps(tickers)

    rows = []
    for c in constituents:
        t = c["ticker"]
        try:
            closes = hist[t]["Close"].dropna()
            if len(closes) < 2:
                continue
            prev, last = closes.iloc[-2], closes.iloc[-1]
            pct = round((last - prev) / prev * 100, 2)
        except Exception:
            continue
        cap = caps.get(t) or 1
        rows.append({
            "ticker": t,
            "name": c["name"],
            "sector": c["sector"],
            "pct_change": pct,
            "market_cap": cap,
        })

    result = {"as_of": int(time.time()), "stocks": rows}
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(result))
    return result


if __name__ == "__main__":
    data = fetch()
    print(f"{len(data['stocks'])} stocks fetched")
    print(sorted(data["stocks"], key=lambda r: -r["market_cap"])[:10])
