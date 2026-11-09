"""VIX level + recent history, and S&P 500 index level, via Yahoo Finance."""
import json
import time
from pathlib import Path

import yfinance as yf
from market.fetchers.paths import DATA_DIR

CACHE_PATH = DATA_DIR / "vix.json"


def fetch():
    vix = yf.Ticker("^VIX").history(period="6mo", interval="1d")
    spx = yf.Ticker("^GSPC").history(period="6mo", interval="1d")

    vix_hist = [
        {"date": ts.strftime("%Y-%m-%d"), "close": round(row["Close"], 2)}
        for ts, row in vix.iterrows()
    ]
    spx_hist = [
        {"date": ts.strftime("%Y-%m-%d"), "close": round(row["Close"], 2)}
        for ts, row in spx.iterrows()
    ]

    latest_vix = vix_hist[-1]["close"]
    prev_vix = vix_hist[-2]["close"] if len(vix_hist) > 1 else latest_vix
    latest_spx = spx_hist[-1]["close"]
    prev_spx = spx_hist[-2]["close"] if len(spx_hist) > 1 else latest_spx

    result = {
        "as_of": int(time.time()),
        "vix": {
            "latest": latest_vix,
            "pct_change": round((latest_vix - prev_vix) / prev_vix * 100, 2),
            "history": vix_hist,
        },
        "spx": {
            "latest": latest_spx,
            "pct_change": round((latest_spx - prev_spx) / prev_spx * 100, 2),
            "history": spx_hist,
        },
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(result))
    return result


if __name__ == "__main__":
    data = fetch()
    print("VIX:", data["vix"]["latest"], data["vix"]["pct_change"], "%")
    print("SPX:", data["spx"]["latest"], data["spx"]["pct_change"], "%")
    print(f"{len(data['vix']['history'])} days of history")
