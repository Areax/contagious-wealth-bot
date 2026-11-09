"""Self-computed replica of CME FedWatch's "Aggregated" view: for each
upcoming FOMC meeting, the cumulative probability of each possible target
rate range, derived from 30-Day Fed Funds futures prices.

CME's own description of "Aggregated": it "compares the rates implied by
CME's Fed Funds futures with the CURRENT target rate range... a view into
the cumulative number of hikes or cuts the market is pricing by a certain
point in the future." That's a direct one-shot comparison of each future
month's contract against today's rate — not a meeting-to-meeting tree. (An
earlier attempt using the open-source `pyfedwatch` library, which builds a
full combinatorial tree across meetings, did NOT reproduce CME's numbers —
that's their separate "Conditional" view. This module was reverse-engineered
and validated against a real CME FedWatch screenshot instead.)

Data, all free and license-clean:
  - 30-Day Fed Funds futures prices (ZQ contracts): Yahoo Finance. These
    trade on CME's own exchange (CME Globex) — Yahoo doesn't estimate them,
    it redistributes the same exchange data. Verified to match CME's own
    displayed prices to the tick.
  - Current EFFR + target rate band, and a fallback for already-expired
    contract months Yahoo has delisted: FRED (official Fed data).

Known imprecision: for a meeting immediately followed by ANOTHER meeting the
next month (no clean gap month to read a single resolved rate from), the
resolved post-meeting rate has to be backed out from that shared month's own
blended price. Small rounding in the inputs (FRED EFFR is only published to
2 decimals) gets amplified there, worst-case by a few percentage points when
the true answer sits very close to 0% or 100%. Validated to be exact (within
0.1pt) on meetings that DO have a clean gap month, and within roughly 1-3pts
on the rest.
"""
import calendar
import json
import time
from datetime import datetime
from pathlib import Path

import yfinance as yf

from market.fetchers.fred_utils import latest_value, month_average
from market.fetchers.paths import DATA_DIR

CACHE_PATH = DATA_DIR / "fedwatch.json"

CME_MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
                    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}

# Known FOMC meeting dates (decision/announcement day). The Fed publishes
# these ~1-2 years ahead; update this list when they announce further out.
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_DATES = [
    "2026-01-27", "2026-03-17", "2026-04-28", "2026-06-16", "2026-07-28",
    "2026-09-16", "2026-10-28", "2026-12-09",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28",
    "2027-09-15", "2027-10-27", "2027-12-08",
]
NUM_UPCOMING = 10


def _month_key(date_str):
    return date_str[:7]


def _next_month_key(month_key):
    y, m = map(int, month_key.split("-"))
    return f"{y+1:04d}-01" if m == 12 else f"{y:04d}-{m+1:02d}"


def _has_meeting(month_key):
    return any(_month_key(d) == month_key for d in FOMC_DATES)


def _contract_symbol(month_key):
    y, m = map(int, month_key.split("-"))
    return f"ZQ{CME_MONTH_CODES[m]}{y % 100:02d}"


def _implied_rate(month_key, cache):
    if month_key in cache:
        return cache[month_key]
    symbol = _contract_symbol(month_key)
    price = None
    try:
        hist = yf.Ticker(symbol + ".CBT").history(period="5d")
        if hist is not None and not hist.empty:
            price = float(hist["Close"].iloc[-1])
    except Exception:
        price = None
    if price is None:
        # contract already expired & delisted from Yahoo -> derive from FRED's
        # actual EFFR average for that month (that IS the settlement basis)
        y, m = map(int, month_key.split("-"))
        avg_effr = month_average("DFF", y, m)
        price = 100 - avg_effr
    rate = 100 - price
    cache[month_key] = rate
    return rate


def _rate_bucket_label(low, high):
    # low/high are percent rates like 3.75/4.00 -> displayed CME-style as "375-400"
    return f"{low*100:.0f}-{high*100:.0f}"


def fetch():
    watch_date = datetime.now()
    watch_date_str = watch_date.strftime("%Y-%m-%d")

    _, ll = latest_value("DFEDTARL")
    ll_date, ul = latest_value("DFEDTARU")
    _, effr = latest_value("DFF")

    upcoming = [d for d in sorted(FOMC_DATES) if d > watch_date_str][:NUM_UPCOMING]

    rate_cache = {}
    resolved = {}
    prev_rate = effr
    for date in upcoming:
        month_key = _month_key(date)
        nxt = _next_month_key(month_key)
        try:
            if not _has_meeting(nxt):
                r_after = _implied_rate(nxt, rate_cache)
            else:
                n = calendar.monthrange(int(month_key[:4]), int(month_key[5:7]))[1]
                meeting_day = int(date[8:10])
                pre_days = meeting_day - 1
                post_days = n - pre_days
                avg = _implied_rate(month_key, rate_cache)
                r_after = (avg * n - pre_days * prev_rate) / post_days
        except Exception:
            continue
        resolved[date] = r_after
        prev_rate = r_after

    rows = []
    all_k = set()
    for date in upcoming:
        if date not in resolved:
            continue
        ek = (resolved[date] - effr) / 0.25
        k_lo = int(ek // 1)
        frac = ek - k_lo
        p_lo, p_hi = round((1 - frac) * 100, 2), round(frac * 100, 2)
        has_hi = p_hi > 0.005
        d = datetime.strptime(date, "%Y-%m-%d")
        rows.append({
            "date": date, "date_label": d.strftime("%-m/%-d/%Y"),
            "k_lo": k_lo, "p_lo": p_lo, "p_hi": p_hi if has_hi else 0.0,
        })
        all_k.add(k_lo)
        if has_hi:
            all_k.add(k_lo + 1)

    # fixed rate-range columns (CME's own "Aggregated" table layout), sorted low to high
    columns = []
    for k in sorted(all_k):
        band = (ll + 0.25 * k, ul + 0.25 * k)
        columns.append({"k": k, "label": _rate_bucket_label(*band)})

    meetings = []
    for row in rows:
        cells = []
        for col in columns:
            if col["k"] == row["k_lo"]:
                prob = row["p_lo"]
            elif col["k"] == row["k_lo"] + 1:
                prob = row["p_hi"]
            else:
                prob = 0.0
            nonzero = [v for v in (row["p_lo"], row["p_hi"]) if v > 0.005]
            if prob > 0.005 and nonzero and prob == max(nonzero):
                highlight = "blue"
            elif prob > 0.005 and nonzero and prob == min(nonzero):
                highlight = "yellow"
            else:
                highlight = ""
            cells.append({"label": col["label"], "probability": round(prob, 2), "highlight": highlight})
        meetings.append({"date": row["date"], "date_label": row["date_label"], "cells": cells})

    result = {
        "as_of": int(time.time()),
        "watch_date": watch_date_str,
        "current_target": f"{ll:.2f}-{ul:.2f}",
        "effr": effr,
        "columns": [c["label"] for c in columns],
        "meetings": meetings,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    data = fetch()
    print(f"Current target: {data['current_target']}  EFFR: {data['effr']}")
    print("Columns:", data["columns"])
    for m in data["meetings"]:
        parts = ", ".join(f"{c['label']}={c['probability']}%({c['highlight'] or '-'})" for c in m["cells"])
        print(f"  {m['date_label']:<12} {parts}")
