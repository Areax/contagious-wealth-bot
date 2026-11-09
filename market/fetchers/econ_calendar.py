"""US economic calendar (PPI, CPI, and other releases).

MarketWatch's calendar sits behind an active DataDome CAPTCHA challenge for any
automated browser — that's real bot-detection, not just a missing-JS problem,
so it isn't something to script around. Instead this uses:
  - the public, unauthenticated ForexFactory calendar feed for the week's
    schedule + consensus forecast + previous reading (no login, no bot wall;
    widely used by trading tools for exactly this data)
  - FRED's key-less CSV endpoint (official Fed data) for the latest actual
    CPI / PPI month-over-month change, since the calendar feed doesn't carry
    "actual" until well after release
"""
import json
import time
from datetime import datetime
from pathlib import Path

import requests

from market.fetchers.fred_utils import fetch_series
from market.fetchers.paths import DATA_DIR

CACHE_PATH = DATA_DIR / "econ_calendar.json"
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; local-market-dashboard/1.0)"}

FRED_SERIES = {
    "CPI (m/m)": "CPIAUCSL",
    "PPI Final Demand (m/m)": "PPIFIS",
}
HIGHLIGHT_KEYWORDS = ["cpi", "consumer price", "ppi", "producer price"]


def _fred_latest_mom_pct(series_id):
    rows = fetch_series(series_id)
    if len(rows) < 2:
        return None
    (date_prev, val_prev), (date_last, val_last) = rows[-2], rows[-1]
    pct = (float(val_last) - float(val_prev)) / float(val_prev) * 100
    return {"date": date_last, "value": round(pct, 2)}


def _fetch_fred_actuals():
    out = {}
    for label, series_id in FRED_SERIES.items():
        try:
            out[label] = _fred_latest_mom_pct(series_id)
        except Exception:
            out[label] = None
    return out


def fetch():
    resp = requests.get(CALENDAR_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    events = resp.json()

    upcoming = []
    for e in events:
        if e.get("country") != "USD":
            continue
        if not any(k in e["title"].lower() for k in HIGHLIGHT_KEYWORDS):
            continue
        dt = datetime.fromisoformat(e["date"])
        upcoming.append({
            "date": dt.strftime("%A, %b %-d"),
            "time": dt.strftime("%-I:%M %p ET"),
            "title": e["title"],
            "impact": e["impact"],
            "forecast": e.get("forecast") or "",
            "previous": e.get("previous") or "",
        })
    upcoming.sort(key=lambda e: e["date"])

    result = {
        "as_of": int(time.time()),
        "fred_actuals": _fetch_fred_actuals(),
        "upcoming": upcoming,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    data = fetch()
    print("FRED actuals:", data["fred_actuals"])
    print("\nUpcoming PPI/CPI releases (this week's window):")
    if not data["upcoming"]:
        print("  none scheduled this week")
    for ev in data["upcoming"]:
        print(f"  {ev['date']} {ev['time']:>12} [{ev['impact']:<6}] {ev['title']:<35} "
              f"forecast={ev['forecast']:<8} prev={ev['previous']}")
