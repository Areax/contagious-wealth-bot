"""Earnings calendar from Yahoo Finance — both the upcoming week and the
previous week, so the dashboard can toggle between them.

Yahoo's calendar page ignores from/to alone for picking which day's table to
show (every day looked identical when tested) — the actual day filter is a
separate `day=YYYY-MM-DD` query param used by its own day-tabs. So this does
one page load per weekday with that param, tagging each row with its real
earnings date.
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from market.fetchers.browser_utils import browser_page
from market.fetchers.paths import DATA_DIR

CACHE_PATH = DATA_DIR / "earnings_calendar.json"
DAYS_SPAN = 7
TOP_N_PER_RANGE = 60


def _parse_day_table(text):
    header, _, body = text.partition("Follow\n")
    blocks = [b.strip() for b in body.split("\n\n\n") if b.strip()]
    rows = []
    for block in blocks:
        fields = [f.strip() for f in block.replace("\n", "\t").split("\t") if f.strip()]
        if not fields or fields[0].lower() == "symbol":
            continue
        if len(fields) < 7:
            continue
        symbol, company, event, call_time = fields[0], fields[1], fields[2], fields[3]
        eps_est, reported_eps = fields[4], fields[5]
        surprise = fields[6] if fields[6] not in ("-",) else ""
        market_cap = fields[7] if len(fields) > 7 else ""
        rows.append({
            "symbol": symbol,
            "company": company,
            "call_time": call_time,
            "eps_estimate": eps_est,
            "reported_eps": reported_eps,
            "surprise_pct": surprise,
            "market_cap": market_cap,
        })
    return rows


def _cap_to_number(cap_str):
    if not cap_str or cap_str in ("-", "--"):
        return 0
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    suffix = cap_str[-1]
    try:
        if suffix in mult:
            return float(cap_str[:-1]) * mult[suffix]
        return float(cap_str)
    except ValueError:
        return 0


def _fetch_range(page, start, end):
    range_start_str = start.strftime("%Y-%m-%d")
    range_end_str = end.strftime("%Y-%m-%d")

    rows = []
    day = start
    while day <= end:
        if day.weekday() < 5:  # Mon-Fri only, US markets closed on weekends
            day_str = day.strftime("%Y-%m-%d")
            url = (f"https://finance.yahoo.com/calendar/earnings"
                   f"?from={range_start_str}&to={range_end_str}&day={day_str}")
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            try:
                page.wait_for_selector("table", timeout=15000)
                page.wait_for_timeout(700)
                text = page.eval_on_selector("table", "el => el.innerText")
            except Exception:
                text = ""
            for r in _parse_day_table(text):
                r["date"] = day_str
                r["date_label"] = day.strftime("%a, %b %-d")
                rows.append(r)
        day += timedelta(days=1)

    # dedupe by (symbol, date) - same company can appear with multiple share classes
    seen = set()
    unique_rows = []
    for r in rows:
        key = (r["symbol"], r["date"])
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(r)
    # keep the top N by market cap, then sort those for display: date, then symbol
    unique_rows.sort(key=lambda r: -_cap_to_number(r["market_cap"]))
    top_rows = unique_rows[:TOP_N_PER_RANGE]
    top_rows.sort(key=lambda r: (r["date"], r["symbol"]))

    return {
        "range": f"{start.strftime('%b %-d')} - {end.strftime('%b %-d')}",
        "companies": top_rows,
    }


def fetch():
    today = datetime.now().date()
    next_start, next_end = today, today + timedelta(days=DAYS_SPAN - 1)
    last_start, last_end = today - timedelta(days=DAYS_SPAN), today - timedelta(days=1)

    with browser_page() as page:
        next_week = _fetch_range(page, next_start, next_end)
        last_week = _fetch_range(page, last_start, last_end)

    result = {
        "as_of": int(time.time()),
        "next_week": next_week,
        "last_week": last_week,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    data = fetch()
    for key in ("next_week", "last_week"):
        block = data[key]
        print(f"\n{key}: {block['range']} ({len(block['companies'])} shown)")
        for c in block["companies"][:8]:
            print(f"  {c['date_label']:<12} {c['symbol']:<8} {c['company']:<35} {c['call_time']:<5} cap={c['market_cap']}")
