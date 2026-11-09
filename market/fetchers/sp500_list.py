"""S&P 500 constituent list (ticker, name, sector) from Wikipedia, cached locally.

Membership/sector rarely changes, so this is only refetched if the cache is stale.
"""
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from market.fetchers.paths import DATA_DIR

CACHE_PATH = DATA_DIR / "sp500_constituents.json"
CACHE_MAX_AGE_SECONDS = 7 * 24 * 3600
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; local-market-dashboard/1.0)"}


def _fetch_from_wikipedia():
    resp = requests.get(WIKI_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    rows = table.find_all("tr")[1:]
    out = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        ticker = cells[0].text.strip().replace(".", "-")  # BRK.B -> BRK-B for yfinance
        name = cells[1].text.strip()
        sector = cells[2].text.strip()
        out.append({"ticker": ticker, "name": name, "sector": sector})
    return out


def get_constituents(force_refresh=False):
    if not force_refresh and CACHE_PATH.exists():
        age = time.time() - CACHE_PATH.stat().st_mtime
        if age < CACHE_MAX_AGE_SECONDS:
            return json.loads(CACHE_PATH.read_text())

    constituents = _fetch_from_wikipedia()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(constituents, indent=2))
    return constituents


if __name__ == "__main__":
    data = get_constituents(force_refresh=True)
    print(f"Fetched {len(data)} constituents")
    print(data[:5])
