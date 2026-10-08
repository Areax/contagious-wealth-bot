"""
S&P 500 ticker universe. Scraped from Wikipedia and cached locally so the
tool still works offline / if Wikipedia's table markup changes.
"""

import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

CACHE_PATH = Path(__file__).resolve().parent.parent / "cache" / "sp500_universe.json"
CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60  # 1 week
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _fetch_from_wikipedia() -> list[dict]:
    # Wikipedia rejects urllib's default user-agent with a 403; fetch with a
    # normal browser-style header via requests, then hand the HTML to pandas.
    resp = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0 (compatible; contagious-wealth-bot/0.1)"}, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    df = df.rename(columns={
        "Symbol": "symbol",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
    })
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B (yfinance convention)
    records = df[["symbol", "name", "sector", "sub_industry"]].to_dict("records")
    return records


def get_sp500_universe(force_refresh: bool = False) -> list[dict]:
    """Returns list of {symbol, name, sector, sub_industry} dicts."""
    if not force_refresh and CACHE_PATH.exists():
        age = time.time() - CACHE_PATH.stat().st_mtime
        if age < CACHE_MAX_AGE_SECONDS:
            return json.loads(CACHE_PATH.read_text())

    try:
        records = _fetch_from_wikipedia()
    except Exception as exc:
        if CACHE_PATH.exists():
            print(f"[universe] Wikipedia fetch failed ({exc}); using stale cache.")
            return json.loads(CACHE_PATH.read_text())
        raise RuntimeError(
            "Could not fetch S&P 500 list from Wikipedia and no cache exists. "
            f"Original error: {exc}"
        )

    _write_cache(CACHE_PATH, records)
    return records


def _write_cache(path: Path, records: list[dict]) -> None:
    # A failed cache write must not fail the caller: the list was already
    # fetched fresh. Seen 2026-10: a long-running server lost write access
    # to cache/ ([Errno 1] Operation not permitted) and every scan errored
    # out after the cache expired, leaving a week-old result on screen.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2))
    except OSError as exc:
        print(f"[universe] couldn't write cache {path.name} ({exc}); using fresh list without caching it.")


def get_sp500_symbols(force_refresh: bool = False) -> list[str]:
    return [r["symbol"] for r in get_sp500_universe(force_refresh=force_refresh)]


# ---------------------------------------------------------------------------
# Russell 2000 / 3000. Your "Wheel It" list doesn't map to any single named
# index (it mixes small-cap momentum, materials/uranium cyclicals, and
# leveraged ETFs like SOXL/GGLL, which can never belong to any real index) —
# but Russell 2000 (small-cap) and Russell 3000 (~98% of the investable US
# market) are the closest real, freely-obtainable broader universes, and
# would catch most of the real stocks on that list. Sourced from iShares'
# public daily holdings CSV for their Russell-tracking ETFs (IWM / IWV) —
# legitimate, official fund holdings data, not a scrape of a third party's
# derived list.
# ---------------------------------------------------------------------------

RUSSELL_FUNDS = {
    "2000": {"fund_id": "239710", "slug": "ishares-russell-2000-etf", "cache_name": "russell2000_universe.json"},
    "3000": {"fund_id": "239714", "slug": "ishares-russell-3000-etf", "cache_name": "russell3000_universe.json"},
}


def _fetch_russell_from_ishares(index: str) -> list[dict]:
    fund = RUSSELL_FUNDS[index]
    url = f"https://www.ishares.com/us/products/{fund['fund_id']}/{fund['slug']}/latest-holdings.csv"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; contagious-wealth-bot/0.1)"}, timeout=30)
    resp.raise_for_status()

    lines = resp.text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith("Ticker,Name,")), None)
    if header_idx is None:
        raise RuntimeError(f"Unexpected iShares CSV format for Russell {index} (no header row found)")

    df = pd.read_csv(io.StringIO(resp.text), skiprows=header_idx)
    df = df[df["Asset Class"] == "Equity"].copy()
    df = df.dropna(subset=["Ticker"])
    df["symbol"] = df["Ticker"].astype(str).str.strip().str.replace(" ", "-", regex=False).str.upper()
    df = df.rename(columns={"Name": "name", "Sector": "sector"})
    df["sub_industry"] = None  # this CSV doesn't break out sub-industry the way the S&P 500 Wikipedia table does
    df = df[df["symbol"].str.len() > 0]
    records = df[["symbol", "name", "sector", "sub_industry"]].drop_duplicates(subset="symbol").to_dict("records")
    return records


def get_russell_universe(index: str = "2000", force_refresh: bool = False) -> list[dict]:
    """index: '2000' or '3000'. Returns list of {symbol, name, sector, sub_industry} dicts."""
    if index not in RUSSELL_FUNDS:
        raise ValueError(f"index must be '2000' or '3000', got {index!r}")

    cache_path = Path(__file__).resolve().parent.parent / "cache" / RUSSELL_FUNDS[index]["cache_name"]

    if not force_refresh and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < CACHE_MAX_AGE_SECONDS:
            return json.loads(cache_path.read_text())

    try:
        records = _fetch_russell_from_ishares(index)
    except Exception as exc:
        if cache_path.exists():
            print(f"[universe] iShares Russell {index} fetch failed ({exc}); using stale cache.")
            return json.loads(cache_path.read_text())
        raise RuntimeError(
            f"Could not fetch Russell {index} list from iShares and no cache exists. "
            f"Original error: {exc}"
        )

    _write_cache(cache_path, records)
    return records


def get_russell_symbols(index: str = "2000", force_refresh: bool = False) -> list[str]:
    return [r["symbol"] for r in get_russell_universe(index=index, force_refresh=force_refresh)]


if __name__ == "__main__":
    universe = get_sp500_universe()
    print(f"Loaded {len(universe)} S&P 500 constituents.")
    print(universe[:5])

    r2k = get_russell_universe("2000")
    print(f"Loaded {len(r2k)} Russell 2000 constituents.")
    print(r2k[:5])
