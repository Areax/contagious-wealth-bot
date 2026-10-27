"""
Fundamentals for the Section 12 watchlist screen. yfinance-backed.

CALL-COUNT NOTE: current_price/previous_close are pulled from the same
.info dict as everything else here, NOT from a separate quote call —
.info already contains them (verified directly). engine/csp_dashboard.py
used to also call data.market_data.get_quote() per symbol for this, which
was a fully redundant third network call per symbol. At Russell-3000 scale
(~1300+ symbols surviving the uptrend prefilter in a typical scan), that
redundancy meant thousands of avoidable requests — cutting it removes
roughly a third of the fundamentals-phase call volume with no loss of
data, which matters a lot for how likely a large scan is to get rate
limited.

KNOWN GAP: the doc's "double beat" (EPS beat AND revenue beat) needs a
consensus *revenue* estimate to check against. yfinance's free data only
reliably exposes EPS estimate vs. actual (via get_earnings_dates), not a
revenue estimate. revenue_beat is therefore returned as None ("unknown")
rather than guessed — the screener treats None as "does not disqualify,
but flag it" rather than silently treating it as a pass. If you want true
double-beat verification, that requires a paid fundamentals feed (Financial
Modeling Prep, Finnhub premium, etc.) — swap it in here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class Fundamentals:
    symbol: str
    pe_ratio: float | None
    free_cash_flow: float | None      # trailing twelve months, dollars
    total_debt: float | None          # dollars
    total_cash: float | None          # dollars — used for the Curated-tier cash-runway check
    market_cap: float | None
    eps_beat_last_quarter: bool | None
    eps_surprise_pct: float | None
    eps_actual_last_quarter: float | None    # last reported EPS, dollars/share
    eps_estimate_last_quarter: float | None  # consensus estimate it was measured against
    next_earnings_date: str | None    # "YYYY-MM-DD", soonest future reported/estimated date, or None
    revenue_beat_last_quarter: None   # always None — see module docstring
    sector: str | None
    # Additional quality-factor fields — NOT in the source doc, added per
    # user request for extra rigor beyond Section 12's minimal P/E/FCF/EPS
    # check. See engine/checks.py for how they're applied.
    debt_to_equity: float | None      # yfinance reports this as a percent, e.g. 78.4 = 0.784x
    return_on_equity: float | None    # fraction, e.g. 0.15 = 15%
    revenue_growth_yoy: float | None  # fraction, most recent quarter YoY
    current_ratio: float | None       # current assets / current liabilities
    analyst_recommendation_mean: float | None  # 1.0 (strong buy) - 5.0 (sell)
    beta: float | None  # volatility relative to the market (1.0 = moves with it, >1 = more volatile).
    # Same info-dict pull as everything else above — costs nothing extra,
    # and specifically involves zero options-chain calls. Useful as a cheap
    # proxy for "will this name carry rich option premium" without having
    # to hit the options chain to find out.
    current_price: float | None   # also from .info — see module docstring on why this replaced a separate quote call
    previous_close: float | None
    # Squeeze-screening signal, same free .info pull as everything else
    # here — zero extra cost. FINRA settles short interest twice a month
    # and Yahoo reports it ~1-2 weeks after that, so this is NEVER
    # real-time; treat it as a slow-moving signal, not current-day data.
    short_pct_float: float | None  # fraction, e.g. 0.084 = 8.4% of float sold short
    name: str | None = None  # company name, for display next to the ticker


def _earnings_dates_df(t: yf.Ticker) -> pd.DataFrame | None:
    try:
        df = t.get_earnings_dates(limit=12)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    return df


def _latest_reported_earnings_row(df: pd.DataFrame) -> pd.Series | None:
    reported = df[df["Reported EPS"].notna()]
    if reported.empty:
        return None
    return reported.sort_index(ascending=False).iloc[0]


def _next_earnings_date(df: pd.DataFrame) -> str | None:
    future = df[df["Reported EPS"].isna()]
    if future.empty:
        return None
    return future.sort_index(ascending=True).index[0].strftime("%Y-%m-%d")


def get_fundamentals(symbol: str) -> Fundamentals:
    t = yf.Ticker(symbol)
    info = t.info or {}

    df = _earnings_dates_df(t)
    eps_beat = None
    eps_surprise = None
    eps_actual = None
    eps_estimate = None
    next_earnings = None
    if df is not None:
        row = _latest_reported_earnings_row(df)
        if row is not None:
            surprise = row.get("Surprise(%)")
            if pd.notna(surprise):
                eps_surprise = float(surprise)
                eps_beat = eps_surprise > 0
            if pd.notna(row.get("Reported EPS")):
                eps_actual = float(row["Reported EPS"])
            if pd.notna(row.get("EPS Estimate")):
                eps_estimate = float(row["EPS Estimate"])
        next_earnings = _next_earnings_date(df)

    return Fundamentals(
        symbol=symbol,
        pe_ratio=info.get("trailingPE"),
        free_cash_flow=info.get("freeCashflow"),
        total_debt=info.get("totalDebt"),
        total_cash=info.get("totalCash"),
        market_cap=info.get("marketCap"),
        eps_beat_last_quarter=eps_beat,
        eps_surprise_pct=eps_surprise,
        eps_actual_last_quarter=eps_actual,
        eps_estimate_last_quarter=eps_estimate,
        next_earnings_date=next_earnings,
        revenue_beat_last_quarter=None,
        sector=info.get("sector"),
        debt_to_equity=info.get("debtToEquity"),
        return_on_equity=info.get("returnOnEquity"),
        revenue_growth_yoy=info.get("revenueGrowth"),
        current_ratio=info.get("currentRatio"),
        analyst_recommendation_mean=info.get("recommendationMean"),
        beta=info.get("beta"),
        current_price=info.get("currentPrice") or info.get("regularMarketPrice"),
        previous_close=info.get("previousClose") or info.get("regularMarketPreviousClose"),
        short_pct_float=info.get("shortPercentOfFloat"),
        name=info.get("shortName") or info.get("longName"),
    )


if __name__ == "__main__":
    for sym in ("AAPL", "KO"):
        print(get_fundamentals(sym))
