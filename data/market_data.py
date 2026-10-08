"""
Market data provider. yfinance is the default backend so this runs today
with zero API keys / broker accounts. The Schwab Trader/Market Data API can
be slotted in later as a second implementation of the same interface
(get_price_history / get_option_chain / get_vix / etc.) without touching
engine/ code, since engine/ only depends on this interface's return shapes.

NOTE ON DATA QUALITY: yfinance is free, unofficial, and occasionally rate
limited or stale (especially impliedVolatility on illiquid contracts, and
after-hours quotes). Treat outputs as directionally correct, not
execution-grade. Cross-check anything you're about to act on against
thinkorswim directly before trading it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

_QUOTE_CACHE: dict[str, tuple[float, dict]] = {}
_QUOTE_CACHE_TTL_SECONDS = 60 * 5


@dataclass(frozen=True)
class OptionContract:
    contract_symbol: str
    expiration: str        # "YYYY-MM-DD"
    dte: int
    strike: float
    option_type: str       # "call" | "put"
    bid: float
    ask: float
    last: float
    mid: float
    implied_vol: float | None
    open_interest: int
    volume: int


def get_price_history(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Returns a DataFrame indexed by date with Open/High/Low/Close/Volume."""
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No price history returned for {symbol!r}")
    return df


_BATCH_CHUNK_SIZE = 250
_BATCH_CHUNK_PAUSE_SECONDS = 1.5


def _download_chunk(chunk: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers=" ".join(chunk), period=period, interval=interval,
        group_by="ticker", threads=True, progress=False, auto_adjust=False,
    )
    result: dict[str, pd.DataFrame] = {}
    if raw.empty:
        return result

    # With group_by="ticker", yfinance returns a MultiIndex-columned frame
    # regardless of how many symbols were requested — including exactly
    # one (verified directly: a 1-symbol call still comes back as
    # MultiIndex([(symbol, 'Open'), (symbol, 'Close'), ...])). There's no
    # flat-frame special case to handle.
    top_level = set(raw.columns.get_level_values(0))
    for sym in chunk:
        if sym not in top_level:
            continue
        df = raw[sym].dropna(how="all")
        if not df.empty:
            result[sym] = df
    return result


_MAX_RETRY_ROUNDS = 2
_RETRY_BACKOFF_SECONDS = 20


def get_price_histories_batch(
    symbols: list[str], period: str = "2y", interval: str = "1d",
    cancel_event=None,  # threading.Event | None — checked between chunks/retry rounds so a user-requested cancel actually takes effect during a large (e.g. full Russell 3000) pull, not just after it finishes
) -> dict[str, pd.DataFrame]:
    """Same data as calling get_price_history once per symbol — NOT as one
    HTTP request, despite what an earlier version of this docstring
    claimed. Yahoo's price-history endpoint only ever accepts one ticker
    per request; yf.download(threads=True) fires one real HTTP request per
    symbol through a thread pool, all in parallel. It's much FASTER than
    doing them one at a time (parallelizing the wait, not reducing the
    request count — confirmed: a 250-symbol chunk is 250 real requests to
    Yahoo in a couple seconds), but it is NOT gentler on Yahoo's rate
    limiter, and at large-universe scale (Russell 3000) it visibly trips
    one — see the chunking note below.

    IMPORTANT, verified directly: a per-ticker YFRateLimitError inside
    yf.download() is caught and logged BY YFINANCE ITSELF — it never
    raises out to this function. (Confirmed: requested 600 symbols, one
    ['KO'] logged as rate-limited, get_price_histories_batch still
    returned normally with 599/600 — no exception at any point.) An
    earlier version of this docstring claimed such an error "propagates
    immediately" to the caller's circuit breaker; that was wrong for this
    code path specifically (it's still correct for the single-ticker
    fundamentals/options-chain calls elsewhere, which really do raise it
    synchronously). Because there's nothing to catch here, the retry logic
    below is presence-based — comparing what came back to what was asked
    for — not exception-based.

    Symbols still missing after retries are simply absent from the result
    dict rather than raising, so callers can treat a missing key as "no
    data" per-symbol.

    CHUNKED, not one call for the whole list — verified directly that a
    single yf.download() for ~2600 symbols (full Russell 3000) both gets
    rate-limited AND crashes outright with a real bug inside yfinance
    itself (a thread-safety race in its internal error-tracking dict when
    too many concurrent failures pile up in one call — 'dictionary changed
    size during iteration'). Chunking to _BATCH_CHUNK_SIZE with a short
    pause between chunks keeps each individual yf.download() call in the
    size range that's been verified reliable and avoids triggering that
    crash — but does NOT fully eliminate rate limiting at large scale
    (observed ~10% of a full Russell 3000 pull failing per run even
    chunked). _MAX_RETRY_ROUNDS retries, with a longer backoff than the
    inter-chunk pause, recover most of that for a one-time full-universe
    pull; a few percent missing on any given run is still expected and
    should be treated as normal, not a bug to chase further."""
    if not symbols:
        return {}

    result: dict[str, pd.DataFrame] = {}
    pending = list(symbols)

    for round_num in range(_MAX_RETRY_ROUNDS + 1):
        if not pending or (cancel_event is not None and cancel_event.is_set()):
            break
        if round_num > 0:
            time.sleep(_RETRY_BACKOFF_SECONDS)

        for i in range(0, len(pending), _BATCH_CHUNK_SIZE):
            if cancel_event is not None and cancel_event.is_set():
                break
            chunk = pending[i:i + _BATCH_CHUNK_SIZE]
            result.update(_download_chunk(chunk, period, interval))
            if i + _BATCH_CHUNK_SIZE < len(pending):
                time.sleep(_BATCH_CHUNK_PAUSE_SECONDS)

        pending = [s for s in pending if s not in result]

    return result


def get_quote(symbol: str, fresh: bool = False) -> dict:
    now = time.time()
    cached = _QUOTE_CACHE.get(symbol)
    if not fresh and cached and now - cached[0] < _QUOTE_CACHE_TTL_SECONDS:
        return cached[1]

    t = yf.Ticker(symbol)
    fast = t.fast_info
    quote = {
        "symbol": symbol,
        "last_price": float(fast.get("lastPrice") or fast.get("last_price") or 0.0),
        "prev_close": float(fast.get("previousClose") or fast.get("previous_close") or 0.0),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _QUOTE_CACHE[symbol] = (now, quote)
    return quote


def get_vix() -> float:
    hist = get_price_history("^VIX", period="5d", interval="1d")
    # See engine.technicals.build_snapshot for why: yfinance can serve a
    # trailing row with a NaN Close before Yahoo finalizes the session.
    return float(hist["Close"].dropna().iloc[-1])


def get_spy_qqq_at_all_time_high() -> dict:
    """Returns {'SPY': bool, 'QQQ': bool} — is each within 0.1% of its
    all-time closing high (using available history, ~25y for SPY)."""
    result = {}
    for sym in ("SPY", "QQQ"):
        hist = get_price_history(sym, period="max", interval="1d")
        close = hist["Close"].dropna()  # see engine.technicals.build_snapshot — same NaN-trailing-row issue
        current = close.iloc[-1]
        ath = close.max()
        result[sym] = bool(current >= ath * 0.999)
    return result


def get_option_expirations(symbol: str) -> list[str]:
    return list(yf.Ticker(symbol).options)


def get_option_chain(symbol: str, expiration: str) -> list[OptionContract]:
    """Full chain (calls + puts) for one expiration date."""
    t = yf.Ticker(symbol)
    chain = t.option_chain(expiration)
    today = datetime.now().date()
    exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
    dte = (exp_date - today).days

    contracts = []
    for option_type, df in (("call", chain.calls), ("put", chain.puts)):
        for _, row in df.iterrows():
            bid = float(row.get("bid") or 0.0)
            ask = float(row.get("ask") or 0.0)
            mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else float(row.get("lastPrice") or 0.0)
            iv = row.get("impliedVolatility")
            contracts.append(OptionContract(
                contract_symbol=row.get("contractSymbol", ""),
                expiration=expiration,
                dte=dte,
                strike=float(row["strike"]),
                option_type=option_type,
                bid=bid,
                ask=ask,
                last=float(row.get("lastPrice") or 0.0),
                mid=mid,
                implied_vol=float(iv) if iv is not None and iv > 0 else None,
                open_interest=int(row["openInterest"]) if pd.notna(row.get("openInterest")) else 0,
                volume=int(row["volume"]) if pd.notna(row.get("volume")) else 0,
            ))
    return contracts


if __name__ == "__main__":
    print("VIX:", get_vix())
    print("Quote AAPL:", get_quote("AAPL"))
    exps = get_option_expirations("AAPL")
    print("AAPL expirations (first 5):", exps[:5])
    if exps:
        chain = get_option_chain("AAPL", exps[2])
        print(f"Chain for {exps[2]}: {len(chain)} contracts, sample:", chain[0])
