"""
Data layer for the CSP web dashboard (web/app.py). Cash-secured-puts only,
for now — "we'll add other options later" per user request. No LLM
involved; this only calls engine/checks.py + engine/options_selector.py
against real yfinance data.

PERFORMANCE / RATE-LIMIT SURFACE: scanning used to be fully sequential, AND
every symbol's options chain was fetched automatically as part of the main
scan. Both are fixed now:

1. Price history for the whole requested symbol list is fetched in ONE
   batched call (get_price_histories_batch) instead of one call per
   symbol — ~30x faster for that step alone (measured: ~1.5s vs ~50s for
   50 symbols). The uptrend check (cheap: pure computation on data we
   already have) runs on this batch immediately, BEFORE any fundamentals
   call is made.
2. Symbols that fail the uptrend check are dropped right there — no
   fundamentals or quote fetch either. Historically ~45-55% of the S&P 500
   fails this check at any given time, so this alone cuts the remaining
   work roughly in half.
3. The remaining work (fundamentals + quote, for whichever symbols survive
   step 2) runs CONCURRENTLY via a thread pool — I/O-bound network calls,
   not CPU work, so Python's GIL doesn't block real parallelism here.
4. THE OPTIONS CHAIN IS NEVER FETCHED DURING A SCAN AT ALL. scan_universe/
   _full_scan only ever do fundamentals + technicals + quote — cheap,
   fast, and incapable of triggering a rate limit on their own. Options
   data is fetched on demand, one symbol at a time, only when explicitly
   requested via fetch_option_for_symbol (wired to a "Get Option" button
   in the UI) — per user request, since most of the actual rate-limit risk
   came from fetching option chains for every survivor on every scan.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from dataclasses import dataclass, field

from yfinance.exceptions import YFRateLimitError

from config import strategy_rules as rules
from data.fundamentals import get_fundamentals
from data.market_data import get_price_histories_batch, get_quote
from engine.checks import (
    check_csp_entry_trigger,
    check_current_ratio,
    check_debt_to_equity,
    check_eps_beat,
    check_fcf_positive,
    check_pe,
    check_return_on_equity,
    check_revenue_growth,
    check_uptrend,
    check_uptrend_legacy,
    passes_curated_quality_gate,
)
from engine.options_selector import SelectedOption, find_put_at_least_delta
from engine.technicals import build_snapshot

# Circuit breaker for yfinance rate limiting. yfinance raises a single,
# dedicated YFRateLimitError from its central HTTP layer (yfinance/data.py)
# for EVERY call type — price history, options chains, fundamentals,
# earnings dates all go through it — so catching this one specific
# exception type reliably means "Yahoo just rate-limited us," never a
# generic/unrelated failure. Two effects, both per user request:
#   1. Within a scan: the instant any worker sees it, every other symbol
#      still pending in that scan is marked as skipped rather than making
#      its own doomed (and rate-limit-worsening) calls.
#   2. Across scans: the trip is remembered for a cooldown window, so
#      starting a NEW scan shortly after a rate limit doesn't immediately
#      slam the same limited endpoint again.
_RATE_LIMIT_COOLDOWN_SECONDS = 120
_rate_limit_lock = threading.Lock()
_rate_limited_until: float = 0.0  # time.time() value; 0 = not currently tripped


def _trip_rate_limit():
    global _rate_limited_until
    with _rate_limit_lock:
        _rate_limited_until = time.time() + _RATE_LIMIT_COOLDOWN_SECONDS


def rate_limit_active() -> tuple[bool, float]:
    """Returns (is_active, seconds_remaining). Exposed for the web layer so
    it can refuse to start a new scan (and say why) during the cooldown."""
    with _rate_limit_lock:
        remaining = _rate_limited_until - time.time()
    return remaining > 0, max(0.0, remaining)


CHECK_DEFINITIONS = [
    ("uptrend", "30wk Uptrend", lambda t, f: check_uptrend(t)),
    ("uptrend_legacy", "12/18mo Uptrend", lambda t, f: check_uptrend_legacy(t)),
    ("pe_ok", "P/E", lambda t, f: check_pe(f)),
    ("fcf_positive", "FCF+", lambda t, f: check_fcf_positive(f)),
    ("eps_beat", "EPS", lambda t, f: check_eps_beat(f)),
    ("debt_to_equity", "D/E", lambda t, f: check_debt_to_equity(f)),
    ("roe", "ROE", lambda t, f: check_return_on_equity(f)),
    ("rev_growth", "Rev Growth", lambda t, f: check_revenue_growth(f)),
    ("csp_trigger", "Down Days", lambda t, f: check_csp_entry_trigger(t)),
]


@dataclass
class CspRow:
    symbol: str
    sector: str | None
    spot: float
    name: str | None = None
    checks: dict = field(default_factory=dict)   # key -> {value: bool|None, reason: str}
    checks_passed: int = 0
    checks_evaluated: int = 0
    next_earnings_date: str | None = None
    day_change_pct: float | None = None   # vs. yesterday's close
    beta: float | None = None
    realized_vol: float | None = None  # 20-day annualized, from the same TechnicalSnapshot already built for the uptrend check — zero extra cost
    realized_vol_pctile: float | None = None  # where current realized vol ranks vs this stock's own trailing 1yr — also already computed, zero extra cost. Closer to true "IV Rank" semantics than the absolute number: elevated FOR THIS STOCK, not elevated in general.
    short_pct_float: float | None = None  # see data.fundamentals.Fundamentals.short_pct_float — squeeze-screening signal, ~1-2wk stale
    put: SelectedOption | None = None
    options_fetched: bool = False   # distinguishes "not looked up yet" from "looked up, nothing qualified"
    options_error: str | None = None  # set only by an on-demand options fetch that failed (e.g. rate limited)
    error: str | None = None
    # Curated mode's hard fundamentals gate (see engine.checks.passes_curated_quality_gate).
    # Computed for every row regardless of mode (cheap — reuses the same
    # Fundamentals object already fetched), but only used to actually
    # remove a row when scan_universe(require_curated_fundamentals=True).
    curated_gate_passed: bool | None = None
    curated_gate_reasons: list = field(default_factory=list)
    # True for symbols the user typed in themselves: they bypass the uptrend,
    # volatility and Strict fundamentals filters (see scan_universe's
    # force_include) so a personal pick is always shown, never dropped.
    forced: bool = False


_RATE_LIMIT_SKIP_MESSAGE = "Yahoo Finance is limiting requests right now. Wait a few minutes and try again."


def _find_and_filter_put(symbol: str, spot: float, tech, stats: dict | None = None) -> SelectedOption | None:
    """Shared by the on-demand single-symbol fetch and (if ever needed)
    batch code: the actual options-chain search + the same delta-band
    filter used everywhere else in the dashboard. Can raise
    YFRateLimitError or other exceptions — callers handle those.

    Passes max_delta so find_put_at_least_delta's own "stop searching"
    tolerance matches the acceptable band we're about to filter on below —
    see its docstring (max_delta param) for the bug this fixes: without
    this, the search could stop at a nearest-expiration match that's
    inside its OWN generic overshoot tolerance but outside OUR stricter
    band, discard it below, and never even look at a farther expiration
    that would have had a clean in-band match."""
    lo, hi = rules.CSP_DASHBOARD_ACCEPTABLE_DELTA_RANGE
    put = find_put_at_least_delta(
        symbol, spot,
        target_delta=rules.WATCHLIST_TARGET_DELTA,
        min_dte=rules.WATCHLIST_RETURN_ON_RISK_DTE,       # hard floor: 30 days minimum
        max_dte=rules.CSP_DASHBOARD_MAX_DTE_SEARCH,       # how far out to search if 30-45 DTE doesn't reach -0.20 delta
        ror_reference_dte=rules.WATCHLIST_RETURN_ON_RISK_DTE,
        realized_vol_hint=tech.realized_vol,
        max_delta=hi,
        stats=stats,
    )
    if put is not None:
        if not (lo <= abs(put.computed_delta) <= hi):
            # Outside the range we're willing to show at all (e.g. a
            # 0.69-delta fallback pick because the real ~20-delta strikes
            # had no market) — treat exactly like "no put found" rather
            # than surfacing a low-confidence number.
            return None
    return put


def fetch_option_for_symbol(symbol: str, spot: float) -> tuple[SelectedOption | None, str | None, str | None]:
    """On-demand, single-symbol options lookup — the ONLY path that calls
    the options chain now (see module docstring point 4). Wired to a
    per-row "Get Option" button in the UI rather than run automatically
    for every scan survivor. Returns (put_or_None, error_message_or_None, note_or_None); the
    note explains a None put that is NOT an error: "no_live_quotes" (every
    contract had a zero bid/ask, which is what Yahoo serves outside market
    hours) or "no_chain" (no expirations listed in the search window).
    Respects the same rate-limit circuit breaker as scan_universe: refuses
    to run during an active cooldown, and trips it if it hits
    YFRateLimitError itself (one bad symbol still protects the rest of
    your remaining "Get Option" clicks today)."""
    active, remaining_s = rate_limit_active()
    if active:
        return None, f"skipped — rate-limited by Yahoo Finance; cooling down (~{int(remaining_s)}s left)", None

    try:
        daily = get_price_histories_batch([symbol], period="2y").get(symbol)
        if daily is None or daily.empty:
            return None, "no price history available", None
        tech = build_snapshot(symbol, daily)
    except YFRateLimitError:
        _trip_rate_limit()
        return None, "rate limited by Yahoo Finance (price history)", None
    except Exception as exc:
        return None, str(exc), None

    try:
        stats: dict = {}
        put = _find_and_filter_put(symbol, spot, tech, stats=stats)
        note = None
        if put is None:
            if stats.get("expirations", 0) == 0:
                note = "no_chain"
            elif stats.get("contracts_seen", 0) > 0 and stats.get("contracts_live", 0) < max(1, 0.2 * stats["contracts_seen"]):
                # Outside market hours Yahoo zeroes most bids/asks; a few
                # stale two-sided quotes can linger, so use a share, not zero.
                note = "no_live_quotes"
        return put, None, note
    except YFRateLimitError:
        _trip_rate_limit()
        return None, "rate limited by Yahoo Finance (options chain)", None
    except Exception as exc:
        return None, f"options chain error: {exc}", None


def _full_scan(symbol: str, tech, stop_event: threading.Event) -> CspRow:
    """Fundamentals + quote only — no options chain, ever (see module
    docstring). Only ever called for a symbol that already has a
    TechnicalSnapshot (i.e. already passed the cheap uptrend prefilter, or
    require_uptrend was off) — see scan_universe.

    stop_event is the cross-worker rate-limit circuit breaker: checked
    before doing ANY work (so a symbol queued behind an already-tripped
    scan does nothing at all), and set the instant this worker itself hits
    YFRateLimitError, so every other in-flight/pending worker bails on its
    next check too. Fundamentals/quote calls are much less rate-limit-prone
    than options-chain calls in practice, but the same protection applies
    regardless of which call trips it."""
    if stop_event.is_set():
        return CspRow(symbol=symbol, sector=None, spot=0.0, error=_RATE_LIMIT_SKIP_MESSAGE)

    try:
        fund = get_fundamentals(symbol)
        spot = fund.current_price
        prev_close = fund.previous_close
        if spot is None:
            # Rare (halted/delisted-ish names) — .info didn't have a price.
            # Falls back to a real quote call rather than failing the row
            # outright; this is the ONE place that can still cost an extra
            # call per symbol, and only for the exceptions.
            quote = get_quote(symbol)
            spot = quote["last_price"]
            prev_close = quote.get("prev_close") or prev_close
        day_change_pct = ((spot - prev_close) / prev_close) if (prev_close and prev_close > 0) else None
    except YFRateLimitError:
        stop_event.set()
        _trip_rate_limit()
        return CspRow(symbol=symbol, sector=None, spot=0.0, error="Yahoo Finance is limiting requests right now. Wait a few minutes and try again.")
    except Exception as exc:
        return CspRow(symbol=symbol, sector=None, spot=0.0, error=str(exc))

    # Raw numeric value behind each check, where there is one — lets the
    # frontend show the actual number (e.g. "38.6") instead of only a
    # checkmark, for columns where that's more informative than pass/fail.
    raw_values = {
        "pe_ok": fund.pe_ratio,
        "debt_to_equity": fund.debt_to_equity,
        "roe": fund.return_on_equity,
        "rev_growth": fund.revenue_growth_yoy,
        # eps_beat's raw is a small object (actual vs. estimate), not a
        # single number — the frontend renders it specially, not via the
        # generic single-number formatter the others use.
        "eps_beat": {
            "actual": fund.eps_actual_last_quarter,
            "estimate": fund.eps_estimate_last_quarter,
        },
    }

    checks = {}
    passed = 0
    evaluated = 0
    for key, label, fn in CHECK_DEFINITIONS:
        value, reason = fn(tech, fund)
        checks[key] = {"label": label, "value": value, "reason": reason, "raw": raw_values.get(key)}
        # "uptrend" is excluded from the passed/evaluated tally: every row
        # reaching this point already passed it (see scan_universe's
        # require_uptrend filter), so counting it would inflate every row's
        # ratio by a constant +1/+1 for a check the dashboard doesn't even
        # display as its own column anymore.
        if key != "uptrend" and value is not None:
            evaluated += 1
            if value:
                passed += 1

    gate_passed, gate_reasons = passes_curated_quality_gate(fund)

    return CspRow(
        symbol=symbol, sector=fund.sector, spot=spot, checks=checks, name=fund.name,
        checks_passed=passed, checks_evaluated=evaluated,
        next_earnings_date=fund.next_earnings_date,
        day_change_pct=day_change_pct, beta=fund.beta,
        realized_vol=tech.realized_vol,
        realized_vol_pctile=tech.realized_vol_pctile,
        short_pct_float=fund.short_pct_float,
        curated_gate_passed=gate_passed,
        curated_gate_reasons=gate_reasons,
    )


def scan_universe(
    symbols: list[str], progress_cb=None, require_uptrend: bool = True, max_workers: int = 10,
    min_realized_vol: float | None = None, require_curated_fundamentals: bool = False,
    cancel_event: threading.Event | None = None, force_include: set[str] | None = None,
) -> tuple[list[CspRow], int, int, int, dict[str, str]]:
    """Returns (rows, num_removed_for_no_uptrend, num_removed_for_low_vol,
    num_removed_for_fundamentals, removed_reasons) — removed_reasons maps
    each symbol the scan dropped to a plain-English reason, so the UI can
    answer "why is X missing?".

    force_include: symbols the user picked by hand. They skip every filter
    below (uptrend, vol floor, Strict fundamentals gate) and always come
    back as rows marked forced=True, with all the checks still computed so
    the dashboard can show how they actually look. Only a real data error
    (e.g. no price history) can still leave one as an error row.

    Two-phase, per the module docstring: (1) one batched price-history
    fetch + cheap local uptrend check for every symbol, (2) the expensive
    fundamentals/quote/options work only for symbols that pass — run
    concurrently across `max_workers` threads. require_uptrend=True (the
    default) drops any symbol whose Section 12 criterion 1 check (uptrend,
    per engine.technicals.is_in_uptrend) came back False BEFORE phase 2 ever touches it — per user
    request, these don't even show up in the dashboard. Rows where price
    data couldn't be fetched at all (a real data error) are kept as error
    rows, not silently dropped, since that's a data problem rather than a
    real "no uptrend" result — same as before.

    min_realized_vol (Options Trading mode; see config.OPTIONS_MODE_MIN_REALIZED_VOL)
    is a second, equally cheap pre-fundamentals filter: drops any symbol
    whose 20-day annualized realized vol is below the threshold, on the
    reasoning that a low-vol name is very unlikely to carry rich option
    premium regardless of what its fundamentals look like. Applied at the
    SAME stage as the uptrend check — before any fundamentals/quote call —
    so, as a bonus, it also reduces call volume for large-universe scans.

    require_curated_fundamentals (Curated mode) is checked AFTER phase 2,
    since it needs fundamentals data that's only fetched there — see
    engine.checks.passes_curated_quality_gate. Every row still gets that
    gate computed and attached regardless of this flag (cheap, already-
    fetched data); this flag only controls whether a failing row is
    actually removed from the returned list.

    cancel_event, if given, doubles as the internal rate-limit stop_event
    (same object — a user-requested cancel and a rate-limit trip both just
    mean "stop doing more work," so every existing stop_event check below
    handles both for free). Checked before the price-history fetch, between
    its internal chunks (see get_price_histories_batch), in the phase-1
    per-symbol loop, and — already — in the phase-2 thread pool. A symbol
    not yet reached when cancelled simply isn't in the returned rows;
    that's an intentional partial result, not an error."""
    total = len(symbols)
    force_include = {x.upper() for x in (force_include or ())}
    progress_lock = threading.Lock()
    completed = 0

    def report(sym):
        nonlocal completed
        if progress_cb:
            with progress_lock:
                progress_cb(completed, total, sym)
                completed += 1

    active, remaining_s = rate_limit_active()
    if active:
        rows = [CspRow(
            symbol=sym, sector=None, spot=0.0,
            error="Yahoo Finance is limiting requests right now. Wait a few minutes and try again.",
        ) for sym in symbols]
        for sym in symbols:
            report(sym)
        return rows, 0, 0, 0, {}

    stop_event = cancel_event if cancel_event is not None else threading.Event()
    if stop_event.is_set():
        rows = [CspRow(symbol=sym, sector=None, spot=0.0, error="cancelled") for sym in symbols]
        for sym in symbols:
            report(sym)
        return rows, 0, 0, 0, {}
    try:
        price_histories = get_price_histories_batch(symbols, period="2y", cancel_event=stop_event)
    except YFRateLimitError:
        _trip_rate_limit()
        rows = [CspRow(symbol=sym, sector=None, spot=0.0, error="Yahoo Finance is limiting requests right now. Wait a few minutes and try again.") for sym in symbols]
        for sym in symbols:
            report(sym)
        return rows, 0, 0, 0, {}

    survivors: list[tuple[str, object]] = []  # (symbol, TechnicalSnapshot)
    rows: list[CspRow] = []
    removed = 0
    removed_low_vol = 0
    removed_reasons: dict[str, str] = {}

    for sym in symbols:
        if stop_event.is_set():
            rows.append(CspRow(symbol=sym, sector=None, spot=0.0, error="cancelled"))
            report(sym)
            continue
        daily = price_histories.get(sym)
        if daily is None or daily.empty:
            rows.append(CspRow(symbol=sym, sector=None, spot=0.0, error="no price history returned (batch fetch)"))
            report(sym)
            continue
        try:
            tech = build_snapshot(sym, daily)
        except Exception as exc:
            rows.append(CspRow(symbol=sym, sector=None, spot=0.0, error=f"technicals error: {exc}"))
            report(sym)
            continue

        if sym.upper() not in force_include:
            if require_uptrend and not check_uptrend(tech)[0]:
                removed += 1
                removed_reasons[sym] = "its price is not rising (it is below its 30-week average)"
                report(sym)
                continue

            if min_realized_vol is not None and (tech.realized_vol is None or tech.realized_vol < min_realized_vol):
                removed_low_vol += 1
                removed_reasons[sym] = f"it barely moves (under {min_realized_vol * 100:.0f}% volatility a year), so the strictness level dropped it"
                report(sym)
                continue

        survivors.append((sym, tech))

    if survivors:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_symbol = {pool.submit(_full_scan, sym, tech, stop_event): sym for sym, tech in survivors}
            for future in concurrent.futures.as_completed(future_to_symbol):
                sym = future_to_symbol[future]
                try:
                    rows.append(future.result())
                except concurrent.futures.CancelledError:
                    rows.append(CspRow(symbol=sym, sector=None, spot=0.0, error=_RATE_LIMIT_SKIP_MESSAGE))
                except Exception as exc:
                    rows.append(CspRow(symbol=sym, sector=None, spot=0.0, error=str(exc)))
                report(sym)

                if stop_event.is_set():
                    # Best-effort: futures the pool hasn't started on a
                    # worker thread yet can be cancelled outright, saving
                    # them from making a now-pointless call. Ones already
                    # mid-flight will hit their own stop_event check inside
                    # _full_scan and bail with the same skipped message.
                    for other_future in future_to_symbol:
                        if not other_future.done():
                            other_future.cancel()

    for r in rows:
        if r.symbol.upper() in force_include:
            r.forced = True

    removed_fundamentals = 0
    if require_curated_fundamentals:
        kept_rows = []
        for r in rows:
            if r.error is not None or r.forced or r.curated_gate_passed is not False:
                kept_rows.append(r)
            else:
                removed_fundamentals += 1
                removed_reasons[r.symbol] = "it did not pass the company-health checks of the High strictness level"
        rows = kept_rows

    return rows, removed, removed_low_vol, removed_fundamentals, removed_reasons


def row_to_dict(r: CspRow) -> dict:
    d = {
        "symbol": r.symbol,
        "sector": r.sector,
        "name": r.name,
        "spot": r.spot,
        "error": r.error,
        "checks": r.checks,
        "checks_passed": r.checks_passed,
        "checks_evaluated": r.checks_evaluated,
        "next_earnings_date": r.next_earnings_date,
        "day_change_pct": r.day_change_pct,
        "beta": r.beta,
        "realized_vol": r.realized_vol,
        "realized_vol_pctile": r.realized_vol_pctile,
        "short_pct_float": r.short_pct_float,
        "options_fetched": r.options_fetched,
        "options_error": r.options_error,
        "curated_gate_passed": r.curated_gate_passed,
        "curated_gate_reasons": r.curated_gate_reasons,
        "forced": r.forced,
        "put": None,
    }
    if r.put:
        p = r.put
        d["put"] = {
            "strike": p.contract.strike,
            "expiration": p.contract.expiration,
            "dte": p.contract.dte,
            "delta": p.computed_delta,
            "delta_abs": abs(p.computed_delta),
            "at_or_above_target_delta": p.at_or_above_target_delta,
            "is_fallback_pick": p.is_fallback_pick,
            "bid": p.contract.bid,
            "ask": p.contract.ask,
            "mid": p.contract.mid,
            "premium": p.premium,
            "capital_at_risk": p.capital_at_risk,
            "return_on_risk": p.return_on_risk,
            "return_on_risk_30d_normalized": p.return_on_risk_30d_normalized,
            "open_interest": p.contract.open_interest,
            "volume": p.contract.volume,
        }
    return d
