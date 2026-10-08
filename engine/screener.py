"""
Section 12: BUILD YOUR WATCHLIST — implemented as a mechanical filter over
the S&P 500 universe.

Criterion 3 ("options must pay well": 20-delta / 2% RoR at 30 DTE) is
deliberately NOT checked here — it requires pulling a full option chain per
symbol, which is expensive to do for 500 names up front. It's checked in
engine/options_selector.py, only for symbols that already pass this
(cheap) fundamentals + technicals screen.

Criterion 4 ("personal connection") is inherently subjective and NOT scored
here; it's left to the user / to the AI suggestion layer's qualitative pass
(see ai_suggest/).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import strategy_rules as rules
from data.fundamentals import Fundamentals, get_fundamentals
from data.market_data import get_price_history
from engine.checks import (
    check_analyst_sentiment,
    check_current_ratio,
    check_debt_to_equity,
    check_eps_beat,
    check_fcf_positive,
    check_pe,
    check_return_on_equity,
    check_revenue_growth,
    check_uptrend,
)
from engine.technicals import TechnicalSnapshot, build_snapshot


@dataclass
class ScreenResult:
    symbol: str
    passed: bool
    reasons_failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    technicals: TechnicalSnapshot | None = None
    fundamentals: Fundamentals | None = None


def _check_fundamentals(f: Fundamentals) -> tuple[list[str], list[str]]:
    fails: list[str] = []
    warns: list[str] = []

    pe_ok, pe_reason = check_pe(f)
    if pe_ok is None:
        warns.append(f"P/E check skipped: {pe_reason}")
    elif not pe_ok:
        fails.append(pe_reason)

    fcf_ok, fcf_reason = check_fcf_positive(f)
    if fcf_ok is None:
        warns.append(f"FCF check skipped: {fcf_reason}")
    elif not fcf_ok:
        fails.append(f"free cash flow is not positive ({fcf_reason})")

    eps_ok, eps_reason = check_eps_beat(f)
    if eps_ok is None:
        warns.append(f"EPS-beat check skipped: {eps_reason}")
    elif not eps_ok:
        fails.append(f"missed EPS estimate last quarter ({eps_reason})")

    warns.append(
        "revenue-beat not verifiable from free data — 'double beat' criterion is "
        "approximated as EPS-beat only (see data/fundamentals.py)"
    )

    if rules.QUALITY_FILTERS_ENABLED:
        de_ok, de_reason = check_debt_to_equity(f)
        if de_ok is None:
            warns.append(f"debt/equity check skipped: {de_reason}")
        elif not de_ok:
            if f.sector in rules.DEBT_TO_EQUITY_SECTOR_EXCEPTIONS:
                warns.append(f"{de_reason} (over threshold, but {f.sector} sector runs structurally higher D/E — not disqualifying)")
            else:
                fails.append(f"debt load too high: {de_reason} [extra filter, not in source doc]")

        roe_ok, roe_reason = check_return_on_equity(f)
        if roe_ok is None:
            warns.append(f"ROE check skipped: {roe_reason}")
        elif not roe_ok:
            fails.append(f"return on equity too low: {roe_reason} [extra filter, not in source doc]")

        rev_ok, rev_reason = check_revenue_growth(f)
        if rev_ok is None:
            warns.append(f"revenue growth check skipped: {rev_reason}")
        elif not rev_ok:
            fails.append(f"revenue shrinking: {rev_reason} [extra filter, not in source doc]")

        cr_ok, cr_reason = check_current_ratio(f)
        if cr_ok is None:
            warns.append(f"current ratio check skipped: {cr_reason}")
        elif not cr_ok:
            fails.append(f"weak short-term liquidity: {cr_reason} [extra filter, not in source doc]")

        analyst_ok, analyst_reason = check_analyst_sentiment(f)
        if analyst_ok is False:
            warns.append(f"soft flag (not disqualifying): {analyst_reason} [extra filter, not in source doc]")

    return fails, warns


def _check_technicals(t: TechnicalSnapshot) -> tuple[list[str], list[str]]:
    fails: list[str] = []
    warns: list[str] = []

    uptrend_ok, uptrend_reason = check_uptrend(t)
    if not uptrend_ok:
        fails.append(f"not in a confirmed uptrend ({uptrend_reason})")

    if not rules.STRICT_MODE:
        if t.realized_vol_pctile is not None and t.realized_vol_pctile < rules.IV_PERCENTILE_MIN:
            fails.append(
                f"realized-vol percentile {t.realized_vol_pctile:.0f} < {rules.IV_PERCENTILE_MIN:.0f} "
                "(premium likely too thin to hit the RoR target — extra filter, not in source doc)"
            )

    return fails, warns


def screen_symbol(symbol: str) -> ScreenResult:
    try:
        daily = get_price_history(symbol, period="2y")
        tech = build_snapshot(symbol, daily)
    except Exception as exc:
        return ScreenResult(symbol=symbol, passed=False, reasons_failed=[f"price data error: {exc}"])

    try:
        fund = get_fundamentals(symbol)
    except Exception as exc:
        return ScreenResult(symbol=symbol, passed=False, reasons_failed=[f"fundamentals data error: {exc}"], technicals=tech)

    fund_fails, fund_warns = _check_fundamentals(fund)
    tech_fails, tech_warns = _check_technicals(tech)

    all_fails = fund_fails + tech_fails
    all_warns = fund_warns + tech_warns

    return ScreenResult(
        symbol=symbol,
        passed=(len(all_fails) == 0),
        reasons_failed=all_fails,
        warnings=all_warns,
        technicals=tech,
        fundamentals=fund,
    )


def screen_universe(symbols: list[str], progress: bool = True) -> list[ScreenResult]:
    results = []
    for i, sym in enumerate(symbols):
        if progress and i % 25 == 0:
            print(f"[screener] {i}/{len(symbols)} ({sym})...")
        results.append(screen_symbol(sym))
    return results
