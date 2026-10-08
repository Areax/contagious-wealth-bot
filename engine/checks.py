"""
Individual named boolean checks, factored out so engine/screener.py (hard
pass/fail) and engine/scorecard.py (checklist display + ranking) share one
definition of each rule instead of two copies that can drift apart.

Every check returns (bool | None, str). None means "insufficient data to
evaluate" — never silently treated as pass or fail by callers.
"""

from __future__ import annotations

from config import strategy_rules as rules
from data.fundamentals import Fundamentals
from engine.technicals import TechnicalSnapshot


def check_pe(f: Fundamentals) -> tuple[bool | None, str]:
    if f.pe_ratio is None:
        return None, "P/E unavailable"
    if f.pe_ratio < 0:
        return False, f"negative P/E ({f.pe_ratio:.1f})"
    if f.pe_ratio <= rules.WATCHLIST_MAX_PE:
        return True, f"P/E {f.pe_ratio:.1f} <= {rules.WATCHLIST_MAX_PE:.0f}"
    if f.free_cash_flow is not None and f.total_debt is not None and f.free_cash_flow > 0:
        years = f.total_debt / f.free_cash_flow
        if years <= rules.WATCHLIST_PE_OVERRIDE_FCF_YEARS:
            return True, f"P/E {f.pe_ratio:.1f} > {rules.WATCHLIST_MAX_PE:.0f} but FCF retires debt in ~{years:.1f}yr"
        return False, f"P/E {f.pe_ratio:.1f} > {rules.WATCHLIST_MAX_PE:.0f}, FCF override needs ~{years:.1f}yr"
    return False, f"P/E {f.pe_ratio:.1f} > {rules.WATCHLIST_MAX_PE:.0f}, no FCF/debt data to check override"


def check_fcf_positive(f: Fundamentals) -> tuple[bool | None, str]:
    if f.free_cash_flow is None:
        return None, "FCF unavailable"
    return f.free_cash_flow > 0, f"FCF ${f.free_cash_flow:,.0f}"


def check_debt_to_equity(f: Fundamentals) -> tuple[bool | None, str]:
    """NOT in source doc. Sector exception (Financials/Utilities) is
    applied by the caller (screener), not here — this returns the raw
    check result plus enough info for the caller to decide."""
    if f.debt_to_equity is None:
        return None, "debt/equity unavailable"
    ok = f.debt_to_equity <= rules.MAX_DEBT_TO_EQUITY
    return ok, f"D/E {f.debt_to_equity:.0f}% (max {rules.MAX_DEBT_TO_EQUITY:.0f}%)"


def check_return_on_equity(f: Fundamentals) -> tuple[bool | None, str]:
    """NOT in source doc."""
    if f.return_on_equity is None:
        return None, "ROE unavailable"
    ok = f.return_on_equity >= rules.MIN_RETURN_ON_EQUITY
    return ok, f"ROE {f.return_on_equity:.1%} (min {rules.MIN_RETURN_ON_EQUITY:.0%})"


def check_revenue_growth(f: Fundamentals) -> tuple[bool | None, str]:
    """NOT in source doc."""
    if f.revenue_growth_yoy is None:
        return None, "revenue growth unavailable"
    ok = f.revenue_growth_yoy >= rules.MIN_REVENUE_GROWTH_YOY
    return ok, f"revenue growth {f.revenue_growth_yoy:+.1%} YoY"


def check_current_ratio(f: Fundamentals) -> tuple[bool | None, str]:
    """NOT in source doc."""
    if f.current_ratio is None:
        return None, "current ratio unavailable"
    ok = f.current_ratio >= rules.MIN_CURRENT_RATIO
    return ok, f"current ratio {f.current_ratio:.2f} (min {rules.MIN_CURRENT_RATIO:.1f})"


def check_analyst_sentiment(f: Fundamentals) -> tuple[bool | None, str]:
    """NOT in source doc. Soft/informational only — never a hard fail,
    see config/strategy_rules.py for why."""
    if f.analyst_recommendation_mean is None:
        return None, "analyst rating unavailable"
    ok = f.analyst_recommendation_mean <= rules.ANALYST_RECOMMENDATION_WARN_ABOVE
    return ok, f"analyst mean {f.analyst_recommendation_mean:.2f} (1=Strong Buy, 5=Sell)"


def check_eps_beat(f: Fundamentals) -> tuple[bool | None, str]:
    if f.eps_beat_last_quarter is None:
        return None, "no recent EPS surprise data"
    return f.eps_beat_last_quarter, f"EPS surprise {f.eps_surprise_pct:.1f}%"


def passes_curated_quality_gate(f: Fundamentals) -> tuple[bool, list[str]]:
    """HARD gate for Curated mode only — unlike pe_ok/fcf_positive/
    debt_to_equity/roe above (which are informational-only everywhere,
    never removing a row), this actually excludes a symbol. Built to
    mechanically reproduce what "no leveraged crypto-treasury vehicles,
    no pre-revenue biotechs" looks like using only numeric fundamentals —
    see config.strategy_rules CURATED_* constants for why these specific
    thresholds. Missing data fails closed (treated as not passing) rather
    than being silently skipped, since this gate is meant to be strict.
    Returns (passed, reasons_failed)."""
    reasons = []

    if rules.CURATED_REQUIRE_POSITIVE_PE and not (f.pe_ratio is not None and f.pe_ratio > 0):
        reasons.append("not profitable (no positive trailing P/E)")

    fcf_ok = False
    if f.free_cash_flow is not None:
        if f.free_cash_flow > 0:
            fcf_ok = True
        elif f.total_cash is not None:
            runway_years = f.total_cash / abs(f.free_cash_flow) if f.free_cash_flow != 0 else float("inf")
            fcf_ok = runway_years >= rules.CURATED_MIN_CASH_RUNWAY_YEARS
    if not fcf_ok:
        reasons.append(f"negative FCF without >= {rules.CURATED_MIN_CASH_RUNWAY_YEARS:.0f}yr cash runway")

    if f.debt_to_equity is not None and f.debt_to_equity > rules.CURATED_MAX_DEBT_TO_EQUITY:
        reasons.append(f"D/E {f.debt_to_equity:.0f}% > {rules.CURATED_MAX_DEBT_TO_EQUITY:.0f}%")

    if not (f.return_on_equity is not None and f.return_on_equity >= rules.CURATED_MIN_ROE):
        reasons.append(f"ROE below {rules.CURATED_MIN_ROE:.0%}")

    return len(reasons) == 0, reasons


def check_uptrend(t: TechnicalSnapshot) -> tuple[bool, str]:
    return t.is_uptrend, "price above a rising 30-week MA (Weinstein Stage 2)"


def check_uptrend_legacy(t: TechnicalSnapshot) -> tuple[bool, str]:
    """Informational only — does NOT gate the dashboard (check_uptrend
    does). Shown side by side purely for comparison against the project's
    original 12/18-month approach."""
    return t.is_uptrend_legacy_12_18mo, "legacy: SMA stack + higher-highs/higher-lows vs 12mo or 18mo ago"


def check_csp_entry_trigger(t: TechnicalSnapshot) -> tuple[bool, str]:
    ok = (
        t.consecutive_down_days >= rules.CSP_ENTRY_DOWN_DAYS
        or t.single_day_move_pct <= -rules.CSP_ENTRY_SINGLE_DAY_DROP_PCT
    )
    return ok, f"{t.consecutive_down_days} down day(s), last move {t.single_day_move_pct:.2%}"


def check_cc_entry_trigger(t: TechnicalSnapshot) -> tuple[bool, str]:
    ok = (
        t.consecutive_up_days >= rules.CC_ENTRY_UP_DAYS
        or t.single_day_move_pct >= rules.CC_ENTRY_SINGLE_DAY_GAIN_PCT
    )
    return ok, f"{t.consecutive_up_days} up day(s), last move {t.single_day_move_pct:.2%}"


def check_near_lower_bb(t: TechnicalSnapshot, tolerance_pct: float = 0.02) -> tuple[bool | None, str]:
    if t.bb_lower is None:
        return None, "Bollinger Band unavailable"
    near = t.price <= t.bb_lower * (1 + tolerance_pct)
    return near, f"price {t.price:.2f} vs lower band {t.bb_lower:.2f}"


def check_rsi_oversold(t: TechnicalSnapshot, threshold: float = rules.LEAPS_RSI_OVERSOLD) -> tuple[bool | None, str]:
    if t.rsi14 is None:
        return None, "RSI unavailable"
    return t.rsi14 <= threshold, f"RSI14 {t.rsi14:.1f}"


def check_macd_bullish_pending(t: TechnicalSnapshot) -> tuple[bool, str]:
    return t.macd_bullish_crossover_pending, "MACD histogram rising toward a bullish cross"
