"""
Step 2: for a given list of tickers (typically the AI suggestion layer's
output from ai_suggest/, but works on any list), run every qualitative/
technical check as a row of checkmarks, find the real contract nearest
-0.20 delta on the put side, and rank tickers by the premium that contract
yields (return-on-risk, 30-day-normalized) — highest premium-for-the-risk
first.

This is diagnostic/comparison output, not a suggestion to trade — it
answers "which of these names is currently showing the most checks AND
paying the most for roughly the same (~20-delta) risk," which strategies.py
then turns into an actual suggestion with an entry-trigger gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import strategy_rules as rules
from data.fundamentals import Fundamentals, get_fundamentals
from data.market_data import get_price_history, get_quote
from engine.checks import (
    check_csp_entry_trigger,
    check_current_ratio,
    check_debt_to_equity,
    check_eps_beat,
    check_fcf_positive,
    check_macd_bullish_pending,
    check_near_lower_bb,
    check_pe,
    check_return_on_equity,
    check_revenue_growth,
    check_rsi_oversold,
    check_uptrend,
)
from engine.options_selector import SelectedOption, find_contract_near_delta
from engine.technicals import TechnicalSnapshot, build_snapshot

# (key, display_label, check_fn(tech, fund) -> (bool|None, str))
# First 4 + uptrend are Section 12 (source doc); the rest are the extra
# quality/technical checks layered on top — see config/strategy_rules.py.
CHECK_DEFINITIONS = [
    ("uptrend", "Uptrend", lambda t, f: check_uptrend(t)),
    ("pe_ok", "P/E", lambda t, f: check_pe(f)),
    ("fcf_positive", "FCF+", lambda t, f: check_fcf_positive(f)),
    ("eps_beat", "EPS Beat", lambda t, f: check_eps_beat(f)),
    ("debt_to_equity", "D/E", lambda t, f: check_debt_to_equity(f)),
    ("roe", "ROE", lambda t, f: check_return_on_equity(f)),
    ("rev_growth", "Rev Growth", lambda t, f: check_revenue_growth(f)),
    ("csp_trigger", "3 Down Days", lambda t, f: check_csp_entry_trigger(t)),
    ("near_lower_bb", "Near Lower BB", lambda t, f: check_near_lower_bb(t)),
    ("rsi_oversold", "RSI Oversold", lambda t, f: check_rsi_oversold(t)),
    ("macd_pending", "MACD Pending", lambda t, f: check_macd_bullish_pending(t)),
]


@dataclass
class ScorecardRow:
    symbol: str
    spot: float
    checks: dict            # key -> (bool | None, reason str)
    checks_passed: int
    checks_evaluated: int
    put_option: SelectedOption | None
    error: str | None = None
    ai_theme: str | None = None
    ai_reason: str | None = None


def score_symbol(symbol: str, ai_theme: str | None = None, ai_reason: str | None = None) -> ScorecardRow:
    try:
        daily = get_price_history(symbol, period="2y")
        tech = build_snapshot(symbol, daily)
        fund = get_fundamentals(symbol)
        spot = get_quote(symbol)["last_price"]
    except Exception as exc:
        return ScorecardRow(
            symbol=symbol, spot=0.0, checks={}, checks_passed=0, checks_evaluated=0,
            put_option=None, error=str(exc), ai_theme=ai_theme, ai_reason=ai_reason,
        )

    checks = {}
    passed = 0
    evaluated = 0
    for key, _label, fn in CHECK_DEFINITIONS:
        result, reason = fn(tech, fund)
        checks[key] = (result, reason)
        if result is not None:
            evaluated += 1
            if result:
                passed += 1

    put_option = find_contract_near_delta(
        symbol, spot, "put",
        target_delta=rules.WATCHLIST_TARGET_DELTA,
        dte_range=(rules.WATCHLIST_RETURN_ON_RISK_DTE - 9, rules.WATCHLIST_RETURN_ON_RISK_DTE + 15),
        min_return_on_risk=None,  # scorecard reports RoR, doesn't gate on it — ranking IS the gate
        ror_reference_dte=rules.WATCHLIST_RETURN_ON_RISK_DTE,
        realized_vol_hint=tech.realized_vol,
    )

    return ScorecardRow(
        symbol=symbol, spot=spot, checks=checks, checks_passed=passed, checks_evaluated=evaluated,
        put_option=put_option, ai_theme=ai_theme, ai_reason=ai_reason,
    )


def build_scorecard(
    symbols: list[str],
    ai_meta: dict[str, tuple[str, str]] | None = None,  # symbol -> (theme, reason)
    progress: bool = True,
) -> list[ScorecardRow]:
    rows = []
    for i, sym in enumerate(symbols):
        if progress:
            print(f"[scorecard] {i + 1}/{len(symbols)} {sym}...")
        theme, reason = ai_meta.get(sym, (None, None)) if ai_meta else (None, None)
        rows.append(score_symbol(sym, ai_theme=theme, ai_reason=reason))

    def rank_key(row: ScorecardRow):
        # Rows with a valid put quote, ranked by 30d-normalized RoR (highest
        # premium for ~20-delta risk first); rows without one sort last.
        if row.put_option is None:
            return (1, 0.0)
        return (0, -row.put_option.return_on_risk_30d_normalized)

    return sorted(rows, key=rank_key)


def format_table(rows: list[ScorecardRow]) -> str:
    from tabulate import tabulate

    def mark(v):
        if v is None:
            return "?"
        return "✓" if v else "✗"

    headers = ["Symbol", "Spot"] + [label for _key, label, _fn in CHECK_DEFINITIONS] + [
        "Checks", "Put Strike", "Put DTE", "Put Δ", "Premium", "RoR(30d-norm)",
    ]
    table_rows = []
    for r in rows:
        if r.error:
            table_rows.append([r.symbol, "ERROR: " + r.error] + [""] * (len(headers) - 2))
            continue
        row = [r.symbol, f"{r.spot:.2f}"]
        for key, _label, _fn in CHECK_DEFINITIONS:
            row.append(mark(r.checks[key][0]))
        row.append(f"{r.checks_passed}/{r.checks_evaluated}")
        if r.put_option:
            p = r.put_option
            row += [
                f"{p.contract.strike:.2f}",
                p.contract.dte,
                f"{abs(p.computed_delta):.2f}",
                f"${p.premium:,.0f}",
                f"{p.return_on_risk_30d_normalized:.2%}",
            ]
        else:
            row += ["-", "-", "-", "-", "-"]
        table_rows.append(row)

    return tabulate(table_rows, headers=headers, tablefmt="simple")
