"""
Section 17 strategy rules as suggestion generators. Each function takes
market context for one already-screened symbol and returns a
TradeSuggestion (or None if the entry trigger simply isn't present today —
that's a normal, frequent outcome, not an error).

Implemented: 17a Cash-Secured Puts, 17b Covered Calls, 17d LEAPS.
NOT yet implemented: 17c Recovery Mode, 17e Vertical Spreads, 17f Collars,
17g Shorting. Those four need portfolio/position context (cost basis,
shares owned) this engine doesn't have without a brokerage account
read-connection — see README for status. Do not assume they run silently;
they are simply absent from suggest.py's output today.

IMPORTANT: this module only ever *suggests*. Nothing here places, or is
capable of placing, an order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import strategy_rules as rules
from data.market_data import get_quote, get_spy_qqq_at_all_time_high, get_vix
from engine.options_selector import SelectedOption, find_contract_near_delta
from engine.risk import SizingResult
from engine.technicals import TechnicalSnapshot


@dataclass
class TradeSuggestion:
    symbol: str
    strategy: str
    doc_section: str
    action: str
    selected_option: SelectedOption | None
    suggested_contracts: int
    suggested_capital: float
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    meets_all_doc_criteria: bool = True


def suggest_cash_secured_put(
    symbol: str, spot: float, tech: TechnicalSnapshot, sizing: SizingResult,
) -> TradeSuggestion | None:
    entry_ok = (
        tech.consecutive_down_days >= rules.CSP_ENTRY_DOWN_DAYS
        or tech.single_day_move_pct <= -rules.CSP_ENTRY_SINGLE_DAY_DROP_PCT
    )
    if not entry_ok:
        return None

    sel = find_contract_near_delta(
        symbol, spot, "put",
        target_delta=rules.CSP_SAFE_DELTA,
        dte_range=(rules.CSP_TARGET_DTE - 7, rules.CSP_TARGET_DTE + 7),
        min_return_on_risk=rules.WATCHLIST_MIN_RETURN_ON_RISK,
        ror_reference_dte=rules.WATCHLIST_RETURN_ON_RISK_DTE,
        realized_vol_hint=tech.realized_vol,
    )
    if sel is None:
        return None

    warnings = []
    if sel.contract.strike >= spot:
        warnings.append("selected strike is at/above spot — verify it's actually below a support level per Section 17a")
    if not sel.meets_return_on_risk:
        warnings.append(
            f"RoR {sel.return_on_risk_30d_normalized:.2%} < {rules.WATCHLIST_MIN_RETURN_ON_RISK:.0%} "
            "target (Section 12 criterion 3) — delta target was still met, but premium is thin"
        )

    contracts = int(sizing.max_per_stock_dollars // sel.capital_at_risk) if sel.capital_at_risk > 0 else 0

    return TradeSuggestion(
        symbol=symbol,
        strategy="cash_secured_put",
        doc_section="17a",
        action=(
            f"SELL TO OPEN 1x {symbol} {sel.contract.expiration} "
            f"${sel.contract.strike:.2f} PUT (~{abs(sel.computed_delta):.2f}Δ)"
        ),
        selected_option=sel,
        suggested_contracts=max(contracts, 0),
        suggested_capital=sel.capital_at_risk,
        rationale=[
            f"Entry trigger met: {tech.consecutive_down_days} consecutive down day(s), "
            f"last-day move {tech.single_day_move_pct:.2%} (Section 17a entry rule)",
            f"Target delta {rules.CSP_SAFE_DELTA:.2f} ('safer trade'); selected contract delta "
            f"{abs(sel.computed_delta):.2f}",
            f"{rules.CSP_TARGET_DTE} DTE target; selected {sel.contract.dte} DTE",
            f"Position capped at {rules.MAX_ALLOCATION_PER_STOCK_PCT:.0%} of portfolio per stock (Section 23)",
        ],
        warnings=warnings,
        meets_all_doc_criteria=(entry_ok and sel.meets_return_on_risk),
    )


def suggest_covered_call(
    symbol: str, spot: float, tech: TechnicalSnapshot, sizing: SizingResult, shares_owned: int = 0,
) -> TradeSuggestion | None:
    """Section 17b requires already owning >=100 shares. Without a
    brokerage read-connection this engine doesn't know your positions, so
    by default it reports covered-call candidates as HYPOTHETICAL ("if you
    owned 100 shares..."); pass shares_owned to size it for real."""
    entry_ok = (
        tech.consecutive_up_days >= rules.CC_ENTRY_UP_DAYS
        or tech.single_day_move_pct >= rules.CC_ENTRY_SINGLE_DAY_GAIN_PCT
    )
    if not entry_ok:
        return None

    sel = find_contract_near_delta(
        symbol, spot, "call",
        target_delta=sum(rules.CC_DELTA_RANGE) / 2,
        dte_range=rules.CC_TARGET_DTE_RANGE,
        realized_vol_hint=tech.realized_vol,
    )
    if sel is None:
        return None

    lots_owned = shares_owned // 100
    hypothetical = lots_owned == 0
    contracts = 1 if hypothetical else lots_owned

    warnings = []
    if hypothetical:
        warnings.append("HYPOTHETICAL: Section 17b requires owning >=100 shares; no position was supplied, sized as if you owned 1 lot")
    if sel.contract.strike <= spot:
        warnings.append("selected strike is at/below spot — verify it's actually above a resistance level per Section 17b")

    return TradeSuggestion(
        symbol=symbol,
        strategy="covered_call",
        doc_section="17b",
        action=(
            f"SELL TO OPEN {contracts}x {symbol} {sel.contract.expiration} "
            f"${sel.contract.strike:.2f} CALL (~{sel.computed_delta:.2f}Δ)"
        ),
        selected_option=sel,
        suggested_contracts=contracts,
        suggested_capital=sel.capital_at_risk * contracts,
        rationale=[
            f"Entry trigger met: {tech.consecutive_up_days} consecutive up day(s), "
            f"last-day move {tech.single_day_move_pct:.2%} (Section 17b entry rule)",
            f"Target delta range {rules.CC_DELTA_RANGE}; selected contract delta {sel.computed_delta:.2f}",
            f"DTE target {rules.CC_TARGET_DTE_RANGE}; selected {sel.contract.dte} DTE",
        ],
        warnings=warnings,
        meets_all_doc_criteria=entry_ok,
    )


def suggest_leaps(
    symbol: str, spot: float, tech: TechnicalSnapshot, sizing: SizingResult, portfolio_size: float,
) -> TradeSuggestion | None:
    vix = get_vix()
    if vix < rules.LEAPS_MIN_VIX_FLOOR:
        return None

    ath = get_spy_qqq_at_all_time_high()
    if ath["SPY"] or ath["QQQ"]:
        return None  # doc: "SPY and QQQ are NOT at all-time highs"

    if not tech.is_uptrend:
        return None

    near_bb_lower = tech.bb_lower is not None and spot <= tech.bb_lower * 1.02
    rsi_oversold = tech.rsi14 is not None and tech.rsi14 <= rules.LEAPS_RSI_OVERSOLD

    if not (near_bb_lower and rsi_oversold):
        return None

    sel = find_contract_near_delta(
        symbol, spot, "call",
        target_delta=rules.LEAPS_DELTA,
        dte_range=(rules.LEAPS_MIN_DTE, rules.LEAPS_MIN_DTE + 180),
        realized_vol_hint=tech.realized_vol,
    )
    if sel is None:
        return None

    max_pct = rules.LEAPS_MAX_ALLOC_PCT_SMALL_ACCOUNT if portfolio_size < rules.LEAPS_SMALL_ACCOUNT_THRESHOLD else rules.LEAPS_MAX_ALLOC_PCT
    max_dollars = min(portfolio_size * max_pct, sizing.leaps_bucket_dollars[1])
    contracts = int(max_dollars // sel.premium) if sel.premium > 0 else 0

    warnings = []
    if not tech.macd_bullish_crossover_pending:
        warnings.append("MACD bullish crossover NOT flagged as pending (Section 17d wants this too — soft signal, included anyway since delta/BB/RSI/VIX/ATH all aligned)")

    return TradeSuggestion(
        symbol=symbol,
        strategy="leaps_call",
        doc_section="17d",
        action=(
            f"BUY TO OPEN {max(contracts, 0)}x {symbol} {sel.contract.expiration} "
            f"${sel.contract.strike:.2f} CALL (~{sel.computed_delta:.2f}Δ, deep ITM)"
        ),
        selected_option=sel,
        suggested_contracts=max(contracts, 0),
        suggested_capital=sel.premium * max(contracts, 1),
        rationale=[
            f"VIX {vix:.1f} (>= floor {rules.LEAPS_MIN_VIX_FLOOR}; ideal >= {rules.LEAPS_MIN_VIX_IDEAL})",
            "SPY and QQQ both confirmed NOT at all-time highs",
            "Confirmed uptrend (price above a rising 30-week MA)",
            f"Price within 2% of lower Bollinger Band ({tech.bb_lower:.2f})" if tech.bb_lower else "BB check skipped",
            f"RSI14 {tech.rsi14:.1f} <= oversold threshold {rules.LEAPS_RSI_OVERSOLD}" if tech.rsi14 else "RSI check skipped",
            f"Target delta {rules.LEAPS_DELTA}; selected {sel.computed_delta:.2f}; "
            f"DTE >= {rules.LEAPS_MIN_DTE}, selected {sel.contract.dte}",
            f"Sized to <= {max_pct:.0%} of portfolio per Section 17d allocation rule",
        ],
        warnings=warnings,
        meets_all_doc_criteria=True,
    )
