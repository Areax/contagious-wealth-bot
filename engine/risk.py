"""
Section 15 (VIX allocations) + Section 16 (portfolio size tiers) + Section 23
(15% max per stock) turned into position-sizing math. This module never
touches a broker — it just answers "given my portfolio size and today's
VIX, how big should this specific suggestion be?"
"""

from __future__ import annotations

from dataclasses import dataclass

from config import strategy_rules as rules


@dataclass(frozen=True)
class SizingContext:
    portfolio_size: float
    vix_level: float


@dataclass(frozen=True)
class SizingResult:
    vix_tier_label: str
    capital_deployed_pct_range: tuple
    cash_reserved_pct_range: tuple
    portfolio_tier_label: str
    wheel_bucket_dollars: tuple      # (low, high) dollars available for CSP/CC/wheel-style trades
    leaps_bucket_dollars: tuple
    spreads_bucket_dollars: tuple
    max_per_stock_dollars: float
    max_per_sector_dollars: float    # advisory only — not in source doc, see strategy_rules.STRICT_MODE


def size_portfolio(ctx: SizingContext) -> SizingResult:
    vix_tier = rules.vix_tier_for(ctx.vix_level)
    port_tier = rules.portfolio_tier_for(ctx.portfolio_size)

    deployed_low = ctx.portfolio_size * vix_tier.capital_deployed_pct[0] / 100
    deployed_high = ctx.portfolio_size * vix_tier.capital_deployed_pct[1] / 100

    def bucket(pct_range: tuple) -> tuple:
        return (
            deployed_low * pct_range[0] / 100,
            deployed_high * pct_range[1] / 100,
        )

    return SizingResult(
        vix_tier_label=vix_tier.label,
        capital_deployed_pct_range=vix_tier.capital_deployed_pct,
        cash_reserved_pct_range=vix_tier.cash_reserved_pct,
        portfolio_tier_label=port_tier.label,
        wheel_bucket_dollars=bucket(port_tier.wheel_pct),
        leaps_bucket_dollars=bucket(port_tier.leaps_pct),
        spreads_bucket_dollars=bucket(port_tier.spreads_pct),
        max_per_stock_dollars=ctx.portfolio_size * rules.MAX_ALLOCATION_PER_STOCK_PCT,
        max_per_sector_dollars=ctx.portfolio_size * rules.MAX_ALLOCATION_PER_SECTOR_PCT,
    )


def max_contracts_for_capital(capital_at_risk_per_contract: float, available_dollars: float) -> int:
    if capital_at_risk_per_contract <= 0:
        return 0
    return int(available_dollars // capital_at_risk_per_contract)


def leaps_max_allocation_pct(portfolio_size: float) -> float:
    if portfolio_size < rules.LEAPS_SMALL_ACCOUNT_THRESHOLD:
        return rules.LEAPS_MAX_ALLOC_PCT_SMALL_ACCOUNT
    return rules.LEAPS_MAX_ALLOC_PCT
