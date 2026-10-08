"""
Orchestrator: universe -> screen -> size -> per-strategy suggestions.
This is Mode A, the deterministic engine. It never calls an LLM and never
places an order — see engine/strategies.py's module docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

from data.market_data import get_quote, get_vix
from engine.risk import SizingContext, SizingResult, size_portfolio
from engine.screener import ScreenResult, screen_universe
from engine.strategies import (
    TradeSuggestion,
    suggest_cash_secured_put,
    suggest_covered_call,
    suggest_leaps,
)


@dataclass
class SuggestionRun:
    vix_level: float
    sizing: SizingResult
    screened: list[ScreenResult]
    suggestions: list[TradeSuggestion]


def run_suggestions(symbols: list[str], portfolio_size: float, progress: bool = True) -> SuggestionRun:
    vix = get_vix()
    sizing = size_portfolio(SizingContext(portfolio_size=portfolio_size, vix_level=vix))

    screened = screen_universe(symbols, progress=progress)
    passed = [r for r in screened if r.passed]

    suggestions: list[TradeSuggestion] = []
    for r in passed:
        try:
            spot = get_quote(r.symbol)["last_price"]
        except Exception as exc:
            if progress:
                print(f"[suggest] {r.symbol}: quote error ({exc}), skipping")
            continue

        for fn in (suggest_cash_secured_put, suggest_covered_call):
            try:
                s = fn(r.symbol, spot, r.technicals, sizing)
            except Exception as exc:
                if progress:
                    print(f"[suggest] {r.symbol} {fn.__name__} error: {exc}")
                s = None
            if s is not None:
                suggestions.append(s)

        try:
            leaps = suggest_leaps(r.symbol, spot, r.technicals, sizing, portfolio_size)
            if leaps is not None:
                suggestions.append(leaps)
        except Exception as exc:
            if progress:
                print(f"[suggest] {r.symbol} suggest_leaps error: {exc}")

    return SuggestionRun(vix_level=vix, sizing=sizing, screened=screened, suggestions=suggestions)
