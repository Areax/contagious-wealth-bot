"""
Black-Scholes pricing and Greeks. Used two ways:

1. Live suggestions: yfinance option chains give bid/ask/impliedVolatility
   per contract but no delta column, so we compute delta ourselves from the
   chain's own implied volatility.
2. Backtesting: no historical option chain is available for free, so we
   *simulate* a chain at each historical date from the underlying's price,
   a risk-free rate, and a realized-volatility proxy for IV. This is a
   documented approximation, not real historical quotes — see backtest/.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    theta: float  # per calendar day
    vega: float   # per 1.0 (100 vol points) change in IV
    price: float


def _d1_d2(spot: float, strike: float, t_years: float, rate: float, iv: float):
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return None, None
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv ** 2) * t_years) / (iv * math.sqrt(t_years))
    d2 = d1 - iv * math.sqrt(t_years)
    return d1, d2


def price_and_greeks(
    option_type: str,       # "call" or "put"
    spot: float,
    strike: float,
    t_years: float,
    iv: float,
    rate: float = 0.045,    # approx risk-free rate; not critical for delta
) -> Greeks:
    option_type = option_type.lower()
    d1, d2 = _d1_d2(spot, strike, t_years, rate, iv)

    if d1 is None:
        # Degenerate case (expired / zero vol) -> intrinsic value, delta 0 or 1
        if option_type == "call":
            intrinsic = max(spot - strike, 0.0)
            delta = 1.0 if spot > strike else 0.0
        else:
            intrinsic = max(strike - spot, 0.0)
            delta = -1.0 if spot < strike else 0.0
        return Greeks(delta=delta, gamma=0.0, theta=0.0, vega=0.0, price=intrinsic)

    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        delta = norm.cdf(d1)
        price = spot * norm.cdf(d1) - strike * math.exp(-rate * t_years) * norm.cdf(d2)
        theta_annual = (
            -(spot * pdf_d1 * iv) / (2 * math.sqrt(t_years))
            - rate * strike * math.exp(-rate * t_years) * norm.cdf(d2)
        )
    elif option_type == "put":
        delta = norm.cdf(d1) - 1.0
        price = strike * math.exp(-rate * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        theta_annual = (
            -(spot * pdf_d1 * iv) / (2 * math.sqrt(t_years))
            + rate * strike * math.exp(-rate * t_years) * norm.cdf(-d2)
        )
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    gamma = pdf_d1 / (spot * iv * math.sqrt(t_years))
    vega = spot * pdf_d1 * math.sqrt(t_years) / 100.0  # per 1 vol point (1%)
    theta = theta_annual / 365.0

    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, price=price)


def implied_vol_from_price(
    option_type: str,
    spot: float,
    strike: float,
    t_years: float,
    market_price: float,
    rate: float = 0.045,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> float | None:
    """Bisection solver — used only as a fallback when a chain provides a
    price but no usable impliedVolatility field."""
    lo, hi = 1e-4, 5.0
    if market_price <= 0:
        return None
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        g = price_and_greeks(option_type, spot, strike, t_years, mid, rate)
        if abs(g.price - market_price) < tol:
            return mid
        if g.price > market_price:
            hi = mid
        else:
            lo = mid
    return mid
