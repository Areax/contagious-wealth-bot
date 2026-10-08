"""
Given a symbol, finds the option contract that best matches a target
delta / DTE window, and computes Return on Risk (RoR) the way the doc uses
it: premium collected / capital at risk.

Delta is computed via Black-Scholes from an implied vol WE solve ourselves
(engine.black_scholes.implied_vol_from_price) against the contract's real
bid/ask midpoint — NOT the chain's own pre-computed `impliedVolatility`
field. We found in practice that Yahoo's per-contract IV field can be
badly wrong even when bid/ask look populated (observed: deep-ITM strikes
with 200%+ "implied vol" and open interest reading 0 across an entire
chain, most likely a stale/placeholder value rather than a real quote,
especially outside regular market hours when this free feed doesn't
reliably refresh). Solving IV ourselves from a bid/ask pair we've already
validated as a genuine two-sided, uncrossed market is more trustworthy
than relying on a vendor field we can't audit — and if OUR solver can't
land on a sane value either, we skip the contract entirely rather than
compute a delta from it. See _has_real_market and _solve_reliable_iv.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from data.market_data import OptionContract, get_option_chain, get_option_expirations
from engine.black_scholes import implied_vol_from_price, price_and_greeks

# Reject any contract whose implied vol (ours or the chain's) falls
# outside this band — even a genuinely volatile stock rarely justifies
# outside this in a 30-90 day option; anything past it is far more likely
# to be a data artifact than real market pricing.
_MIN_SANE_IV = 0.03
_MAX_SANE_IV = 2.0


@dataclass
class SelectedOption:
    contract: OptionContract
    computed_delta: float
    return_on_risk: float          # premium / capital at risk, for the actual DTE
    return_on_risk_30d_normalized: float  # RoR scaled to a 30-day-equivalent, for cross-comparison
    capital_at_risk: float
    premium: float
    meets_return_on_risk: bool     # whether it cleared the caller's min_return_on_risk, if one was given
    at_or_above_target_delta: bool | None = None  # set only by find_put_at_least_delta; literal fact: |delta| >= target_delta
    is_fallback_pick: bool = False  # find_put_at_least_delta only: True if no expiration in the search window had a clean (non-overshooting) match, and this is the best validated contract found regardless — treat with more caution than a clean pick, even if at_or_above_target_delta happens to be True


def _has_real_market(c: OptionContract) -> bool:
    """True only if the contract has a genuine live two-sided quote. When
    yfinance returns bid=ask=0 (happens for thinly/never-traded strikes,
    or transient data gaps even on liquid names), our `mid` silently falls
    back to a stale `lastPrice` — and Yahoo's own impliedVolatility for
    that contract degrades toward its solver's floor (we've seen it hit
    ~0.00001), which produces a nonsense delta (often exactly -1.0 or 0.0)
    instead of erroring. A contract with no real bid/ask also can't
    actually be traded at the price we'd be computing RoR from. So: no
    bid+ask, no trust — skip it rather than risk it winning a delta match
    and getting presented as real. Also rejects a crossed market (bid >
    ask) — seen in practice on the same broken/stale contracts, and never
    a real tradeable quote."""
    return c.bid > 0 and c.ask > 0 and c.ask >= c.bid


def _solve_reliable_iv(option_type: str, spot: float, c: OptionContract) -> float | None:
    """Solves implied vol ourselves from the contract's real bid/ask
    midpoint (caller must have already confirmed _has_real_market(c)).
    Returns None if the solver can't find a sane value — that's a signal
    the quote itself is unreliable (e.g. a price inconsistent with any
    reasonable vol given intrinsic value), not just a formality."""
    iv = implied_vol_from_price(option_type, spot, c.strike, c.dte / 365.0, c.mid)
    if iv is None or iv < _MIN_SANE_IV or iv > _MAX_SANE_IV:
        return None
    return iv


# A candidate's solved IV must fall within this multiple of the reference
# IV to be trusted. Real volatility smiles are smooth and IV normally sits
# somewhat above realized vol (a risk premium) — but an 80%+ solved IV next
# to a ~25% reference is corrupted/stale data, not real skew or premium. A
# single global IV band (_MIN/MAX_SANE_IV) can't catch this on its own
# because a fake price can still solve to something that looks "sane" in
# isolation — it just won't be consistent with a real reference point.
_IV_SKEW_BAND = (0.4, 3.0)

# In find_put_at_least_delta: a delta that clears target_delta by more than
# this multiple isn't treated as "the closest match at or above target" —
# see the comment at its use site. A real, liquidly-quoted chain has
# strikes every $1-5 near the target zone, so a genuine closest-above match
# is rarely more than ~50% over target; a 3x+ jump straight to a deep-ITM
# strike means the real near-target strikes just have no market right now.
_MAX_DELTA_OVERSHOOT_MULTIPLE = 2.0


def _reference_iv(chain: list[OptionContract], spot: float, option_type: str, realized_vol_hint: float | None) -> float | None:
    """The anchor every candidate strike's solved IV gets sanity-checked
    against for this expiration.

    Prefers `realized_vol_hint` (the stock's own recent realized
    volatility, computed independently from price history — see
    engine/technicals.current_realized_vol) when available, since it can't
    be corrupted by the same data-quality issues that can hit the options
    feed. We found in practice that the options-chain-only fallback below
    isn't safe on its own: when an ENTIRE expiration's near-the-money
    quotes are stale/zeroed (common outside market hours), the "closest
    available" ATM contract can itself be one of the bad ones, which then
    falsely validates its equally-bad neighbors as self-consistent.

    Falls back to the closest-to-the-money contract with a real market
    only when no realized-vol hint was supplied by the caller. Returns
    None if neither is available — the caller should then treat this
    expiration as unverifiable rather than guess."""
    if realized_vol_hint is not None and _MIN_SANE_IV <= realized_vol_hint <= _MAX_SANE_IV:
        return realized_vol_hint

    real_market = [c for c in chain if _has_real_market(c)]
    if not real_market:
        return None
    atm = min(real_market, key=lambda c: abs(c.strike - spot))
    return _solve_reliable_iv(option_type, spot, atm)


def _nearest_expirations_in_range(symbol: str, min_dte: int, max_dte: int) -> list[str]:
    today = datetime.now().date()
    out = []
    for exp in get_option_expirations(symbol):
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if min_dte <= dte <= max_dte:
            out.append(exp)
    return out


def find_contract_near_delta(
    symbol: str,
    spot: float,
    option_type: str,           # "put" or "call"
    target_delta: float,        # magnitude, e.g. 0.20 (sign handled internally)
    dte_range: tuple[int, int],
    min_return_on_risk: float | None = None,
    ror_reference_dte: int = 30,
    realized_vol_hint: float | None = None,  # see _reference_iv — strongly recommended, pass tech.realized_vol
) -> SelectedOption | None:
    """Finds the contract whose delta is closest to `target_delta`, within
    the DTE window — full stop. `min_return_on_risk` is NOT used to widen
    the delta search (that would silently trade a different risk profile
    than the one targeted); it only sets `meets_return_on_risk` on the
    result so the caller can apply Section 12 criterion 3 ("20-delta AND
    >=2% RoR") as its own pass/fail, exactly as the doc states it — two
    independent conditions on the SAME contract, not a search over both."""
    min_dte, max_dte = dte_range
    expirations = _nearest_expirations_in_range(symbol, min_dte, max_dte)
    if not expirations:
        return None

    # Prefer the expiration closest to the midpoint of the DTE window.
    midpoint = (min_dte + max_dte) / 2
    expirations.sort(key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d").date() - datetime.now().date()).days - midpoint))

    best: OptionContract | None = None
    best_greeks = None
    best_delta_diff = float("inf")

    for exp in expirations[:3]:  # don't hammer the data source; nearest few are enough
        chain = [c for c in get_option_chain(symbol, exp) if c.option_type == option_type]
        ref_iv = _reference_iv(chain, spot, option_type, realized_vol_hint)
        if ref_iv is None:
            continue  # this expiration's data isn't trustworthy enough to validate any strike against
        for c in chain:
            if c.dte <= 0 or not _has_real_market(c):
                continue
            iv = _solve_reliable_iv(option_type, spot, c)
            if iv is None or not (_IV_SKEW_BAND[0] * ref_iv <= iv <= _IV_SKEW_BAND[1] * ref_iv):
                continue  # our own solver couldn't land on a sane IV — skip rather than compute nonsense delta
            g = price_and_greeks(option_type, spot, c.strike, c.dte / 365.0, iv)
            diff = abs(abs(g.delta) - target_delta)

            if diff < best_delta_diff:
                best_delta_diff = diff
                best = c
                best_greeks = g

    if best is None:
        return None

    # Puts: net of premium received (strike*100 - premium), matching
    # thinkorswim's convention — see find_put_at_least_delta for the
    # verification. Calls (covered call / LEAPS long call): capital_at_risk
    # is the share cost basis, a different concept — left as spot*100.
    premium = best.mid * 100
    capital_at_risk = (best.strike * 100 - premium) if option_type == "put" else spot * 100
    ror = premium / capital_at_risk
    ror_normalized = ror * (ror_reference_dte / best.dte)
    meets_ror = True if min_return_on_risk is None else (ror_normalized >= min_return_on_risk)

    return SelectedOption(
        contract=best,
        computed_delta=best_greeks.delta,
        return_on_risk=ror,
        return_on_risk_30d_normalized=ror_normalized,
        capital_at_risk=capital_at_risk,
        premium=premium,
        meets_return_on_risk=meets_ror,
    )


def find_put_at_least_delta(
    symbol: str,
    spot: float,
    target_delta: float,        # magnitude, e.g. 0.20
    min_dte: int = 30,          # hard floor — never select a contract under this many days out
    max_dte: int = 90,          # how far out we're willing to search if nearer expirations don't reach target_delta
    ror_reference_dte: int = 30,
    max_expirations_checked: int = 6,
    realized_vol_hint: float | None = None,  # see _reference_iv — strongly recommended, pass tech.realized_vol
    max_delta: float | None = None,  # magnitude ceiling — see docstring; None keeps the old 2x-overshoot tolerance
) -> SelectedOption | None:
    """For the CSP dashboard: 'the nearest-or-greater-than-target-delta put,
    at least min_dte days out, as close to min_dte as possible.'

    Expirations are scanned in ascending DTE order starting at min_dte (so
    the very first one checked is the closest-to-30-days contract), and we
    STOP at the first expiration whose chain actually contains a put with
    |delta| >= target_delta — that's the result. We only move to a
    farther-dated expiration when the nearer one doesn't reach target_delta
    at any strike (a longer-dated contract carries more time value, so it
    reaches a given delta at a more OTM/safer strike — this is what lets
    the search actually find a real ~20-delta put on a low-IV name instead
    of always landing on the same near-the-money contract from the nearest
    expiration). If NO expiration within [min_dte, max_dte] ever reaches
    target_delta, falls back to the single best (highest-delta) contract
    seen across the whole search and sets at_or_above_target_delta=False,
    so the caller can flag it rather than silently presenting it as a real
    match.

    max_delta: BUG FOUND IN PRACTICE (LYFT, live chain, 2026-09-25) — the
    "stop searching" tolerance used to be hardcoded to target_delta * 2.0
    (_MAX_DELTA_OVERSHOOT_MULTIPLE), regardless of what ceiling the CALLER
    actually accepts. csp_dashboard.py separately rejects anything outside
    CSP_DASHBOARD_ACCEPTABLE_DELTA_RANGE (0.20-0.30) — so a name whose
    nearest expiration's best match landed at, say, 0.30-0.40 would get
    "found" here (0.40 <= 2x0.20), stop the search, then get thrown away
    entirely by the caller's stricter band check — even when the VERY NEXT
    expiration had a clean in-band match that this function never even
    looked at, because it already thought it was done. Confirmed on live
    data: LYFT's nearest (Oct 30) expiration best-matches at -0.302 (just
    over 0.30) and stopped there; the next (Nov 20) expiration had -0.218,
    a clean in-band match, sitting unexamined one loop iteration later.
    Passing max_delta lets the caller align this function's stopping
    tolerance with its own actual acceptance ceiling, so it keeps
    searching farther-dated expirations instead of settling for a match
    it's just going to discard anyway."""
    expirations = _nearest_expirations_in_range(symbol, min_dte, max_dte)
    if not expirations:
        return None

    expirations.sort(key=lambda e: (datetime.strptime(e, "%Y-%m-%d").date() - datetime.now().date()).days)

    fallback_contract: OptionContract | None = None
    fallback_greeks = None
    fallback_delta = -float("inf")  # largest delta magnitude seen, across ALL scanned expirations

    for exp in expirations[:max_expirations_checked]:
        chain = [c for c in get_option_chain(symbol, exp) if c.option_type == "put"]
        ref_iv = _reference_iv(chain, spot, "put", realized_vol_hint)
        if ref_iv is None:
            continue  # this expiration's data isn't trustworthy enough to validate any strike against — try the next one

        best_here: OptionContract | None = None
        best_here_greeks = None
        best_here_delta = float("inf")  # smallest delta magnitude >= target, within THIS expiration

        for c in chain:
            if c.dte <= 0 or not _has_real_market(c):
                continue
            iv = _solve_reliable_iv("put", spot, c)
            if iv is None or not (_IV_SKEW_BAND[0] * ref_iv <= iv <= _IV_SKEW_BAND[1] * ref_iv):
                continue
            g = price_and_greeks("put", spot, c.strike, c.dte / 365.0, iv)
            delta_mag = abs(g.delta)

            if delta_mag >= target_delta and delta_mag < best_here_delta:
                best_here_delta = delta_mag
                best_here = c
                best_here_greeks = g

            if delta_mag > fallback_delta:
                fallback_delta = delta_mag
                fallback_contract = c
                fallback_greeks = g

        stop_tolerance = max_delta if max_delta is not None else target_delta * _MAX_DELTA_OVERSHOOT_MULTIPLE
        if best_here is not None and best_here_delta <= stop_tolerance:
            # This expiration (the closest-to-min_dte one checked so far
            # that has a qualifying strike) reaches target_delta without a
            # huge jump — stop here rather than searching farther than
            # necessary.
            chosen, greeks, is_fallback = best_here, best_here_greeks, False
            break
        # else: best_here (if any) was >= target_delta but only by a huge
        # margin (e.g. a 0.69 delta "match" for a 0.20 target) — that jump
        # is itself a sign of a sparse/broken chain (the real near-target
        # strikes have no market right now), not a genuine closest match.
        # Don't stop here; keep searching farther-dated expirations. It's
        # already covered by the fallback-of-last-resort tracking above if
        # nothing better ever turns up.
    else:
        if fallback_contract is None:
            return None
        chosen, greeks, is_fallback = fallback_contract, fallback_greeks, True

    # Computed directly from the chosen contract's own delta rather than
    # hardcoded per code path — a fallback pick can still legitimately have
    # cleared target_delta (just by a margin big enough that we didn't stop
    # the search early for it), and the flag should say so accurately.
    met = bool(abs(greeks.delta) >= target_delta)

    # capital_at_risk is NET of premium received (strike*100 - premium),
    # matching thinkorswim's convention — not the gross strike*100. Verified
    # directly against a live ToS screenshot (TNK, 2026-09-25: our strike/
    # DTE/delta/bid/ask/OI matched exactly; RoR only matched once this net
    # basis was used — gross gave 2.86%, net gives 2.94%, ToS showed 2.94%).
    premium = chosen.mid * 100
    capital_at_risk = chosen.strike * 100 - premium
    ror = premium / capital_at_risk
    ror_normalized = ror * (ror_reference_dte / chosen.dte)

    return SelectedOption(
        contract=chosen,
        computed_delta=greeks.delta,
        return_on_risk=ror,
        return_on_risk_30d_normalized=ror_normalized,
        capital_at_risk=capital_at_risk,
        premium=premium,
        meets_return_on_risk=True,  # not gated here; dashboard sorts by RoR directly
        at_or_above_target_delta=met,
        is_fallback_pick=is_fallback,
    )


def find_contract_in_delta_range(
    symbol: str,
    spot: float,
    option_type: str,
    delta_range: tuple[float, float],
    dte_range: tuple[int, int],
    min_return_on_risk: float | None = None,
    ror_reference_dte: int = 30,
    realized_vol_hint: float | None = None,
) -> SelectedOption | None:
    """Like find_contract_near_delta but targets the midpoint of a delta
    range (used for the doc's ranged deltas, e.g. 25-30-delta covered calls)."""
    mid_delta = sum(delta_range) / 2
    return find_contract_near_delta(
        symbol, spot, option_type, mid_delta, dte_range,
        min_return_on_risk=min_return_on_risk, ror_reference_dte=ror_reference_dte,
        realized_vol_hint=realized_vol_hint,
    )
