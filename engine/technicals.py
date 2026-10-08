"""
Technical indicators used by Sections 9, 12, 14, 17 of the guide.

IV PERCENTILE NOTE: the doc's RoR/delta targets implicitly want elevated
option premium, but there's no free source of *historical option* IV to
compute a real IV-rank/IV-percentile off of. We proxy it with a 1-year
percentile rank of 20-day realized (historical) volatility of the
underlying — realized vol and implied vol are correlated but NOT the same
thing (IV usually sits above realized vol, and can diverge sharply around
events). This is clearly labeled `realized_vol_percentile`, not `iv_percentile`,
everywhere in the code so it's never confused with the real thing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window).mean()


def ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(close, window)
    std = close.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return lower, mid, upper


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def macd_bullish_crossover_pending(close: pd.Series) -> bool:
    """True if MACD is below signal but the histogram is rising toward zero
    over the last 3 bars — i.e. a crossover looks imminent, not confirmed."""
    macd_line, signal_line, hist = macd(close)
    if len(hist.dropna()) < 4:
        return False
    recent = hist.dropna().iloc[-4:]
    below_signal = macd_line.iloc[-1] < signal_line.iloc[-1]
    rising = recent.is_monotonic_increasing
    return bool(below_signal and rising)


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window).std() * np.sqrt(252)


def current_realized_vol(close: pd.Series, window: int = 20) -> float | None:
    """Latest 20-day (by default) annualized realized vol as a single
    number. Used as an independent sanity anchor for options-chain implied
    vol — it's computed purely from historical stock prices, so it can't be
    corrupted by the same data-quality issues that hit the options feed
    (e.g. stale/zeroed bid-ask outside market hours)."""
    rv = realized_vol(close, window=window).dropna()
    if rv.empty:
        return None
    return float(rv.iloc[-1])


def realized_vol_percentile(close: pd.Series, window: int = 20, lookback: int = 252) -> float | None:
    rv = realized_vol(close, window=window).dropna()
    if len(rv) < lookback // 2:
        return None
    rv_window = rv.iloc[-lookback:]
    current = rv_window.iloc[-1]
    return float((rv_window < current).mean() * 100)


def consecutive_down_days(close: pd.Series) -> int:
    """Count of trailing consecutive daily closes lower than the prior close."""
    diffs = close.diff()
    count = 0
    for val in reversed(diffs.iloc[1:].tolist()):
        if val < 0:
            count += 1
        else:
            break
    return count


def consecutive_up_days(close: pd.Series) -> int:
    diffs = close.diff()
    count = 0
    for val in reversed(diffs.iloc[1:].tolist()):
        if val > 0:
            count += 1
        else:
            break
    return count


def single_day_move_pct(close: pd.Series) -> float:
    """Most recent day's close-to-close % move (signed)."""
    if len(close) < 2:
        return 0.0
    return float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2])


UPTREND_MA_WEEKS = 30          # Stan Weinstein's Stage Analysis: 30-week SMA on weekly closes
UPTREND_RISING_LOOKBACK_WEEKS = 4  # "rising" = the MA itself is higher than it was ~1 month ago


def is_in_uptrend(daily_close: pd.Series, weekly_close: pd.Series) -> bool:
    """Whether the stock is in an uptrend, per Stan Weinstein's Stage
    Analysis: price is above a RISING 30-week (~7 month) simple moving
    average on weekly closes.

    This REPLACES the project's original approach (SMA50/100/200 stack +
    a 12-to-18-month higher-high/higher-low point comparison, meant to
    approximate the source doc's "uptrend lasting 12 to 18 months").
    Explicit user decision, made after: (1) researching that neither 18
    nor 24 months is an established convention anywhere — the two actual
    standards are a 12-month academic momentum lookback and Weinstein's
    30-week MA, which is structurally a continuously-updating trend
    filter, not a fixed-point comparison at all; (2) the 12/18-month
    point-comparison version, even after being patched to check both
    endpoints, was still exposed as brittle by real cases (IREN: massive
    run-up finishing before the 12-month mark, partial pullback since —
    read as "uptrend" on a chart, failed the rigid point comparison); (3)
    explicit instruction to prioritize this well-grounded, continuously-
    updating definition over the doc's own unvalidated duration language,
    and to let stocks fall out of the list on that basis rather than
    special-casing known candidates (SOFI, RDDT, etc.) to keep them in."""
    if len(weekly_close) < UPTREND_MA_WEEKS + UPTREND_RISING_LOOKBACK_WEEKS:
        return False

    ma = sma(weekly_close, UPTREND_MA_WEEKS)
    ma_now = ma.iloc[-1]
    ma_before = ma.iloc[-1 - UPTREND_RISING_LOOKBACK_WEEKS]
    if pd.isna(ma_now) or pd.isna(ma_before):
        return False

    price = daily_close.iloc[-1]
    above_ma = price > ma_now
    ma_rising = ma_now > ma_before

    return bool(above_ma and ma_rising)


def is_uptrend_legacy_12_18mo(daily_close: pd.Series, weekly_close: pd.Series, min_months: int = 12, max_months: int = 18) -> bool:
    """The PROJECT'S ORIGINAL uptrend check, kept only as an informational,
    non-gating comparison column — is_in_uptrend() (the 30-week MA check
    above) is what actually filters the dashboard now. Reconstructed
    exactly as it was before being replaced: price above its 50/100/200
    daily SMA stack AND weekly closes show a higher high AND higher low
    than EITHER 12 months ago OR 18 months ago. Approximates the source
    doc's literal "uptrend lasting 12 to 18 months" language. Displayed
    side by side with the new check purely so you can see how often they
    agree or disagree — see conversation notes for why the project moved
    off this as the actual filter (IREN false-negative, no established
    convention for 18mo specifically)."""
    if len(daily_close) < 210:
        return False

    sma50 = sma(daily_close, 50).iloc[-1]
    sma100 = sma(daily_close, 100).iloc[-1]
    sma200 = sma(daily_close, 200).iloc[-1]
    price = daily_close.iloc[-1]
    above_all_smas = price > sma50 > sma100 or price > sma200
    if not above_all_smas:
        return False

    recent_window = weekly_close.iloc[-13:]  # last ~3 months
    any_endpoint_had_enough_history = False

    for months in (min_months, max_months):
        weeks_needed = months * 4
        older_window = weekly_close.iloc[-weeks_needed - 13:-weeks_needed]
        if len(older_window) < 13:
            continue
        any_endpoint_had_enough_history = True
        higher_high = recent_window.max() > older_window.max()
        higher_low = recent_window.min() > older_window.min()
        if higher_high and higher_low:
            return True

    if not any_endpoint_had_enough_history:
        return True  # not enough weekly history to test either endpoint; SMA stack already confirmed above

    return False


def resample_weekly(daily_close: pd.Series) -> pd.Series:
    return daily_close.resample("W").last().dropna()


@dataclass(frozen=True)
class TechnicalSnapshot:
    symbol: str
    price: float
    sma50: float | None
    sma100: float | None
    sma200: float | None
    ema10: float | None
    ema20: float | None
    ema50: float | None
    rsi14: float | None
    bb_lower: float | None
    bb_mid: float | None
    bb_upper: float | None
    macd_bullish_crossover_pending: bool
    realized_vol_pctile: float | None
    realized_vol: float | None      # current 20-day annualized realized vol, absolute (not percentile)
    consecutive_down_days: int
    consecutive_up_days: int
    single_day_move_pct: float
    is_uptrend: bool
    is_uptrend_legacy_12_18mo: bool  # informational only — see is_uptrend_legacy_12_18mo() docstring


def build_snapshot(symbol: str, daily_df: pd.DataFrame) -> TechnicalSnapshot:
    # yfinance can serve an incomplete trailing row — e.g. Open/High/Low/
    # Volume populated but Close still NaN (observed live, 2026-09-25:
    # today's bar had a NaN Close well after market close, presumably a
    # feed lag on Yahoo's side finalizing the session). An unfiltered
    # close.iloc[-1] silently becomes NaN, which makes is_in_uptrend()
    # (price > moving average) False for EVERY symbol at once — a scan
    # returning zero results across the entire universe, not a real "no
    # uptrends today." Drop any NaN-close rows before any calculation
    # touches this series, so a bad trailing row is treated as "not
    # posted yet" rather than "the stock crashed to zero."
    close = daily_df["Close"].dropna()
    weekly_close = resample_weekly(close)
    bb_l, bb_m, bb_u = bollinger_bands(close)

    return TechnicalSnapshot(
        symbol=symbol,
        price=float(close.iloc[-1]),
        sma50=_last_or_none(sma(close, 50)),
        sma100=_last_or_none(sma(close, 100)),
        sma200=_last_or_none(sma(close, 200)),
        ema10=_last_or_none(ema(close, 10)),
        ema20=_last_or_none(ema(close, 20)),
        ema50=_last_or_none(ema(close, 50)),
        rsi14=_last_or_none(rsi(close, 14)),
        bb_lower=_last_or_none(bb_l),
        bb_mid=_last_or_none(bb_m),
        bb_upper=_last_or_none(bb_u),
        macd_bullish_crossover_pending=macd_bullish_crossover_pending(close),
        realized_vol_pctile=realized_vol_percentile(close),
        realized_vol=current_realized_vol(close),
        consecutive_down_days=consecutive_down_days(close),
        consecutive_up_days=consecutive_up_days(close),
        single_day_move_pct=single_day_move_pct(close),
        is_uptrend=is_in_uptrend(close, weekly_close),
        is_uptrend_legacy_12_18mo=is_uptrend_legacy_12_18mo(close, weekly_close),
    )


def _last_or_none(series: pd.Series) -> float | None:
    if series.empty or pd.isna(series.iloc[-1]):
        return None
    return float(series.iloc[-1])
