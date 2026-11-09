"""Indicator math for the Sectors tab.

Conventions follow TA-Lib, the reference implementation other libraries test
against: EMA is seeded with the SMA of its first n values (not the first
price), and ATR uses Wilder smoothing seeded with the SMA of the first n true
ranges. Warm-up bars stay NaN - never backfilled - so no value ever depends
on data from a later bar.
"""
import numpy as np
import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    x = series.to_numpy(dtype=float)
    if np.isnan(x).any():
        raise ValueError("ema() needs a gap-free series; drop or fill NaN rows first")
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return pd.Series(out, index=series.index)
    out[n - 1] = x[:n].mean()
    alpha = 2.0 / (n + 1)
    for i in range(n, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=series.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    h, l, c = (s.to_numpy(dtype=float) for s in (high, low, close))
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return pd.Series(out, index=close.index)
    prev_c = c[:-1]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))
    # tr[i] is the true range of bar i+1; the first ATR covers bars 1..n
    out[n] = tr[:n].mean()
    for i in range(n + 1, len(c)):
        out[i] = (out[i - 1] * (n - 1) + tr[i - 1]) / n
    return pd.Series(out, index=close.index)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI, seeded like TA-Lib: the first average gain/loss is the
    plain mean of the first n changes."""
    c = close.to_numpy(dtype=float)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return pd.Series(out, index=close.index)
    d = np.diff(c)
    gain, loss = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = gain[:n].mean(), loss[:n].mean()
    for i in range(n, len(c)):
        if i > n:
            ag = (ag * (n - 1) + gain[i - 1]) / n
            al = (al * (n - 1) + loss[i - 1]) / n
        out[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return pd.Series(out, index=close.index)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    """MACD line (fast EMA - slow EMA) and its signal line (EMA of the line)."""
    line = ema(close, fast) - ema(close, slow)
    ready = line.dropna()
    sig = ema(ready, signal).reindex(close.index) if len(ready) else line * np.nan
    return line, sig


def swing_points(values: np.ndarray, k: int, kind: str) -> list[int]:
    """Indices of swing highs (kind='high') or lows (kind='low'): the bar is
    the extreme of the 2k+1 bars centred on it. A swing at index i is only
    knowable at bar i + k - callers must not act on it earlier.

    Ties resolve to the leftmost bar so a flat top yields one swing, not two."""
    idx = []
    for i in range(k, len(values) - k):
        left, right = values[i - k:i], values[i + 1:i + k + 1]
        v = values[i]
        if kind == "high" and v > left.max() and v >= right.max():
            idx.append(i)
        elif kind == "low" and v < left.min() and v <= right.min():
            idx.append(i)
    return idx
