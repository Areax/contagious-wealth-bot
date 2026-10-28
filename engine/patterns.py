"""
"Obvious" chart signals on daily candles, for the dashboard's Sector Signals
tab. Deliberately simple, rule-based detectors with every threshold named
below, so a reader can see exactly why a signal fired and tune it.

What each detector means (plain English first, then the rule):

- Double top (bearish): the price hit about the same high twice, with a dip
  between, and then fell below that dip. Rule: any two swing highs within
  DT_TOP_TOL of each other, DT_MIN_SEP..DT_MAX_SEP bars apart (a small bounce
  inside the dip is allowed), the lowest low between them at least
  DT_MIN_DEPTH below the lower top, and nothing higher in the CONTEXT_BARS
  before the first top or between the two (a reversal pattern sits at the
  top of the move). "Confirmed" once a
  close drops below that low (the neckline) within as many bars as separated
  the two tops; "forming" while the window is still open and no close has
  gone above the tops. A close above the tops voids the pattern; a window
  that expires without a break makes it stale and it is dropped.
- Double bottom (bullish): the mirror image, built from swing lows.
- Head and shoulders (bearish): three peaks, the middle one tallest, the two
  outer ones about level, and a close below the lower of the two dips
  between them (a conservative, horizontal neckline). Rule: three
  consecutive swing highs, head at least HS_HEAD_MIN above both shoulders,
  shoulders within HS_SHOULDER_TOL of each other. Confirmation must come
  within (right shoulder - left shoulder) bars after the right shoulder; a
  close above the right shoulder voids it.
- Inverse head and shoulders (bullish): the mirror image.
- Bull retest gap (bullish): the price jumped up leaving a gap, later came
  back down into that gap and held, closing back above it. Rule: an up-gap
  (today's low above yesterday's high by at least GAP_MIN_PCT) inside the
  last GAP_LOOKBACK bars, then at least GAP_MIN_WAIT bars later a bar whose
  low enters the gap zone and whose close is at or above the gap's top. The
  first touch decides it. If any later close falls back inside the gap, the
  signal is cancelled (the floor failed after all).
- Bear retest gap (bearish): the mirror image; the gap zone acted as a
  ceiling. "Retest gap" is an informal name; this is the usual "break and
  retest" idea applied to a price gap.
- First red candle above the 21-day EMA (watch): after at least
  EMA_PRIOR_RUN green candles, the latest candle closed down, with its whole
  range (wicks included) above the 21-day exponential moving average and a
  body of at least EMA_MIN_BODY_PCT. Many traders read this as the first
  pause in an uptrend. Rule: close < open, the previous EMA_PRIOR_RUN
  candles all green, low > EMA21, |close - open| / close >= EMA_MIN_BODY_PCT.
- First green candle below the 21-day EMA (watch): the mirror: close > open,
  previous candles all red, high < EMA21, same body minimum. A first bounce
  up day after two down days while the price sits under its 21-day average.

The two EMA candle signals are "watch" rather than bullish/bearish because
the same candle is read both ways in practice (a dip to buy, or the start of
a slide). They only fire for the most recent candle, which is the only bar
where "first" is a live fact.

A pattern only counts as current if its key bar (the confirming close, the
retest, or the candle itself) is within RECENT_BARS of the latest bar, or,
for "forming" patterns, if the last pivot is within 2 * RECENT_BARS bars.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

PIVOT_WINDOW = 5       # a swing high/low must be the extreme of the 5 bars on each side
LOOKBACK_BARS = 150    # patterns may start this far back
RECENT_BARS = 10       # key bar must be within this many bars of today to count as current

DT_TOP_TOL = 0.03      # double top/bottom: the two extremes within 3% of each other
DT_MIN_DEPTH = 0.04    # the dip/bump between them at least 4% away from the extremes
DT_MIN_SEP = 10        # bars between the two extremes
DT_MAX_SEP = 80

CONTEXT_BARS = 40       # a top/bottom pattern must be the extreme of the bars since this far before its first pivot (it reverses a trend)

HS_SHOULDER_TOL = 0.05  # shoulders within 5% of each other
HS_HEAD_MIN = 0.02      # head at least 2% beyond the taller shoulder

GAP_MIN_PCT = 0.005     # a gap must be at least 0.5% of price to count
GAP_LOOKBACK = 40       # the gap itself must be within this many bars
GAP_MIN_WAIT = 2        # the retest must come at least this many bars after the gap day

EMA_LEN = 21
EMA_PRIOR_RUN = 2       # how many opposite-colour candles must precede the "first" candle
EMA_MIN_BODY_PCT = 0.002  # ignore near-doji candles (body under 0.2% of price)


@dataclass
class Signal:
    name: str         # e.g. "Double top"
    direction: str    # "bullish" | "bearish" | "watch"
    status: str       # "confirmed" | "forming" | "latest"
    date: str         # YYYY-MM-DD of the key bar
    detail: str       # plain-English explanation with the numbers that fired it
    key_level: float | None = None  # neckline / gap edge, when there is one
    bars: tuple = ()                # pivot bar indexes, used to suppress overlapping patterns

    def to_dict(self) -> dict:
        return asdict(self)


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["Open", "High", "Low", "Close"]].dropna()
    return out.tail(LOOKBACK_BARS + PIVOT_WINDOW + EMA_LEN * 3)


def swing_points(high: np.ndarray, low: np.ndarray, k: int = PIVOT_WINDOW) -> tuple[list[int], list[int]]:
    """Indexes of swing highs and swing lows: a bar whose high (low) is the
    extreme of the k bars on each side. The last k bars can never be pivots
    yet, which is why "forming" patterns lag a little by design."""
    highs, lows = [], []
    n = len(high)
    for i in range(k, n - k):
        # Strictly above the k bars before it and at least level with the k
        # bars after it, so a flat-topped plateau yields one pivot (its first
        # bar) instead of none.
        if high[i] > high[i - k:i].max() and high[i] >= high[i + 1:i + k + 1].max():
            highs.append(i)
        if low[i] < low[i - k:i].min() and low[i] <= low[i + 1:i + k + 1].min():
            lows.append(i)
    return highs, lows


def _fmt_date(idx) -> str:
    return pd.Timestamp(idx).strftime("%Y-%m-%d")


def _recent(i: int, n: int, bars: int = RECENT_BARS) -> bool:
    return (n - 1 - i) <= bars


def _resolve(C: np.ndarray, start: int, end: int, break_fn, void_fn) -> tuple[int | None, bool, bool]:
    """Walk closes in [start, end): return (break_bar, voided, window_open).
    The first bar that breaks wins; a void before any break kills the pattern.
    A confirmed break that is later undone (any following close back on the
    wrong side of the neckline) also counts as voided: the pattern failed.
    window_open is True when the window extends past the latest bar without
    a decision, i.e. the pattern is still live ("forming")."""
    n = len(C)
    live = end > n
    end = min(end, n)
    for t in range(start, end):
        if break_fn(C[t]):
            if any(not break_fn(C[u]) for u in range(t + 1, n)):
                return None, True, False
            return t, False, False
        if void_fn(C[t]):
            return None, True, False
    return None, False, live


def detect_double(df: pd.DataFrame, highs: list[int], lows: list[int]) -> list[Signal]:
    """Double top (from swing highs) and double bottom (from swing lows).
    Returns at most one of each: the most recent that is still current."""
    out: list[Signal] = []
    H, L, C = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)

    def scan(pivs: list[int], is_top: bool) -> Signal | None:
        best = None
        # Every pair of swing points in range, not just neighbours: a small
        # bounce inside the dip adds an extra pivot but does not undo the
        # pattern. What does undo it is a peak between the two tops that
        # is higher than both (mirror for bottoms).
        pairs = [(pivs[a], pivs[b]) for a in range(len(pivs)) for b in range(a + 1, len(pivs))
                 if DT_MIN_SEP <= pivs[b] - pivs[a] <= DT_MAX_SEP]
        pairs.sort(key=lambda ij: (ij[1], ij[0]))  # later patterns win ties
        for i, j in pairs:
            sep = j - i
            p1, p2 = (H[i], H[j]) if is_top else (L[i], L[j])
            if abs(p1 - p2) / max(p1, p2) > DT_TOP_TOL:
                continue
            if is_top and H[max(0, i - CONTEXT_BARS):j + 1].max() > max(p1, p2):
                continue  # something higher nearby: these are not the tops of the move
            if not is_top and L[max(0, i - CONTEXT_BARS):j + 1].min() < min(p1, p2):
                continue
            if is_top:
                neck = L[i:j + 1].min()
                if neck > min(p1, p2) * (1 - DT_MIN_DEPTH):
                    continue
                extreme = max(p1, p2)
                broke, voided, live = _resolve(C, j + 1, j + 1 + sep, lambda c: c < neck, lambda c: c > extreme)
            else:
                neck = H[i:j + 1].max()
                if neck < max(p1, p2) * (1 + DT_MIN_DEPTH):
                    continue
                extreme = min(p1, p2)
                broke, voided, live = _resolve(C, j + 1, j + 1 + sep, lambda c: c > neck, lambda c: c < extreme)
            if voided:
                continue
            name = "Double top" if is_top else "Double bottom"
            direction = "bearish" if is_top else "bullish"
            kind = "highs" if is_top else "lows"
            if best is not None and best.status == "confirmed" and broke is None:
                continue  # a later forming pair must not replace an earlier confirmed one
            if broke is not None and _recent(broke, n):
                detail = (f"Two {kind} near {p1:.2f} and {p2:.2f} ({_fmt_date(df.index[i])} and {_fmt_date(df.index[j])}), "
                          f"then a close {'below' if is_top else 'above'} the {'dip' if is_top else 'bump'} between them at {neck:.2f}.")
                best = Signal(name, direction, "confirmed", _fmt_date(df.index[broke]), detail, float(neck), (i, j))
            elif broke is None and live and _recent(j, n, RECENT_BARS * 2):
                detail = (f"Two {kind} near {p1:.2f} and {p2:.2f}; not confirmed until a close "
                          f"{'below' if is_top else 'above'} {neck:.2f}. A close {'above' if is_top else 'below'} "
                          f"{extreme:.2f} would cancel it.")
                best = Signal(name, direction, "forming", _fmt_date(df.index[j]), detail, float(neck), (i, j))
        return best

    top = scan(highs, True)
    bottom = scan(lows, False)
    if top:
        out.append(top)
    if bottom:
        out.append(bottom)
    return out


def detect_head_shoulders(df: pd.DataFrame, highs: list[int], lows: list[int]) -> list[Signal]:
    """Head and shoulders (three swing highs) and inverse (three swing lows)."""
    out: list[Signal] = []
    H, L, C = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)

    def scan(pivs: list[int], regular: bool) -> Signal | None:
        best = None
        triples = [(pivs[a], pivs[b], pivs[c]) for a in range(len(pivs)) for b in range(a + 1, len(pivs))
                   for c in range(b + 1, len(pivs)) if pivs[c] - pivs[a] <= DT_MAX_SEP * 2]
        triples.sort(key=lambda t: (t[2], t[1], t[0]))
        for s1, head, s2 in triples:
            v1, vh, v2 = (H[s1], H[head], H[s2]) if regular else (L[s1], L[head], L[s2])
            # the head must be the extreme of the whole span; otherwise some
            # other peak between the shoulders is the real head
            if regular and H[max(0, s1 - CONTEXT_BARS):s2 + 1].max() > vh:
                continue  # the head must be the top of the whole move, not a bump inside a bigger one
            if not regular and L[max(0, s1 - CONTEXT_BARS):s2 + 1].min() < vh:
                continue
            if regular:
                if not (vh > v1 * (1 + HS_HEAD_MIN) and vh > v2 * (1 + HS_HEAD_MIN)):
                    continue
            else:
                if not (vh < v1 * (1 - HS_HEAD_MIN) and vh < v2 * (1 - HS_HEAD_MIN)):
                    continue
            if abs(v1 - v2) / max(v1, v2) > HS_SHOULDER_TOL:
                continue
            # Conservative horizontal neckline: the lower of the two dips
            # (or the higher of the two bumps for the inverse). Sloped
            # necklines extrapolated far past the pattern produced
            # impossible levels in practice.
            if regular:
                neck = min(L[s1:head + 1].min(), L[head:s2 + 1].min())
                if neck > min(v1, v2) * (1 - DT_MIN_DEPTH):
                    continue  # dips too shallow to be a real pattern
                broke, voided, live = _resolve(C, s2 + 1, s2 + 1 + (s2 - s1), lambda c: c < neck, lambda c: c > v2)
            else:
                neck = max(H[s1:head + 1].max(), H[head:s2 + 1].max())
                if neck < max(v1, v2) * (1 + DT_MIN_DEPTH):
                    continue
                broke, voided, live = _resolve(C, s2 + 1, s2 + 1 + (s2 - s1), lambda c: c > neck, lambda c: c < v2)
            if voided:
                continue
            name = "Head and shoulders" if regular else "Inverse head and shoulders"
            direction = "bearish" if regular else "bullish"
            if best is not None and best.status == "confirmed" and broke is None:
                continue
            if broke is not None and _recent(broke, n):
                detail = (f"Shoulders near {v1:.2f} and {v2:.2f}, head at {vh:.2f} ({_fmt_date(df.index[head])}), "
                          f"then a close {'below' if regular else 'above'} the neckline at {neck:.2f}.")
                best = Signal(name, direction, "confirmed", _fmt_date(df.index[broke]), detail, float(neck), (s1, head, s2))
            elif broke is None and live and _recent(s2, n, RECENT_BARS * 2):
                detail = (f"Shoulders near {v1:.2f} and {v2:.2f}, head at {vh:.2f}; not confirmed until a close "
                          f"{'below' if regular else 'above'} {neck:.2f}. A close {'above' if regular else 'below'} "
                          f"the right shoulder ({v2:.2f}) would cancel it.")
                best = Signal(name, direction, "forming", _fmt_date(df.index[s2]), detail, float(neck), (s1, head, s2))
        return best

    reg = scan(highs, True)
    inv = scan(lows, False)
    if reg:
        out.append(reg)
    if inv:
        out.append(inv)
    return out


def detect_gap_retests(df: pd.DataFrame) -> list[Signal]:
    """Bull retest gap: up-gap, later pullback into the gap that closes back
    above it. Bear retest gap: down-gap, later rally into the gap that closes
    back below it. The first touch decides it; a later close back inside the
    gap cancels the signal. Reports the most recent live retest of each kind."""
    out: list[Signal] = []
    H, L, C = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    start = max(1, n - GAP_LOOKBACK)
    bull = bear = None          # (bar_index, Signal)
    for g in range(start, n):
        if L[g] > H[g - 1] * (1 + GAP_MIN_PCT):          # up-gap: zone is [H[g-1], L[g]]
            bottom, top = H[g - 1], L[g]
            if any(C[t] < top for t in range(g + 1, min(g + GAP_MIN_WAIT, n))):
                continue  # filled right away, before a retest could even count
            for r in range(g + GAP_MIN_WAIT, n):
                if L[r] <= top:                           # price came back into the gap
                    held = C[r] >= top and L[r] >= bottom * 0.995
                    if held and not any(C[t] < top for t in range(r + 1, n)):
                        bull = (r, Signal("Bull retest gap", "bullish", "confirmed", _fmt_date(df.index[r]),
                                          f"Gapped up on {_fmt_date(df.index[g])} (gap {bottom:.2f} to {top:.2f}); on "
                                          f"{_fmt_date(df.index[r])} the price dipped into the gap and closed back above it. "
                                          f"Would be cancelled by a close back below {top:.2f}.",
                                          float(top)))
                    break                                  # first touch decides it (held or filled)
        elif H[g] < L[g - 1] * (1 - GAP_MIN_PCT):        # down-gap: zone is [H[g], L[g-1]]
            bottom, top = H[g], L[g - 1]
            if any(C[t] > bottom for t in range(g + 1, min(g + GAP_MIN_WAIT, n))):
                continue
            for r in range(g + GAP_MIN_WAIT, n):
                if H[r] >= bottom:
                    held = C[r] <= bottom and H[r] <= top * 1.005
                    if held and not any(C[t] > bottom for t in range(r + 1, n)):
                        bear = (r, Signal("Bear retest gap", "bearish", "confirmed", _fmt_date(df.index[r]),
                                          f"Gapped down on {_fmt_date(df.index[g])} (gap {bottom:.2f} to {top:.2f}); on "
                                          f"{_fmt_date(df.index[r])} the price rose into the gap and closed back below it. "
                                          f"Would be cancelled by a close back above {bottom:.2f}.",
                                          float(bottom)))
                    break
    for item in (bull, bear):
        if item is not None and _recent(item[0], n):
            out.append(item[1])
    return out


def detect_ema_candles(df: pd.DataFrame) -> list[Signal]:
    """The two 'first candle' signals, for the latest bar only."""
    out: list[Signal] = []
    if len(df) < EMA_LEN + EMA_PRIOR_RUN + 1:
        return out
    e = ema(df["Close"], EMA_LEN).values
    O, H, L, C = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    i = len(df) - 1
    body_ok = abs(C[i] - O[i]) / C[i] >= EMA_MIN_BODY_PCT
    prior_green = all(C[i - k] > O[i - k] for k in range(1, EMA_PRIOR_RUN + 1))
    prior_red = all(C[i - k] < O[i - k] for k in range(1, EMA_PRIOR_RUN + 1))
    if body_ok and C[i] < O[i] and prior_green and L[i] > e[i]:
        out.append(Signal("First red candle above 21-day EMA", "watch", "latest", _fmt_date(df.index[i]),
                          f"Closed down ({O[i]:.2f} to {C[i]:.2f}) after {EMA_PRIOR_RUN} green candles, with the whole "
                          f"candle (low {L[i]:.2f}) above the 21-day EMA at {e[i]:.2f}. First pause in an uptrend: "
                          f"some buy the dip, some see early weakness.",
                          float(e[i])))
    if body_ok and C[i] > O[i] and prior_red and H[i] < e[i]:
        out.append(Signal("First green candle below 21-day EMA", "watch", "latest", _fmt_date(df.index[i]),
                          f"Closed up ({O[i]:.2f} to {C[i]:.2f}) after {EMA_PRIOR_RUN} red candles, with the whole "
                          f"candle (high {H[i]:.2f}) below the 21-day EMA at {e[i]:.2f}. First up day after two down days while the price sits under its 21-day average.",
                          float(e[i])))
    return out


def _dedupe(signals: list[Signal], d: pd.DataFrame) -> list[Signal]:
    """Two clean-ups for overlapping shapes: a forming double top and a
    forming double bottom on the same chart just describe a sideways range,
    so they become one 'watch' signal; and a head-and-shoulders' two dips
    also look like a double bottom, so that double bottom is dropped."""
    by = {s.name: s for s in signals}
    top, bot = by.get("Double top"), by.get("Double bottom")
    if top and bot and top.status == "forming" and bot.status == "forming":
        dt_neck, db_neck = top.key_level, bot.key_level  # dip of the double top (lower), bump of the double bottom (higher)
        H, L = d["High"].values, d["Low"].values
        rng_hi = max(H[b] for b in top.bars)
        rng_lo = min(L[b] for b in bot.bars)
        signals = [s for s in signals if s.name not in ("Double top", "Double bottom")]
        signals.append(Signal("Sideways range", "watch", "forming", max(top.date, bot.date),
                              f"Both a double top (near {rng_hi:.2f}) and a double bottom (near {rng_lo:.2f}) are forming, "
                              f"so the price is moving sideways between about {rng_lo:.2f} and {rng_hi:.2f}. "
                              f"A close below {dt_neck:.2f} (confirming the double top) or above {db_neck:.2f} (confirming the double bottom) would break it.", None, tuple(sorted(top.bars + bot.bars))))
    for hs_name, dup_name in (("Head and shoulders", "Double bottom"), ("Inverse head and shoulders", "Double top")):
        hs, dup = by.get(hs_name), by.get(dup_name)
        if hs and dup and dup.bars and hs.bars and hs.bars[0] <= min(dup.bars) and max(dup.bars) <= hs.bars[-1]:
            signals = [s for s in signals if s is not dup]
    return signals


def analyze(df: pd.DataFrame) -> dict:
    """All signals for one symbol's daily history, plus context the table
    shows (last close, change, EMA position)."""
    d = _clean(df)
    if len(d) < EMA_LEN + 2 * PIVOT_WINDOW + 2:
        return {"signals": [], "last_close": None, "change_pct": None, "ema21": None, "above_ema21": None, "bars": len(d), "as_of": None,
                "error": f"not enough price history to check ({len(d)} day{'s' if len(d) != 1 else ''})"}
    highs, lows = swing_points(d["High"].values, d["Low"].values)
    signals: list[Signal] = []
    signals += detect_double(d, highs, lows)
    signals += detect_head_shoulders(d, highs, lows)
    signals += detect_gap_retests(d)
    signals += detect_ema_candles(d)
    signals = _dedupe(signals, d)
    e21 = float(ema(d["Close"], EMA_LEN).iloc[-1])
    last, prev = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])
    order = {"confirmed": 0, "latest": 1, "forming": 2}
    for sg in signals:
        sg.bars = tuple(int(b) for b in sg.bars)
    signals.sort(key=lambda s: (order.get(s.status, 3), s.date))
    return {
        "signals": [s.to_dict() for s in signals],
        "last_close": last,
        "change_pct": (last - prev) / prev if prev else None,
        "ema21": e21,
        "above_ema21": last > e21,
        "bars": len(d),
        "as_of": _fmt_date(d.index[-1]),
        "error": None,
    }
