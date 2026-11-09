"""Per-stock signal detection for the Sectors tab.

Every signal at bar t uses only bars 0..t (closes included): a swing high or
low found at bar i is treated as unknown until bar i + PIVOT_K, so chart
patterns fire only once all their pivots are confirmed. The event study
enters on the NEXT bar's open, so nothing here may peek past bar t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market.fetchers.indicators import atr, ema, sma, swing_points

PATTERN_LOOKBACK = 60  # a bottom/top pattern must be the extreme of this many sessions before it
PRIOR_MOVE = 0.10      # and price must have moved at least this much into it
DOUBLE_TOL = 0.01      # double top/bottom: the two peaks (troughs) within 1% of each other
DOUBLE_GAP = (10, 120) # sessions between the two peaks (troughs)
PIVOT_K = 5            # bars each side that confirm a swing high/low
PATTERN_MAX_WAIT = 30  # bars after the last pivot is confirmed to wait for the breakout
EMA_NEAR = 0.01        # EMA candle signals: the wick must reach within 1% of the 21 EMA
GAP_MIN = 0.01         # gap between two same-colour candles: at least 1% of price
BREAKOUT_COOLDOWN = 10  # ignore repeat 52w breakouts within this many bars
TEMPLATE_RESET = 20     # trend template must be off this many bars before a new entry counts
ENTRY_RESET = 21        # momentum / near-high / reversal: condition off this many bars before a new entry counts
MOMENTUM_TOP = 0.80     # 12-1 month momentum rank that counts as "strong"
NEAR_HIGH = 0.95        # close within 5% of the 52-week high
REVERSAL_BOTTOM = 0.10  # 1-month return rank that counts as a "loser"
SURPRISE_MIN = 0.02     # EPS surprise of at least 2% either way


@dataclass(frozen=True)
class Signal:
    key: str
    label: str
    direction: int  # +1 bullish, -1 bearish
    group: str      # "research": published evidence of an edge; "yours": your chart-pattern/candle signals, unproven
    definition: str
    short: str      # compact badge label
    plain: str      # one-line plain-English summary shown above the exact rule


SIGNALS = [
    Signal("trend_template", "Trend template entry", +1, "research",
           f"First day all Minervini trend-template conditions hold after at least {TEMPLATE_RESET} "
           "sessions without them: close > 50 > 150 > 200-day SMA, "
           "200-day SMA higher than 21 sessions ago, close at least 30% above its 52-week low and within "
           "25% of its 52-week high, and 12-month return in the top 30% of this ~110-stock universe.", 'Trend entry',
           'The stock just moved into a healthy long-term uptrend and sits near its 1-year high.'),
    Signal("breakout_52w", "52-week-high breakout", +1, "research",
           "Close above the highest high of the prior 252 sessions on at least 2x the prior 50-day average volume. "
           f"Repeats within {BREAKOUT_COOLDOWN} sessions are ignored.", '52w breakout',
           'The stock closed at a new 1-year high on unusually heavy trading.'),
    Signal("momentum_12_1", "Strong 12-month momentum", +1, "research",
           "12-month return excluding the latest month (close 21 sessions ago vs 252 sessions ago) ranks in the "
           f"top {1 - MOMENTUM_TOP:.0%} of this universe, after at least {ENTRY_RESET} sessions outside it. "
           "Jegadeesh & Titman (1993) and many replications.", 'Momentum',
           "Over the past year, ignoring the latest month, the stock just became one of the strongest fifth here."),
    Signal("near_52w_high", "Near 52-week high", +1, "research",
           f"Close within {1 - NEAR_HIGH:.0%} of the 52-week high, after at least {ENTRY_RESET} sessions further "
           "away than that. George & Hwang (2004).", 'Near high',
           "The stock just climbed back to within 5% of its 1-year high."),
    Signal("reversal_1m", "1-month loser bounce", +1, "research",
           f"21-session return ranks in the bottom {REVERSAL_BOTTOM:.0%} of this universe, after at least "
           f"{ENTRY_RESET} sessions outside it. Short-term reversal (Jegadeesh 1990, Lehmann 1990).", '1-mo bounce',
           "The stock was one of the worst tenth here over the past month; such stocks have tended to bounce."),
    Signal("earnings_beat", "Earnings beat drift", +1, "research",
           f"Reported EPS beat the estimate by at least {SURPRISE_MIN:.0%} and the stock closed up on its first "
           "session after the news. Fires on that session. Post-earnings-announcement drift (Bernard & Thomas 1989).",
           'Earnings beat', "Earnings beat forecasts and the stock rose on the news; prices have tended to keep drifting up."),
    Signal("earnings_miss", "Earnings miss drift", -1, "research",
           f"Mirror of earnings beat: EPS missed by at least {SURPRISE_MIN:.0%} and the stock closed down on its "
           "first session after the news.", 'Earnings miss',
           "Earnings missed forecasts and the stock fell on the news; prices have tended to keep drifting down."),
    Signal("double_bottom", "Double bottom", +1, "yours",
           f"Two confirmed swing lows {DOUBLE_GAP[0]}-{DOUBLE_GAP[1]} sessions apart with no lower price between them, "
           f"within {DOUBLE_TOL:.0%} of each other, with a peak at least 5% above them in between. Both lows must be the "
           f"lowest prices of the {PATTERN_LOOKBACK} sessions before the first one, after a fall of at least "
           f"{PRIOR_MOVE:.0%} into it. "
           f"Fires on the first close above that peak, within {PATTERN_MAX_WAIT} "
           f"sessions after the second low is confirmed ({PIVOT_K} sessions after it). Skipped if price closes above "
           "the peak before then (the breakout happened before the pattern was knowable) or makes a new low first.", 'Dbl bottom',
           'Price fell to about the same low twice, then broke above the bounce in between.'),
    Signal("inverse_hs", "Inverse head & shoulders", +1, "yours",
           "Three back-to-back confirmed swing lows, each 5-60 sessions after the previous one, where the "
           "middle (head) is at least 3% below both shoulders and the shoulders are within 5% of each other. "
           f"Fires on the first close above the neckline drawn through the two peaks between them, within "
           f"{PATTERN_MAX_WAIT} sessions after the right shoulder is confirmed ({PIVOT_K} sessions after it). "
           "Skipped if that close came before confirmation or price undercuts the head first.", 'Inv H&S',
           'Three dips with the middle one deepest, then price broke above the line joining the bounces.'),
    Signal("bullish_engulfing", "Bullish engulfing", +1, "yours",
           "A green candle whose body (open to close) covers the whole red candle before it, wicks "
           "included: it opens at or below that candle's low and closes at or above its high. If its body "
           "also covers more red candles in a row before that, all of them count toward how many it engulfed.", 'Bull engulf',
           'An up day whose body swallowed the whole down day (or days) before it, wicks and all.'),
    Signal("first_red_above_ema21", "First red candle above 21 EMA", +1, "yours",
           "After a close crosses above the 21-day EMA, the first red candle (close < open) that still "
           f"closes above it, if its low comes within {EMA_NEAR:.0%} of the EMA. If that first red candle "
           "stays farther away, there's no signal for that run. Only one per run: the next one needs a close back below the EMA and a new "
           "cross above.", '1st red > EMA21',
           'Price crossed above its 21-day average and pulled back to touch near it on a down day, still above it.'),
    Signal("double_top", "Double top", -1, "yours",
           f"Mirror of double bottom: two confirmed swing highs {DOUBLE_GAP[0]}-{DOUBLE_GAP[1]} sessions apart, no higher price between them, within {DOUBLE_TOL:.0%} "
           "of each other, a trough at least 5% below them, fires on the first close below that trough "
           "(same confirmation, wait and cancel rules).", 'Dbl top',
           'Price hit about the same high twice, then broke below the dip in between.'),
    Signal("head_shoulders", "Head & shoulders", -1, "yours",
           "Mirror of inverse head & shoulders: three back-to-back swing highs 5-60 sessions apart, head at "
           "least 3% above both shoulders, fires on the first close below the neckline through the two "
           "troughs (same confirmation, wait and cancel rules).", 'H&S',
           'Three peaks with the middle one highest, then price broke below the line joining the dips.'),
    Signal("bearish_engulfing", "Bearish engulfing", -1, "yours",
           "A red candle whose body (open to close) covers the whole green candle before it, wicks "
           "included: it opens at or above that candle's high and closes at or below its low. If its body "
           "also covers more green candles in a row before that, all of them count toward how many it engulfed.", 'Bear engulf',
           'A down day whose body swallowed the whole up day (or days) before it, wicks and all.'),
    Signal("green_below_ema21", "Green candle below 21 EMA", -1, "yours",
           "After a close crosses below the 21-day EMA, the first green candle (close > open) that still "
           f"closes below it, if its high comes within {EMA_NEAR:.0%} of the EMA. If that first green candle "
           "stays farther away, there's no signal for that run. Only one per run: the next one needs a close back above the EMA and a new "
           "cross below.", 'Green < EMA21',
           'Price crossed below its 21-day average and bounced up near it on an up day, still below it.'),
]


def _is_reversal_extreme(highs, lows, first, last, extreme, kind) -> bool:
    """A reversal pattern needs something to reverse: its extreme is the most
    extreme price from PATTERN_LOOKBACK sessions before its first point through
    its last point, and price moved at least PRIOR_MOVE into it."""
    lo = max(0, first - PATTERN_LOOKBACK)
    if first - lo < 10:
        return False
    if kind == "bottom":
        return lows[lo:last + 1].min() >= extreme * 0.995 and highs[lo:first].max() >= extreme * (1 + PRIOR_MOVE)
    return highs[lo:last + 1].max() <= extreme * 1.005 and lows[lo:first].min() <= extreme / (1 + PRIOR_MOVE)


def _double(highs, lows, closes, swings, kind, details=None):
    """Double top (kind='top', swings are swing-high indices) or double
    bottom (kind='bottom', swing lows). Returns fire-bar indices; fills
    details[t] with the points to draw when given a dict."""
    fires = []
    used = set()
    for bi, b in enumerate(swings):
        # pair b with the nearest earlier swing that matches it, as long as no
        # price between them pokes above the two tops (below the two bottoms);
        # smaller swings in between are fine, so long patterns still count
        a = None
        for a_ in reversed(swings[:bi]):
            gap = b - a_
            if gap < DOUBLE_GAP[0]:
                continue
            if gap > DOUBLE_GAP[1] or a_ in used:
                break
            p1_, p2_ = (highs[a_], highs[b]) if kind == "top" else (lows[a_], lows[b])
            if kind == "top":
                between = highs[a_ + 1:b].max()
                if between > p2_ * (1 + DOUBLE_TOL):
                    break  # a clearly higher top sits between: no older swing can pair either
                inside = between <= max(p1_, p2_)
            else:
                between = lows[a_ + 1:b].min()
                if between < p2_ * (1 - DOUBLE_TOL):
                    break
                inside = between >= min(p1_, p2_)
            if inside and abs(p1_ - p2_) / max(p1_, p2_) <= DOUBLE_TOL:
                a = a_
                break
        if a is None:
            continue
        if kind == "top":
            p1, p2 = highs[a], highs[b]
            level = lows[a + 1:b].min()
            if (min(p1, p2) - level) / min(p1, p2) < 0.05:
                continue
            extreme = max(p1, p2)
        else:
            p1, p2 = lows[a], lows[b]
            level = highs[a + 1:b].max()
            if (level - max(p1, p2)) / max(p1, p2) < 0.05:
                continue
            extreme = min(p1, p2)
        if not _is_reversal_extreme(highs, lows, a, b, extreme, kind):
            continue
        start = b + PIVOT_K
        for t in range(b + 1, min(start + PATTERN_MAX_WAIT, len(closes))):
            if kind == "top" and highs[t] > extreme or kind == "bottom" and lows[t] < extreme:
                break  # pattern failed before it triggered
            broke = kind == "top" and closes[t] < level or kind == "bottom" and closes[t] > level
            if broke and t < start:
                break  # broke out before the second pivot was confirmed: not tradable as this pattern
            if broke:
                fires.append(t)
                used.update((a, b))
                if details is not None:
                    details[t] = {"points": [[a, float(p1)], [b, float(p2)]], "line": [[a, float(level)], [t, float(level)]]}
                break
    return fires


def _head_shoulders(highs, lows, closes, swings, kind, details=None):
    """Head & shoulders top (kind='top', swing highs) or inverse
    (kind='bottom', swing lows). Neckline runs through the two opposite
    extremes between the shoulders and the head."""
    fires = []
    for l, h, r in zip(swings, swings[1:], swings[2:]):
        if not (5 <= h - l <= 60 and 5 <= r - h <= 60):
            continue
        if kind == "top":
            pl, ph, pr = highs[l], highs[h], highs[r]
            if ph < 1.03 * max(pl, pr) or abs(pl - pr) / max(pl, pr) > 0.05:
                continue
            n1 = l + 1 + int(np.argmin(lows[l + 1:h]))
            n2 = h + 1 + int(np.argmin(lows[h + 1:r]))
            v1, v2 = lows[n1], lows[n2]
        else:
            pl, ph, pr = lows[l], lows[h], lows[r]
            if ph > 0.97 * min(pl, pr) or abs(pl - pr) / min(pl, pr) > 0.05:
                continue
            n1 = l + 1 + int(np.argmax(highs[l + 1:h]))
            n2 = h + 1 + int(np.argmax(highs[h + 1:r]))
            v1, v2 = highs[n1], highs[n2]
        if not _is_reversal_extreme(highs, lows, l, r, ph, kind):
            continue
        slope = (v2 - v1) / (n2 - n1)
        start = r + PIVOT_K
        for t in range(r + 1, min(start + PATTERN_MAX_WAIT, len(closes))):
            if kind == "top" and highs[t] > ph or kind == "bottom" and lows[t] < ph:
                break
            neck = v1 + slope * (t - n1)
            broke = kind == "top" and closes[t] < neck or kind == "bottom" and closes[t] > neck
            if broke and t < start:
                break  # broke the neckline before the right shoulder was confirmed
            if broke:
                fires.append(t)
                if details is not None:
                    details[t] = {"points": [[l, float(pl)], [h, float(ph)], [r, float(pr)]],
                                  "line": [[n1, float(v1)], [t, float(neck)]]}
                break
    return fires


def find_gaps(opens, highs, lows, closes) -> list[dict]:
    """Price gaps worth keeping an eye on (not trade signals). A gap up: two
    green candles in a row with the second day's low at least GAP_MIN above
    the first day's high, so no trading happened in between. A gap down is
    the mirror with two red candles. A gap counts as filled once price trades
    back to the far edge (the first day's high for a gap up, its low for a
    gap down). Each gap: i (second day), dir, top, bottom, filled (bar index
    or None) and rem (the part still untouched)."""
    out = []
    n = len(closes)
    for i in range(1, n):
        a_green, b_green = closes[i - 1] > opens[i - 1], closes[i] > opens[i]
        a_red, b_red = closes[i - 1] < opens[i - 1], closes[i] < opens[i]
        if a_green and b_green and lows[i] - highs[i - 1] >= GAP_MIN * closes[i - 1]:
            d, top, bottom = 1, lows[i], highs[i - 1]
        elif a_red and b_red and lows[i - 1] - highs[i] >= GAP_MIN * closes[i - 1]:
            d, top, bottom = -1, lows[i - 1], highs[i]
        else:
            continue
        filled, rem = None, [bottom, top]
        for j in range(i + 1, n):
            if d > 0:
                if lows[j] <= bottom:
                    filled = j
                    break
                rem[1] = min(rem[1], lows[j])
            else:
                if highs[j] >= top:
                    filled = j
                    break
                rem[0] = max(rem[0], highs[j])
        out.append({"i": i, "dir": d, "top": float(top), "bottom": float(bottom), "filled": filled,
                    "rem": None if filled is not None else [float(rem[0]), float(rem[1])]})
    return out


def _fresh_entries(cond: np.ndarray, ready: np.ndarray, reset: int) -> list[int]:
    """Bars where cond turns true after being false (with its inputs ready)
    on each of the prior `reset` bars."""
    return [t for t in range(reset, len(cond))
            if cond[t] and not cond[t - reset:t].any() and ready[t - reset:t].all()]


def _earnings_events(df: pd.DataFrame, earnings, closes) -> tuple[list[int], list[int]]:
    """earnings: [{"ts": announcement time (ET, ISO), "surprise": EPS surprise
    as a fraction}]. The reaction session is the announcement day if it came
    before the close, else the next session; announcements with no time of
    day are skipped because their reaction session is unknown."""
    beats, misses = [], []
    dates = df.index.normalize()
    for row in earnings or []:
        ts = pd.Timestamp(row["ts"])
        surprise = row.get("surprise")
        if surprise is None or not np.isfinite(surprise) or (ts.hour, ts.minute) == (0, 0):
            continue
        day = pd.Timestamp(ts.date())
        t = int(dates.searchsorted(day, side="right" if ts.hour >= 16 else "left"))
        if t < 1 or t >= len(closes):
            continue
        move = closes[t] / closes[t - 1] - 1
        if surprise >= SURPRISE_MIN and move > 0:
            beats.append(t)
        elif surprise <= -SURPRISE_MIN and move < 0:
            misses.append(t)
    return beats, misses


def engulfed_count(o, h, l, c, t: int, direction: int, limit: int = 10) -> int:
    """How many candles in a row right before t, all of the opposite colour,
    candle t's body (open to close) covers completely, wicks included.
    0 = not an engulfing candle. direction +1: t must be green and engulf
    red candles."""
    if (c[t] - o[t]) * direction <= 0:
        return 0
    top, bottom = max(o[t], c[t]), min(o[t], c[t])
    k = 0
    for j in range(t - 1, max(-1, t - 1 - limit), -1):
        if (c[j] - o[j]) * direction >= 0 or h[j] > top or l[j] < bottom:
            break
        k += 1
    return k


def _engulfing(o, h, l, c, direction: int) -> tuple[list[int], dict]:
    fires, counts = [], {}
    for t in range(1, len(c)):
        k = engulfed_count(o, h, l, c, t, direction)
        if k:
            fires.append(t)
            counts[t] = k
    return fires, counts


def _first_against_ema(o: np.ndarray, c: np.ndarray, e: np.ndarray,
                       l: np.ndarray | None = None, h: np.ndarray | None = None,
                       near: float = EMA_NEAR) -> tuple[list[int], list[int]]:
    """One firing per run on each side of the EMA. A run starts when the close
    crosses the EMA (from a real EMA value on the prior bar); the first red
    candle of an above-run that still closes above the EMA, with its low
    within `near` of the EMA, fires the rise signal. Mirror for the first
    green candle of a below-run (its high within `near`). Only that first
    candle counts: if its wick is farther away, the run ends with no signal.
    Later candles in the same run never fire. Without l/h there's no
    distance check."""
    reds, greens = [], []
    armed_up = armed_down = False
    for t in range(1, len(c)):
        if np.isnan(e[t]) or np.isnan(e[t - 1]):
            armed_up = armed_down = False
            continue
        above, was_above = c[t] > e[t], c[t - 1] > e[t - 1]
        below, was_below = c[t] < e[t], c[t - 1] < e[t - 1]
        if above and not was_above:
            armed_up = True
        if not above:
            armed_up = False
        if below and not was_below:
            armed_down = True
        if not below:
            armed_down = False
        # only the first red (green) candle of a run counts: if its wick is
        # too far from the EMA, the run is spent without a signal
        if armed_up and c[t] < o[t]:
            if l is None or l[t] <= e[t] * (1 + near):
                reds.append(t)
            armed_up = False
        if armed_down and c[t] > o[t]:
            if h is None or h[t] >= e[t] * (1 - near):
                greens.append(t)
            armed_down = False
    return reds, greens


def compute(df: pd.DataFrame, rs_pct: pd.Series, mom_pct: pd.Series | None = None,
            rev_pct: pd.Series | None = None, earnings: list | None = None,
            ) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """df: one symbol's Open/High/Low/Close/Volume, no NaN rows. rs_pct: that
    symbol's cross-sectional 12-month-return percentile (0-1) per date;
    mom_pct: same for 12-1 month momentum; rev_pct: same for the 21-session
    return; earnings: see _earnings_events. Ranks must use data up to each
    date only. Missing inputs leave those signals empty.

    Returns (features frame, {signal_key: [bar indices where it fired]})."""
    o, h, l, c, v = (df[col] for col in ("Open", "High", "Low", "Close", "Volume"))
    feat = pd.DataFrame(index=df.index)
    feat["sma50"], feat["sma150"], feat["sma200"] = sma(c, 50), sma(c, 150), sma(c, 200)
    feat["ema21"] = ema(c, 21)
    feat["atr14"] = atr(h, l, c, 14)
    feat["high_52w"] = h.rolling(252, min_periods=252).max()
    feat["low_52w"] = l.rolling(252, min_periods=252).min()
    feat["rs_pct"] = rs_pct.reindex(df.index)

    template = (
        (c > feat["sma50"]) & (feat["sma50"] > feat["sma150"]) & (feat["sma150"] > feat["sma200"])
        & (feat["sma200"] > feat["sma200"].shift(21))
        & (c >= 1.3 * feat["low_52w"]) & (c >= 0.75 * feat["high_52w"])
        & (feat["rs_pct"] >= 0.70)
    )
    feat["trend_template"] = template

    oa, ha, la, ca = (s.to_numpy(dtype=float) for s in (o, h, l, c))
    events: dict[str, list[int]] = {s.key: [] for s in SIGNALS}

    # A fresh entry: template true today after being false (with all inputs
    # real numbers, so the end of warm-up doesn't look like an entry) for
    # each of the prior TEMPLATE_RESET sessions. Without that, a stock
    # flickering across one condition registers as many separate entries.
    inputs_ready = (
        feat[["sma50", "sma150", "sma200", "high_52w", "low_52w", "rs_pct"]].notna().all(axis=1)
        & feat["sma200"].shift(21).notna()
    ).to_numpy()
    tpl = template.to_numpy()
    events["trend_template"] = [
        t for t in range(TEMPLATE_RESET, len(tpl))
        if tpl[t] and not tpl[t - TEMPLATE_RESET:t].any() and inputs_ready[t - TEMPLATE_RESET:t].all()
    ]

    prior_high = h.shift(1).rolling(252, min_periods=252).max().to_numpy()
    prior_vol = v.shift(1).rolling(50, min_periods=50).mean().to_numpy()
    va = v.to_numpy(dtype=float)
    last_breakout = -10**9
    for t in range(len(ca)):
        if ca[t] > prior_high[t] and va[t] >= 2 * prior_vol[t] and t - last_breakout > BREAKOUT_COOLDOWN:
            events["breakout_52w"].append(t)
            last_breakout = t

    sh = swing_points(ha, PIVOT_K, "high")
    sl = swing_points(la, PIVOT_K, "low")
    patterns = {k: {} for k in ("double_top", "double_bottom", "head_shoulders", "inverse_hs")}
    events["double_top"] = _double(ha, la, ca, sh, "top", patterns["double_top"])
    events["double_bottom"] = _double(ha, la, ca, sl, "bottom", patterns["double_bottom"])
    events["head_shoulders"] = _head_shoulders(ha, la, ca, sh, "top", patterns["head_shoulders"])
    events["inverse_hs"] = _head_shoulders(ha, la, ca, sl, "bottom", patterns["inverse_hs"])
    feat.attrs["patterns"] = patterns
    feat.attrs["gaps"] = find_gaps(oa, ha, la, ca)
    events["bullish_engulfing"], bull_n = _engulfing(oa, ha, la, ca, +1)
    events["bearish_engulfing"], bear_n = _engulfing(oa, ha, la, ca, -1)
    feat.attrs["engulfed"] = {"bullish_engulfing": bull_n, "bearish_engulfing": bear_n}

    events["first_red_above_ema21"], events["green_below_ema21"] = _first_against_ema(
        oa, ca, feat["ema21"].to_numpy(), la, ha)

    near = (c >= NEAR_HIGH * feat["high_52w"]).to_numpy()
    events["near_52w_high"] = _fresh_entries(near, feat["high_52w"].notna().to_numpy(), ENTRY_RESET)
    for key, ranks, test in (("momentum_12_1", mom_pct, lambda r: r >= MOMENTUM_TOP),
                             ("reversal_1m", rev_pct, lambda r: r <= REVERSAL_BOTTOM)):
        if ranks is None:
            continue
        r = ranks.reindex(df.index)
        events[key] = _fresh_entries(test(r).to_numpy() & r.notna().to_numpy(), r.notna().to_numpy(), ENTRY_RESET)
    events["earnings_beat"], events["earnings_miss"] = _earnings_events(df, earnings, ca)

    for key in events:
        events[key] = sorted(set(events[key]))
    return feat, events
