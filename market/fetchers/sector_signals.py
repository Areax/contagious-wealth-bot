"""Sectors tab: SPDR sector ETF trend regime, today's signals on each ETF's
top 10 holdings, and a historical event study of every signal.

Design follows the evidence on indicator use (see README):
  - Regime: ensemble of three slow trend rules per ETF, evaluated on
    completed month-end closes only.
  - Signals: computed from completed daily bars only (an in-progress
    session bar is dropped), see fetchers/signals.py.
  - Event study: every past firing is entered at the NEXT session's open and
    measured against SPY over the same dates, net of a round-trip cost, and
    compared to the same stocks' return on any random day. That last
    comparison matters because the universe is TODAY's top holdings - stocks
    that got there by rising - so raw returns are biased upward.
"""
from __future__ import annotations

import json
import time
import warnings
from statistics import NormalDist
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from market.fetchers.cache_utils import is_fresh
from market.fetchers.indicators import atr, ema, sma
from market.fetchers.combos import run as run_combos
from market.fetchers.signals import SIGNALS, compute
from market.fetchers.paths import DATA_DIR

CACHE_PATH = DATA_DIR / "sector_signals.json"
# Price history for the click-to-open charts. Kept out of CACHE_PATH (which
# is embedded in the page) because it's ~1-2 MB; the page loads it on demand.
CHARTS_PATH = DATA_DIR / "sector_charts.json"
CHART_SESSIONS = 260  # about a year of daily bars per chart
HOLDINGS_CACHE_PATH = DATA_DIR / "sector_holdings.json"
# Committed snapshot, used per ETF only when both the live fetch and the
# local cache fail - CI starts with no cache, so without this one flaky
# funds_data call would blank the whole tab.
HOLDINGS_SEED_PATH = Path(__file__).resolve().parent / "sector_holdings_seed.json"
HOLDINGS_TTL = 7 * 24 * 3600
EARNINGS_CACHE_PATH = DATA_DIR / "sector_earnings.json"
EARNINGS_TTL = 7 * 24 * 3600
COMBOS_PATH = DATA_DIR / "sector_combos.json"

SECTOR_ETFS = {
    "XLB": "Materials", "XLC": "Communication Services", "XLE": "Energy",
    "XLF": "Financials", "XLI": "Industrials", "XLK": "Technology",
    "XLP": "Consumer Staples", "XLRE": "Real Estate", "XLU": "Utilities",
    "XLV": "Health Care", "XLY": "Consumer Discretionary",
}
# Second share class -> the one kept in the event study and RS ranking, so
# one company doesn't count twice. Both still show in the holdings table.
SHARE_CLASS_PRIMARY = {"GOOG": "GOOGL", "BRK-A": "BRK-B", "FOX": "FOXA", "NWS": "NWSA"}
BENCHMARK = "SPY"
HORIZONS = (5, 20, 60)
ROUND_TRIP_COST = 0.002
# Trade test: wait for price to break the signal candle's far wick (its high
# for a rise signal, its low for a drop signal) within TRADE_TRIGGER_DAYS
# sessions, and enter there (at the open if it gaps through). Stop at the
# opposite wick, target TRADE_REWARD times the distance to the stop. No trade
# if price reaches the stop level first or never breaks the wick. Exit at the
# close after TRADE_MAX_HOLD sessions if neither level is hit first.
# The stop sits at least TRADE_MIN_STOP_ATR x the 14-day average range away,
# so a tiny-range signal candle can't produce a stop inside normal noise.
TRADE_REWARD = 1.2
TRADE_MAX_HOLD = 20
TRADE_MIN_STOP_ATR = 0.5
TRADE_TRIGGER_DAYS = 3
# The two 21 EMA candle signals work as standing orders instead: the order
# waits below/above the candle until it fills or the opposite EMA signal
# prints (that cancels it), at most TRADE_MAX_HOLD sessions. Touching the
# stop level first does not cancel these.
EMA_OPPOSITE = {"first_red_above_ema21": "green_below_ema21", "green_below_ema21": "first_red_above_ema21"}
STUDY_YEARS = 5
RECENT_SESSIONS = 5
# Verdict rules. Each signal's result is compared with placebo runs of the
# same signal shifted to other dates (see _placebo), giving a p-value. 30
# tests run (10 signals x 3 horizons), so p of ~0.05 turns up by chance; the
# strong bar matches Harvey & Liu's t >= 3 for a new finding. Few firing
# dates, or firings bunched into a handful of overlapping windows, make any
# statistic unreliable, so those get no verdict at all.
STRONG_P = 0.0027  # two-sided, = |z| >= 3
HINT_P = 0.0455    # two-sided, = |z| >= 2
MIN_VERDICT_DATES = 30
MIN_INDEPENDENT_WINDOWS = 10
MIN_COVERAGE = 0.9  # skip the study if fewer than this share of stocks have a full year of data
MIN_COVERAGE_BARS = 260
# Placebo shifts closer than this many sessions (either direction) are
# skipped: a signal is built from its recent past, so a placebo landing on
# that same stretch isn't luck, it's the signal's own setup.
PLACEBO_EXCLUSION = {"trend_template": 260, "breakout_52w": 260, "momentum_12_1": 260, "near_52w_high": 260}
PLACEBO_EXCLUSION_DEFAULT = 60
WARMUP_SESSIONS = 260  # no signal can fire before ~a year of history
MIN_STUDY_SESSIONS = 252  # skip the study if under a year of usable window remains
ET = ZoneInfo("America/New_York")
# Approximates the NYSE calendar for "is there another session this month?";
# a misjudged day (e.g. Good Friday) only delays the regime by one session.
_NEXT_SESSION = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _fetch_holdings() -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Top 10 holdings per ETF and where each list came from: "cached" (fetched
    within the TTL), "live", or a fallback after a failed fetch - "stale"
    (older cached list) or "seed" (committed snapshot). Each ETF falls back
    on its own, so one failed call only affects that sector. The cache keeps
    a fetch time per ETF so an old list is never re-labelled as fresh."""
    cache = _read_json(HOLDINGS_CACHE_PATH)
    now = time.time()
    fresh = {e: c["rows"] for e, c in cache.items()
             if isinstance(c, dict) and c.get("rows") and now - c.get("fetched_at", 0) < HOLDINGS_TTL}
    seed = _read_json(HOLDINGS_SEED_PATH)
    out, source = {}, {}
    for etf in SECTOR_ETFS:
        if etf in fresh:  # only refetch lists whose cache expired
            out[etf], source[etf] = fresh[etf], "cached"
            continue
        rows = []
        try:
            top = yf.Ticker(etf).funds_data.top_holdings
            for sym, row in top.head(10).iterrows():
                weight = float(row["Holding Percent"])
                rows.append({"symbol": str(sym).replace(".", "-"), "name": str(row["Name"]),
                             "weight": None if np.isnan(weight) else weight})
        except Exception:
            rows = []
        old = cache.get(etf)
        old_rows = old.get("rows") if isinstance(old, dict) else None
        for label, candidate in (("live", rows), ("stale", old_rows), ("seed", seed.get(etf))):
            if candidate:
                out[etf], source[etf] = candidate, label
                break
        else:
            out[etf], source[etf] = [], "missing"
        if source[etf] == "live":
            cache[etf] = {"rows": rows, "fetched_at": now}
    if any(src == "live" for src in source.values()):
        HOLDINGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        HOLDINGS_CACHE_PATH.write_text(json.dumps(cache, indent=2))
    return out, source


def _fetch_earnings(stocks: list[str]) -> tuple[dict[str, list[dict]], int]:
    """Past EPS surprises per stock from Yahoo, cached per stock for a week.
    A failed fetch falls back to that stock's older cache, else no earnings
    signals for it. Returns ({symbol: rows}, number of stocks with data)."""
    cache = _read_json(EARNINGS_CACHE_PATH)
    now = time.time()
    out = {}
    for sym in stocks:
        entry = cache.get(sym)
        if not (entry and now - entry.get("fetched", 0) < EARNINGS_TTL):
            try:
                df = yf.Ticker(sym).get_earnings_dates(limit=40)
                rows = [] if df is None else [
                    {"ts": ts.isoformat(), "surprise": float(r["Surprise(%)"]) / 100}
                    for ts, r in df.iterrows() if pd.notna(r.get("Surprise(%)"))
                ]
                entry = cache[sym] = {"fetched": now, "rows": rows}
            except Exception:
                pass  # keep whatever (possibly stale) entry the cache had
        out[sym] = entry["rows"] if entry else []
    try:
        EARNINGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        EARNINGS_CACHE_PATH.write_text(json.dumps(cache))
    except OSError:
        pass
    return out, sum(1 for rows in out.values() if rows)


def _download_once(tickers: list[str]) -> dict[str, pd.DataFrame]:
    raw = yf.download(tickers, period="7y", interval="1d", auto_adjust=True,
                      group_by="ticker", threads=True, progress=False)
    now_et = datetime.now(ET)
    frames = {}
    for t in tickers:
        if raw is None or t not in raw.columns.get_level_values(0):
            continue
        df = raw[t][["Open", "High", "Low", "Close", "Volume"]].dropna()
        # drop corrupt bars (zero/negative prices, high below low) - one of
        # these would otherwise poison every average it touches
        ohlc = df[["Open", "High", "Low", "Close"]]
        df = df[(ohlc > 0).all(axis=1) & (df["High"] >= df["Low"]) & np.isfinite(ohlc).all(axis=1)]
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        # Drop today's bar while the session is still open - it isn't a
        # completed candle yet and every signal assumes completed bars.
        if len(df) and df.index[-1].date() == now_et.date() and (now_et.hour, now_et.minute) < (16, 15):
            df = df.iloc[:-1]
        if len(df) >= 2:  # short histories still show; their warm-up indicators stay blank
            frames[t] = df
    return frames


def _download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """yfinance drops individual tickers under rate limiting without raising,
    so retry whatever came back missing once, after a short pause."""
    frames = _download_once(tickers)
    missing = [t for t in tickers if t not in frames]
    if missing:
        time.sleep(5)
        frames.update(_download_once(missing))
    return frames


def _regime(close: pd.Series) -> dict:
    """Score 0-3 from three slow trend rules, all read at the last COMPLETED
    month-end: close above its 10-month SMA, close above its 200-day SMA,
    and a positive 12-month return. The in-progress month is ignored."""
    last = close.index[-1]
    month_done = (last + _NEXT_SESSION).month != last.month
    cutoff = last if month_done else last.replace(day=1) - pd.Timedelta(days=1)
    daily = close[close.index <= cutoff]
    monthly = daily.resample("ME").last().dropna()
    if len(monthly) < 13 or len(daily) < 200:
        return {"score": None}
    above_10m = bool(monthly.iloc[-1] > monthly.rolling(10).mean().iloc[-1])
    above_200d = bool(daily.iloc[-1] > sma(daily, 200).iloc[-1])
    ret_12m = float(monthly.iloc[-1] / monthly.iloc[-13] - 1)
    return {
        "score": int(above_10m) + int(above_200d) + int(ret_12m > 0),
        "above_10m": above_10m, "above_200d": above_200d,
        "ret_1m": float(monthly.iloc[-1] / monthly.iloc[-2] - 1), "ret_12m": ret_12m,
        "as_of": daily.index[-1].strftime("%Y-%m-%d"),
    }


def _forward_excess(df: pd.DataFrame, spy: pd.DataFrame, h: int) -> pd.Series:
    """For a signal on date d: buy at the next session's open, sell at the
    close h sessions after d, minus SPY over the identical dates. Computed on
    SPY's calendar so a session the stock is missing gives NaN instead of
    silently stretching its window. Indexed like df (signal date)."""
    on_cal = df.reindex(spy.index)
    stock = on_cal["Close"].shift(-h) / on_cal["Open"].shift(-1) - 1
    bench = spy["Close"].shift(-h) / spy["Open"].shift(-1) - 1
    return (stock - bench).reindex(df.index)


def _by_date(pos: np.ndarray, vals: np.ndarray, n_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Average values that share a trading day. Correlated stocks firing on
    the same day aren't independent bets, so each day counts once."""
    counts = np.bincount(pos, minlength=n_days)
    sums = np.bincount(pos, weights=vals, minlength=n_days)
    days = np.nonzero(counts)[0]
    return days, sums[days] / counts[days]


def _hac_t(days: np.ndarray, means: np.ndarray, n_days: int, lag: int) -> float | None:
    """t-stat of the mean of per-day values, with a Bartlett-kernel
    (Newey-West style) variance over trading-day distance: days within `lag`
    sessions share part of their holding window, so they're not independent.
    Autocovariances via FFT, so the placebo can afford to run every shift."""
    n = len(means)
    if n <= 2:
        return None
    resid = np.zeros(n_days)
    resid[days] = means - means.mean()
    size = 1 << int(np.ceil(np.log2(2 * n_days)))
    spec = np.fft.rfft(resid, size)
    autocov = np.fft.irfft(spec * np.conj(spec), size)[:min(lag, n_days - 1) + 1]
    k = np.arange(len(autocov))
    weights = 1 - k / (lag + 1)
    var = float(weights[0] * autocov[0] + 2 * (weights[1:] @ autocov[1:])) / n**2
    return float(means.mean() / np.sqrt(var)) if var > 0 else None


def _independent_windows(pos: np.ndarray, h: int) -> int:
    """How many firing days are at least h sessions apart (greedy), i.e. how
    many non-overlapping bets the sample really holds."""
    count, last = 0, None
    for p in np.unique(pos):
        if last is None or p - last >= h:
            count, last = count + 1, p
    return count


def _verdict(p: float | None, score: float | None, edge: float | None, n_dates: int, windows: int) -> str:
    """p: two-sided placebo p-value of the timing edge; score: the same as a
    signed z-score; edge: placebo-centred edge before costs."""
    if p is None or score is None or n_dates < MIN_VERDICT_DATES or windows < MIN_INDEPENDENT_WINDOWS:
        return "too_few"
    if p <= STRONG_P:
        if score < 0:
            return "reverses"
        return "works" if edge is not None and edge > ROUND_TRIP_COST else "too_small"
    if p <= HINT_P:
        return "hint_works" if score > 0 else "hint_backwards"
    return "none"


def _placebo(stock_idx, pos, abnormal, start, n_days, lag, exclusion):
    """Raw t-stats and mean edges of the signal re-run at every allowed
    shift: all firing days move together by s sessions, wrapping around the
    study window. Moving them together keeps the real bunching of firings
    (within a stock's hot and cold stretches, and across stocks), which the
    Newey-West correction alone underestimates; only the link to the actual
    dates is broken. Shifts within `exclusion` sessions either way are
    skipped (see PLACEBO_EXCLUSION)."""
    width = n_days - start
    shifts = [d for d in range(1, width) if min(d, width - d) >= exclusion]
    ts, edges = [], []
    for d in shifts:
        shifted = start + (pos - start + d) % width
        vals = abnormal[stock_idx, shifted]
        keep = np.isfinite(vals)
        if keep.sum() <= 2:
            continue
        days, means = _by_date(shifted[keep], vals[keep], n_days)
        t = _hac_t(days, means, n_days, lag)
        if t is not None:
            ts.append(t)
            edges.append(means.mean())
    return np.asarray(ts), np.asarray(edges)


def _signal_stats(stock_idx, pos, direction, excess, abnormal, start, n_days, h, exclusion) -> dict | None:
    """One signal at one horizon. excess[i, d] is stock i's forward return
    minus SPY's for a signal on calendar day d; abnormal additionally has the
    same-day average of all stocks and stock i's own average removed.
    net = excess in the signal's direction after costs, averaged per day.
    edge = abnormal in the signal's direction (before costs), centred on the
    placebo, so 0 = no better than the same signal at random other dates.
    p = share of placebo runs at least as extreme as the real one."""
    ex = excess[stock_idx, pos]
    keep = np.isfinite(ex)
    stock_idx, pos, ex = stock_idx[keep], pos[keep], ex[keep]
    if len(pos) == 0:
        return None
    net_vals = direction * ex - ROUND_TRIP_COST
    days, net_means = _by_date(pos, net_vals, n_days)
    abn = direction * abnormal[stock_idx, pos]
    ok = np.isfinite(abn)
    abn_days, abn_means = _by_date(pos[ok], abn[ok], n_days)
    raw_t = _hac_t(abn_days, abn_means, n_days, h - 1)
    placebo_t, placebo_edge = _placebo(stock_idx[ok], pos[ok], direction * abnormal, start, n_days, h - 1, exclusion)
    p = score = edge = None
    if raw_t is not None and len(placebo_t) >= 100:
        center = float(np.median(placebo_t))
        extreme = np.abs(placebo_t - center) >= abs(raw_t - center)
        p = float((extreme.sum() + 1) / (len(placebo_t) + 1))
        score = float(np.copysign(NormalDist().inv_cdf(1 - p / 2), raw_t - center))
        edge = float(abn_means.mean() - np.median(placebo_edge))
    windows = _independent_windows(pos, h)
    net = float(net_means.mean())
    return {
        "n": int(len(pos)), "n_dates": int(len(days)), "windows": windows, "net": net,
        "hit_rate": float((net_vals > 0).mean()), "edge": edge, "score": score, "p": p,
        "placebo_runs": int(len(placebo_t)), "verdict": _verdict(p, score, edge, len(days), windows),
    }


def _paths(stocks, frames, spy, event_pos, start, max_h) -> dict:
    """Average path after each signal, for the event-study charts: on day k
    after the signal, the stock's return from the next session's open to
    that day's close, minus SPY's over the same dates, in the signal's
    direction and before costs. Firings on the same day are averaged first,
    the same way "Net vs SPY" is, so the line at k = h equals that column
    plus the cost. "baseline" is the same for every stock on every day."""
    cal = spy.index
    n_days = len(cal)
    close = np.vstack([frames[s]["Close"].reindex(cal).to_numpy() for s in stocks])
    entry = np.vstack([frames[s]["Open"].reindex(cal).shift(-1).to_numpy() for s in stocks])
    spy_close = spy["Close"].to_numpy()
    spy_entry = spy["Open"].shift(-1).to_numpy()
    ks = np.arange(1, max_h + 1)

    def excess_after(stock_idx, pos):
        """[event, k] forward excess; NaN where the window runs past the data."""
        idx = pos[:, None] + ks[None, :]
        inside = idx < n_days
        idx = np.where(inside, idx, n_days - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ex = (close[stock_idx[:, None], idx] / entry[stock_idx, pos][:, None]
                  - spy_close[idx] / spy_entry[pos][:, None])
        ex[~inside] = np.nan
        return ex

    def date_averaged(pos, ex):
        out = [0.0]
        for j in range(ex.shape[1]):
            ok = np.isfinite(ex[:, j])
            if ok.sum() == 0:
                out.append(None)
                continue
            _, means = _by_date(pos[ok], ex[ok, j], n_days)
            out.append(round(float(means.mean()), 6))
        return out

    all_i, all_p = np.nonzero(np.isfinite(close[:, start:]))
    all_p = all_p + start
    rows = {}
    for key, (stock_idx, pos, direction) in event_pos.items():
        rows[key] = None if len(pos) == 0 else [
            None if v is None else direction * v for v in date_averaged(pos, excess_after(stock_idx, pos))]
    return {"days": [0, *ks.tolist()], "baseline": date_averaged(all_p, excess_after(all_i, all_p)), "rows": rows}


def _event_study(stocks, frames, spy, events_by_stock, study_start) -> dict | None:
    """Everything here lives on SPY's trading calendar: arrays are
    [stock, calendar day], and a day the stock is missing is NaN."""
    if not stocks:
        return None
    cal = spy.index
    n_days = len(cal)
    # Signals need ~a year of history (52-week high/low, 200-day average)
    # before they can fire, so the window can't start earlier than that.
    start = max(int(cal.searchsorted(study_start)), WARMUP_SESSIONS)
    if n_days - start < MIN_STUDY_SESSIONS:
        return None  # caller reports why
    event_pos = {}
    for i, sym in enumerate(stocks):
        on_cal = cal.get_indexer(frames[sym].index)
        for sig in SIGNALS:
            pts = on_cal[events_by_stock[sym][sig.key]] if events_by_stock[sym][sig.key] else np.array([], int)
            event_pos[(i, sig.key)] = pts[pts >= start]

    rows = {sig.key: {} for sig in SIGNALS}
    baseline = {}
    for h in HORIZONS:
        excess = np.full((len(stocks), n_days), np.nan)
        for i, sym in enumerate(stocks):
            excess[i] = _forward_excess(frames[sym], spy, h).reindex(cal).to_numpy()
        excess[:, :start] = np.nan
        excess[~np.isfinite(excess)] = np.nan
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            day_avg = np.nanmean(excess, axis=0)
            stock_avg = np.nanmean(excess - day_avg, axis=1)
        abnormal = excess - day_avg - stock_avg[:, None]
        valid = ~np.isnan(excess)
        bull_days, bull_means = _by_date(np.nonzero(valid)[1], excess[valid] - ROUND_TRIP_COST, n_days)
        bear_days, bear_means = _by_date(np.nonzero(valid)[1], -excess[valid] - ROUND_TRIP_COST, n_days)
        baseline[str(h)] = {
            "bull": {"net": float(bull_means.mean()), "n": int(valid.sum())},
            "bear": {"net": float(bear_means.mean()), "n": int(valid.sum())},
        }
        for sig in SIGNALS:
            idx = [(i, p) for i in range(len(stocks)) for p in event_pos[(i, sig.key)]]
            stock_idx = np.array([i for i, _ in idx], dtype=int)
            pos = np.array([p for _, p in idx], dtype=int)
            exclusion = max(h, PLACEBO_EXCLUSION.get(sig.key, PLACEBO_EXCLUSION_DEFAULT))
            rows[sig.key][str(h)] = _signal_stats(stock_idx, pos, sig.direction, excess, abnormal,
                                                  start, n_days, h, exclusion)

    path_events = {}
    for sig in SIGNALS:
        idx = [(i, p) for i in range(len(stocks)) for p in event_pos[(i, sig.key)]]
        path_events[sig.key] = (np.array([i for i, _ in idx], dtype=int),
                                np.array([p for _, p in idx], dtype=int), sig.direction)
    paths = _paths(stocks, frames, spy, path_events, start, max(HORIZONS))

    study_rows = [{"key": sig.key, "label": sig.label, "short": sig.short, "plain": sig.plain,
                   "direction": sig.direction, "group": sig.group, "definition": sig.definition,
                   "horizons": rows[sig.key]} for sig in SIGNALS]

    def found(*verdicts):
        return [{"label": r["label"], "horizons": [h for h in HORIZONS
                 if r["horizons"][str(h)] and r["horizons"][str(h)]["verdict"] in verdicts]}
                for r in study_rows
                if any(st and st["verdict"] in verdicts for st in r["horizons"].values())]

    return {
        "start": cal[start].strftime("%Y-%m-%d"), "end": cal[-1].strftime("%Y-%m-%d"),
        "years": round((cal[-1] - cal[start]).days / 365.25, 1),
        "cost": ROUND_TRIP_COST, "horizons": list(HORIZONS), "n_stocks": len(stocks),
        "strong_p": STRONG_P, "hint_p": HINT_P,
        "min_dates": MIN_VERDICT_DATES, "min_windows": MIN_INDEPENDENT_WINDOWS,
        "baseline": baseline, "rows": study_rows, "paths": paths,
        "summary": {"n_signals": len(SIGNALS), "works": found("works"), "reverses": found("reverses"),
                    "too_small": found("too_small"), "hint_works": found("hint_works"),
                    "hint_backwards": found("hint_backwards")},
    }


def _ohlc_atr(df: pd.DataFrame) -> list[np.ndarray]:
    return [df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close")] + \
           [atr(df["High"], df["Low"], df["Close"], 14).to_numpy(float)]


def _cancel_at(events: dict, key: str, t: int) -> int | None:
    """For an EMA candle signal on bar t: the bar of the next opposite EMA
    signal (which cancels its order), a large number if none yet. None for
    every other signal (they use the fixed entry window)."""
    if key not in EMA_OPPOSITE:
        return None
    opp = events.get(EMA_OPPOSITE[key], [])
    k = int(np.searchsorted(opp, t, side="right"))
    return int(opp[k]) if k < len(opp) else 10**9


def _simulate(o, h, l, c, a, t: int, direction: int, cancel_at: int | None = None, *,
              reward: float = TRADE_REWARD, max_hold: int = TRADE_MAX_HOLD,
              stop_buffer_atr: float = 0.0, cost: float = ROUND_TRIP_COST,
              max_gap_atr: float | None = None) -> dict | None:
    """One trade for a signal on bar t of a stock's own arrays. None while the
    entry window is still open with no entry. Entry waits for a break of the
    signal candle's far wick; "skipped" (with a "why") means no trade. When a
    bar touches two levels, the extreme nearer the open counts as first. A
    bar that opens beyond either level exits at that open. reward / max_hold / stop_buffer_atr (stop that many ATRs
    beyond the wick) / cost override the site rule."""
    n = len(c)
    if t + 1 >= n:
        return None
    if not a[t] > 0:
        return {"result": "skipped", "why": "no_range", "e": t + 1}
    trigger = h[t] if direction > 0 else l[t]   # break of this wick = entry
    wick = (l[t] - stop_buffer_atr * a[t]) if direction > 0 else (h[t] + stop_buffer_atr * a[t])  # opposite wick = stop
    e, entry = None, None
    standing = cancel_at is not None
    end = min(cancel_at, t + max_hold) if standing else t + TRADE_TRIGGER_DAYS
    # Daily bars hide the order of the high and the low. Assume the extreme
    # nearer the open came first (open -> low -> high -> close when the open
    # sits nearer the low, else open -> high -> low -> close).
    def adverse_first(j):
        return (o[j] - l[j] <= h[j] - o[j]) if direction > 0 else (h[j] - o[j] <= o[j] - l[j])

    gap_entry = False
    for j in range(t + 1, min(end + 1, n)):
        if not standing and (o[j] - wick) * direction <= 0:
            return {"result": "skipped", "why": "stop_first", "e": j}
        if (o[j] - trigger) * direction >= 0:
            if max_gap_atr is not None and (o[j] - trigger) * direction > max_gap_atr * a[t]:
                return {"result": "skipped", "why": "gapped", "e": j}   # opened too far past the entry: limit not filled
            e, entry, gap_entry = j, o[j], True   # gapped through the wick: fill at the open
            break
        far = h[j] if direction > 0 else l[j]
        near = l[j] if direction > 0 else h[j]
        hit_far, hit_near = (far - trigger) * direction >= 0, (near - wick) * direction <= 0
        if hit_far and not (not standing and hit_near and adverse_first(j)):
            e, entry = j, trigger
            break
        if not standing and hit_near:
            return {"result": "skipped", "why": "stop_first", "e": j}
    if e is None:
        if end >= n:
            return None                 # still waiting for the break
        if standing and cancel_at <= t + max_hold:
            return {"result": "skipped", "why": "cancelled", "e": t + 1}
        return {"result": "skipped", "why": "no_break", "e": t + 1}
    risk = max((entry - wick) * direction, TRADE_MIN_STOP_ATR * a[t])
    stop = entry - direction * risk
    target = entry + direction * reward * risk
    last = min(e + max_hold - 1, n - 1)
    result, exit_px, exit_i = None, None, last
    for j in range(e, last + 1):
        if j > e:
            if (o[j] - stop) * direction <= 0:
                result, exit_px = "stop", o[j]
            elif (o[j] - target) * direction >= 0:
                result, exit_px = "target", o[j]
        if result is None:
            adverse = l[j] if direction > 0 else h[j]
            favorable = h[j] if direction > 0 else l[j]
            hit_stop = (adverse - stop) * direction <= 0
            hit_target = (favorable - target) * direction >= 0
            if j == e and not gap_entry and adverse_first(j):
                # bought on the way from the low to the high: the low came
                # before the fill, so only the close leg can reach the stop
                hit_stop = (c[j] - stop) * direction <= 0
                stop_first = False
            else:
                stop_first = adverse_first(j)
            if hit_stop and (stop_first or not hit_target):
                result, exit_px = "stop", stop
            elif hit_target:
                result, exit_px = "target", target
        if result:
            exit_i = j
            break
    if result is None:
        exit_px = c[last]
        result = "timeout" if last == e + max_hold - 1 else "open"
    r = direction * (exit_px - entry) / risk
    ret = direction * (exit_px - entry) / entry
    return {"result": result, "e": e, "x": exit_i, "entry": float(entry), "stop": float(stop),
            "target": float(target), "exit": float(exit_px), "risk_pct": float(risk / entry),
            "r": float(r), "r_net": float(r - cost * entry / risk), "ret_net": float(ret - cost)}


def _trade_stats(trades: list[tuple[pd.Timestamp, dict]]) -> dict | None:
    """Closed trades only. The spread uses one average per entry date, since
    trades entered the same day share the market's move."""
    closed = [(d, tr) for d, tr in trades if tr and tr["result"] in ("target", "stop", "timeout")]
    if not closed:
        return None
    r = np.array([tr["r_net"] for _, tr in closed])
    ret = np.array([tr["ret_net"] for _, tr in closed])
    by_day = pd.Series(r, index=[d for d, _ in closed]).groupby(level=0).mean()
    se = float(by_day.std(ddof=1) / np.sqrt(len(by_day))) if len(by_day) > 1 else None
    res = [tr["result"] for _, tr in closed]
    n = len(closed)
    return {
        "n": n, "n_dates": int(len(by_day)),
        "target": res.count("target") / n, "stop": res.count("stop") / n, "timeout": res.count("timeout") / n,
        "win": float((r > 0).mean()), "avg_r": float(r.mean()), "avg_ret": float(ret.mean()),
        "se_r": se, "z": float(r.mean() / se) if se else None, "median_risk": float(np.median([tr["risk_pct"] for _, tr in closed])),
    }


def _trade_study(stocks, frames, events_by_stock, start_date) -> dict:
    """Every firing in the study window traded with the stop-and-target rule,
    plus the same rule entered on every day (the baseline a signal must beat)."""
    rows = {sig.key: [] for sig in SIGNALS}
    base = {1: [], -1: []}
    for sym in stocks:
        df = frames[sym]
        o, h, l, c, a = _ohlc_atr(df)
        idx = df.index
        first = max(int(idx.searchsorted(start_date)), WARMUP_SESSIONS)
        for sig in SIGNALS:
            for t in events_by_stock[sym][sig.key]:
                if t >= first:
                    rows[sig.key].append((idx[t], _simulate(o, h, l, c, a, t, sig.direction,
                                                            _cancel_at(events_by_stock[sym], sig.key, t))))
        for t in range(first, len(c) - 1):
            for d in (1, -1):
                base[d].append((idx[t], _simulate(o, h, l, c, a, t, d)))
    out = {}
    for sig in SIGNALS:
        st = _trade_stats(rows[sig.key])
        b = _trade_stats(base[sig.direction])
        if st:
            beat = None
            if st["se_r"] is not None and b and st["n_dates"] >= MIN_VERDICT_DATES:
                z = (st["avg_r"] - b["avg_r"]) / st["se_r"] if st["se_r"] > 0 else 0.0
                beat = "better" if z >= 2 else "worse" if z <= -2 else "same"
            st["vs_any_day"] = beat
        out[sig.key] = st
    return {"reward": TRADE_REWARD, "max_hold": TRADE_MAX_HOLD, "cost": ROUND_TRIP_COST,
            "min_stop_atr": TRADE_MIN_STOP_ATR, "breakeven": 1 / (1 + TRADE_REWARD),
            "rows": out, "baseline": {"rise": _trade_stats(base[1]), "drop": _trade_stats(base[-1])}}


GAP_FILL_WINDOWS = (20, 60, 250)


def _gap_stats(stocks, frames, feats, start_date) -> dict | None:
    """How often a gap got filled within 1, 3 and 12 months. Only gaps with
    a full 12 months of prices after them count, so recent ones can't drag
    the shares down."""
    n, filled = 0, {w: 0 for w in GAP_FILL_WINDOWS}
    for sym in stocks:
        idx = frames[sym].index
        last = len(idx) - 1
        for g in feats[sym].attrs.get("gaps", []):
            if idx[g["i"]] < start_date or last - g["i"] < GAP_FILL_WINDOWS[-1]:
                continue
            n += 1
            for w in GAP_FILL_WINDOWS:
                if g["filled"] is not None and g["filled"] - g["i"] <= w:
                    filled[w] += 1
    if not n:
        return None
    return {"n": n, "filled": {str(w): filled[w] / n for w in GAP_FILL_WINDOWS}}


def _round_list(series: pd.Series) -> list:
    return [None if pd.isna(v) else round(float(v), 2) for v in series]


def _chart_gaps(gaps, idx, first_day) -> list:
    """Gaps that formed inside the chart window (open, or filled since).
    Older gaps are left out even when still open."""
    day = lambda i: idx[i].strftime("%Y-%m-%d")
    out = []
    for g in gaps:
        if idx[g["i"] - 1] < first_day:
            continue
        out.append({"dir": g["dir"], "a": day(g["i"] - 1), "b": day(g["i"]),
                    "top": round(g["top"], 2), "bottom": round(g["bottom"], 2),
                    "filled": None if g["filled"] is None else day(g["filled"]),
                    "rem": None if g["rem"] is None else [round(v, 2) for v in g["rem"]]})
    return out


def _chart_data(stocks, frames, feats, events_by_stock, spy) -> dict:
    """Last CHART_SESSIONS bars on SPY's calendar for every holding and ETF,
    with the lines each chart draws. A day a stock didn't trade is null."""
    cal = spy.index[-CHART_SESSIONS:]
    out = {"dates": [d.strftime("%Y-%m-%d") for d in cal], "stocks": {}, "etfs": {}}
    for sym in stocks:
        df = frames[sym].reindex(cal)
        feat = feats[sym].reindex(cal)
        idx = frames[sym].index
        arrays = _ohlc_atr(frames[sym])
        direction = {sig.key: sig.direction for sig in SIGNALS}
        events = []
        for key, ts in events_by_stock[sym].items():
            for t in ts:
                if idx[t] < cal[0]:
                    continue
                tr = _simulate(*arrays, t, direction[key], _cancel_at(events_by_stock[sym], key, t))
                if tr and tr["result"] != "skipped":
                    tr = {"result": tr["result"], "entry_date": idx[tr["e"]].strftime("%Y-%m-%d"),
                          "exit_date": idx[tr["x"]].strftime("%Y-%m-%d"),
                          **{k: round(tr[k], 2) for k in ("entry", "stop", "target", "exit")},
                          "r": round(tr["r_net"], 2), "ret": round(tr["ret_net"], 4)}
                elif tr:
                    tr = {"result": "skipped", "why": tr.get("why")}
                ev = {"key": key, "date": idx[t].strftime("%Y-%m-%d"), "trade": tr}
                n_engulfed = feats[sym].attrs.get("engulfed", {}).get(key, {}).get(t)
                if n_engulfed:
                    ev["engulfed"] = n_engulfed
                shape = feats[sym].attrs.get("patterns", {}).get(key, {}).get(t)
                if shape:
                    ev["shape"] = {k: [[idx[i].strftime("%Y-%m-%d"), round(p, 2)] for i, p in pts]
                                   for k, pts in shape.items()}
                events.append(ev)
        out["stocks"][sym] = {
            "o": _round_list(df["Open"]), "h": _round_list(df["High"]),
            "l": _round_list(df["Low"]), "c": _round_list(df["Close"]),
            "v": [None if pd.isna(x) else int(x) for x in df["Volume"]],
            "ema21": _round_list(feat["ema21"]), "sma50": _round_list(feat["sma50"]),
            "sma200": _round_list(feat["sma200"]), "events": sorted(events, key=lambda e: e["date"]),
            "gaps": _chart_gaps(feats[sym].attrs.get("gaps", []), idx, cal[0]),
        }
    for etf in [BENCHMARK, *SECTOR_ETFS]:
        if etf not in frames:
            continue
        close = frames[etf]["Close"]
        # same month-end closes and 10-month average the regime score reads
        monthly = close.resample("ME").last().dropna()
        if (close.index[-1] + _NEXT_SESSION).month == close.index[-1].month:
            monthly = monthly.iloc[:-1]  # month still in progress
        monthly.index = [close.index[close.index <= m][-1] for m in monthly.index]
        m10 = monthly.rolling(10).mean()
        m10 = m10[m10.index >= cal[0]]
        df = frames[etf].reindex(cal)
        out["etfs"][etf] = {
            "o": _round_list(df["Open"]), "h": _round_list(df["High"]), "l": _round_list(df["Low"]),
            "v": [None if pd.isna(x) else int(x) for x in df["Volume"]],
            "c": _round_list(close.reindex(cal)), "ema21": _round_list(ema(close, 21).reindex(cal)),
            "sma50": _round_list(sma(close, 50).reindex(cal)), "sma200": _round_list(sma(close, 200).reindex(cal)),
            "m10": [{"date": d.strftime("%Y-%m-%d"), "v": round(float(v), 2)} for d, v in m10.items() if pd.notna(v)],
        }
    return out


def fetch():
    holdings, holdings_source = _fetch_holdings()
    stocks = sorted({h["symbol"] for rows in holdings.values() for h in rows})
    frames = _download(stocks + list(SECTOR_ETFS) + [BENCHMARK])
    if BENCHMARK not in frames:
        raise RuntimeError("no SPY history - can't compute anything")
    spy = frames[BENCHMARK]
    requested = len(stocks)
    stocks = [s for s in stocks if s in frames]
    full_year = [s for s in stocks if len(frames[s]) >= MIN_COVERAGE_BARS]
    coverage = len(full_year) / requested if requested else 0.0
    study_universe = [s for s in stocks if SHARE_CLASS_PRIMARY.get(s) not in stocks]

    # Minervini's relative-strength condition: 12-month return percentile
    # across this universe, ranked per date among stocks with data that day.
    # A second share class borrows its primary's rank.
    ret_12m = pd.DataFrame({s: frames[s]["Close"] / frames[s]["Close"].shift(252) - 1 for s in study_universe})
    rs_pct = ret_12m.rank(axis=1, pct=True)
    # 12-1 momentum (skip the latest month) and 1-month return, ranked the same way
    mom = pd.DataFrame({s: frames[s]["Close"].shift(21) / frames[s]["Close"].shift(252) - 1 for s in study_universe})
    mom_pct = mom.rank(axis=1, pct=True)
    ret_1m = pd.DataFrame({s: frames[s]["Close"] / frames[s]["Close"].shift(21) - 1 for s in study_universe})
    rev_pct = ret_1m.rank(axis=1, pct=True)
    earnings, earnings_coverage = _fetch_earnings(stocks)

    last_date = spy.index[-1]
    recent_cutoff = spy.index[-RECENT_SESSIONS]  # signals on or after this date count as recent
    events_by_stock, per_stock, feats = {}, {}, {}
    for sym in stocks:
        df = frames[sym]
        rs_col = sym if sym in rs_pct else SHARE_CLASS_PRIMARY[sym]
        feat, events = compute(df, rs_pct[rs_col], mom_pct[rs_col], rev_pct[rs_col], earnings.get(sym))
        events_by_stock[sym] = events
        feats[sym] = feat
        last = feat.iloc[-1]
        close = df["Close"]
        n = len(df)
        stop = None if pd.isna(last["atr14"]) else float(close.iloc[-1] - 10 * last["atr14"])
        per_stock[sym] = {
            # set when this stock's data ends before the market's last session
            "stale_since": None if df.index[-1] >= last_date else df.index[-1].strftime("%Y-%m-%d"),
            "close": float(close.iloc[-1]),
            "chg_pct": float(close.iloc[-1] / close.iloc[-2] - 1),
            "short_history": n < 252,
            "trend_template": bool(last["trend_template"]),
            "above_sma200": None if pd.isna(last["sma200"]) else bool(close.iloc[-1] > last["sma200"]),
            "above_ema21": None if pd.isna(last["ema21"]) else bool(close.iloc[-1] > last["ema21"]),
            "ret_12m": None if n < 253 else float(close.iloc[-1] / close.iloc[-253] - 1),
            "atr_pct": None if pd.isna(last["atr14"]) else float(last["atr14"] / close.iloc[-1]),
            "rs_pct": None if pd.isna(last["rs_pct"]) else float(last["rs_pct"]),
            "pct_from_high": None if pd.isna(last["high_52w"]) else float(close.iloc[-1] / last["high_52w"] - 1),
            "stop_10atr": stop if stop is not None and stop > 0 else None,
            "recent": [
                {"key": key, "date": df.index[t].strftime("%Y-%m-%d")}
                for key, idx in events.items() for t in idx if df.index[t] >= recent_cutoff
            ],
        }

    # A heavily rate-limited download would quietly shrink the universe and
    # reshuffle every RS rank, so below MIN_COVERAGE the study is skipped
    # (and says why) rather than published on a partial set.
    study, study_skipped = None, None
    if coverage >= MIN_COVERAGE:
        study = _event_study(study_universe, frames, spy, events_by_stock,
                             last_date - pd.DateOffset(years=STUDY_YEARS))
        if study is None:
            study_skipped = "not enough price history for a study window of at least a year"
        else:
            study["trades"] = _trade_study(study_universe, frames, events_by_stock,
                                           last_date - pd.DateOffset(years=STUDY_YEARS))
            sector_of = {}
            for etf, rows in holdings.items():
                for h in rows:
                    sector_of.setdefault(h["symbol"], etf)
            combos = run_combos(study_universe, frames, feats, events_by_stock, sector_of, spy,
                                _simulate, _ohlc_atr, last_date - pd.DateOffset(years=STUDY_YEARS))
            combos["last_bar"] = last_date.strftime("%Y-%m-%d")
            COMBOS_PATH.parent.mkdir(parents=True, exist_ok=True)
            COMBOS_PATH.write_text(json.dumps(combos, default=str))
    else:
        study_skipped = f"only {len(full_year)} of {requested} stocks had a full year of price data"

    regimes = {etf: _regime(frames[etf]["Close"]) for etf in SECTOR_ETFS if etf in frames}
    regimes[BENCHMARK] = _regime(spy["Close"])

    sectors = []
    for etf, name in SECTOR_ETFS.items():
        rows = []
        for hold in holdings.get(etf, []):
            info = per_stock.get(hold["symbol"])
            rows.append({**hold, **(info or {"missing": True})})
        sectors.append({"etf": etf, "name": name, "regime": regimes.get(etf, {"score": None}),
                        "holdings_source": holdings_source.get(etf, "missing"), "stocks": rows})

    result = {
        "as_of": int(time.time()),
        "gap_stats": _gap_stats(study_universe, frames, feats, last_date - pd.DateOffset(years=STUDY_YEARS)),
        "last_bar": last_date.strftime("%Y-%m-%d"),
        "benchmark_regime": regimes[BENCHMARK],
        "sectors": sectors,
        "signals": [asdict(s) for s in SIGNALS],
        "study": study,
        "study_skipped": study_skipped,
        "coverage": coverage,
        "earnings_coverage": earnings_coverage,
        "n_stocks": len(stocks),
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHARTS_PATH.write_text(json.dumps(_chart_data(stocks, frames, feats, events_by_stock, spy),
                                      separators=(",", ":")))
    CACHE_PATH.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    data = fetch()
    s = data["study"]
    print(f"last bar {data['last_bar']}  gaps {data.get('gap_stats')}")
    if s:
        print(f"study {s['start']}..{s['end']}  companies {s['n_stocks']}  summary {s['summary']}")
        for row in s["rows"]:
            parts = []
            for h in s["horizons"]:
                st = row["horizons"][str(h)]
                if not st:
                    parts.append("-")
                    continue
                z = "-" if st["score"] is None else f"{st['score']:+.2f}"
                e = "-" if st["edge"] is None else f"{st['edge']:+.4f}"
                parts.append(f"{h}d n={st['n']} w={st['windows']} runs={st['placebo_runs']} net={st['net']:+.4f} edge={e} z={z} p={st['p']} {st['verdict']}")
            print(f"{row['label']:<32} {' | '.join(parts)}")
        tr = s.get("trades", {})
        for key, st in tr.get("rows", {}).items():
            if st:
                print(f"trade {key:<24} risk={st['median_risk']:.3f} n={st['n']} win={st['win']:.2f} target={st['target']:.2f} avgR={st['avg_r']:+.3f} se={st['se_r']} vs_any_day={st['vs_any_day']}")
        for k, b in tr.get("baseline", {}).items():
            print(f"baseline {k} n={b['n']} win={b['win']:.2f} avgR={b['avg_r']:+.3f}")
    for sec in data["sectors"]:
        print(sec["etf"], sec["regime"].get("score"), sec["holdings_source"])
