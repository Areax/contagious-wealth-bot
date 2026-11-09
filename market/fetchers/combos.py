"""Signal combinations: does stacking several checks pick better trades?

Every stock-day in the study window is scored against a fixed list of checks
(trend, momentum, volume, Bollinger Bands, relative strength, sector and
market trend, and each of the signals firing in the last few days). Each
stock-day also gets the outcome of the same stop-and-target trade used on the
Sectors page, entered when price breaks the signal candle's wick.

Two questions, answered on separate periods so the answer isn't fitted to the
data that grades it:
  - Ladder: a bot that buys whenever at least k checks agree (one trade per
    stock at a time). How do results change as k rises?
  - Search: which combinations of up to MAX_COMBO checks did best in the
    TRAIN period, and how did those same picks do in the TEST period?

Picking the best of thousands of combinations always finds some that look
great by luck. Only the test-period column is an honest estimate.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from market.fetchers.indicators import ema, macd, rsi, sma
from market.fetchers.signals import SIGNALS

SPLIT = pd.Timestamp("2025-01-01")   # train before, test from here on
RECENT = 5                           # a signal counts for this many sessions (incl. the day it fires)
MAX_COMBO = 4
MIN_TRAIN_TRADES = 150               # a combination needs this many train trades (one at a time per stock) to rank
TOP_N = 10
GREEDY_STEPS = 6

# key, long label, short label
INDICATOR_CHECKS = [
    ("trend200", "Above its 200-day average", "Below its 200-day average"),
    ("trend50", "Above its 50-day average", "Below its 50-day average"),
    ("ema9", "Above its 9-day EMA", "Below its 9-day EMA"),
    ("stack", "9 EMA above 21 EMA above 50-day", "9 EMA below 21 EMA below 50-day"),
    ("rsi", "RSI between 50 and 70", "RSI between 30 and 50"),
    ("rsi_low", "RSI under 30 (oversold)", "RSI over 70 (overbought)"),
    ("macd", "MACD above its signal line", "MACD below its signal line"),
    ("volume", "Volume 1.5x normal on an up day", "Volume 1.5x normal on a down day"),
    ("bb_break", "Closed above the upper Bollinger Band", "Closed below the lower Bollinger Band"),
    ("bb_dip", "Closed below the lower Bollinger Band", "Closed above the upper Bollinger Band"),
    ("rs", "Top 30% for 12-month strength", "Bottom 30% for 12-month strength"),
    ("near_extreme", "Within 5% of its 1-year high", "Within 5% of its 1-year low"),
    ("sector", "Its sector fund above its 200-day", "Its sector fund below its 200-day"),
    ("market", "S&P 500 above its 200-day", "S&P 500 below its 200-day"),
]


def _checks(df: pd.DataFrame, feat: pd.DataFrame, sector_up: pd.Series, market_up: pd.Series,
            events: dict[str, list[int]], direction: int) -> tuple[pd.DataFrame, pd.Series]:
    """Boolean frame (rows = this stock's dates) of every check for one direction,
    and a mask of rows whose inputs are all warmed up."""
    o, c, v = df["Open"], df["Close"], df["Volume"]
    e9 = ema(c, 9)
    r = rsi(c, 14)
    m_line, m_sig = macd(c)
    mid = sma(c, 20)
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    upper, lower = mid + 2 * sd, mid - 2 * sd
    vol20 = v.shift(1).rolling(20, min_periods=20).mean()
    up_day, down_day = c > o, c < o
    big_vol = v >= 1.5 * vol20
    sec = sector_up.reindex(df.index).ffill()
    mkt = market_up.reindex(df.index).ffill()
    ready = (feat[["sma50", "sma200", "ema21", "high_52w", "low_52w", "rs_pct"]].notna().all(axis=1)
             & r.notna() & m_sig.notna() & upper.notna() & vol20.notna() & sec.notna() & mkt.notna())
    long = direction > 0
    x = pd.DataFrame(index=df.index)
    x["trend200"] = (c > feat["sma200"]) if long else (c < feat["sma200"])
    x["trend50"] = (c > feat["sma50"]) if long else (c < feat["sma50"])
    x["ema9"] = (c > e9) if long else (c < e9)
    x["stack"] = ((e9 > feat["ema21"]) & (feat["ema21"] > feat["sma50"])) if long else \
                 ((e9 < feat["ema21"]) & (feat["ema21"] < feat["sma50"]))
    x["rsi"] = r.between(50, 70) if long else r.between(30, 50)
    x["rsi_low"] = (r < 30) if long else (r > 70)
    x["macd"] = (m_line > m_sig) if long else (m_line < m_sig)
    x["volume"] = big_vol & (up_day if long else down_day)
    x["bb_break"] = (c > upper) if long else (c < lower)
    x["bb_dip"] = (c < lower) if long else (c > upper)
    x["rs"] = (feat["rs_pct"] >= 0.7) if long else (feat["rs_pct"] <= 0.3)
    x["near_extreme"] = (c >= 0.95 * feat["high_52w"]) if long else (c <= 1.05 * feat["low_52w"])
    x["sector"] = (sec == 1) if long else (sec == 0)
    x["market"] = (mkt == 1) if long else (mkt == 0)
    n = len(df)
    for sig in SIGNALS:
        if sig.direction != direction:
            continue
        hit = np.zeros(n, bool)
        for t in events[sig.key]:
            hit[t:min(n, t + RECENT)] = True
        x["sig_" + sig.key] = hit
    return x.fillna(False).astype(bool), ready


def _labels(direction: int) -> dict[str, str]:
    out = {k: (lo if direction > 0 else sh) for k, lo, sh in INDICATOR_CHECKS}
    for sig in SIGNALS:
        if sig.direction == direction:
            out["sig_" + sig.key] = sig.label + " (last 5 days)"
    return out


def _stats(trades: list[tuple]) -> dict | None:
    """trades: (date, r). Spread from one average per entry date."""
    if not trades:
        return None
    r = np.array([t[1] for t in trades])
    by_day = pd.Series(r, index=[t[0] for t in trades]).groupby(level=0).mean()
    se = float(by_day.std(ddof=1) / np.sqrt(len(by_day))) if len(by_day) > 2 else None
    return {"n": len(r), "win": float((r > 0).mean()), "avg_r": float(r.mean()),
            "total_r": float(r.sum()), "z": float(r.mean() / se) if se else None}


class _Universe:
    """All candidate stock-days for one direction, grouped by stock."""

    def __init__(self, per_stock: list[dict]):
        self.per = per_stock
        self.X = np.concatenate([p["X"] for p in per_stock]) if per_stock else np.zeros((0, 0), bool)
        self.train = np.concatenate([p["train"] for p in per_stock]) if per_stock else np.zeros(0, bool)
        self.r = np.concatenate([p["r"] for p in per_stock]) if per_stock else np.zeros(0)
        offs = np.cumsum([0] + [len(p["t"]) for p in per_stock])
        self.slices = [slice(offs[i], offs[i + 1]) for i in range(len(per_stock))]

    def bot(self, mask: np.ndarray) -> dict:
        """Buy whenever mask is true and this stock has no open trade."""
        out = {"train": [], "test": []}
        for p, sl in zip(self.per, self.slices):
            free_from = -1
            for j in np.nonzero(mask[sl])[0]:
                if p["t"][j] < free_from:
                    continue
                free_from = p["x"][j]
                if p["train"][j]:
                    out["train"].append((p["date"][j], p["r"][j]))
                elif p["test"][j]:
                    out["test"].append((p["date"][j], p["r"][j]))
        return {k: _stats(v) for k, v in out.items()}


def _direction(stocks, frames, feats, events_by_stock, sector_up, market_up, simulate, ohlc_atr,
               start, direction) -> dict:
    keys = None
    per = []
    for sym in stocks:
        df = frames[sym]
        x, ready = _checks(df, feats[sym], sector_up[sym], market_up, events_by_stock[sym], direction)
        keys = list(x.columns)
        arrays = ohlc_atr(df)
        idx = df.index
        rows = {"t": [], "x": [], "r": [], "date": [], "train": [], "test": [], "X": []}
        xv, rv = x.to_numpy(), ready.to_numpy()
        for t in range(len(df) - 1):
            if idx[t] < start or not rv[t]:
                continue
            tr = simulate(*arrays, t, direction)
            if not tr or tr["result"] not in ("target", "stop", "timeout"):
                continue
            exit_date = idx[tr["x"]]
            rows["t"].append(t)
            rows["x"].append(tr["x"])
            rows["r"].append(tr["r_net"])
            rows["date"].append(idx[tr["e"]])
            rows["train"].append(exit_date < SPLIT)           # trade fully inside the train period
            rows["test"].append(idx[tr["e"]] >= SPLIT)
            rows["X"].append(xv[t])
        if rows["t"]:
            per.append({k: np.array(v) for k, v in rows.items()})
    if not per:
        return {}
    u = _Universe(per)
    labels = _labels(direction)
    n_checks = len(keys)
    score = u.X.sum(axis=1)

    base = u.bot(np.ones(len(u.r), bool))
    ladder = []
    for k in range(1, n_checks + 1):
        res = u.bot(score >= k)
        if not res["train"] and not res["test"]:
            break
        ladder.append({"k": k, **res})

    # Fast screen with overlapping trades, then the one-at-a-time bot for the survivors.
    X, train, r = u.X, u.train, u.r
    candidates = []
    for size in range(1, MAX_COMBO + 1):
        for combo in combinations(range(n_checks), size):
            m = X[:, combo].all(axis=1) if size > 1 else X[:, combo[0]]
            mt = m & train
            n = int(mt.sum())
            if n < MIN_TRAIN_TRADES:
                continue
            candidates.append((float(r[mt].mean()), combo))
    top, picked_masks = [], []
    for _, combo in sorted(candidates, key=lambda x: (-x[0], len(x[1])))[:TOP_N * 50]:
        m = X[:, combo].all(axis=1)
        # skip a combination that picks almost the same train days as one already listed
        mt = m & train
        if any((mt & p).sum() >= 0.9 * max(mt.sum(), p.sum()) for p in picked_masks):
            continue
        res = u.bot(m)
        if not res["train"] or res["train"]["n"] < MIN_TRAIN_TRADES:
            continue
        picked_masks.append(mt)
        top.append({"idx": list(combo), "checks": [keys[i] for i in combo], **res})
        if len(top) == TOP_N:
            break
    # All the picks traded together: one larger sample, so a fairer read on luck.
    union = u.bot(np.any([X[:, t["idx"]].all(axis=1) for t in top], axis=0)) if top else None
    for t in top:
        t.pop("idx")

    # Greedy build-up: each step adds the check that most improves the train result.
    greedy, chosen = [], []
    for _ in range(GREEDY_STEPS):
        best = None
        for i in range(n_checks):
            if i in chosen:
                continue
            combo = chosen + [i]
            mt = X[:, combo].all(axis=1) & train
            if mt.sum() < MIN_TRAIN_TRADES:
                continue
            avg = float(r[mt].mean())
            if best is None or avg > best[0]:
                best = (avg, i)
        if best is None:
            break
        chosen.append(best[1])
        res = u.bot(X[:, chosen].all(axis=1))
        if greedy and res["train"] and greedy[-1]["train"] and res["train"]["n"] == greedy[-1]["train"]["n"]:
            break  # the new check filtered out nothing
        greedy.append({"checks": [keys[i] for i in chosen], **res})

    return {"checks": [{"key": k, "label": labels[k]} for k in keys], "baseline": base,
            "ladder": ladder, "top": top, "union": union, "greedy": greedy, "n_combos": len(candidates)}


def today_matches(stocks, frames, feats, events_by_stock, sector_up, market_up, result) -> dict:
    """Which stocks meet each top combination on the last bar, and every stock's score."""
    out = {}
    for dkey, direction in (("long", 1), ("short", -1)):
        res = result.get(dkey) or {}
        if not res:
            continue
        combos = [t["checks"] for t in res.get("top", [])] + [g["checks"] for g in res.get("greedy", [])]
        rows = {}
        for sym in stocks:
            x, ready = _checks(frames[sym], feats[sym], sector_up[sym], market_up, events_by_stock[sym], direction)
            if not len(x) or not bool(ready.iloc[-1]):
                continue
            last = x.iloc[-1]
            rows[sym] = {"score": int(last.sum()), "on": [k for k, v in last.items() if v],
                         "combos": [i for i, c in enumerate(combos) if all(last[k] for k in c)]}
        out[dkey] = rows
    return out


def run(stocks, frames, feats, events_by_stock, sector_of, spy, simulate, ohlc_atr, start) -> dict:
    def above200(close):
        s = sma(close, 200)
        return (close > s).astype(float).where(s.notna())

    market_up = above200(spy["Close"])
    etf_up = {etf: above200(frames[etf]["Close"]) for etf in set(sector_of.values()) if etf in frames}
    sector_up = {s: etf_up.get(sector_of.get(s), market_up) for s in stocks}
    result = {"split": SPLIT.strftime("%Y-%m-%d"), "start": start.strftime("%Y-%m-%d"),
              "max_combo": MAX_COMBO, "min_train_trades": MIN_TRAIN_TRADES}
    for dkey, direction in (("long", 1), ("short", -1)):
        result[dkey] = _direction(stocks, frames, feats, events_by_stock, sector_up, market_up,
                                  simulate, ohlc_atr, start, direction)
    result["today"] = today_matches(stocks, frames, feats, events_by_stock, sector_up, market_up, result)
    return result
