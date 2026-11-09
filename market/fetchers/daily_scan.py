"""Daily scan of the S&P 500 for the signals that held up in testing.

After each close, finds every S&P 500 stock where one of SCAN_SIGNALS fired
on the last completed session (plus earlier sessions this week whose order is
still alive), with order levels from the rule that tested best on the S&P 500
(RULE below): buy if price breaks the signal candle's high within 3
sessions, stop one ATR below its low, target 3x the risk, up to 60 sessions. Also writes chart data for those stocks so the Stock lookup
page can show them, and each signal's 5-year record on these stocks.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from market.fetchers import sector_signals as ss
from market.fetchers.signals import SIGNALS, compute
from market.fetchers.sp500_list import get_constituents
from market.fetchers.paths import DATA_DIR

CACHE_PATH = DATA_DIR / "daily_scan.json"
CHARTS_PATH = DATA_DIR / "daily_scan_charts.json"
EARNINGS_PATH = DATA_DIR / "sp500_earnings.json"
# Rise signals that beat buying on a random day in both 2021-24 and 2025-26
# on the S&P 500 (see the exit/stop grid test).
SCAN_SIGNALS = ["bullish_engulfing", "trend_template", "earnings_beat"]
RECENT_SESSIONS = 5
HISTORY_YEARS = 5
MIN_ATR_PCT = 0.005
RULE = {"reward": 3.0, "max_hold": 60, "stop_buffer_atr": 1.0, "cost": 0.0005}
RANDOM_PER_STOCK = 20   # random entry days per stock for the "any day" comparison   # a stock barely moving (e.g. pinned by a takeover offer) is left out
SECTOR_TO_ETF = {name: etf for etf, name in ss.SECTOR_ETFS.items()}
SECTOR_TO_ETF["Information Technology"] = "XLK"


def _status(tr: dict | None) -> str:
    if tr is None:
        return "waiting"
    if tr["result"] == "skipped":
        return "no_break" if tr.get("why") == "no_break" else "stop_first"
    return tr["result"]  # open / target / stop / timeout


def _r(v, nd=2):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)


def fetch() -> dict:
    members = {r["ticker"]: r for r in get_constituents()}
    frames = ss._download(sorted(members) + [ss.BENCHMARK])
    if ss.BENCHMARK not in frames:
        raise RuntimeError("no SPY history")
    spy = frames[ss.BENCHMARK]
    stocks = sorted(s for s in members if s in frames and len(frames[s]) >= 30)
    uni = [s for s in stocks if ss.SHARE_CLASS_PRIMARY.get(s) not in stocks]
    close = {s: frames[s]["Close"] for s in uni}
    rs_pct = pd.DataFrame({s: c / c.shift(252) - 1 for s, c in close.items()}).rank(axis=1, pct=True)
    mom_pct = pd.DataFrame({s: c.shift(21) / c.shift(252) - 1 for s, c in close.items()}).rank(axis=1, pct=True)
    rev_pct = pd.DataFrame({s: c / c.shift(21) - 1 for s, c in close.items()}).rank(axis=1, pct=True)
    orig = ss.EARNINGS_CACHE_PATH
    ss.EARNINGS_CACHE_PATH = EARNINGS_PATH
    try:
        earnings, _ = ss._fetch_earnings(stocks)
    finally:
        ss.EARNINGS_CACHE_PATH = orig

    last_bar = spy.index[-1]
    recent = set(spy.index[-RECENT_SESSIONS:])
    hist_start = last_bar - pd.DateOffset(years=HISTORY_YEARS)
    labels = {s.key: s.label for s in SIGNALS}
    history = {k: [] for k in SCAN_SIGNALS}
    history["random"] = []
    rng = np.random.default_rng(0)
    hits, feats, events_by_stock, lite = [], {}, {}, {}
    for sym in stocks:
        df = frames[sym]
        col = sym if sym in rs_pct else ss.SHARE_CLASS_PRIMARY[sym]
        feat, events = compute(df, rs_pct[col], mom_pct[col], rev_pct[col], earnings.get(sym))
        o, h, l, c, a = ss._ohlc_atr(df)
        idx = df.index
        first = int(idx.searchsorted(hist_start))
        n_before = len(hits)
        if sym in uni and len(c) - 61 > max(first, 252):
            for t in rng.choice(np.arange(max(first, 252), len(c) - 61), RANDOM_PER_STOCK, replace=False):
                tr = ss._simulate(o, h, l, c, a, int(t), +1, **RULE)
                if tr and tr["result"] in ("target", "stop", "timeout"):
                    history["random"].append((idx[t], tr))
        for key in SCAN_SIGNALS:
            for t in events[key]:
                if t < first:
                    continue
                tr = ss._simulate(o, h, l, c, a, t, +1, **RULE)
                if sym in uni and idx[t] < last_bar - pd.Timedelta(days=100) and tr and tr["result"] in ("target", "stop", "timeout"):
                    history[key].append((idx[t], tr))
                if idx[t] not in recent or not a[t] > MIN_ATR_PCT * c[t]:
                    continue
                status = _status(tr)
                # earlier sessions only while the order can still matter
                if idx[t] != last_bar and status not in ("waiting", "open"):
                    continue
                risk = max(h[t] - (l[t] - RULE["stop_buffer_atr"] * a[t]), ss.TRADE_MIN_STOP_ATR * a[t])
                hit = {"symbol": sym, "key": key, "label": labels[key], "date": idx[t].strftime("%Y-%m-%d"),
                       "entry": _r(h[t]), "stop": _r(h[t] - risk), "target": _r(h[t] + RULE["reward"] * risk),
                       "risk_pct": _r(risk / h[t], 4), "status": status}
                n_eng = feat.attrs.get("engulfed", {}).get(key, {}).get(t)
                if n_eng:
                    hit["engulfed"] = n_eng
                if tr and tr["result"] != "skipped":
                    hit["r"] = _r(tr["r_net"])
                hits.append(hit)
        if len(hits) > n_before:
            feats[sym], events_by_stock[sym] = feat, events
            last = feat.iloc[-1]
            n = len(df)
            sector = members[sym]["sector"]
            lite[sym] = {
                "symbol": sym, "name": members[sym]["name"], "close": float(c[-1]),
                "chg_pct": float(c[-1] / c[-2] - 1),
                "above_sma200": None if pd.isna(last["sma200"]) else bool(c[-1] > last["sma200"]),
                "above_ema21": None if pd.isna(last["ema21"]) else bool(c[-1] > last["ema21"]),
                "ret_12m": None if n < 253 else float(c[-1] / c[-253] - 1),
                "atr_pct": None if pd.isna(last["atr14"]) else float(last["atr14"] / c[-1]),
                "rs_pct": None if pd.isna(last["rs_pct"]) else float(last["rs_pct"]),
                "pct_from_high": None if pd.isna(last["high_52w"]) else float(c[-1] / last["high_52w"] - 1),
                "sectors": [{"etf": SECTOR_TO_ETF.get(sector, ""), "name": sector}],
                "sp500_only": True,
            }

    # newest first; within a day, in SCAN_SIGNALS order, then most candles engulfed
    hits.sort(key=lambda x: (x["date"], -SCAN_SIGNALS.index(x["key"]), x.get("engulfed", 0)), reverse=True)
    charts = ss._chart_data(sorted(lite), frames, feats, events_by_stock, spy)
    charts.pop("etfs", None)
    result = {
        "as_of": int(time.time()),
        "last_bar": last_bar.strftime("%Y-%m-%d"),
        "n_scanned": len(stocks),
        "signals": SCAN_SIGNALS,
        "hits": hits,
        "stocks": lite,
        "history": {k: ss._trade_stats(v) for k, v in history.items()},
        "rule": {**RULE, "trigger_days": ss.TRADE_TRIGGER_DAYS, "years": HISTORY_YEARS},
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(result, separators=(",", ":")))
    CHARTS_PATH.write_text(json.dumps(charts, separators=(",", ":")))
    return result


if __name__ == "__main__":
    data = fetch()
    print(f"last bar {data['last_bar']}  scanned {data['n_scanned']}  charted {len(data['stocks'])}")
    for k, st in data["history"].items():
        if st:
            print(f"  {k:<18} n={st['n']} win={st['win']:.2f} avgR={st['avg_r']:+.3f}")
    for hit in data["hits"]:
        print(" ", hit["date"], hit["symbol"], hit["key"], hit["entry"], hit["stop"], hit["target"], hit["status"])
