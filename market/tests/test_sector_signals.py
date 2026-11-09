"""Offline checks for the Sectors tab: no look-ahead, indicator math, event
study statistics, regime timing, failure handling, and page rendering.
Synthetic prices and mocked downloads only, so it runs without network.

Run: python -m unittest discover -s tests -t .
"""
import json
import math
import tempfile
import time
import zlib
from datetime import datetime
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from market.fetchers import sector_signals as ss
from market.fetchers.indicators import atr, ema
from market.fetchers.signals import SIGNALS, compute


def synthetic_ohlcv(n=1500, seed=7, drift=0.0004):
    """Random walk with volatility clustering, overnight gaps and volume
    spikes, so gap, pattern and breakout signals all actually fire."""
    rng = np.random.default_rng(seed)
    vol = 0.01 + 0.01 * (np.sin(np.arange(n) / 60) ** 2)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol)))
    gaps = rng.choice([0, 0.03, -0.03], size=n, p=[0.94, 0.03, 0.03])
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * (1 + gaps[1:] + rng.normal(0, 0.002, n - 1))
    span = np.abs(rng.normal(0, vol)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    volume = rng.lognormal(15, 0.5, n) * np.where(rng.random(n) < 0.08, 3.0, 1.0)
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


class NoLookAhead(unittest.TestCase):
    def test_truncated_history_gives_identical_past(self):
        df = synthetic_ohlcv(seed=9, drift=0.0015)
        # a cross-sectional rank that genuinely varies over time
        rs = pd.Series(np.clip(0.5 + 0.4 * np.sin(np.arange(len(df)) / 90), 0, 1), index=df.index)
        full_feat, full_events = compute(df, rs)
        fired = {k for k, v in full_events.items() if v}
        self.assertIn("breakout_52w", fired)
        self.assertIn("trend_template", fired)
        self.assertGreaterEqual(len(fired), 8)
        for cut in (400, 700, 1000, 1300, 1499):
            feat, events = compute(df.iloc[:cut], rs.iloc[:cut])
            for key in full_events:
                self.assertEqual([t for t in full_events[key] if t < cut], events[key], f"{key} at cut {cut}")
            pd.testing.assert_frame_equal(full_feat.iloc[:cut], feat, check_freq=False)


class ResearchSignals(unittest.TestCase):
    def test_new_signals_fire_without_look_ahead(self):
        df = synthetic_ohlcv(seed=9, drift=0.0015)
        n = len(df)
        mom = pd.Series(np.clip(0.5 + 0.45 * np.sin(np.arange(n) / 70), 0, 1), index=df.index)
        rev = pd.Series(np.clip(0.5 + 0.45 * np.cos(np.arange(n) / 25), 0, 1), index=df.index)
        rs = pd.Series(0.5, index=df.index)
        earnings = [{"ts": (df.index[i] + pd.Timedelta(hours=16)).isoformat(), "surprise": 0.05 * (1 if i % 2 else -1)}
                    for i in range(300, n - 1, 63)]
        full_feat, full = compute(df, rs, mom, rev, earnings)
        for key in ("momentum_12_1", "near_52w_high", "reversal_1m"):
            self.assertTrue(full[key], key)
        self.assertTrue(full["earnings_beat"] or full["earnings_miss"])
        for cut in (600, 1000, 1499):
            _, part = compute(df.iloc[:cut], rs.iloc[:cut], mom.iloc[:cut], rev.iloc[:cut], earnings)
            for key in full:
                self.assertEqual([t for t in full[key] if t < cut], part[key], f"{key} at cut {cut}")

    def test_entries_need_a_reset(self):
        from market.fetchers.signals import ENTRY_RESET, _fresh_entries
        cond = np.zeros(80, bool)
        cond[30:35] = True       # first entry
        cond[40] = True          # too soon after the last one: ignored
        cond[70] = True          # off for long enough: counts
        ready = np.ones(80, bool)
        self.assertEqual(_fresh_entries(cond, ready, ENTRY_RESET), [30, 70] if 70 - 40 > ENTRY_RESET else [30])
        ready[:25] = False       # inputs not ready right before bar 30
        self.assertNotIn(30, _fresh_entries(cond, ready, ENTRY_RESET))

    def test_earnings_reaction_session(self):
        from market.fetchers.signals import _earnings_events
        idx = pd.bdate_range("2024-01-01", periods=10)
        close = np.array([100, 101, 102, 103, 110, 111, 112, 100, 99, 98], float)
        df = pd.DataFrame({"Close": close}, index=idx)
        rows = [
            {"ts": "2024-01-04T16:00:00-05:00", "surprise": 0.10},  # after close Jan 4: reacts Jan 5 (+6.8%)
            {"ts": "2024-01-10T08:00:00-05:00", "surprise": -0.05},  # before open Jan 10: reacts Jan 10 (-10.7%)
            {"ts": "2024-01-11T00:00:00-05:00", "surprise": -0.05},  # no time of day: skipped
            {"ts": "2024-01-02T16:00:00-05:00", "surprise": 0.01},  # surprise too small
            {"ts": "2024-01-11T16:00:00-05:00", "surprise": 0.10},  # beat but stock fell on Jan 12: no beat signal
        ]
        beats, misses = _earnings_events(df, rows, close)
        self.assertEqual(beats, [4])
        self.assertEqual(misses, [7])


class IndicatorMath(unittest.TestCase):
    """Independent references: pandas' ewm seeded with the first-n SMA (a
    different code path from the loop under test), and Wilder's recursion."""

    def setUp(self):
        self.df = synthetic_ohlcv(400)

    def test_ema_matches_talib_convention(self):
        c = self.df["Close"]
        n = 21
        seeded = c.copy()
        seeded.iloc[n - 1] = c.iloc[:n].mean()
        want = seeded.iloc[n - 1:].ewm(span=n, adjust=False).mean()
        got = ema(c, n)
        np.testing.assert_allclose(got.iloc[n - 1:].to_numpy(), want.to_numpy(), rtol=1e-12)
        self.assertTrue(got.iloc[:n - 1].isna().all())

    def test_atr_matches_wilder(self):
        h, l, c = (self.df[k] for k in ("High", "Low", "Close"))
        n = 14
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1).iloc[1:]
        seeded = tr.copy()
        seeded.iloc[n - 1] = tr.iloc[:n].mean()
        want = seeded.iloc[n - 1:].ewm(alpha=1 / n, adjust=False).mean()
        got = atr(h, l, c, n)
        np.testing.assert_allclose(got.iloc[n:].to_numpy(), want.to_numpy(), rtol=1e-12)
        self.assertTrue(got.iloc[:n].isna().all())

    def test_ema_rejects_gaps(self):
        c = self.df["Close"].copy()
        c.iloc[50] = np.nan
        with self.assertRaises(ValueError):
            ema(c, 21)


class EventStudyStats(unittest.TestCase):
    def test_forward_excess_entry_exit_and_missing_day(self):
        spy = synthetic_ohlcv(100, seed=1)
        stock = synthetic_ohlcv(100, seed=2)
        ex = ss._forward_excess(stock, spy, 5)
        want = (stock["Close"].iloc[15] / stock["Open"].iloc[11] - 1) - (spy["Close"].iloc[15] / spy["Open"].iloc[11] - 1)
        self.assertAlmostEqual(ex.iloc[10], want, places=12)
        gapped = stock.drop(stock.index[13])  # a session the stock is missing must not stretch its window
        self.assertTrue(np.isnan(ss._forward_excess(gapped, spy, 1).loc[stock.index[12]]))

    def test_hac_t_matches_brute_force(self):
        rng = np.random.default_rng(3)
        days = np.sort(rng.choice(300, 80, replace=False))
        means = rng.normal(0.002, 0.02, 80)
        t = ss._hac_t(days, means, 300, lag=19)
        e = means - means.mean()
        var = sum(e[i] * e[j] * max(0, 1 - abs(days[i] - days[j]) / 20) for i in range(80) for j in range(80)) / 80**2
        self.assertAlmostEqual(t, means.mean() / math.sqrt(var), places=9)

    def test_by_date_averages_same_day(self):
        days, means = ss._by_date(np.array([3, 3, 7]), np.array([1.0, 3.0, 5.0]), 10)
        np.testing.assert_array_equal(days, [3, 7])
        np.testing.assert_allclose(means, [2.0, 5.0])

    def test_independent_windows_greedy(self):
        self.assertEqual(ss._independent_windows(np.array([0, 3, 25, 26, 60, 199]), 20), 4)

    def test_signal_stats_bear_sign_cost_and_adjustment(self):
        n_days, h = 400, 5
        excess = np.full((2, n_days), 0.01)  # both stocks beat SPY by 1% every window
        abnormal = np.random.default_rng(5).normal(0, 0.002, (2, n_days))
        abnormal[0, 100:300:10] -= 0.02      # ...but stock 0 does 2% worse than usual on these days
        pos = np.arange(100, 300, 10)
        st = ss._signal_stats(np.zeros(len(pos), int), pos, -1, excess, abnormal, 0, n_days, h, 60)
        self.assertAlmostEqual(st["net"], -0.01 - ss.ROUND_TRIP_COST)  # shorting a 1% gain, minus cost
        self.assertEqual(st["n"], len(pos))
        self.assertEqual(st["n_dates"], len(pos))
        self.assertAlmostEqual(st["hit_rate"], 0.0)
        # edge is centred on the placebo: shifted runs mostly miss the -2% days, so it lands near +2%
        self.assertGreater(st["edge"], 0.015)

    def test_verdict_guards(self):
        v = ss._verdict
        self.assertEqual(v(0.001, 3.3, 0.01, ss.MIN_VERDICT_DATES - 1, 100), "too_few")
        self.assertEqual(v(0.001, 3.3, 0.01, 100, ss.MIN_INDEPENDENT_WINDOWS - 1), "too_few")
        self.assertEqual(v(None, None, None, 100, 100), "too_few")
        self.assertEqual(v(0.001, 3.3, 0.01, 100, 100), "works")
        self.assertEqual(v(0.001, 3.3, 0.001, 100, 100), "too_small")  # real but under the 0.2% cost
        self.assertEqual(v(0.001, -3.3, 0.01, 100, 100), "reverses")
        self.assertEqual(v(0.03, 2.2, 0.01, 100, 100), "hint_works")
        self.assertEqual(v(0.03, -2.2, 0.01, 100, 100), "hint_backwards")
        self.assertEqual(v(0.3, 1.0, 0.01, 100, 100), "none")

    @staticmethod
    def _null_world(seed, n_stocks=30, n_days=1500, h=5):
        """A world with no edge to find: daily returns with volatility
        clustering but no predictable drift, and a signal that fires after a
        stock's trailing 10-day run-up is in its top 5% - built from its
        recent past like any real signal. Forward h-day returns overlap each
        other but are independent of the run-up, so the only way to 'find'
        an edge is a placebo landing on the run-up itself."""
        rng = np.random.default_rng(seed)
        vol = 0.01 * np.exp(0.5 * np.sin(np.arange(n_days)[None, :] / 40 + rng.uniform(0, 6, (n_stocks, 1))))
        daily = rng.normal(0, 1, (n_stocks, n_days)) * vol
        cum = np.cumsum(daily, axis=1)
        forward = np.full((n_stocks, n_days), np.nan)
        forward[:, :-h] = cum[:, h:] - cum[:, :-h]
        abnormal = forward - np.nanmean(forward, axis=1, keepdims=True)
        trailing = pd.DataFrame(daily.T).rolling(10).sum().to_numpy().T
        stock_idx, pos = np.nonzero(trailing > np.nanpercentile(trailing, 95, axis=1, keepdims=True))
        keep = pos < n_days - h
        return abnormal, stock_idx[keep], pos[keep]

    def test_placebo_null_past_conditioned_signal(self):
        """No edge planted: past-conditioning and bunched firings must not
        produce a strong verdict."""
        strong = 0
        for seed in range(6):
            abnormal, si, pos = self._null_world(seed)
            st = ss._signal_stats(si, pos, 1, abnormal + 0.001, abnormal, 0, abnormal.shape[1], 5, 60)
            strong += st["verdict"] in ("works", "too_small", "reverses")
        self.assertEqual(strong, 0)

    def test_placebo_detects_planted_edge(self):
        """Same world with +1% added to the real firings' forward returns:
        the calibrated test must find it (guards against an over-strict placebo)."""
        abnormal, si, pos = self._null_world(0)
        planted = abnormal.copy()
        planted[si, pos] += 0.01
        st = ss._signal_stats(si, pos, 1, planted + 0.001, planted, 0, planted.shape[1], 5, 60)
        self.assertEqual(st["verdict"], "works")


class FirstCandleAfterCross(unittest.TestCase):
    def test_one_firing_per_run_above_the_ema(self):
        from market.fetchers.signals import _first_against_ema
        e = np.full(10, 100.0)
        #        below  cross  green  RED    green  RED    below  cross  RED    red
        c = np.array([99, 101, 103, 102, 104, 103, 98, 101, 100.5, 100.2])
        o = np.array([99, 100, 102, 103, 103, 104, 99, 100, 101, 100.6])
        reds, greens = _first_against_ema(o, c, e)
        self.assertEqual(reds, [3, 8])   # bar 5 is a later red in the same run

    def test_mirror_below_the_ema(self):
        from market.fetchers.signals import _first_against_ema
        e = np.full(6, 100.0)
        c = np.array([101, 99, 98, 98.5, 97, 98])
        o = np.array([101, 100, 99, 98, 98, 97])
        reds, greens = _first_against_ema(o, c, e)
        self.assertEqual(greens, [3])

    def test_first_red_candle_far_above_the_ema_ends_the_run(self):
        from market.fetchers.signals import _first_against_ema
        e = np.full(6, 100.0)
        #             below cross green  RED far  green  RED near
        c = np.array([99, 101, 104, 104.5, 105, 101.5])
        o = np.array([99, 100, 102, 105.5, 104, 102.5])
        l = np.array([98, 99.5, 101.5, 104, 104, 100.8])   # bar 3's low is 4% above; bar 5's is 0.8%
        h = np.array([100, 102, 105, 106, 106, 103])
        reds, _ = _first_against_ema(o, c, e, l, h)
        self.assertEqual(reds, [])       # bar 3 was the first red and too far; bar 5 is a later red


class DoubleTop(unittest.TestCase):
    """Two tops within 1% up to 120 sessions apart; smaller swings between are fine."""

    def series(self, second_top=109.5, bump=104):
        h = np.full(180, 100.0)
        h[:60] = np.linspace(90, 105, 60)
        h[60:71] = np.linspace(105, 110, 11); h[71:90] = np.linspace(110, 100, 19)
        h[90:100] = np.linspace(100, bump, 10); h[100:110] = np.linspace(bump, 99, 10)
        h[110:131] = np.linspace(99, second_top, 21); h[131:180] = np.linspace(second_top, 90, 49)
        return h

    def fires(self, h, kind="top"):
        from market.fetchers.signals import _double
        from market.fetchers.indicators import swing_points
        lows = h - 0.5
        return _double(h, lows, h - 0.25, swing_points(h if kind == "top" else lows, 5, "high" if kind == "top" else "low"), kind)

    def test_long_pattern_with_a_small_swing_between(self):
        self.assertEqual(self.fires(self.series()), [158])

    def test_higher_peak_between_is_not_a_double_top(self):
        self.assertEqual(self.fires(self.series(bump=112)), [])

    def test_tops_more_than_one_percent_apart(self):
        self.assertEqual(self.fires(self.series(second_top=107.5)), [])

    def test_double_bottom_mirror(self):
        self.assertEqual(self.fires(200 - self.series(), "bottom"), [158])


class Engulfing(unittest.TestCase):
    def test_body_covers_red_candles_wicks_included(self):
        from market.fetchers.signals import engulfed_count
        #        green  red    red    GREEN body 99.5-104 covers both reds' wicks
        o = np.array([100, 103, 102, 99.5])
        c = np.array([103, 102, 100, 104])
        h = np.array([103.5, 103.5, 102.5, 104.5])
        l = np.array([99.5, 101.5, 99.8, 99.0])
        self.assertEqual(engulfed_count(o, h, l, c, 3, 1), 2)   # stops at bar 0 (green)
        self.assertEqual(engulfed_count(o, h, l, c, 3, -1), 0)  # a green candle is never bearish

    def test_wick_covering_is_not_enough(self):
        from market.fetchers.signals import engulfed_count
        # green wicks (99-105) cover the red candle (99.8-104) but its body (99.6-103) doesn't reach 104
        o = np.array([102, 99.6]); c = np.array([100, 103])
        h = np.array([104, 105]); l = np.array([99.8, 99.0])
        self.assertEqual(engulfed_count(o, h, l, c, 1, 1), 0)

    def test_bearish_mirror(self):
        from market.fetchers.signals import engulfed_count
        o = np.array([100, 103.5]); c = np.array([102, 99.5])
        h = np.array([103, 104]); l = np.array([99.8, 99])
        self.assertEqual(engulfed_count(o, h, l, c, 1, -1), 1)


class TradeSimulation(unittest.TestCase):
    """Stop-and-target trade: entry on a break of the signal candle's far wick,
    stop at its opposite wick."""

    def arrays(self, bars):
        o, h, l, c = (np.array([b[k] for b in bars], float) for k in range(4))
        return o, h, l, c, np.full(len(bars), 1.0)  # ATR 1: minimum stop 0.5

    def test_long_enters_on_break_of_high_and_hits_target(self):
        # signal bar 98-101: entry 101 when broken, stop 98 (risk 3), target 104.6
        bars = [(99, 101, 98, 100), (100, 101.5, 99.5, 101.2), (101.5, 105, 101, 104.8)]
        tr = ss._simulate(*self.arrays(bars), 0, 1)
        self.assertEqual(tr["e"], 1)
        self.assertAlmostEqual(tr["entry"], 101)
        self.assertEqual(tr["result"], "target")
        self.assertAlmostEqual(tr["exit"], 104.6)
        self.assertAlmostEqual(tr["r"], 1.2)

    def test_gap_through_the_wick_fills_at_open(self):
        bars = [(99, 101, 98, 100), (102, 102.5, 101.5, 102), (102, 102.2, 101.8, 102)]
        tr = ss._simulate(*self.arrays(bars), 0, 1)
        self.assertAlmostEqual(tr["entry"], 102)
        self.assertAlmostEqual(tr["stop"], 98)

    def test_waits_up_to_three_sessions(self):
        bars = [(99, 101, 98, 100), (100, 100.5, 99, 100), (100, 100.8, 99.5, 100), (100.5, 101.3, 100, 101)]
        tr = ss._simulate(*self.arrays(bars), 0, 1)
        self.assertEqual(tr["e"], 3)

    def test_never_breaks_is_no_trade(self):
        bars = [(99, 101, 98, 100)] + [(100, 100.5, 99, 100)] * 4
        tr = ss._simulate(*self.arrays(bars), 0, 1)
        self.assertEqual((tr["result"], tr["why"]), ("skipped", "no_break"))

    def test_stop_level_first_is_no_trade(self):
        bars = [(99, 101, 98, 100), (99.5, 100, 97.5, 98)]
        tr = ss._simulate(*self.arrays(bars), 0, 1)
        self.assertEqual((tr["result"], tr["why"]), ("skipped", "stop_first"))

    def test_still_waiting_for_the_break(self):
        bars = [(99, 101, 98, 100), (100, 100.5, 99, 100)]
        self.assertIsNone(ss._simulate(*self.arrays(bars), 0, 1))

    def test_same_bar_touching_both_counts_as_stop(self):
        bars = [(99, 101, 98, 100), (100, 101.5, 99.5, 101), (101, 106, 97, 100)]
        tr = ss._simulate(*self.arrays(bars), 0, 1)
        self.assertEqual(tr["result"], "stop")
        self.assertAlmostEqual(tr["r"], -1.0)

    def test_low_before_the_break_does_not_stop_the_trade(self):
        # SOLV, Nov 2026: the entry bar opened near its low, dipped below the
        # stop, then rallied through the entry and closed above the stop
        bars = [(90.34, 90.50, 85.57, 87.18), (86.45, 93.10, 85.04, 90.38)]
        tr = ss._simulate(*self.arrays(bars), 0, 1, 10**9)
        self.assertEqual(tr["result"], "open")
        self.assertAlmostEqual(tr["entry"], 90.50)

    def test_high_first_bar_reaching_both_counts_the_target(self):
        # bar 2 opens near its high: up to the target first, then down through the stop
        bars = [(99, 101, 98, 100), (100, 101.5, 99.5, 101.2), (104, 105, 97, 99)]
        tr = ss._simulate(*self.arrays(bars), 0, 1)
        self.assertEqual(tr["result"], "target")

    def test_short_gap_through_stop_exits_at_open(self):
        # drop signal 99-102: entry 99 on the break, stop 102 (risk 3)
        bars = [(101, 102, 99, 100), (100, 100.5, 98.8, 99.5), (105, 106, 104, 105)]
        tr = ss._simulate(*self.arrays(bars), 0, -1)
        self.assertAlmostEqual(tr["entry"], 99)
        self.assertEqual(tr["result"], "stop")
        self.assertAlmostEqual(tr["exit"], 105)
        self.assertAlmostEqual(tr["r"], -2.0)

    def test_minimum_stop_distance(self):
        # candle range 0.2: the stop widens to 0.5 (half the ATR) below entry
        bars = [(100, 100.1, 99.9, 100), (100, 100.2, 99.95, 100.1), (100.1, 100.15, 99.9, 100)]
        tr = ss._simulate(*self.arrays(bars), 0, 1)
        self.assertAlmostEqual(tr["entry"], 100.1)
        self.assertAlmostEqual(tr["stop"], 99.6)
        self.assertEqual(tr["result"], "open")

    def test_no_entry_bar_yet(self):
        self.assertIsNone(ss._simulate(*self.arrays([(99, 101, 98, 100)]), 0, 1))

    def test_ema_order_waits_past_three_days_and_ignores_stop_touch(self):
        # standing order: dips to the stop level (98) and keeps waiting; fills on day 5
        bars = [(99, 101, 98, 100), (99.5, 100, 97.5, 98.5), (98.5, 99, 98, 98.6), (98.6, 99.5, 98.2, 99),
                (99, 100.5, 98.8, 100), (100.5, 101.6, 100.2, 101.4)]
        tr = ss._simulate(*self.arrays(bars), 0, 1, 10**9)
        self.assertEqual(tr["e"], 5)
        self.assertAlmostEqual(tr["entry"], 101)

    def test_ema_order_cancelled_by_opposite_signal(self):
        bars = [(99, 101, 98, 100)] + [(100, 100.5, 99, 100)] * 4
        tr = ss._simulate(*self.arrays(bars), 0, 1, 2)
        self.assertEqual((tr["result"], tr["why"]), ("skipped", "cancelled"))

    def test_cancel_at_finds_next_opposite_signal(self):
        ev = {"first_red_above_ema21": [3, 20], "green_below_ema21": [10, 30]}
        self.assertEqual(ss._cancel_at(ev, "first_red_above_ema21", 3), 10)
        self.assertEqual(ss._cancel_at(ev, "green_below_ema21", 10), 20)
        self.assertEqual(ss._cancel_at(ev, "green_below_ema21", 30), 10**9)
        self.assertIsNone(ss._cancel_at(ev, "double_bottom", 3))


class Regime(unittest.TestCase):
    def test_uses_last_completed_month(self):
        idx = pd.bdate_range("2023-01-02", "2026-10-30")
        close = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
        mid_month = close[close.index <= "2026-10-15"]
        self.assertEqual(ss._regime(mid_month)["as_of"], "2026-09-30")  # mid-October: September is the last full month
        self.assertEqual(ss._regime(close)["as_of"], "2026-10-30")      # Fri Oct 30 is October's last session
        self.assertEqual(ss._regime(close)["score"], 3)


class FetchWithMockedData(unittest.TestCase):
    """End to end: holdings fetch failing (seed fallback), download coverage,
    and the rendered page."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        seed = json.loads(ss.HOLDINGS_SEED_PATH.read_text())
        self.tickers = sorted({r["symbol"] for rows in seed.values() for r in rows})
        self.patches = [
            mock.patch.object(ss, "CACHE_PATH", self.tmp / "sector_signals.json"),
            mock.patch.object(ss, "CHARTS_PATH", self.tmp / "sector_charts.json"),
            mock.patch.object(ss, "COMBOS_PATH", self.tmp / "sector_combos.json"),
            mock.patch.object(ss, "EARNINGS_CACHE_PATH", self.tmp / "earnings.json"),
            mock.patch.object(ss, "HOLDINGS_CACHE_PATH", self.tmp / "holdings.json"),
            mock.patch.object(ss.yf, "Ticker", side_effect=RuntimeError("no network in tests")),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def _fake_download(self, keep_share=1.0):
        def download(tickers):
            keep = tickers if keep_share >= 1 else [t for i, t in enumerate(tickers)
                                                    if i % 2 == 0 or t in ss.SECTOR_ETFS or t == ss.BENCHMARK]
            return {t: synthetic_ohlcv(1500, seed=zlib.crc32(t.encode()) % 10_000) for t in keep}
        return download

    def test_seed_fallback_full_study_and_render(self):
        with mock.patch.object(ss, "_download", self._fake_download()):
            data = ss.fetch()
        self.assertTrue(all(sec["holdings_source"] == "seed" for sec in data["sectors"]))
        self.assertIsNotNone(data["study"])
        self.assertTrue(all(s["plain"] for s in data["signals"]))

        # Event-study chart lines agree with the table: at day h the path is
        # "Net vs SPY" plus the cost (both average same-day firings first).
        study = data["study"]
        for row in study["rows"]:
            path = study["paths"]["rows"][row["key"]]
            for h in study["horizons"]:
                st = row["horizons"][str(h)]
                if st:
                    self.assertAlmostEqual(path[h], st["net"] + ss.ROUND_TRIP_COST, places=5)
        self.assertAlmostEqual(study["paths"]["baseline"][20],
                               study["baseline"]["20"]["bull"]["net"] + ss.ROUND_TRIP_COST, places=5)
        charts = json.loads((self.tmp / "sector_charts.json").read_text())
        self.assertEqual(len(charts["dates"]), ss.CHART_SESSIONS)
        self.assertEqual(set(charts["etfs"]), {ss.BENCHMARK, *ss.SECTOR_ETFS})
        some = next(iter(charts["stocks"].values()))
        self.assertEqual(len(some["c"]), ss.CHART_SESSIONS)
        self.assertTrue(any(c["events"] for c in charts["stocks"].values()))

        from web.app import app
        from market.fetchers.sector_view import sectors_view
        combos = json.loads((self.tmp / "sector_combos.json").read_text())
        self.assertIn("ladder", combos["long"])
        view = sectors_view(data, combos)
        ctx = {"data": data, "view": view, "combos": combos, "is_static": True}
        with app.app_context():
            html = app.jinja_env.get_template("sectors.html").render(**ctx)
            stock_html = app.jinja_env.get_template("stock.html").render(**ctx)
            combos_html = app.jinja_env.get_template("combos.html").render(**ctx)
        self.assertIn("Does more agreement help?", combos_html)
        self.assertNotIn("{PATTERN", html)
        self.assertNotIn("{PIVOT", html)
        for sig in SIGNALS:
            self.assertIn(sig.plain.replace("&", "&amp;").replace("'", "&#39;"), html)
        self.assertIn('<option value="AAPL">', stock_html)

    def test_low_coverage_skips_study_but_keeps_page(self):
        with mock.patch.object(ss, "_download", self._fake_download(keep_share=0.5)):
            data = ss.fetch()
        self.assertIsNone(data["study"])
        self.assertIn("full year of price data", data["study_skipped"])
        self.assertTrue(data["sectors"])


class DataHygiene(unittest.TestCase):
    def _raw(self, df, ticker="AAA"):
        df = df.copy()
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
        return df

    def test_corrupt_bars_dropped(self):
        df = synthetic_ohlcv(50)
        df.iloc[10, df.columns.get_loc("Open")] = 0.0
        df.iloc[20, df.columns.get_loc("Low")] = df.iloc[20]["High"] + 1
        with mock.patch.object(ss.yf, "download", return_value=self._raw(df)):
            out = ss._download_once(["AAA"])["AAA"]
        self.assertEqual(len(out), 48)
        self.assertNotIn(df.index[10], out.index)
        self.assertNotIn(df.index[20], out.index)

    def test_in_progress_session_dropped(self):
        df = synthetic_ohlcv(50)
        last = df.index[-1]

        class Clock(datetime):
            hour_minute = (11, 0)

            @classmethod
            def now(cls, tz=None):
                return datetime(last.year, last.month, last.day, *cls.hour_minute, tzinfo=tz)

        with mock.patch.object(ss.yf, "download", return_value=self._raw(df)), mock.patch.object(ss, "datetime", Clock):
            self.assertEqual(ss._download_once(["AAA"])["AAA"].index[-1], df.index[-2])  # 11am: still open
            Clock.hour_minute = (17, 0)
            self.assertEqual(ss._download_once(["AAA"])["AAA"].index[-1], last)  # after the close


class HoldingsSources(unittest.TestCase):
    def setUp(self):
        self.cache = Path(tempfile.mkdtemp()) / "holdings.json"
        self.p = [mock.patch.object(ss, "HOLDINGS_CACHE_PATH", self.cache),
                  mock.patch.object(ss.yf, "Ticker", side_effect=RuntimeError("offline"))]
        for p in self.p:
            p.start()

    def tearDown(self):
        for p in self.p:
            p.stop()

    def _write(self, age_seconds):
        rows = [{"symbol": "X", "name": "X", "weight": 0.1}]
        self.cache.write_text(json.dumps({e: {"rows": rows, "fetched_at": time.time() - age_seconds}
                                          for e in ss.SECTOR_ETFS}))

    def test_labels(self):
        self.assertEqual(set(ss._fetch_holdings()[1].values()), {"seed"})    # no cache, fetch fails
        self._write(60)
        self.assertEqual(set(ss._fetch_holdings()[1].values()), {"cached"})  # fresh cache, no fetch
        self._write(ss.HOLDINGS_TTL + 60)
        self.assertEqual(set(ss._fetch_holdings()[1].values()), {"stale"})   # expired, fetch fails


if __name__ == "__main__":
    unittest.main()
