# Reference (optional)

Deeper notes for people who want more than the dashboard. The README has everything you need to run it.


Everything below is optional reference material for people who want more than the dashboard.

### Command line

The dashboard is the main interface. These commands print to the terminal instead:

```bash
python main.py screen --limit 30          # just the watchlist screen, no option lookups
python main.py screen --symbols AAPL,MSFT,KO --verbose
python main.py suggest --portfolio-size 50000 --symbols AAPL,NVDA,KO
python main.py scorecard --symbols AAPL,KO,NVDA,MSFT,JPM
```

`--limit N` means "only the first N S&P 500 stocks". Leaving it out runs all ~500, which can take 10 minutes or more.

### How it's built

- `web/app.py`: the dashboard (a small Flask web server plus the page itself).
- `engine/`: the screening rules, uptrend math, option selection and Black-Scholes delta.
- `data/`: the only code that talks to Yahoo Finance, Wikipedia and iShares.
- `config/strategy_rules.py`: every number the strategy uses (thresholds, delta target, DTE), each commented with where it came from.
- `cache/`: created at runtime. Stock lists, saved universes and your last scan.

### Optional, off-path: AI ticker brainstorming

`ai_suggest/suggest_tickers.py` (+ `python main.py ai-suggest`, then
`scorecard --from-ai`) still exists as an opt-in, secondary tool if you
ever want an LLM-proposed *shortlist* to run the real screen against
instead of the full S&P 500 — it is never the thing that decides a ticker
is worthy, and it's not part of the default flow above. It's the only file
in the project that imports `anthropic`; nothing else does or needs to.

## What's implemented

- **Section 12 screener** — uptrend (12-18mo SMA stack + weekly higher-highs/
  higher-lows), P/E<100 (with the FCF-covers-debt override), positive FCF,
  EPS-beat last quarter, and (checked later, per-symbol, in the options
  step) 20-delta / 2% RoR at 30 DTE.
- **Quality filters (not in the doc, added per request)** — debt-to-equity
  <= 200% (Financials/Utilities exempted — see below), return on equity
  >= 15%, non-negative YoY revenue growth, current ratio >= 1.0, plus a
  soft/non-disqualifying analyst-sentiment flag. All in
  `config/strategy_rules.py` under "Quality filters", all togglable via
  `QUALITY_FILTERS_ENABLED`.
- **Section 15 VIX allocation** + **Section 16 portfolio-size tiers** + the
  15%-per-stock cap → dollar-denominated sizing for any given portfolio size.
- **Section 17a Cash-Secured Puts**, **17b Covered Calls**, **17d LEAPS** —
  entry-trigger check, real-chain delta targeting via Black-Scholes off the
  chain's own quoted IV, RoR math, position sizing.
- **`scorecard`** — full checklist (all of the above, as ✓/✗/?) plus the
  real chain's contract nearest -0.20 delta and its premium/RoR, for any
  ticker list, ranked by RoR — a comparison view, distinct from `suggest`
  which also gates on the entry trigger and prints an order-shaped result.
- **`web` dashboard** (`web/app.py`) — localhost, sortable, CSP-only,
  ranked by RoR% by default. Uses `engine/options_selector.py`'s
  `find_put_at_least_delta`: the put contract closest to -0.20 delta from
  *at or above* it specifically (not just nearest in either direction —
  see below), since selling below-target delta for CSPs takes on more
  assignment risk than asked for. Flags with "below target Δ" the rare
  case where no contract in the chain reaches -0.20 delta at all, rather
  than silently presenting a lower-delta contract as if it were a real match.
- All numeric rules live in one place: `config/strategy_rules.py`, each
  constant commented with the doc section it came from (or marked as an
  addition if it isn't in the doc at all).

## Known gaps (deliberate, not oversights)

- **Revenue-beat ("double beat")** — free data only gives EPS-beat reliably;
  revenue-beat is reported as unknown, not silently assumed true. See
  `data/fundamentals.py`.
- **IV percentile** — no free historical *option* IV feed exists, so
  `realized_vol_percentile` (in `engine/technicals.py`) is a realized-vol
  proxy, not true IV rank. It's off by default (`STRICT_MODE=True` in
  `config/strategy_rules.py`); flip that to enable it as an extra filter.
- **Not yet implemented: 17c Recovery Mode, 17e Vertical Spreads, 17f
  Collars, 17g Shorting.** These need to know your actual open positions
  (cost basis, shares owned) — covered calls currently default to a
  hypothetical 1-lot when no position is supplied. Wiring in a real
  position feed (Schwab Trader API, read-only) is the natural next step.
- **No backtest engine, by decision.** Real historical option chains
  aren't freely available and a Black-Scholes-simulated backtest was judged
  not worth the false precision — dropped from scope entirely rather than
  shipped as something that looks more rigorous than it is.
- **Schwab Trader API swap-in.** `data/market_data.py` and
  `data/fundamentals.py` are the only two files that know about yfinance;
  a Schwab-backed implementation of the same functions drops in without
  touching `engine/`. Useful once you want account-aware sizing (real
  cash balance/positions) instead of a manually-typed `--portfolio-size`.
