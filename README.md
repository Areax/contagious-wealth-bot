# Contagious Wealth — deterministic suggestion engine

Suggest-only tool built from the "Contagious Wealth" guidebook. **It never
places an order.** It screens the S&P 500 against Section 12's watchlist
criteria, pulls live option chains, and prints/saves specific CSP /
covered-call / LEAPS suggestions with the delta, DTE, and rationale tied
back to the exact section of the doc that produced them. You execute
manually in thinkorswim.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No API keys or broker accounts needed to run today — market data comes
from yfinance (free, unofficial, occasionally rate-limited/stale; treat
as directional, cross-check in thinkorswim before acting).

## Data sources — no thinkorswim/Schwab key needed for any of this

Everything (price history, quotes, VIX, fundamentals, and **live option
chains** — real strikes/bid/ask/OI/IV) comes from `yfinance`, free, no key.
SMA/EMA/RSI/Bollinger Bands/MACD aren't fetched from anywhere — they're
computed in `engine/technicals.py` directly from that real price history.
Delta isn't in yfinance's chain, so `engine/black_scholes.py` derives it
from each contract's own quoted IV — the strike/premium/IV feeding that
are real market quotes, not simulated. A Schwab Trader API key would only
matter later for account-aware data (your real cash/positions) or
execution-grade quotes instead of Yahoo's free (occasionally delayed)
feed — nothing currently built needs it.

## Usage

The mechanical screen is the **only** way tickers get selected — nothing
here is AI-curated.

### Dashboard (start here)

```bash
python main.py web --port 8899
```

Opens a localhost dashboard at `http://127.0.0.1:8899` — a table of
cash-secured-put candidates (checklist columns + the put nearest-or-above
-0.20 delta), **sorted by RoR% by default** (premium ÷ net capital at risk
over the contract's actual DTE — the same convention thinkorswim uses),
click any column header to re-sort. Scans run in a background thread, so
the page keeps responding, shows progress, and has a Cancel button.

- **Universes** — S&P 500, Russell 2000/3000, or a saved "generated
  universe" (a pre-filtered snapshot made with *Create New Universe*, so you
  don't re-screen ~2,600 symbols every time). Your last scan config is
  restored on restart.
- **Modes** — General, or Options Trading (adds a realized-vol floor; an
  optional Strict checkbox adds hard fundamentals gates).
- **Options are fetched on demand** (per row, or "Fetch options for all
  shown", which resumes where it stopped), never during the scan itself.
- **Hide toggles** — far contracts (>45 DTE or >0.30 delta), RoR < 2.5%,
  unknown/"-" rows, and more. Optional columns include Short % Float.
- **Paste your own list** and either add it to the current list
  (temporary) or replace it with a normal scan.

Every scan only keeps stocks above a rising 30-week moving average
(Weinstein Stage 2). Data is Yahoo Finance via `yfinance`: free, but options
chains for small/illiquid names are sometimes incomplete, and short
interest is always 1-2 weeks stale — cross-check before trading.

### CLI

```bash
# Just the watchlist screen (Section 12 + quality filters), fast, no option-chain calls
python main.py screen --limit 30
python main.py screen --symbols AAPL,MSFT,KO --verbose

# Full pipeline: screen + size + suggest, against a portfolio size
python main.py suggest --portfolio-size 50000 --limit 30
python main.py suggest --portfolio-size 50000 --symbols AAPL,NVDA,KO --save-json

# Checklist + -0.20-delta-put premium ranking across the S&P 500 (or a subset)
python main.py scorecard --limit 30
python main.py scorecard --symbols AAPL,KO,NVDA,MSFT,JPM
```

Omit `--limit`/`--symbols` to run the full S&P 500 (~500 names). Screening
alone takes roughly 1 sec/symbol (~8-9 min for the full index); `suggest`
and `scorecard` additionally pull option chains per symbol, so budget more
time for those. Run the full-universe versions as an overnight/pre-market
batch job, not interactively.

### Optional, off-path: AI ticker brainstorming

`ai_suggest/suggest_tickers.py` (+ `python main.py ai-suggest`, then
`scorecard --from-ai`) still exists as an opt-in, secondary tool if you
ever want an LLM-proposed *shortlist* to run the real screen against
instead of the full S&P 500 — it is never the thing that decides a ticker
is worthy, and it's not part of the default flow above. It's the only file
in the project that imports `anthropic`; nothing else does or needs to.
Say the word if you'd rather this be deleted outright instead of left as
an unused option.

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
