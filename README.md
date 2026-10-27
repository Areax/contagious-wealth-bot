# Contagious Wealth Bot

A dashboard that helps you find stocks worth selling **cash-secured puts** on, and shows what each put would pay. It runs on your own computer and opens in your web browser.

**It never buys or sells anything.** It only shows you information. You place any trade yourself, in your own brokerage.

> **Not financial advice.** This is an educational screening tool. It doesn't know your finances, risk tolerance or taxes, and options can lose more than the premium you collect. Data comes from Yahoo Finance's free feed and can be delayed, incomplete or wrong. Always check a contract in your broker before acting.

## Contents

- [What is a cash-secured put?](#what-is-a-cash-secured-put)
- [Quick start](#quick-start-about-10-minutes) (Mac, Windows, Linux)
- [Starting it again later](#starting-it-again-later)
- [Using the dashboard](#using-the-dashboard)
- [Words you'll see](#words-youll-see)
- [Troubleshooting](#troubleshooting)
- [Advanced: command line, reference, known gaps](#advanced)

## What is a cash-secured put?

You agree to buy 100 shares of a stock at a price you pick (the **strike**) if it falls to that price before a set date. You set aside the cash to cover the purchase, and you get paid up front for making the promise. That payment is the **premium**.

- If the stock stays above the strike, you keep the premium and nothing else happens.
- If it falls below the strike, you buy the shares at the strike price. Your real cost is the strike minus the premium you were paid.

This tool finds stocks in an uptrend and then picks the put that pays the most for a reasonable level of risk.

## Quick start (about 10 minutes)

You need: a computer (Mac, Windows or Linux), an internet connection, and a web browser. No account, API key or broker login is needed.

### Step 1: Install Python

You need Python 3.9 or newer. Check whether you already have it by opening a terminal:

- **Mac:** open the **Terminal** app (press Cmd+Space, type "Terminal").
- **Windows:** open **Command Prompt** (press the Windows key, type "cmd").
- **Linux:** open your terminal.

Then type this and press Enter:

```bash
python3 --version
```

On Windows, use `python --version` instead. If you see `Python 3.9` or higher, skip ahead to Step 2. If you get an error or an older version, download Python from <https://www.python.org/downloads/> and install it.

> **Windows:** on the first installer screen, tick **"Add python.exe to PATH"** before clicking Install. This is the most common thing beginners miss.

Close and reopen your terminal after installing.

### Step 2: Download this project

**Option A, with git:**

```bash
git clone https://github.com/Areax/contagious-wealth-bot.git
```

**Option B, without git:** on the GitHub page, click the green **Code** button, then **Download ZIP**, and unzip it. If the repository is private, you need to be signed in and have access.

### Step 3: Open a terminal in the project folder

In your terminal, go into the folder you just downloaded:

```bash
cd contagious-wealth-bot
```

If you used the ZIP, the folder may be named `contagious-wealth-bot-main`. Use that name instead. If you are not sure you are in the right place, this command should list `main.py`:

- Mac and Linux: `ls`
- Windows: `dir`

### Step 4: Create an isolated Python environment

This keeps the project's packages separate from the rest of your computer. You only do this once.

**Mac and Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt):**

```bat
python -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

When it worked, your terminal line now starts with `(.venv)`. If PowerShell says scripts are disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, and try the activate line again. Or just use Command Prompt instead.

### Step 5: Install the packages

```bash
pip install -r requirements.txt
```

This downloads what the project needs and takes a minute or two. Lots of scrolling text is normal. You only do this once.

### Step 6: Start the dashboard

```bash
python main.py web
```

If `python` isn't found on Mac or Linux, use `python3` instead. Inside the activated `(.venv)` environment, either usually works.

You should see a line like `Starting dashboard at http://127.0.0.1:8899`.

### Step 7: Open it in your browser

Go to <http://127.0.0.1:8899> in any browser.

The first time, it scans the first 50 stocks of the S&P 500, which takes about a minute. The status line next to the buttons shows progress. When the scan finishes, it automatically looks up option prices for the stocks that passed, one at a time. Give that another minute or two. Stocks appear in the table as they're ready.

To stop the dashboard, go back to the terminal and press **Ctrl+C**.

## Starting it again later

You do not repeat the install steps. Each time you want to use it:

1. Open a terminal and `cd` into the project folder.
2. Activate the environment again:
   - Mac and Linux: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\activate`
3. Run `python main.py web` and open <http://127.0.0.1:8899>.

The dashboard remembers the last scan you ran and starts from it.

To use a different port (for example if 8899 is taken), add `--port 8900` and open that address instead.

## Using the dashboard

The table lists stocks that passed the screen, with the put the tool would sell on each. It is sorted by **RoR%** (return on risk), highest first. Click any column heading to sort by it.

**What a stock must pass to appear:** its price is above a rising 30-week average (an uptrend). In **Options Trading** mode it also needs enough day-to-day price movement (25% annualized volatility) to pay a meaningful premium. The other columns are a checklist (✓ / ✗) you can use to judge quality yourself.

### The controls

| Control | What it does |
|---|---|
| **Mode** | *General* shows everything that passes the uptrend check. *Options Trading* also removes calm, low-premium stocks. |
| **Strict** (in Options Trading) | Also requires solid fundamentals: profitable, healthy cash flow, moderate debt. A much shorter list. |
| **Universe** | Which group of stocks to scan: S&P 500 (~500), Russell 2000 (small companies), Russell 3000 (almost the whole US market), or one you saved. |
| **Scan** | How many stocks of that group to scan. Start with a small number. |
| **Run scan** | Scan the chosen universe now. |
| **Fetch options for all shown** | Looks up option prices for every row that doesn't have them yet. If it was interrupted, it picks up where it stopped. The **Force refresh** box next to it is ticked by default, so clicking the button re-fetches everything, including rows that already have prices. Untick it to fetch only what's missing. |
| **Cancel** | Stops a scan or an option lookup that is running. |
| **Create New Universe** | Screens the entire Russell 3000 and saves the stocks that pass as a reusable list. **This takes several minutes and makes thousands of requests. Only use it occasionally.** |
| **Paste your own list** | Type tickers like `AAPL, MSFT, KO`. *Add to current list* scans just those and merges them into the table until your next scan. *Replace current list* does a normal scan of only those tickers. |
| **↻ next to a row** | Re-fetches that one stock's option, since prices change during the day. |

### The hide toggles

These only change what you see, not what was scanned:

- **Hide DTE > 45 or Δ > 0.30**: hides contracts that are far out in time or riskier than the target.
- **Hide RoR% < 2.5%**: hides puts that pay too little.
- **Hide unknown/-**: hides stocks where no matching option was found.
- **Hide non-uptrend (12/18mo)**: an extra, older uptrend check for comparison.
- **Hide non-Strict**: only appears in Options Trading mode.

The **Show columns** checkboxes turn extra columns on and off, including **Short % Float** (the share of a stock's tradeable shares sold short, which is updated only about twice a month).

## Words you'll see

| Term | Meaning |
|---|---|
| **Strike** | The price you agree to buy the stock at. |
| **Premium** | What you're paid up front, per contract (100 shares). |
| **DTE** | Days to expiration. How long the contract lasts. |
| **Delta** (Δ) | Roughly the market's estimate of the chance the put ends up in the money. The tool targets about 0.20, meaning about 20%. |
| **RoR%** | Return on risk: premium divided by the cash you tie up (strike × 100 minus the premium), over that contract's life. The same method thinkorswim uses. A longer contract naturally shows a bigger number, so compare like with like. |
| **OI** | Open interest: how many of that contract exist. Very low numbers mean it can be hard to get a fair price. |
| **Uptrend** | Price above a rising 30-week average. |
| **Vol** | Realized volatility: how much the stock has been moving recently. Higher usually means richer premium and more risk. |
| **-** | No matching option was found, or none has been fetched yet. Not necessarily a bad stock. |

## Troubleshooting

**`python` or `python3` is not recognized or not found.** Python isn't installed or isn't on your PATH. Reinstall it and, on Windows, tick "Add python.exe to PATH". Then open a new terminal.

**`ModuleNotFoundError: No module named 'flask'` (or another package).** The environment isn't active, or Step 5 didn't finish. Run the activate command from Step 4 (your line should start with `(.venv)`), then `pip install -r requirements.txt` again.

**`Address already in use` or the port is busy.** Another program is using port 8899, possibly an earlier copy of this one. Close it, or start with `python main.py web --port 8900` and open that address.

**The page loads but the table is empty.** Wait for the first scan to finish. The status line shows progress. If it says an error, read the message. Most often it's a temporary internet or Yahoo problem. Click **Run scan** again.

**Lots of rows show `-` for the option.** Common, and expected:
- Outside market hours Yahoo often has no live quotes, so many contracts look empty. Try again while the US market is open (9:30am to 4pm Eastern, Monday to Friday).
- Yahoo's free data is thin for smaller or less-traded stocks, even when your broker shows a full list. This is a data limit, not a bug in the tool.
- Rows can say `-` simply because no put landed in the 0.20 to 0.30 delta range.

**Yahoo says you're rate limited, or many rows error out.** You asked for too much too quickly. Stop, wait a few minutes, and try again with a smaller scan. Avoid Russell 3000 scans and **Create New Universe** unless you really need them.

**A warning about `urllib3` or `LibreSSL` when starting.** Harmless on older Macs. You can ignore it.

**I closed the terminal and the page stopped working.** The dashboard only runs while the terminal window is running it. Start it again as described above.

**Where is my data stored?** Locally, in the `cache/` folder (the universe lists and your last scan). Nothing about you is sent anywhere. The only network requests are for market data and stock lists: Yahoo Finance (prices, fundamentals, options), Wikipedia (the S&P 500 list) and iShares (the Russell lists). Deleting `cache/` resets everything safely.

## Advanced

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
