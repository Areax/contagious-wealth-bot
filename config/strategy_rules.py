"""
Numeric rules transcribed directly from the "Contagious Wealth" guidebook.
Every constant here should be traceable to a specific section of the source
document. If you change the strategy, change it here — not by hand-editing
numbers inside engine/ or ai_suggest/.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Section 12: BUILD YOUR WATCHLIST
# ---------------------------------------------------------------------------

WATCHLIST_UPTREND_MIN_MONTHS = 12
WATCHLIST_UPTREND_MAX_MONTHS = 18

WATCHLIST_MAX_PE = 100.0
# If P/E > WATCHLIST_MAX_PE, the doc requires the company be able to retire
# ALL debt using free cash flow alone. We approximate this as:
#   total_debt <= WATCHLIST_PE_OVERRIDE_FCF_YEARS * annual_free_cash_flow
WATCHLIST_PE_OVERRIDE_FCF_YEARS = 1.0

WATCHLIST_TARGET_DELTA = 0.20          # "target a 20-delta strike price"
WATCHLIST_MIN_RETURN_ON_RISK = 0.02    # "return on risk of at least 2%"
WATCHLIST_RETURN_ON_RISK_DTE = 30      # "for a 30-day trade" — also the hard DTE floor for the CSP dashboard

# How far past the 30-day floor the CSP dashboard's put selector is willing
# to search for a contract that actually reaches -0.20 delta, before it
# gives up and flags a fallback. Not in the source doc — an implementation
# choice: the nearest (>=30 DTE) expiration is tried first and used
# whenever it already has a qualifying strike; this is just the ceiling.
CSP_DASHBOARD_MAX_DTE_SEARCH = 90

# The CSP dashboard only ever displays a put whose delta falls in this
# range — anything outside it (e.g. a fallback pick that overshot to 0.69
# delta because the real ~20-delta strikes had no market) is shown as N/A
# instead, exactly like a symbol where no put was found at all. Per user
# request: rather than surface a low-confidence fallback with a warning
# label, just don't show a put for that row.
CSP_DASHBOARD_ACCEPTABLE_DELTA_RANGE = (0.20, 0.30)

# "Options Trading" mode's hard volatility floor — NOT in the source doc.
# Realized volatility is a more direct proxy for option premium richness
# than beta (beta only captures market-correlated risk; a stock can carry
# rich premium from purely idiosyncratic volatility beta won't reflect —
# see conversation notes). 25% annualized is a judgment call, not a
# calculated optimum: SPY itself typically realizes ~12-20% in calm
# periods, so 25% is meaningfully above "moves like the market" without
# being so high it excludes everything but the most extreme names. Applied
# BEFORE the fundamentals fetch (same stage as the uptrend filter), so it
# also reduces call volume for large-universe scans, not just display noise.
OPTIONS_MODE_MIN_REALIZED_VOL = 0.25

# "Curated" mode: a much smaller, more selective tier meant to survive
# real due-diligence, not just a raw top-of-universe volatility cut.
# History: an earlier version of this tier used an 85th-percentile-of-
# Russell-3000 absolute vol floor, which excluded all but 2 of the 27
# manually-curated "Wheel It" reference stocks and skewed the survivor
# list toward crypto-treasury vehicles (MSTR, MARA, RIOT, HUT, BMNR,
# SBET) and pre-revenue biotechs — high volatility, but for the wrong
# reasons (balance-sheet leverage / binary trial risk, not chronic
# options-premium richness). Recalibrated against the actual Wheel-It
# stocks' real realized-vol range (~38-107%, median ~58%) instead of an
# arbitrary universe percentile.
CURATED_MODE_MIN_REALIZED_VOL = 0.35

# Curated mode also HARD-GATES on fundamentals (General/Options modes
# leave these as informational-only columns — see engine/checks.py).
# Goal: mechanically exclude balance-sheet-leveraged / pre-earnings names
# (the crypto-treasury and biotech pattern above) without ever referring
# to a ticker or sector by name — sector/industry labels aren't reliable
# for this (yfinance classifies MSTR as "Software - Application" and
# crypto miners/treasuries as "Capital Markets", which also contains
# legitimate brokers/exchanges).
CURATED_REQUIRE_POSITIVE_PE = True   # trailing P/E must exist and be > 0 (i.e. actually profitable)
CURATED_MIN_CASH_RUNWAY_YEARS = 2.0  # if FCF is negative, total_cash / |FCF| must clear this instead
CURATED_MAX_DEBT_TO_EQUITY = 150.0   # tighter than MAX_DEBT_TO_EQUITY (200) above
CURATED_MIN_ROE = 0.10               # lower than MIN_RETURN_ON_EQUITY (0.15) — 15% excluded AMD/INTC
                                      # for capex/restructuring-driven ROE dips, not real distress


# ---------------------------------------------------------------------------
# Section 15: VIX ALLOCATIONS
# Ordered high -> low. `min_vix` is inclusive lower bound for the bucket.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VixTier:
    label: str
    min_vix: float
    capital_deployed_pct: tuple  # (low, high)
    cash_reserved_pct: tuple     # (low, high)
    notes: str


VIX_TIERS = (
    VixTier("Extreme fear / crisis", 30.0, (100, 100), (0, 0),
            "Aggressive premium selling, widest spreads, highest IV edge"),
    VixTier("Elevated volatility", 20.0, (80, 100), (0, 20),
            "Strong premium, core wheel and spreads"),
    VixTier("Normal volatility", 15.0, (40, 80), (20, 60),
            "Selective trades, tighter risk control"),
    VixTier("Low volatility / complacency", 10.0, (20, 40), (60, 80),
            "Capital preservation, fewer setups"),
    # Below 10 isn't defined in the doc; we extend the lowest tier's floor
    # rather than invent a new one.
    VixTier("Below-range complacency", 0.0, (20, 40), (60, 80),
            "Undefined in source doc — using the VIX~10 tier as floor"),
)


def vix_tier_for(vix_level: float) -> VixTier:
    for tier in VIX_TIERS:
        if vix_level >= tier.min_vix:
            return tier
    return VIX_TIERS[-1]


# ---------------------------------------------------------------------------
# Section 16: PORTFOLIO SIZE TIERS
# Allocation percentages of *deployed* capital across strategy buckets.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PortfolioTier:
    label: str
    min_size: float
    max_size: float
    wheel_pct: tuple
    leaps_pct: tuple
    spreads_pct: tuple
    goal: str


PORTFOLIO_TIERS = (
    PortfolioTier("$10K-$30K", 10_000, 30_000, (30, 40), (30, 40), (20, 30),
                  "Capital growth and risk control"),
    PortfolioTier("$30K-$75K", 30_000, 75_000, (40, 50), (30, 40), (10, 20),
                  "Growth plus income balance"),
    PortfolioTier("$75K-$150K", 75_000, 150_000, (55, 65), (25, 35), (5, 10),
                  "Consistent income plus upside"),
    PortfolioTier("$150K-$300K", 150_000, 300_000, (65, 75), (20, 30), (0, 5),
                  "Predictable cash flow"),
    PortfolioTier("$300K+", 300_000, float("inf"), (70, 85), (15, 30), (0, 5),
                  "Wealth preservation plus steady income"),
)


def portfolio_tier_for(portfolio_size: float) -> PortfolioTier:
    for tier in PORTFOLIO_TIERS:
        if tier.min_size <= portfolio_size < tier.max_size:
            return tier
    return PORTFOLIO_TIERS[-1]


# ---------------------------------------------------------------------------
# Section 23 (default trading plan) / general risk caps
# ---------------------------------------------------------------------------

MAX_ALLOCATION_PER_STOCK_PCT = 0.15   # "maximum of 15% of my portfolio to any single stock"

# NOT in the source doc explicitly (see research notes) — added as a
# deterministic risk-engine safeguard, off by default, tune per user.
MAX_ALLOCATION_PER_SECTOR_PCT = 0.35


# ---------------------------------------------------------------------------
# Section 17a: CASH-SECURED PUTS
# ---------------------------------------------------------------------------

CSP_ENTRY_DOWN_DAYS = 3
CSP_ENTRY_SINGLE_DAY_DROP_PCT = 0.05
CSP_TARGET_DTE = 30
CSP_SAFE_DELTA = 0.20
CSP_AGGRESSIVE_DELTA_RANGE = (0.25, 0.30)


# ---------------------------------------------------------------------------
# Section 17b: COVERED CALLS
# ---------------------------------------------------------------------------

CC_ENTRY_UP_DAYS = 3
CC_ENTRY_SINGLE_DAY_GAIN_PCT = 0.05
CC_TARGET_DTE_RANGE = (14, 21)   # "2 to 3 weeks out"
CC_DELTA_RANGE = (0.25, 0.30)


# ---------------------------------------------------------------------------
# Section 17c: COVERED CALLS BELOW COST BASIS (Recovery Mode)
# ---------------------------------------------------------------------------

RECOVERY_DTE_RANGE = (7, 10)
RECOVERY_DELTA_RANGE = (0.08, 0.12)
RECOVERY_WEEKLY_INCOME_TARGET_PCT = (0.003, 0.005)


# ---------------------------------------------------------------------------
# Section 17d: LEAPS
# ---------------------------------------------------------------------------

LEAPS_MIN_VIX_IDEAL = 20.0
LEAPS_MIN_VIX_FLOOR = 15.0
LEAPS_DELTA = 0.70
LEAPS_MIN_DTE = 365
LEAPS_RSI_OVERSOLD = 30.0
LEAPS_MAX_ALLOC_PCT = 0.10
LEAPS_MAX_ALLOC_PCT_SMALL_ACCOUNT = 0.33   # if portfolio < $25K
LEAPS_SMALL_ACCOUNT_THRESHOLD = 25_000
LEAPS_EXIT_WEEK1_2_PROFIT_PCT = 0.10
LEAPS_EXIT_WEEK2_4_PROFIT_PCT = 0.20


# ---------------------------------------------------------------------------
# Section 17e: VERTICAL SPREADS
# ---------------------------------------------------------------------------

SPREAD_WIDTH_HIGH_PRICE_THRESHOLD = 500.0
SPREAD_WIDTH_HIGH = 10.0
SPREAD_WIDTH_LOW = 5.0


# ---------------------------------------------------------------------------
# Section 17g: SHORTING
# ---------------------------------------------------------------------------

SHORT_PUT_DTE_RANGE = (45, 60)
PROTECTIVE_PUT_OTM_RANGE = (0.10, 0.20)
PROTECTIVE_PUT_DELTA_RANGE = (0.10, 0.25)


# ---------------------------------------------------------------------------
# Section 19: PROFIT TARGETS (used by a separate position-monitor job, not
# the suggestion engine, since it requires knowing open positions)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProfitTargetRule:
    min_dte: int
    profit_pct: float = None
    close_at_price: float = None


PROFIT_TARGETS = (
    ProfitTargetRule(min_dte=21, profit_pct=0.50),
    ProfitTargetRule(min_dte=14, profit_pct=0.75),
    ProfitTargetRule(min_dte=0, close_at_price=0.05),
)


# ---------------------------------------------------------------------------
# Additions NOT in the source doc — recommended tightenings from the
# strategy review. All default OFF/permissive so "strict mode" reproduces
# the doc exactly; flip via config to opt into the stricter behavior.
# ---------------------------------------------------------------------------

STRICT_MODE = True  # if True, the IV-percentile / earnings-blackout filters below are ignored entirely

IV_PERCENTILE_MIN = 30.0            # skip premium-selling entries below this
EARNINGS_BLACKOUT_DAYS = 3          # skip new defined-risk entries this close to earnings
WHEEL_INVALIDATION_BREAK_200SMA = True  # exit recovery-mode wheel on a confirmed 200 SMA breakdown


# ---------------------------------------------------------------------------
# Quality filters — NOT in the source doc. Added on top of Section 12's
# minimal P/E + FCF + EPS-beat check, per user request for more rigor,
# following common "quality factor" stock-screening heuristics (the kind
# used in quality/value screens à la Buffett-style or CANSLIM-adjacent
# criteria): profitable relative to equity, not overloaded on debt,
# growing rather than shrinking, and able to cover near-term obligations.
#
# These ARE enabled by default (unlike the IV/earnings-blackout additions
# above, which stay off under STRICT_MODE) because they were explicitly
# requested. Toggle QUALITY_FILTERS_ENABLED to fall back to doc-only rigor.
# ---------------------------------------------------------------------------

QUALITY_FILTERS_ENABLED = True

MAX_DEBT_TO_EQUITY = 200.0   # yfinance reports this as a percent (200 = 2.0x equity)
# Financials and Utilities structurally run much higher D/E as part of their
# business model (banks/insurers hold debt-like liabilities, utilities are
# capital-intensive with regulated returns) — a fail here for those sectors
# is downgraded to a warning rather than a hard disqualifier.
DEBT_TO_EQUITY_SECTOR_EXCEPTIONS = ("Financial Services", "Financials", "Utilities")

MIN_RETURN_ON_EQUITY = 0.15   # 15% ROE — common quality-factor bar
MIN_REVENUE_GROWTH_YOY = 0.0  # non-negative YoY revenue growth (not shrinking)
MIN_CURRENT_RATIO = 1.0       # current assets cover current liabilities

# Analyst consensus is lagging/crowd-following, not a "quality" signal in
# the same sense as the above — kept as a soft/warning-only check, never a
# hard fail, so it never silently overrides the doc's own criteria.
ANALYST_RECOMMENDATION_WARN_ABOVE = 3.0  # yfinance scale: 1=Strong Buy ... 5=Sell
