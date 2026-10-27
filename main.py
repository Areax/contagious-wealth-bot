#!/usr/bin/env python3
"""
CLI entry point.

    python main.py screen --limit 30
    python main.py suggest --portfolio-size 50000 --limit 30
    python main.py suggest --portfolio-size 50000 --symbols AAPL,MSFT,KO

This tool NEVER places orders. Every output is a suggestion for you to
review and execute yourself (in thinkorswim or elsewhere).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from data.universe import get_sp500_symbols
from engine.suggest import run_suggestions

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def cmd_ai_suggest(args):
    from ai_suggest.suggest_tickers import get_candidates, save_candidates

    candidates = get_candidates(count=args.count, themes_hint=args.theme)
    for c in candidates:
        print(f"{c.symbol:6s} [{c.sector}] {c.theme} — {c.reason}")
    path = save_candidates(candidates)
    print(f"\nSaved {len(candidates)} candidate(s): {path}")
    print("Latest pointer: cache/ai_candidates_latest.json (used by `scorecard --from-ai`)")


def cmd_web(args):
    from web.app import run_dashboard

    limit = None if (args.limit and str(args.limit).lower() == "all") else args.limit
    print(f"Starting dashboard at http://127.0.0.1:{args.port}  (first run scans the first {limit or 50} S&P 500 stocks; after that it resumes your last scan. Change anything in the page once it loads.)")
    run_dashboard(port=args.port, initial_limit=limit)


def cmd_scorecard(args):
    from engine.scorecard import build_scorecard, format_table

    ai_meta = None
    if args.from_ai:
        from ai_suggest.suggest_tickers import load_latest_candidates

        cands = load_latest_candidates()
        symbols = [c.symbol for c in cands]
        ai_meta = {c.symbol: (c.theme, c.reason) for c in cands}
    elif args.symbols:
        symbols = args.symbols.split(",")
    else:
        symbols = get_sp500_symbols()[: args.limit or 30]

    rows = build_scorecard(symbols, ai_meta=ai_meta)
    print()
    print(format_table(rows))

    if ai_meta:
        print("\nAI rationale:")
        for r in rows:
            if r.ai_reason:
                print(f"  {r.symbol}: [{r.ai_theme}] {r.ai_reason}")


def cmd_screen(args):
    from engine.screener import screen_universe

    symbols = args.symbols.split(",") if args.symbols else get_sp500_symbols()[: args.limit or None]
    results = screen_universe(symbols)
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    print(f"\n{len(passed)}/{len(results)} passed Section 12 screen.\n")
    for r in passed:
        warn_str = f"  (warnings: {'; '.join(r.warnings)})" if r.warnings else ""
        print(f"  PASS  {r.symbol}{warn_str}")
    if args.verbose:
        print()
        for r in failed:
            print(f"  FAIL  {r.symbol}: {'; '.join(r.reasons_failed)}")


def cmd_suggest(args):
    symbols = args.symbols.split(",") if args.symbols else get_sp500_symbols()[: args.limit or None]
    run = run_suggestions(symbols, portfolio_size=args.portfolio_size)

    print(f"\nVIX: {run.vix_level:.2f}  ({run.sizing.vix_tier_label})")
    print(f"Portfolio tier: {run.sizing.portfolio_tier_label}")
    print(
        f"Capital deployed target: {run.sizing.capital_deployed_pct_range[0]}-"
        f"{run.sizing.capital_deployed_pct_range[1]}% "
        f"(cash reserved {run.sizing.cash_reserved_pct_range[0]}-{run.sizing.cash_reserved_pct_range[1]}%)"
    )
    print(f"Wheel bucket: ${run.sizing.wheel_bucket_dollars[0]:,.0f}-${run.sizing.wheel_bucket_dollars[1]:,.0f}")
    print(f"LEAPS bucket: ${run.sizing.leaps_bucket_dollars[0]:,.0f}-${run.sizing.leaps_bucket_dollars[1]:,.0f}")
    print(f"Spreads bucket: ${run.sizing.spreads_bucket_dollars[0]:,.0f}-${run.sizing.spreads_bucket_dollars[1]:,.0f}")
    print(f"Max per stock: ${run.sizing.max_per_stock_dollars:,.0f}\n")

    screened_passed = sum(1 for r in run.screened if r.passed)
    print(f"{screened_passed}/{len(run.screened)} symbols passed Section 12 screen.")
    print(f"{len(run.suggestions)} trade suggestion(s) generated.\n")

    for s in run.suggestions:
        flag = "OK" if s.meets_all_doc_criteria else "PARTIAL"
        print(f"[{flag}] {s.doc_section}  {s.action}")
        print(f"        contracts: {s.suggested_contracts}  capital: ${s.suggested_capital:,.0f}")
        for r in s.rationale:
            print(f"        - {r}")
        for w in s.warnings:
            print(f"        ! {w}")
        print()

    if args.save_json:
        REPORTS_DIR.mkdir(exist_ok=True)
        out_path = REPORTS_DIR / f"suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "generated_at": datetime.now().isoformat(),
            "vix": run.vix_level,
            "sizing": {
                "vix_tier": run.sizing.vix_tier_label,
                "portfolio_tier": run.sizing.portfolio_tier_label,
                "max_per_stock_dollars": run.sizing.max_per_stock_dollars,
            },
            "suggestions": [
                {
                    "symbol": s.symbol,
                    "strategy": s.strategy,
                    "doc_section": s.doc_section,
                    "action": s.action,
                    "contracts": s.suggested_contracts,
                    "capital": s.suggested_capital,
                    "meets_all_doc_criteria": s.meets_all_doc_criteria,
                    "rationale": s.rationale,
                    "warnings": s.warnings,
                }
                for s in run.suggestions
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Contagious Wealth deterministic suggestion engine (suggest-only, never trades).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_screen = sub.add_parser("screen", help="Run the Section 12 watchlist screen over the S&P 500 (or a subset).")
    p_screen.add_argument("--symbols", type=str, default=None, help="Comma-separated tickers instead of the full S&P 500.")
    p_screen.add_argument("--limit", type=int, default=None, help="Only scan the first N S&P 500 symbols (for speed while testing).")
    p_screen.add_argument("--verbose", action="store_true", help="Also print failures and why.")
    p_screen.set_defaults(func=cmd_screen)

    p_suggest = sub.add_parser("suggest", help="Screen + generate CSP/covered-call/LEAPS suggestions.")
    p_suggest.add_argument("--portfolio-size", type=float, required=True, help="Your total portfolio size in dollars, for VIX/tier sizing.")
    p_suggest.add_argument("--symbols", type=str, default=None, help="Comma-separated tickers instead of the full S&P 500.")
    p_suggest.add_argument("--limit", type=int, default=None, help="Only scan the first N S&P 500 symbols (for speed while testing).")
    p_suggest.add_argument("--save-json", action="store_true", help="Save the suggestion report to reports/.")
    p_suggest.set_defaults(func=cmd_suggest)

    p_ai = sub.add_parser("ai-suggest", help="Step 1: LLM proposes S&P 500 ticker candidates (writes cache/ai_candidates_latest.json). Requires ANTHROPIC_API_KEY.")
    p_ai.add_argument("--count", type=int, default=15, help="How many candidate tickers to request.")
    p_ai.add_argument("--theme", type=str, default=None, help="Optional free-text focus for this run, e.g. 'defensive sectors' or 'AI infrastructure'.")
    p_ai.set_defaults(func=cmd_ai_suggest)

    p_score = sub.add_parser("scorecard", help="Step 2: checklist + -0.20-delta-put premium ranking for a set of tickers.")
    p_score.add_argument("--from-ai", action="store_true", help="Use the latest AI-suggested candidates (cache/ai_candidates_latest.json).")
    p_score.add_argument("--symbols", type=str, default=None, help="Comma-separated tickers instead of AI candidates.")
    p_score.add_argument("--limit", type=int, default=None, help="If neither --from-ai nor --symbols given, scan the first N S&P 500 symbols (default 30).")
    p_score.set_defaults(func=cmd_scorecard)

    p_web = sub.add_parser("web", help="Local sortable dashboard of cash-secured-put candidates, ranked by RoR%.")
    p_web.add_argument("--port", type=int, default=8899)
    p_web.add_argument("--limit", type=int, default=None, help="On a first-ever run, scan the first N S&P 500 symbols (default: 50). Ignored once you have a saved last scan.")
    p_web.set_defaults(func=cmd_web)

    args = parser.parse_args()
    try:
        args.func(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
