"""
Mode B, Step 1: AI ticker-suggestion layer.

This is the ONLY module in the whole project allowed to call an LLM. It
runs as a separate, offline/batch process — never inside the deterministic
engine's scan loop (engine/suggest.py, engine/scorecard.py never import
this module or anthropic). It does one job: propose a shortlist of S&P 500
tickers worth running through the mechanical Section 12 screen, with a
short qualitative reason for each (sector narrative, "products people
use", thematic idea) — the kind of judgment call Section 12 criterion 4
("personal connection") explicitly leaves to a human/qualitative pass.

It NEVER decides deltas, strikes, position size, or whether to trade —
that's 100% engine/. It also can't invent tickers: every symbol returned is
checked against the real S&P 500 universe and dropped (with a warning) if
it doesn't match, since an LLM can hallucinate or suggest a delisted/
non-index name.

Requires ANTHROPIC_API_KEY in the environment. Get one at
https://console.anthropic.com/ — this is a separate key from your Claude
Code session; the deterministic engine works fully without it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from data.universe import get_sp500_universe

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
LATEST_PATH = CACHE_DIR / "ai_candidates_latest.json"

DEFAULT_MODEL = os.environ.get("AI_SUGGEST_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = """You help build a candidate shortlist for a mechanical options-income \
screening tool. The tool itself enforces all hard numeric rules (P/E, free cash flow, \
uptrend confirmation, option delta/return-on-risk) - your job is narrower: propose \
tickers worth spending API calls to check, based on qualitative judgment the mechanical \
screen can't make:
- sector/theme narratives currently in favor (without predicting short-term price moves)
- companies with broad consumer/business familiarity ("products people use")
- diversification across sectors, not a pile of the same 3 mega-caps every run

You are NOT deciding whether to trade anything. You are NOT checking technicals or \
fundamentals yourself - assume the caller will verify everything mechanically afterward. \
Do not recommend a ticker based on a specific news event, earnings prediction, or price \
target; that's exactly the kind of thing the mechanical layer (and the human) should \
verify with real data, not take on your say-so.
"""


@dataclass
class TickerCandidate:
    symbol: str
    sector: str
    theme: str
    reason: str


PROPOSE_TOOL = {
    "name": "propose_tickers",
    "description": "Propose a shortlist of S&P 500 tickers as screening candidates.",
    "input_schema": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Ticker symbol, must be an S&P 500 constituent"},
                        "theme": {"type": "string", "description": "Short theme/narrative tag, e.g. 'AI infrastructure', 'consumer staples defensiveness'"},
                        "reason": {"type": "string", "description": "1-2 sentence qualitative reason, no price predictions or earnings forecasts"},
                    },
                    "required": ["symbol", "theme", "reason"],
                },
            }
        },
        "required": ["candidates"],
    },
}


def _build_user_prompt(universe: list[dict], count: int, themes_hint: str | None) -> str:
    sectors = sorted(set(r["sector"] for r in universe))
    lines = [
        f"Propose {count} S&P 500 tickers as screening candidates for an options-income "
        "strategy (cash-secured puts / covered calls / LEAPS on the underlying).",
        "",
        f"Available sectors in the universe: {', '.join(sectors)}.",
        "Spread candidates across at least 4 different sectors unless you have a strong reason not to.",
    ]
    if themes_hint:
        lines.append(f"\nUser-supplied focus/theme for this run: {themes_hint}")
    lines.append(
        "\nUse the propose_tickers tool to return your answer. Every symbol must be one of "
        "the S&P 500 tickers provided in the system context — do not invent or suggest "
        "delisted/non-index symbols."
    )
    return "\n".join(lines)


def get_candidates(
    count: int = 15,
    themes_hint: str | None = None,
    model: str = DEFAULT_MODEL,
) -> list[TickerCandidate]:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("pip install anthropic (see requirements.txt)") from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. This is the one module in the project that "
            "calls an LLM — get a key at https://console.anthropic.com/ and "
            "`export ANTHROPIC_API_KEY=...` before running ai-suggest."
        )

    universe = get_sp500_universe()
    valid_symbols = {r["symbol"] for r in universe}
    symbol_to_sector = {r["symbol"]: r["sector"] for r in universe}

    universe_listing = "\n".join(f"{r['symbol']} ({r['sector']}): {r['name']}" for r in universe)

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT + "\n\nS&P 500 universe (symbol (sector): name):\n" + universe_listing,
        tools=[PROPOSE_TOOL],
        tool_choice={"type": "tool", "name": "propose_tickers"},
        messages=[{"role": "user", "content": _build_user_prompt(universe, count, themes_hint)}],
    )

    tool_use_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use_block is None:
        raise RuntimeError(f"Model did not return a tool_use block: {response.content}")

    raw_candidates = tool_use_block.input.get("candidates", [])

    candidates: list[TickerCandidate] = []
    for c in raw_candidates:
        symbol = c.get("symbol", "").strip().upper()
        if symbol not in valid_symbols:
            print(f"[ai_suggest] dropping non-S&P-500 or hallucinated symbol: {symbol!r}")
            continue
        candidates.append(TickerCandidate(
            symbol=symbol,
            sector=symbol_to_sector.get(symbol, "unknown"),
            theme=c.get("theme", ""),
            reason=c.get("reason", ""),
        ))

    return candidates


def save_candidates(candidates: list[TickerCandidate]) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "candidates": [asdict(c) for c in candidates],
    }
    timestamped = CACHE_DIR / f"ai_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    timestamped.write_text(json.dumps(payload, indent=2))
    LATEST_PATH.write_text(json.dumps(payload, indent=2))
    return timestamped


def load_latest_candidates() -> list[TickerCandidate]:
    if not LATEST_PATH.exists():
        raise FileNotFoundError(
            f"No AI candidates found at {LATEST_PATH}. Run `python main.py ai-suggest` first."
        )
    payload = json.loads(LATEST_PATH.read_text())
    return [TickerCandidate(**c) for c in payload["candidates"]]


if __name__ == "__main__":
    cands = get_candidates(count=10)
    for c in cands:
        print(f"{c.symbol:6s} [{c.sector}] {c.theme} — {c.reason}")
    path = save_candidates(cands)
    print(f"\nSaved: {path}")
