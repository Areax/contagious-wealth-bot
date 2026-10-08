"""
Named, timestamped snapshots of a full-universe screen. Clicking "Create
New Universe" (while in some mode) runs a full Russell 3000 scan under
that mode's live criteria (uptrend + vol floor, and for Strict, hard
fundamentals gates too) and saves the SURVIVING symbol list as a new
entry here — e.g. "127 Symbols from Russell 3000 filtered 9/25".

From that point on it behaves exactly like any other universe (S&P 500,
Russell 2000, Russell 3000) in the Universe dropdown: a flat, named list
of tickers you can pick and scan. It's deliberately decoupled from
whichever mode it was generated under — re-scanning it later still
re-applies whatever mode is CURRENTLY selected as a live filter on top,
the same way any universe works. The point of generating one is just to
start from a much smaller, pre-filtered pool instead of re-screening the
full ~2600-symbol Russell 3000 every time.

Never overwritten in place. Each "Create New Universe" click adds a new,
independently removable entry, so an old snapshot can sit alongside a
fresh one for comparison, or just get deleted when it's no longer wanted.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from config import strategy_rules as rules
from data.universe import get_russell_symbols
from engine.csp_dashboard import scan_universe

STORE_DIR = Path(__file__).resolve().parent.parent / "cache" / "generated_universes"

# What "under mode X" means for the discovery scan — mirrors the live
# filters each mode already applies on any scan (see web/app.py).
_MODE_CRITERIA = {
    "general": {"min_realized_vol": None, "require_fundamentals_gate": False},
    "options": {"min_realized_vol": rules.OPTIONS_MODE_MIN_REALIZED_VOL, "require_fundamentals_gate": False},
    "curated": {"min_realized_vol": rules.CURATED_MODE_MIN_REALIZED_VOL, "require_fundamentals_gate": True},
}


def create_universe(mode: str, progress_cb=None, cancel_event=None) -> dict:
    """Runs the expensive full-Russell-3000 discovery scan and persists a
    new universe record. Returns the full record (including symbols).
    cancel_event: threading.Event | None — see scan_universe."""
    criteria = _MODE_CRITERIA[mode]
    source_symbols = get_russell_symbols("3000")
    rows, removed_uptrend, removed_vol, removed_fundamentals = scan_universe(
        source_symbols,
        progress_cb=progress_cb,
        require_uptrend=True,
        min_realized_vol=criteria["min_realized_vol"],
        require_curated_fundamentals=criteria["require_fundamentals_gate"],
        max_workers=10,
        cancel_event=cancel_event,
    )
    symbols = sorted(r.symbol for r in rows if r.error is None)
    uid = uuid.uuid4().hex[:10]
    date_str = time.strftime("%m/%d", time.localtime()).lstrip("0").replace("/0", "/")
    record = {
        "id": uid,
        "name": f"{len(symbols)} Symbols from Russell 3000 filtered {date_str}",
        "symbols": symbols,
        "generated_at": time.time(),
        "generated_under_mode": mode,
        "source_universe_size": len(source_symbols),
        "removed_no_uptrend": removed_uptrend,
        "removed_low_vol": removed_vol,
        "removed_failed_fundamentals": removed_fundamentals,
        "errors": len(rows) - len(symbols),
    }
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    (STORE_DIR / f"{uid}.json").write_text(json.dumps(record, indent=2))
    return record


def list_universes() -> list[dict]:
    """Lightweight metadata only (no symbol lists) — newest first."""
    if not STORE_DIR.exists():
        return []
    records = []
    for f in STORE_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        records.append({
            "id": data["id"],
            "name": data["name"],
            "size": len(data["symbols"]),
            "generated_at": data["generated_at"],
            "generated_under_mode": data.get("generated_under_mode"),
        })
    records.sort(key=lambda r: r["generated_at"], reverse=True)
    return records


def get_universe_symbols(uid: str) -> list[str] | None:
    path = STORE_DIR / f"{uid}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["symbols"]


def delete_universe(uid: str) -> bool:
    path = STORE_DIR / f"{uid}.json"
    if not path.exists():
        return False
    path.unlink()
    return True
