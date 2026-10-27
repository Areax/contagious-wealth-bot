"""
Local dashboard for cash-secured-put candidates, sortable by RoR% (the
premium yield of the put nearest-or-above -0.20 delta). No LLM, no
brokerage account, no order placement — read-only view over
engine/csp_dashboard.py.

Run via: python main.py web --port 8899  (add --limit N to scan fewer than the full list on startup)
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request

from config import strategy_rules as rules
from data.market_data import get_quote
from data.universe import get_russell_symbols, get_sp500_symbols
from engine.csp_dashboard import fetch_option_for_symbol, row_to_dict, scan_universe
from data.sector_etfs import SECTOR_ETFS, get_top_holdings
from engine.patterns import analyze as analyze_patterns
from data.market_data import get_price_histories_batch
from engine.generated_universes import (
    create_universe,
    delete_universe,
    get_universe_symbols,
    list_universes,
)

app = Flask(__name__)

# Remembers the last scan you actually ran (mode/universe/limit/custom
# list) so a server restart resumes where you left off instead of always
# falling back to a hardcoded default — a restart used to silently show
# stale/wrong data (e.g. a leftover tiny test scan) until you noticed and
# re-ran it yourself.
LAST_SCAN_CONFIG_PATH = Path(__file__).resolve().parent.parent / "cache" / "last_scan_config.json"


def _load_last_scan_config() -> dict:
    if LAST_SCAN_CONFIG_PATH.exists():
        try:
            return json.loads(LAST_SCAN_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_last_scan_config(limit, custom_symbols, universe, mode):
    try:
        LAST_SCAN_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LAST_SCAN_CONFIG_PATH.write_text(json.dumps({
            "limit": limit, "custom_symbols": custom_symbols, "universe": universe, "mode": mode,
        }))
    except OSError:
        pass  # best-effort — losing the "remember last config" convenience is not worth crashing a scan over

UNIVERSE_LOADERS = {
    "sp500": get_sp500_symbols,
    "russell2000": lambda: get_russell_symbols("2000"),
    "russell3000": lambda: get_russell_symbols("3000"),
}

# "Screening mode" per user request: which columns matter and how
# aggressively to filter depends on whether you're hunting options-income
# candidates or something else (e.g. a future swing-trading mode). Today
# only "general" (no extra hard filter — today's long-standing behavior)
# and "options" (adds the realized-vol floor below, on top of the existing
# uptrend filter) are implemented; the mode concept is deliberately generic
# so a third mode can be added later without restructuring this.
MODE_MIN_REALIZED_VOL = {
    "general": None,
    "options": rules.OPTIONS_MODE_MIN_REALIZED_VOL,
    "curated": rules.CURATED_MODE_MIN_REALIZED_VOL,
}

# Curated mode additionally hard-gates on fundamentals (profitability,
# cash runway, D/E, ROE — see engine.checks.passes_curated_quality_gate)
# on top of the vol floor above. General/Options modes leave those checks
# informational-only, same as before.
MODE_REQUIRE_CURATED_FUNDAMENTALS = {
    "general": False,
    "options": False,
    "curated": True,
}

_state_lock = threading.Lock()
_state = {
    "status": "idle",        # idle | scanning | creating_universe | done | error
    "progress": 0,
    "total": 0,
    "current_symbol": None,
    "generated_at": None,
    "rows": [],
    "removed_no_uptrend": 0,
    "removed_low_vol": 0,
    "removed_failed_fundamentals": 0,
    "error": None,
    "requested_limit": None,
    "custom_symbols": None,
    "universe": "russell3000",
    "mode": "options",
    "last_created_universe": None,  # {id, name, size} — set once, right after Create New Universe finishes
    "append_mode": False,  # True only while/just after an "Add to current list" run — see _run_scan(append=...)
    "removed_reasons": {},   # symbol -> plain-English reason the last full scan dropped it (for "where is X?")
    "scanned_symbols": [],   # every symbol the last full scan looked at
    "base_total": 0,  # symbols in the last full (non-append) scan; total/progress get reused by small add-ticker runs
    "last_append_added": 0,  # how many of the requested symbols actually passed the live gate and got merged in
}


_BUSY_STATUSES = {"scanning", "creating_universe"}

# Sector Signals tab: its own small state, so it never blocks or clobbers
# the put finder. Results are also written to disk so a restart still shows
# the last scan.
SECTOR_CACHE_PATH = Path(__file__).resolve().parent.parent / "cache" / "sector_signals.json"
_sector_lock = threading.Lock()
_sector_state = {"status": "idle", "phase": "", "progress": 0, "total": len(SECTOR_ETFS), "error": None, "generated_at": None, "results": []}


def _load_sector_cache():
    if SECTOR_CACHE_PATH.exists():
        try:
            data = json.loads(SECTOR_CACHE_PATH.read_text())
            _sector_state["results"] = data.get("results", [])
            _sector_state["generated_at"] = data.get("generated_at")
            _sector_state["status"] = "done" if _sector_state["results"] else "idle"
        except (json.JSONDecodeError, OSError):
            pass


def _run_sector_scan():
    with _sector_lock:
        if _sector_state["status"] == "scanning":
            return
        _sector_state.update(status="scanning", phase="prices", progress=0, error=None)
    try:
        holdings = {etf: get_top_holdings(etf) for etf, _ in SECTOR_ETFS}
        symbols = sorted({etf for etf, _ in SECTOR_ETFS} | {h["symbol"] for hs in holdings.values() for h in hs})
        hist = get_price_histories_batch(symbols, period="1y")
        with _sector_lock:
            _sector_state["phase"] = "patterns"

        def one(symbol, name, weight=None):
            df = hist.get(symbol)
            if df is None or df.empty:
                return {"symbol": symbol, "name": name, "weight": weight, "error": "no price history", "signals": []}
            r = analyze_patterns(df)
            r.update(symbol=symbol, name=name, weight=weight)
            r.setdefault("error", None)
            return r

        results = []
        for i, (etf, sector) in enumerate(SECTOR_ETFS):
            entry = {"etf": etf, "sector": sector, "fund": one(etf, f"{sector} sector fund"),
                     "holdings": [one(h["symbol"], h["name"], h["weight"]) for h in holdings[etf]]}
            results.append(entry)
            with _sector_lock:
                _sector_state["progress"] = i + 1
        generated_at = datetime.now().isoformat()
        with _sector_lock:
            _sector_state.update(status="done", results=results, generated_at=generated_at)
        try:
            SECTOR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            SECTOR_CACHE_PATH.write_text(json.dumps({"results": results, "generated_at": generated_at}))
        except OSError:
            pass
    except Exception as exc:
        with _sector_lock:
            _sector_state.update(status="error", error=str(exc))

# Shared by both scan and create-universe background threads — a user-
# requested cancel (POST /api/cancel) sets this, and scan_universe (plus
# the price-history batch fetch it calls) checks it at several points, not
# just at the very end, so it actually takes effect during a large/slow
# pull instead of only stopping the NEXT one. Cleared at the start of every
# new run.
_cancel_event = threading.Event()


def _run_scan(
    limit: int | None, custom_symbols: list[str] | None = None, universe: str = "russell3000",
    mode: str = "options", append: bool = False,
):
    """append=True is the "Add to current list" path: scans ONLY
    custom_symbols and merges the results into the existing _state["rows"]
    (by symbol) instead of replacing them, and never touches
    universe/custom_symbols/last-scan-config — so the addition is purely
    client-visible-until-the-next-real-scan, never promoted to "the current
    list" the way a real scan or universe would be."""
    with _state_lock:
        if _state["status"] in _BUSY_STATUSES:
            return
        _cancel_event.clear()
        _state["status"] = "scanning"
        _state["progress"] = 0
        _state["error"] = None
        _state["append_mode"] = append
        if not append:
            _state["requested_limit"] = limit
            _state["custom_symbols"] = custom_symbols
            _state["universe"] = universe
            _state["mode"] = mode

    if not append:
        _save_last_scan_config(limit, custom_symbols, universe, mode)

    try:
        if custom_symbols:
            symbols = custom_symbols
        elif universe.startswith("generated:"):
            # A previously "Create New Universe"-generated list (see
            # engine/generated_universes.py) — just a flat, named symbol
            # pool at this point, same as any other universe. Whatever
            # mode is selected NOW still applies its own live uptrend/vol/
            # fundamentals filter on top, same as for any universe; it
            # doesn't have to match the mode this list was generated under.
            uid = universe.split(":", 1)[1]
            symbols = get_universe_symbols(uid)
            if symbols is None:
                raise RuntimeError(f"generated universe {uid!r} no longer exists (was it deleted?)")
        else:
            loader = UNIVERSE_LOADERS.get(universe, get_sp500_symbols)
            symbols = loader()
            if limit:
                symbols = symbols[:limit]

        # Set immediately, not just inside progress_cb below — otherwise
        # /api/status reports the PREVIOUS scan's total (e.g. "0/50" for a
        # newly-started full-index scan) for the several seconds it takes
        # to do the initial batch price-history fetch, before the first
        # progress callback ever fires.
        with _state_lock:
            _state["total"] = len(symbols)
            if not append:
                _state["base_total"] = len(symbols)

        def progress_cb(i, total, sym):
            with _state_lock:
                _state["progress"] = i
                _state["total"] = total
                _state["current_symbol"] = sym

        min_realized_vol = MODE_MIN_REALIZED_VOL.get(mode)
        require_curated_fundamentals = MODE_REQUIRE_CURATED_FUNDAMENTALS.get(mode, False)
        rows, removed, removed_low_vol, removed_failed_fundamentals, removed_reasons = scan_universe(
            symbols, progress_cb=progress_cb, require_uptrend=True, min_realized_vol=min_realized_vol,
            require_curated_fundamentals=require_curated_fundamentals, cancel_event=_cancel_event,
            force_include=set(custom_symbols) if custom_symbols else None,
        )
        row_dicts = [row_to_dict(r) for r in rows]

        with _state_lock:
            if append:
                # Merge by symbol on top of whatever's already shown —
                # re-adding a symbol that's already there just refreshes
                # it. removed_*/generated_at describe the base list and
                # are deliberately left alone. Symbols that failed the
                # mode's live gate (uptrend/vol/fundamentals) simply won't
                # be in row_dicts — same as any other scan — so they're
                # silently not added; last_append_added reports the real
                # count so the UI doesn't claim more were added than were.
                merged = {r["symbol"]: r for r in _state["rows"]}
                for r in row_dicts:
                    merged[r["symbol"]] = r
                _state["rows"] = list(merged.values())
                _state["last_append_added"] = len(row_dicts)
            else:
                _state["rows"] = row_dicts
                _state["removed_reasons"] = removed_reasons
                _state["scanned_symbols"] = list(symbols)
                _state["removed_no_uptrend"] = removed
                _state["removed_low_vol"] = removed_low_vol
                _state["removed_failed_fundamentals"] = removed_failed_fundamentals
                _state["generated_at"] = datetime.now().isoformat()
            _state["status"] = "cancelled" if _cancel_event.is_set() else "done"
            _state["current_symbol"] = None
    except Exception as exc:
        with _state_lock:
            _state["status"] = "error"
            _state["error"] = str(exc)


@app.route("/api/scan", methods=["POST"])
def api_scan():
    limit_raw = request.args.get("limit", "50")
    limit = None if limit_raw.lower() == "all" else int(limit_raw)
    universe = request.args.get("universe", "sp500")
    if universe not in UNIVERSE_LOADERS and not universe.startswith("generated:"):
        return jsonify({"ok": False, "message": f"unknown universe {universe!r}"}), 400
    mode = request.args.get("mode", "general")
    if mode not in MODE_MIN_REALIZED_VOL:
        return jsonify({"ok": False, "message": f"unknown mode {mode!r}"}), 400
    append = request.args.get("append") in ("1", "true", "yes")

    custom_symbols = None
    body = request.get_json(silent=True) or {}
    raw_list = body.get("symbols") or request.args.get("symbols")
    if raw_list:
        # Accept comma- and/or newline-separated, case-insensitive, dedup
        # while preserving order — lets you paste a watchlist straight in.
        parts = [p.strip().upper() for chunk in raw_list.splitlines() for p in chunk.split(",")]
        seen = set()
        custom_symbols = [p for p in parts if p and not (p in seen or seen.add(p))]
        if not custom_symbols:
            return jsonify({"ok": False, "message": "no valid symbols found in the list"}), 400
    if append and not custom_symbols:
        return jsonify({"ok": False, "message": "append requires a symbol list"}), 400

    with _state_lock:
        already_running = _state["status"] in _BUSY_STATUSES
    if already_running:
        return jsonify({"ok": False, "message": "scan already in progress"}), 409

    t = threading.Thread(target=_run_scan, args=(limit, custom_symbols, universe, mode, append), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/api/cancel", methods=["POST"])
def api_cancel():
    """Signals the currently-running scan or universe-create to stop as
    soon as it next checks (between price-history chunks, between symbols,
    or between thread-pool futures — see scan_universe) rather than only
    at the very end. A no-op, not an error, if nothing is running."""
    with _state_lock:
        running = _state["status"] in _BUSY_STATUSES
    if not running:
        return jsonify({"ok": True, "message": "nothing running"})
    _cancel_event.set()
    return jsonify({"ok": True})


@app.route("/api/universes", methods=["GET"])
def api_universes_list():
    return jsonify(list_universes())


def _run_create_universe(mode: str):
    with _state_lock:
        if _state["status"] in _BUSY_STATUSES:
            return
        _cancel_event.clear()
        _state["status"] = "creating_universe"
        _state["progress"] = 0
        _state["error"] = None
        _state["append_mode"] = False
        _state["last_created_universe"] = None

    def progress_cb(i, total, sym):
        with _state_lock:
            _state["progress"] = i
            _state["total"] = total
            _state["current_symbol"] = sym

    try:
        # A cancelled run still SAVES whatever it screened before the cancel
        # (rows for symbols never reached simply aren't in the result) —
        # a partial universe is more useful than nothing, but the status
        # below makes clear it's partial, not a full clean completion.
        record = create_universe(mode, progress_cb=progress_cb, cancel_event=_cancel_event)
        with _state_lock:
            _state["last_created_universe"] = {"id": record["id"], "name": record["name"], "size": len(record["symbols"])}
            _state["status"] = "cancelled" if _cancel_event.is_set() else "done"
            _state["current_symbol"] = None
    except Exception as exc:
        with _state_lock:
            _state["status"] = "error"
            _state["error"] = str(exc)


@app.route("/api/universes", methods=["POST"])
def api_universes_create():
    """Runs the full Russell 3000 discovery scan (~3 min) under the given
    mode's live criteria and saves the result as a new, named, independent
    universe — see engine/generated_universes.py. Never overwrites an
    existing one."""
    mode = request.args.get("mode", "options")
    if mode not in MODE_MIN_REALIZED_VOL:
        return jsonify({"ok": False, "message": f"unknown mode {mode!r}"}), 400

    with _state_lock:
        already_running = _state["status"] in _BUSY_STATUSES
    if already_running:
        return jsonify({"ok": False, "message": "scan already in progress"}), 409

    t = threading.Thread(target=_run_create_universe, args=(mode,), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/api/universes/<uid>", methods=["DELETE"])
def api_universes_delete(uid):
    ok = delete_universe(uid)
    if not ok:
        return jsonify({"ok": False, "message": f"universe {uid!r} not found"}), 404
    return jsonify({"ok": True})


@app.route("/api/option", methods=["POST"])
def api_option():
    """On-demand, single-symbol options lookup — the only endpoint that
    ever touches the options chain. Never called automatically; only from
    the per-row "Get Option" button (or a user-driven batch of them)."""
    symbol = (request.args.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"ok": False, "message": "symbol required"}), 400
    force = (request.args.get("force") or "").strip().lower() in ("1", "true", "yes")

    with _state_lock:
        row = next((r for r in _state["rows"] if r["symbol"] == symbol), None)
        # Already have a real result for this symbol — skip re-fetching
        # entirely rather than hit Yahoo again, UNLESS the caller explicitly
        # asked for a refresh (force=1, from the per-row refresh button or
        # the "Force refresh" bulk toggle — premiums move during the day,
        # so a result fetched hours ago can go stale). This isn't just an
        # efficiency shortcut: without `force`, two concurrent viewers (e.g.
        # two browser tabs) each running their own "fetch all" loop can race
        # to re-fetch the same already-successful symbol — and since a
        # later call unconditionally overwrites the row, a coincidental
        # non-match on the redundant fetch (quotes shift between the two
        # attempts) silently clobbers a perfectly good earlier result.
        # Confirmed happening live, 2026-09-25: two tabs' independent
        # auto-fetch loops racing against the same 127-symbol scan.
        if row is not None and row.get("options_fetched") and row.get("put") is not None and not force:
            return jsonify({"ok": True, "put": row["put"], "options_error": row.get("options_error"), "options_note": row.get("options_note")})
    if row is None:
        return jsonify({"ok": False, "message": f"{symbol} not in current results — run a scan first"}), 404
    if row.get("error"):
        return jsonify({"ok": False, "message": f"{symbol} has a scan error, can't fetch options for it: {row['error']}"}), 400

    # Live quote at click time: the scan's spot/day-change can be hours or
    # days old by the time options are fetched, and the put selection keys
    # off spot. Falls back to the scan-time spot if the quote call fails.
    spot, day_change_pct, quote_as_of = row["spot"], row.get("day_change_pct"), None
    try:
        quote = get_quote(symbol, fresh=True)
        if quote["last_price"] > 0:
            spot = quote["last_price"]
            prev_close = quote["prev_close"]
            day_change_pct = (spot - prev_close) / prev_close if prev_close > 0 else None
            quote_as_of = quote["fetched_at"]
    except Exception:
        pass

    put, error, note = fetch_option_for_symbol(symbol, spot)
    put_dict = None
    if put:
        put_dict = {
            "strike": put.contract.strike,
            "expiration": put.contract.expiration,
            "dte": put.contract.dte,
            "delta": put.computed_delta,
            "delta_abs": abs(put.computed_delta),
            "at_or_above_target_delta": put.at_or_above_target_delta,
            "is_fallback_pick": put.is_fallback_pick,
            "bid": put.contract.bid,
            "ask": put.contract.ask,
            "mid": put.contract.mid,
            "premium": put.premium,
            "capital_at_risk": put.capital_at_risk,
            "return_on_risk": put.return_on_risk,
            "return_on_risk_30d_normalized": put.return_on_risk_30d_normalized,
            "open_interest": put.contract.open_interest,
            "volume": put.contract.volume,
        }

    with _state_lock:
        for r in _state["rows"]:
            if r["symbol"] == symbol:
                r["put"] = put_dict
                r["options_fetched"] = True
                r["options_error"] = error
                r["options_note"] = note
                r["spot"] = spot
                r["day_change_pct"] = day_change_pct
                if quote_as_of:
                    r["quote_as_of"] = quote_as_of
                break

    return jsonify({
        "ok": True, "put": put_dict, "options_error": error, "options_note": note,
        "spot": spot, "day_change_pct": day_change_pct, "quote_as_of": quote_as_of,
    })


@app.route("/api/sectors/scan", methods=["POST"])
def api_sectors_scan():
    with _sector_lock:
        running = _sector_state["status"] == "scanning"
    if running:
        return jsonify({"ok": False, "message": "sector scan already running"}), 409
    threading.Thread(target=_run_sector_scan, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/sectors/status")
def api_sectors_status():
    with _sector_lock:
        return jsonify({k: _sector_state[k] for k in ("status", "phase", "progress", "total", "error", "generated_at")})


@app.route("/api/sectors")
def api_sectors():
    with _sector_lock:
        return jsonify({"generated_at": _sector_state["generated_at"], "results": _sector_state["results"]})


@app.route("/api/why")
def api_why():
    """Answers "where is stock X?" for a symbol that is NOT in the current
    rows: dropped by the last scan (with the reason), never in the scanned
    list, or unknown. The UI handles symbols that ARE in the rows itself."""
    symbol = (request.args.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"ok": False, "message": "symbol required"}), 400
    with _state_lock:
        if any(r["symbol"] == symbol for r in _state["rows"]):
            return jsonify({"ok": True, "status": "in_table"})
        reason = _state["removed_reasons"].get(symbol)
        if reason:
            return jsonify({"ok": True, "status": "removed", "reason": reason})
        scanned = _state["scanned_symbols"]
        if scanned and symbol not in scanned:
            return jsonify({"ok": True, "status": "not_in_list"})
    return jsonify({"ok": True, "status": "unknown"})


@app.route("/api/status")
def api_status():
    with _state_lock:
        return jsonify({
            "status": _state["status"],
            "progress": _state["progress"],
            "total": _state["total"],
            "current_symbol": _state["current_symbol"],
            "generated_at": _state["generated_at"],
            "row_count": len(_state["rows"]),
            "removed_no_uptrend": _state["removed_no_uptrend"],
            "removed_low_vol": _state["removed_low_vol"],
            "removed_failed_fundamentals": _state["removed_failed_fundamentals"],
            "error": _state["error"],
            "requested_limit": _state["requested_limit"],
            "custom_symbols_count": len(_state["custom_symbols"]) if _state["custom_symbols"] else None,
            "universe": _state["universe"],
            "mode": _state["mode"],
            "last_created_universe": _state["last_created_universe"],
            "append_mode": _state["append_mode"],
            "last_append_added": _state["last_append_added"],
            "base_total": _state["base_total"],
            "forced_count": sum(1 for r in _state["rows"] if r.get("forced")),
        })


@app.route("/api/data")
def api_data():
    with _state_lock:
        return jsonify({
            "generated_at": _state["generated_at"],
            "rows": _state["rows"],
        })


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Cash-Secured Put Finder</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; padding: 20px; background: #f7f7f8; color: #1a1a1a; }
  @media (prefers-color-scheme: dark) { body { background: #16171a; color: #e8e8ea; } }
  h1 { font-size: 20px; margin: 0 0 4px 0; }
  .sub { color: #888; font-size: 13px; margin-bottom: 6px; }
  .as-of { color: #555; font-size: 13px; font-weight: 600; margin-bottom: 16px; }
  @media (prefers-color-scheme: dark) { .as-of { color: #bbb; } }
  .controls { display: flex; gap: 10px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }
  .controls input, .controls select { padding: 6px 8px; border-radius: 6px; border: 1px solid #ccc; }
  .toggle { display: flex; align-items: center; gap: 4px; font-size: 13px; cursor: pointer; font-weight: normal; }
  .toggle input { padding: 0; width: auto; }
  #customSymbols { width: 320px; height: 32px; resize: vertical; font-family: inherit; font-size: 13px; }
  .get-option-btn { padding: 4px 10px; border-radius: 5px; border: 1px solid #2563eb; background: transparent; color: #2563eb; cursor: pointer; font-size: 12px; }
  .get-option-btn:hover { background: #2563eb; color: white; }
  .get-option-btn:disabled { opacity: 0.5; cursor: default; }
  @media (prefers-color-scheme: dark) { .get-option-btn { color: #7ab0ff; border-color: #7ab0ff; } .get-option-btn:hover { background: #7ab0ff; color: #111; } }
  .refresh-option-btn { padding: 0 4px; margin-left: 4px; border: none; background: transparent; color: #2563eb; cursor: pointer; font-size: 15px; line-height: 1; }
  .refresh-option-btn:hover { color: #2563eb; }
  .refresh-option-btn:disabled { opacity: 0.4; cursor: wait; }
  @media (prefers-color-scheme: dark) { .refresh-option-btn:hover { color: #7ab0ff; } }
  .controls button { padding: 7px 14px; border-radius: 6px; border: none; background: #2563eb; color: white; cursor: pointer; font-size: 13px; }
  .controls button:disabled { background: #93c5fd; cursor: default; }
  /* Tactile "pressed" feedback on every button, native :active — instant,
     needs no JS, works even for a click that turns out to be a no-op. */
  button:active:not(:disabled) { transform: scale(0.96); }
  button { transition: transform 0.06s ease-out; }
  /* "Actually doing something" feedback — a small spinner + dimmed label,
     toggled by JS the instant a button's action starts (setBusy/clearBusy
     below), not only once a response comes back. */
  button.busy { cursor: wait; opacity: 0.85; }
  .spinner { display: inline-block; width: 10px; height: 10px; margin-right: 6px; vertical-align: -1px; border: 2px solid rgba(255,255,255,0.4); border-top-color: #fff; border-radius: 50%; animation: spin 0.6s linear infinite; }
  .get-option-btn .spinner { border-color: rgba(37,99,235,0.35); border-top-color: #2563eb; }
  @media (prefers-color-scheme: dark) { .get-option-btn .spinner { border-color: rgba(122,176,255,0.35); border-top-color: #7ab0ff; } }
  @keyframes spin { to { transform: rotate(360deg); } }
  #status { font-size: 13px; color: #666; }
  table { border-collapse: collapse; width: 100%; font-size: 12px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-radius: 8px; overflow: hidden; }
  @media (prefers-color-scheme: dark) { table { background: #222327; } }
  th, td { padding: 6px 3px; text-align: right; border-bottom: 1px solid #eee; white-space: nowrap; }
  @media (prefers-color-scheme: dark) { th, td { border-bottom: 1px solid #333; } }
  th:first-child, td:first-child { text-align: left; }
  th { cursor: pointer; user-select: none; background: #fafafa; font-weight: 600; position: sticky; top: 0; border-bottom: 1px dotted #999; }
  th:hover { background: #eef2ff; }
  @media (prefers-color-scheme: dark) { th:hover { background: #2a2f45; } }
  @media (prefers-color-scheme: dark) { th { background: #1c1d21; } }
  th.sorted { color: #2563eb; }
  /* JS-positioned tooltip (not a CSS ::after anchored to the header) so it
     can be clamped to the viewport — a header-relative tooltip bleeds off
     the right edge for right-hand columns like OI/RoR%. See showTooltip(). */
  #tooltipBox {
    position: fixed;
    display: none;
    width: 260px;
    max-width: 60vw;
    white-space: normal;
    text-align: left;
    font-weight: normal;
    font-size: 12px;
    line-height: 1.4;
    color: #eee;
    background: #26272c;
    padding: 8px 10px;
    border-radius: 6px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35);
    z-index: 200;
    pointer-events: none;
  }
  tr:hover { background: #f0f4ff; }
  @media (prefers-color-scheme: dark) { tr:hover { background: #202634; } }
  .check-y { color: #16a34a; font-weight: bold; }
  .check-n { color: #dc2626; font-weight: bold; }
  .check-u { color: #999; }
  .fallback { color: #d97706; font-size: 11px; }
  .err { color: #dc2626; }
  .ror-cell { font-weight: 600; }
  .table-wrap { overflow-x: auto; }
  .scanbar { margin-bottom: 10px; }
  .scanbar .controls { margin-bottom: 4px; }
  .scanbar select { max-width: 460px; }
  details.more { font-size: 13px; margin-top: 6px; }
  details.more summary { cursor: pointer; color: #2563eb; }
  @media (prefers-color-scheme: dark) { details.more summary { color: #7ab0ff; } }
  details.more .controls { margin-top: 8px; margin-bottom: 4px; }
  .summary { max-width: 980px; margin: 0 0 10px 0; }
  .summary p { margin: 0 0 3px 0; }
  .whererow { font-size: 13px; display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .whererow input { padding: 4px 6px; border-radius: 6px; border: 1px solid #ccc; font-size: 13px; }
  .asof-line { font-size: 13px; color: #666; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  @media (prefers-color-scheme: dark) { .asof-line { color: #a0a0a8; } }
  button.small { padding: 4px 10px; border-radius: 6px; border: 1px solid #2563eb; background: transparent; color: #2563eb; cursor: pointer; font-size: 12px; }
  button.small:hover:not(:disabled) { background: #2563eb; color: white; }
  button.small:disabled { opacity: 0.6; cursor: default; }
  @media (prefers-color-scheme: dark) { button.small { color: #7ab0ff; border-color: #7ab0ff; } button.small:hover:not(:disabled) { background: #7ab0ff; color: #111; } }
  button.small.stop { color: #dc2626; border-color: #dc2626; }
  button.small.stop:hover:not(:disabled) { background: #dc2626; color: white; }
  button.small .spinner { border-color: rgba(37,99,235,0.35); border-top-color: #2563eb; }
  .field { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
  .scanbar .ownrow { border-top: 1px solid #e3e3e8; margin-top: 8px; padding-top: 10px; font-size: 13px; }
  @media (prefers-color-scheme: dark) { .scanbar .ownrow { border-top-color: #33343b; } }
  .low-oi { color: #d97706; }
  .sub { display: block; font-size: 10px; color: #888; line-height: 1.1; }
  .warn-cell { color: #d97706; }
  .warn-cell .sub { color: #d97706; }
  .co { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; font-size: 11px; font-weight: normal; color: #888; max-width: 100px; overflow: hidden; white-space: normal; line-height: 1.2; }
  td:first-child { font-weight: 600; }
  .ownrow textarea { width: 200px; height: 26px; min-height: 26px; }
  .ownrow select { padding: 5px 8px; border-radius: 6px; border: 1px solid #ccc; font-size: 13px; }
  .ownrow button { padding: 6px 14px; border-radius: 6px; border: none; background: #2563eb; color: white; cursor: pointer; font-size: 13px; }
  .ownrow button.warn { background: #d97706; }
  .ownrow button:disabled { background: #93c5fd; }
  .hint.inline { display: inline; margin: 0; }
  .filterbar { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; font-size: 13px; max-width: 980px; margin: 0 0 6px 0; }
  .filterlabel { font-weight: 600; }
  .filterbar details.columns { margin: 0; }
  .filterbar details.columns .controls { margin-top: 6px; }
  .tabs { display: flex; gap: 6px; margin-bottom: 14px; border-bottom: 1px solid #ddd; }
  @media (prefers-color-scheme: dark) { .tabs { border-bottom-color: #33343b; } }
  .tab { padding: 8px 16px; border: 1px solid transparent; border-bottom: none; background: transparent; color: #666; cursor: pointer; font-size: 14px; border-radius: 8px 8px 0 0; }
  .tab.active { background: white; color: #1a1a1a; border-color: #ddd; font-weight: 600; }
  @media (prefers-color-scheme: dark) { .tab { color: #a0a0a8; } .tab.active { background: #1e1f24; color: #e8e8ea; border-color: #33343b; } }
  .badge { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 11px; font-weight: 600; margin: 1px 4px 1px 0; white-space: nowrap; }
  .badge.bullish { background: #dcfce7; color: #166534; }
  .badge.bearish { background: #fee2e2; color: #991b1b; }
  .badge.watch { background: #fef3c7; color: #92400e; }
  @media (prefers-color-scheme: dark) { .badge.bullish { background: #14532d; color: #bbf7d0; } .badge.bearish { background: #7f1d1d; color: #fecaca; } .badge.watch { background: #78350f; color: #fde68a; } }
  .badge .st { font-weight: normal; opacity: 0.8; }
  tr.etf-row td { background: #eef2ff; font-weight: 600; }
  @media (prefers-color-scheme: dark) { tr.etf-row td { background: #1f2a44; } }
  #sectorTable td.sigs { text-align: left; white-space: normal; max-width: 200px; }
  #sectorTable td:first-child { max-width: 135px; white-space: normal; }
  #sectorTable .co { max-width: 125px; }
  .sig { display: block; line-height: 1.5; white-space: normal; }
  .sig .st { font-size: 10px; color: #888; margin-left: 3px; white-space: nowrap; }
  .sig .why { display: block; font-size: 10px; color: #888; white-space: normal; line-height: 1.3; margin-bottom: 3px; }
  #sectorTable th:first-child, #sectorTable td:first-child { position: static; }
  .lead { font-size: 15px; margin: 0 0 10px 0; max-width: 760px; }
  .hint { color: #666; font-size: 13px; margin: 6px 0 0 0; }
  @media (prefers-color-scheme: dark) { .hint { color: #a0a0a8; } }
  .card { background: white; border: 1px solid #e3e3e8; border-radius: 10px; padding: 14px 16px; margin-bottom: 14px; max-width: 980px; }
  @media (prefers-color-scheme: dark) { .card { background: #1e1f24; border-color: #33343b; } }
  .card h2 { font-size: 15px; margin: 0 0 6px 0; display: flex; align-items: center; gap: 8px; }
  .card .controls { margin-bottom: 8px; }
  .step { display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px; border-radius: 50%; background: #2563eb; color: white; font-size: 13px; }
  .glossary, .advanced, .columns { margin-bottom: 12px; font-size: 13px; max-width: 980px; }
  .glossary summary, .advanced summary, .columns summary { cursor: pointer; color: #2563eb; font-weight: 600; }
  @media (prefers-color-scheme: dark) { .glossary summary, .advanced summary, .columns summary { color: #7ab0ff; } }
  .glossary dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 14px; margin: 10px 0 0 0; }
  .glossary dt { font-weight: 600; }
  .glossary dd { margin: 0; color: #666; }
  @media (prefers-color-scheme: dark) { .glossary dd { color: #a0a0a8; } }
  .advanced { border: 1px dashed #c9c9d1; border-radius: 10px; padding: 10px 16px; }
  .advanced[open] summary { margin-bottom: 10px; }
  .columns { margin: 8px 0; }
  .columns .controls { margin-top: 8px; }
  .card h3 { font-size: 13px; margin: 14px 0 4px 0; }
  .listlabel { font-weight: 600; margin: 0 0 4px 0; font-size: 14px; }
  .cue { font-size: 13px; color: #b45309; font-weight: 600; margin: 4px 0 0 0; }
  .statusline { font-size: 13px; margin: 4px 0 0 0; color: #1a1a1a; }
  @media (prefers-color-scheme: dark) { .statusline { color: #e8e8ea; } }
  #status { color: #1a1a1a; font-weight: 500; }
  @media (prefers-color-scheme: dark) { #status { color: #e8e8ea; } }
  section.card.own { border-style: dashed; }
  .optional { font-weight: normal; color: #888; font-size: 13px; }
  .step.alt { background: #888; width: auto; padding: 0 8px; border-radius: 11px; }
  .radios { display: flex; flex-direction: column; gap: 4px; }
  .n { color: #888; font-size: 12px; }
  th:first-child, td:first-child { position: sticky; left: 0; z-index: 2; background: #fff; }
  @media (prefers-color-scheme: dark) { th:first-child, td:first-child { background: #222327; } }
  .group-row th { cursor: default; background: transparent; text-align: center; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; color: #2563eb; border-bottom: 2px solid #2563eb; position: static; }
  .group-row th:hover { background: transparent; }
  tr.divider td { text-align: left; background: #eef2ff; color: #1e3a8a; font-weight: 600; font-size: 12px; padding: 6px 10px; white-space: normal; }
  td[colspan="6"] { white-space: normal; }
  @media (prefers-color-scheme: dark) { tr.divider td { background: #1f2a44; color: #bcd0ff; } }
  .mine { font-size: 11px; color: #2563eb; }
  ul.key { margin: 8px 0 0 0; padding-left: 18px; }
  ul.key li { margin-bottom: 2px; }
  .subhead { font-size: 12px; font-weight: 600; color: #666; margin: 8px 0 2px 0; text-transform: uppercase; letter-spacing: 0.03em; }
  @media (prefers-color-scheme: dark) { .subhead { color: #a0a0a8; } }
  label.toggle.dim { opacity: 0.5; }
  @keyframes flash { 0%,100% { background: transparent; } 30% { background: #fde68a; } }
  tr.flashrow td { animation: flash 2s ease; }
  label.toggle.flash { animation: flash 1.6s ease; border-radius: 4px; }
  a.reason { color: inherit; text-decoration: underline dotted; cursor: pointer; }
  .rowcount { font-size: 13px; font-weight: 600; margin: 8px 0 0 0; }
  p.legend { max-width: 980px; }
  .legend { font-size: 12px; color: #666; margin: 8px 0 0 0; }
  @media (prefers-color-scheme: dark) { .legend { color: #a0a0a8; } }
  .controls button.primary { padding: 9px 22px; font-size: 14px; font-weight: 600; }
  .controls button.warn { background: #d97706; }
  .controls button.danger-fill { background: #dc2626; }
  .controls button.danger-outline { background: transparent; color: #dc2626; border: 1px solid #dc2626; }
  .controls button.danger-outline:disabled { background: transparent; color: #aaa; border-color: #ccc; opacity: 0.7; }
  .controls button.danger-outline:not(:disabled):hover { background: #dc2626; color: white; }
</style>
</head>
<body>
  <nav class="tabs">
    <button class="tab active" id="tabBtnPut" onclick="showTab('put')">Put finder</button>
    <button class="tab" id="tabBtnSectors" onclick="showTab('sectors')">Sector signals</button>
  </nav>
  <div id="tabPut">
  <h1>Cash-Secured Put Finder</h1>
  <p class="lead">Finds stocks whose price is rising and shows how much you would be paid to promise to buy them at a lower price. &ldquo;Cash-secured&rdquo; means you keep enough cash on hand to buy the shares if asked. Nothing here buys or sells anything.</p>
  <details class="glossary">
    <summary>New here? What do these words mean?</summary>
    <p class="subhead">The basics</p>
    <dl>
      <dt>Ticker</dt><dd>A stock's short code, like AAPL for Apple. The company name is shown under it.</dd>
      <dt>Stock list</dt><dd>The group of stocks the scan looks through, such as the S&amp;P 500.</dd>
      <dt>Scan</dt><dd>Checks every stock in the list and keeps the ones whose price is rising and that pass the strictness level.</dd>
      <dt>Strictness</dt><dd>How picky the scan is. Low keeps every rising stock. Medium also needs enough price movement to pay well. High also needs a healthy company.</dd>
      <dt>Filter</dt><dd>Hides rows from the table. It never changes what was scanned.</dd>
      <dt>Put</dt><dd>A contract where you get paid now for promising to buy 100 shares at a set price if the stock falls to it. &ldquo;Selling a put&rdquo; means making that promise.</dd>
      <dt>Strike price</dt><dd>The price you promise to buy at.</dd>
      <dt>You collect</dt><dd>What you are paid up front for making that promise (per 100 shares).</dd>
      <dt>Days left</dt><dd>How long the promise lasts.</dd>
      <dt>Chance of buying</dt><dd>Roughly the chance you end up having to buy the shares. 20% is the target: low enough that it rarely happens, high enough to still pay well.</dd>
      <dt>Return</dt><dd>What you collect, divided by the cash you set aside minus what you collect. Higher pays more. Not per year.</dd>
      <dt>Easy to trade?</dt><dd>How many of that exact contract exist (also called open interest). Under 10 means few people trade it, so getting a fair price can be hard.</dd>
      <dt>Cash you set aside</dt><dd>The strike price times 100. A $50 strike ties up $5,000 until the put ends.</dd>
      <dt>Uptrend</dt><dd>The price is above a rising 30-week average. Stocks that fail this are removed by the scan. Only tickers you add yourself can fail it.</dd>
    </dl>
    <p class="subhead">Extra columns (off unless you turn them on)</p>
    <dl>
      <dt>Volatility</dt><dd>How much the stock jumps around. More movement usually means you are paid more, and risk more.</dd>
      <dt>Price vs profit (P/E)</dt><dd>The stock price divided by yearly profit per share. Very high can mean expensive.</dd>
      <dt>Beat earnings?</dt><dd>Did the company's last report beat what analysts expected?</dd>
      <dt>Debt OK?</dt><dd>&#10003; means the company's debt is at a manageable level.</dd>
      <dt>Sales growing?</dt><dd>&#10003; means sales grew compared with a year ago.</dd>
      <dt>12-18 month trend?</dt><dd>A second, longer trend check. A stock can pass the main 30-week trend check and still show &#10007; here.</dd>
      <dt>Dip lately?</dt><dd>&#10003; means the stock just fell for a few days or dropped a lot in one day, which many people like as a time to sell a put. A dash just means no recent dip.</dd>
      <dt>Volatility rank</dt><dd>Is the stock moving more than usual for itself right now? 100 means the most active it has been all year.</dd>
      <dt>Shorted %</dt><dd>The share of the stock that traders have bet against. High numbers mean many people expect it to fall.</dd>
      <dt>Beta (market swings)</dt><dd>1.0 moves like the overall market. Higher swings more.</dd>
    </dl>
  </details>

  <section class="card scanbar">
    <div class="controls">
      <span class="field"><label for="universeSelect">Stock list:</label>
      <select id="universeSelect" onchange="onUniverseChange()">
        <option value="sp500" selected>S&amp;P 500 &mdash; 500 large US companies (scan about 1 min, then 1 to 3 min for put prices)</option>
        <option value="russell2000">Russell 2000 &mdash; about 2,000 small companies (slow: several minutes, then put prices)</option>
        <option value="russell3000">Russell 3000 &mdash; about 2,600 stocks (slow: several minutes, then put prices)</option>
        <optgroup id="generatedUniverseGroup" label="Saved lists"></optgroup>
      </select></span>
    </div>
    <p class="hint"><b>Stock list:</b> the group of stocks to scan. <span id="universeNote"></span></p>
    <p id="listChangeCue" class="cue" style="display:none;">You changed the stock list. Press Run scan to use it.</p>
    <div class="controls">
      <span class="field"><label for="modeSelect">Strictness:</label>
      <select id="modeSelect" onchange="applyModePreset()">
        <option value="general">Low &mdash; every stock whose price is rising</option>
        <option value="options" selected>Medium &mdash; only stocks that move enough to pay well</option>
        <option value="curated">High &mdash; only healthy companies that move enough (shortest list)</option>
      </select></span>
    </div>
    <p class="hint"><b>Strictness:</b> <span id="modeNote"></span> <span id="styleCount"></span> Changing it re-filters this table at once; the next scan uses it too.</p>
    <div class="controls">
      <button id="scanBtn" class="primary" onclick="startScan()">Run scan</button>
      <button id="cancelBtn" class="danger-fill" onclick="cancelRun()" style="display:none;">Cancel</button>
      <span id="status">idle</span>
    </div>
    <p class="hint"><b>Run scan:</b> checks every stock in the list (about a minute for the S&amp;P 500), keeps the rising ones that pass the strictness level, then loads put prices by itself (about 1.5 seconds per stock kept, usually 1 to 3 minutes). The line below says when everything is done.</p>
    <p id="fetchStatus" class="statusline"></p>
    <div class="controls ownrow">
      <span class="field"><label for="customSymbols">Or look up just these tickers:</label>
      <textarea id="customSymbols" placeholder="AAPL, MSFT, KO" rows="1"></textarea></span>
      <select id="customScanMode" onchange="updateCustomRunLabel()" title="Add: your tickers join the table. Replace: the table shows only your tickers until the next scan.">
        <option value="add" selected>Add to the table</option>
        <option value="replace">Replace the table with these</option>
      </select>
      <button id="customRunBtn" onclick="startCustomRun()">Run</button>
      <span class="hint inline">Always shown, even where a filter would hide them. Cleared by the next scan. Your stock lists are not changed.</span>
      <span id="customStatus" class="statusline"></span>
    </div>
    <details class="more">
      <summary>More scan options</summary>
      <div class="controls">
        <label for="limitSelect">How many to scan:</label>
        <select id="limitSelect" onchange="updateListChangeCue()">
          <option value="25">Only the first 25 (quick test)</option>
          <option value="50">Only the first 50</option>
          <option value="100">Only the first 100</option>
          <option value="all" selected>Every stock in the list</option>
        </select>
      </div>
      <p class="hint">Saved lists are shortcuts: stocks that already passed a full scan. You can make or delete them under Advanced, at the bottom of the page. (A stock list is sometimes called a &ldquo;universe&rdquo;.)</p>
    </details>
  </section>

  <div class="summary">
    <p id="listLabel" class="listlabel"></p>
    <p id="rowCount" class="rowcount"></p>
    <p class="whererow"><label for="whereInput">Where is a stock?</label> <input id="whereInput" placeholder="ticker, e.g. NVDA" size="12" onkeydown="if (event.key === 'Enter') whereIs()"> <button class="small" onclick="whereIs()">Find it</button> <span id="whereAnswer" class="hint inline"></span></p>
    <p class="asof-line">
      <span id="asOf">Put prices: &mdash;</span>
      <button id="fetchAllBtn" class="small" onclick="fetchAllOptions()">Update put prices</button>
      <button id="cancelFetchBtn" class="small stop" onclick="cancelRun()" style="display:none;">Stop</button>
      <label class="toggle" title="Ticked: every shown stock is looked up again, even ones that already have a price (prices change during the day). Unticked: only stocks still missing a price. Does not apply to the automatic lookup after a scan."><input type="checkbox" id="forceRefreshToggle" checked> also re-check stocks that already have a price</label>
    </p>
  </div>


  <div class="filterbar">
    <span class="filterlabel">Filters <span class="hint inline">(only change what you see)</span></span>
    <label class="toggle" title="Hides puts that expire more than 45 days from now."><input type="checkbox" id="hideFarDteToggle" checked onchange="renderTable()"> Hide puts lasting over 45 days <span class="n" data-count="hideFarDteToggle"></span></label>
    <label class="toggle" title="Delta above 0.30 means roughly a 30%+ chance you end up having to buy the shares."><input type="checkbox" id="hideRiskyDeltaToggle" checked onchange="renderTable()"> Hide over a 30% chance of buying <span class="n" data-count="hideRiskyDeltaToggle"></span></label>
    <label class="toggle" title="Hides puts that pay less than 2.5%. Stocks with no price yet or none found are kept."><input type="checkbox" id="hideLowRorToggle" checked onchange="renderTable()"> Hide paying under 2.5% <span class="n" data-count="hideLowRorToggle"></span></label>
    <label class="toggle" title="Hides stocks where prices were checked but no put matched. Often just a thin options listing on Yahoo's free data, not a bad stock."><input type="checkbox" id="hideUnknownToggle" onchange="renderTable()"> Hide no put found <span class="n" data-count="hideUnknownToggle"></span></label>
    <label class="toggle" title="An extra, stricter trend check: price above its 50/100/200-day averages and higher highs and lows than 12 or 18 months ago. Every stock here already passes the basic 30-week check."><input type="checkbox" id="hideNonUptrendToggle" onchange="renderTable()"> Hide without a 12-18 month trend <span class="n" data-count="hideNonUptrendToggle"></span></label>
  </div>
  <div class="filterbar">
    <span class="filterlabel">Columns</span>
    <details class="columns">
      <summary>Show or hide columns</summary>
      <div class="controls" id="columnToggles">
        <label class="toggle"><input type="checkbox" data-col="change_pct"> Change today</label>
        <label class="toggle"><input type="checkbox" data-col="sector"> Sector</label>
        <label class="toggle"><input type="checkbox" data-col="next_earnings" checked> Next earnings</label>
        <label class="toggle"><input type="checkbox" data-col="eps_beat"> Beat earnings?</label>
        <label class="toggle"><input type="checkbox" data-col="realized_vol"> Volatility</label>
        <label class="toggle"><input type="checkbox" data-col="pe_ok"> Price vs profit (P/E)</label>
        <label class="toggle"><input type="checkbox" data-col="debt_to_equity"> Debt OK?</label>
        <label class="toggle"><input type="checkbox" data-col="fcf_positive"> Makes cash?</label>
        <label class="toggle"><input type="checkbox" data-col="roe"> Efficient? (ROE)</label>
        <label class="toggle"><input type="checkbox" data-col="rev_growth"> Sales growing?</label>
        <label class="toggle"><input type="checkbox" data-col="uptrend_legacy"> 12-18 month trend?</label>
        <label class="toggle"><input type="checkbox" data-col="csp_trigger"> Dip lately?</label>
        <label class="toggle"><input type="checkbox" data-col="beta"> Market swings (Beta)</label>
        <label class="toggle"><input type="checkbox" data-col="vol_pctile"> Volatility rank</label>
        <label class="toggle"><input type="checkbox" data-col="short_pct_float"> Shorted %</label>
      </div>
    </details>
  </div>
  <p class="legend">Best payers first. An orange earnings date means the company reports before the put expires, which can move the price a lot. Click a heading to sort (&#9660; marks the current sort); hover a heading for an explanation. &#9733; = a ticker you added (always shown, filters skipped). Return is not per year, so a longer put shows a bigger number. Easy to trade: <span class="check-n">&#10007;</span> under 10 open contracts means a fair price can be hard to get.</p>
  <p class="legend" id="legendChecks" style="display:none;">Company checks: <span class="check-y">&#10003;</span> yes &nbsp;<span class="check-n">&#10007;</span> no &nbsp;<b>?</b> unknown.</p>

  <p id="scrollHint" class="hint" style="display:none; max-width:980px;">Wider than your window: scroll sideways. The stock name stays put.</p>
  <div class="table-wrap">
  <table id="dataTable">
    <thead>
      <tr id="groupRow" class="group-row"></tr>
      <tr id="headerRow"></tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
  </div>

  <div id="tooltipBox"></div>

  <details class="advanced">
    <summary>Advanced: create or delete saved lists</summary>
    <div class="controls">
      <button id="createUniverseBtn" onclick="createNewUniverse()">Create a new saved list (takes about 3 minutes)</button>
    </div>
    <p class="hint">Scans the whole Russell 3000 (about 2,600 stocks) using your current strictness level and saves the stocks that pass as a new entry under &ldquo;Saved lists&rdquo; in the Stock list dropdown. Handy as a faster starting point later. Use it sparingly: it makes thousands of requests.</p>
    <div class="controls">
      <button id="deleteUniverseBtn" class="danger-outline" onclick="deleteSelectedUniverse()" disabled title="Only lists you created can be deleted, not the built-in S&amp;P 500 or Russell lists.">Delete the selected saved list</button>
    </div>
  </details>
  </div>

  <div id="tabSectors" style="display:none;">
  <h1>Sector Signals</h1>
  <p class="lead">The 11 sector funds that make up the S&amp;P 500 (the &ldquo;XL&rdquo; funds) and the 10 biggest stocks in each, checked for a few well-known chart patterns on daily candles from Yahoo Finance's free end-of-day data (it can lag a day). Patterns describe what the price has done lately; they are hints, not predictions. Nothing here buys or sells anything.</p>

  <section class="card scanbar">
    <div class="controls">
      <button id="sectorScanBtn" class="primary" onclick="sectorScan()">Scan sectors</button>
      <span id="sectorStatus">idle</span>
    </div>
    <p class="hint"><b>Scan sectors:</b> loads about a year of daily prices for the __N_ETF__ funds and their __N_HOLD__ biggest stocks (about 10 seconds), then checks each one for the patterns below. The list of each fund's biggest holdings is refreshed weekly.</p>
  </section>

  <div class="summary">
    <p id="sectorAsOf" class="listlabel"></p>
    <p id="sectorCount" class="rowcount"></p>
  </div>

  <div class="filterbar">
    <span class="filterlabel">Show</span>
    <label class="toggle"><input type="checkbox" id="sectorOnlySignals" checked onchange="renderSectors()"> Only stocks with a signal</label>
    <span class="field"><label for="sectorDirection">Direction:</label>
    <select id="sectorDirection" onchange="renderSectors()">
      <option value="all" selected>All</option>
      <option value="bullish">Bullish only</option>
      <option value="bearish">Bearish only</option>
      <option value="watch">Watch only</option>
    </select></span>
    <label class="toggle"><input type="checkbox" id="sectorHideForming" onchange="renderSectors()"> Hide patterns still forming</label>
    <label class="toggle"><input type="checkbox" id="sectorShowDetails" onchange="renderSectors()"> Show the numbers under each signal</label>
  </div>

  <p class="legend">Red badges lean bearish, green bullish, amber are &ldquo;watch&rdquo; signals that can go either way. <b>confirmed</b> = the pattern completed within the last 10 trading days; <b>forming</b> = the shape is there but not confirmed yet; <b>latest</b> = about the newest candle only. Hover a badge for the exact numbers. &ldquo;Lean&rdquo; adds up a row's signals: confirmed = 2, forming = 1, watch = 0, bullish minus bearish, and ignores the filters. A confirmed pattern that the price later undoes is dropped. Open the list below for what each pattern means.</p>
  <details class="glossary">
    <summary>What the signals mean</summary>
    <dl>
      <dt><span class="badge bearish">Double top</span></dt><dd>The price hit about the same high twice with a dip between, then fell below that dip. Often marks the end of a rise. <i>Bearish.</i></dd>
      <dt><span class="badge bullish">Double bottom</span></dt><dd>The mirror image: two similar lows, then a close above the bump between them. Often marks the end of a fall. <i>Bullish.</i></dd>
      <dt><span class="badge bearish">Head and shoulders</span></dt><dd>Three peaks with the middle one tallest, then a close below the lower of the two dips between them (the neckline). <i>Bearish.</i></dd>
      <dt><span class="badge bullish">Inverse head and shoulders</span></dt><dd>The mirror image, built from three dips. <i>Bullish.</i></dd>
      <dt><span class="badge watch">Sideways range</span></dt><dd>A double top and a double bottom are both forming, so the price is bouncing between a floor and a ceiling. <i>Watch.</i></dd>
      <dt><span class="badge bullish">Bull retest gap</span></dt><dd>The price jumped up leaving a gap on the chart, later dipped back into that gap and held, closing above it. The gap acted as a floor. <i>Bullish.</i></dd>
      <dt><span class="badge bearish">Bear retest gap</span></dt><dd>The mirror image: a gap down, then a rally into the gap that was pushed back. The gap acted as a ceiling. <i>Bearish.</i></dd>
      <dt><span class="badge watch">First red candle above 21-day EMA</span><br><span class="hint inline">shown as &ldquo;1st red above 21 EMA&rdquo;</span></dt><dd>After at least two green days, the latest day closed down with the whole candle, wicks included, above the 21-day average. The first pause in a rise: some buy the dip, some see early weakness. <i>Watch.</i></dd>
      <dt><span class="badge watch">First green candle below 21-day EMA</span><br><span class="hint inline">shown as &ldquo;1st green below 21 EMA&rdquo;</span></dt><dd>After at least two red days, the latest day closed up with the whole candle, wicks included, below the 21-day average. A first up day while the price sits under its average. <i>Watch.</i></dd>
    </dl>
    <p class="hint">A forming pattern is dropped once the price moves past it (for example a close above both tops of a double top), and a pattern that never confirms within its own width is dropped as stale. 21-day EMA = an average of the last 21 closes that weights recent days more. &ldquo;First&rdquo; red or green candle means at least two days of the opposite colour came before it, and tiny (doji) candles are ignored.</p>
  </details>

  <div class="table-wrap">
  <table id="sectorTable">
    <thead>
      <tr id="sectorHeader">
        <th style="text-align:left">Stock</th><th title="How much of the fund this stock makes up">% of fund</th><th>Price ($)</th><th title="Change on the latest trading day">Last day</th><th title="Is the latest close above or below its 21-day exponential moving average?">21-day EMA</th><th title="Bullish minus bearish signals, confirmed ones counting double">Lean</th><th style="text-align:left">Signals (hover for the numbers)</th>
      </tr>
    </thead>
    <tbody id="sectorBody"></tbody>
  </table>
  </div>
  </div>


<script>
const CHECK_KEYS = ["pe_ok","fcf_positive","eps_beat","debt_to_equity","roe","rev_growth","uptrend_legacy","csp_trigger"];
// Which columns the checkboxes above the table can hide. Everything else
// (pe_ok, debt_to_equity, roe, csp_trigger, and all the put/option columns)
// is always shown.
const TOGGLEABLE_COLUMNS = new Set(["change_pct","sector","pe_ok","fcf_positive","eps_beat","debt_to_equity","roe","rev_growth","uptrend_legacy","csp_trigger","next_earnings","beta","realized_vol","vol_pctile","short_pct_float"]);
// Must match which checkboxes above start unchecked in the HTML (fcf_positive, roe) —
// these are the two quality filters NOT in the source guidebook (see ROE tooltip),
// off by default so the table defaults closer to doc-only criteria. short_pct_float
// is off by default everywhere too — squeeze-screening signal, not part of the CSP
// strategy itself, and Yahoo's short interest data is always 1-2 weeks stale.
const hiddenColumns = new Set(["change_pct","sector","pe_ok","debt_to_equity","fcf_positive","roe","rev_growth","uptrend_legacy","csp_trigger","beta","realized_vol","eps_beat","vol_pctile","short_pct_float"]);
let currentRows = [];
let sortKey = "ror";
let sortDir = -1; // descending

// Mirrors config.strategy_rules.OPTIONS_MODE_MIN_REALIZED_VOL — used for
// the CLIENT-SIDE instant filter on already-loaded data when you switch to
// Options Trading mode (the backend applies the same threshold, but only
// takes effect on the NEXT scan; this makes it immediate on what's already
// on screen too, per "immediately eliminate").
const OPTIONS_MODE_MIN_REALIZED_VOL = 0.25;
// Mirrors config.strategy_rules.CURATED_MODE_MIN_REALIZED_VOL / CURATED_*
// gates — same "instant on already-loaded data" purpose as above.
const CURATED_MODE_MIN_REALIZED_VOL = 0.35;

// Per-mode column-visibility presets, applied by applyModePreset(). Kept as
// a plain object (not hardcoded logic) so a third mode (e.g. swing trading)
// is a one-line addition later, not a restructure.
const MODE_COLUMN_PRESETS = {
  general: {change_pct: false, sector: false, pe_ok: false, fcf_positive: false, eps_beat: false, debt_to_equity: false, roe: false, rev_growth: false, uptrend_legacy: false, csp_trigger: false, next_earnings: true, beta: false, realized_vol: false, vol_pctile: false, short_pct_float: false},
  options: {change_pct: false, sector: false, pe_ok: false, fcf_positive: false, eps_beat: false, debt_to_equity: false, roe: false, rev_growth: false, uptrend_legacy: false, csp_trigger: false, next_earnings: true, beta: false, realized_vol: false, vol_pctile: false, short_pct_float: false},
  curated: {change_pct: false, sector: false, pe_ok: false, fcf_positive: true, eps_beat: true, debt_to_equity: true, roe: false, rev_growth: false, uptrend_legacy: false, csp_trigger: false, next_earnings: true, beta: false, realized_vol: false, vol_pctile: false, short_pct_float: false},
};

// User-facing modes are just "General" and "Options Trading" — Curated
// isn't a third peer mode (it's the same purpose as Options Trading, just
// a stricter screen), so it's a checkbox WITHIN Options Trading rather
// than its own dropdown entry. Backend/cache code still uses "options"
// vs "curated" as two distinct screening configs — this is the one place
// that maps the UI's (mode dropdown, strict checkbox) pair to that
// internal string, so every other function can just call this instead of
// reading the raw dropdown.
function getEffectiveMode() {
  return document.getElementById("modeSelect").value;
}

function applyModePreset() {
  const mode = getEffectiveMode();

  const preset = MODE_COLUMN_PRESETS[mode];
  if (preset) {
    Object.entries(preset).forEach(([col, visible]) => {
      const cb = document.querySelector(`#columnToggles input[data-col="${col}"]`);
      if (cb) cb.checked = visible;
      if (visible) hiddenColumns.delete(col); else hiddenColumns.add(col);
    });
  }
  const note = document.getElementById("modeNote");
  if (mode === "options") {
    note.textContent = `Medium hides stocks that barely move (under ${(OPTIONS_MODE_MIN_REALIZED_VOL * 100).toFixed(0)}% a year), since they pay little.`;
  } else if (mode === "curated") {
    note.textContent = `High hides stocks that barely move (under ${(CURATED_MODE_MIN_REALIZED_VOL * 100).toFixed(0)}% a year) and companies that are unprofitable, low on cash or heavily in debt.`;
  } else {
    note.textContent = "Low hides nothing extra: every stock whose price is rising stays.";
  }
  renderTable();
}

function isColumnVisible(key) {
  return !hiddenColumns.has(key);
}

const COLUMN_TOOLTIPS = {
  symbol: "The stock's ticker symbol. A star means you added it yourself. The blue arrow looks that one stock up again.",
  sector: "The industry the company is in.",
  spot: "The stock's current price, in dollars.",
  change_pct: "How much the price moved today, compared with yesterday's close.",
  strike: "The price you promise to buy 100 shares at if the stock falls to it. Multiply by 100 for the cash you set aside.",
  dte: "How many days until the put expires. After that the promise ends.",
  delta: "Roughly the chance you end up having to buy the shares, estimated from the option's own price. The tool aims for about 20% and only shows 20% to 30%. (Technical name: delta.)",
  premium: "What you are paid up front for one contract (100 shares), at the midpoint of the bid and ask prices.",
  ror: "What you collect, divided by the cash you set aside (strike x 100, minus what you collect), over this put's own life. Not per year, so a longer put shows a bigger number. The table is sorted by this. (Same formula thinkorswim calls RoR.)",
  oi: "How many of this exact contract exist (open interest). Under about 10 means few people trade it, so getting a fair price can be hard.",
  next_earnings: "The company's next earnings report date. Big news can move the price sharply, so many people avoid puts that last past it.",
  beta: "How much the stock swings compared with the overall market. 1.0 moves with the market; 2.0 swings about twice as much.",
  realized_vol: "How much the price has been moving lately, as a yearly percentage. More movement usually means you are paid more, and risk more.",
  vol_pctile: "Is the stock moving more than usual for itself right now? 100 means the most active it has been all year.",
  short_pct_float: "The share of the stock that traders have bet against. High numbers mean many expect it to fall. Updated only about twice a month, so always 1 to 2 weeks old.",
  check_pe_ok: "Price divided by yearly profit per share. Green means 100 or less, or the company could pay off its debt from cash flow within a year.",
  check_fcf_positive: "Did the company bring in more cash than it spent over the last year?",
  check_eps_beat: "Did the last earnings report beat what analysts expected? Hover the cell for the numbers.",
  check_debt_to_equity: "Is the company's debt at a manageable level (no more than twice its equity)? Banks and utilities are exempt.",
  check_roe: "Does the company earn a solid return on its shareholders' money (15% or more)?",
  check_rev_growth: "Did sales grow compared with a year ago?",
  check_uptrend_legacy: "An extra, longer trend check over 12 to 18 months. Every stock already passes the basic 30-week check; this one is stricter.",
  check_csp_trigger: "Did the stock just dip (three down days in a row, or a 5% drop in one day)? Many people like that as a time to sell a put. A dash means no recent dip.",
};

function buildColumns() {
  return [
    {key: "symbol", label: "Symbol"},
    {key: "spot", label: "Spot"},
    ...(isColumnVisible("change_pct") ? [{key: "change_pct", label: "Chg %"}] : []),
    {key: "premium", label: "You collect"},
    {key: "ror", label: "Return %"},
    {key: "strike", label: "Strike price"},
    {key: "dte", label: "Days left"},
    {key: "delta", label: "Chance of buying"},
    {key: "oi", label: "Open contracts"},
    ...(isColumnVisible("sector") ? [{key: "sector", label: "Sector"}] : []),
    ...CHECK_KEYS.filter(isColumnVisible).map(k => ({key: "check_" + k, label: null})),
    ...(isColumnVisible("next_earnings") ? [{key: "next_earnings", label: "Earnings"}] : []),
    ...(isColumnVisible("beta") ? [{key: "beta", label: "Beta"}] : []),
    ...(isColumnVisible("realized_vol") ? [{key: "realized_vol", label: "Vol"}] : []),
    ...(isColumnVisible("vol_pctile") ? [{key: "vol_pctile", label: "Vol %ile"}] : []),
    ...(isColumnVisible("short_pct_float") ? [{key: "short_pct_float", label: "Short % Float"}] : []),
  ];
}
let COLUMNS = buildColumns();

function showTooltip(anchorEl, text) {
  const box = document.getElementById("tooltipBox");
  box.textContent = text;
  box.style.display = "block";
  const rect = anchorEl.getBoundingClientRect();
  const margin = 8;
  // Measure after display:block so offsetWidth/Height are accurate, then
  // clamp so the box never runs past the right edge (the original bug) or
  // any other edge of the viewport.
  let left = rect.left;
  let top = rect.bottom + 6;
  const maxLeft = window.innerWidth - box.offsetWidth - margin;
  if (left > maxLeft) left = Math.max(margin, maxLeft);
  if (left < margin) left = margin;
  const maxTop = window.innerHeight - box.offsetHeight - margin;
  if (top > maxTop) top = Math.max(margin, rect.top - box.offsetHeight - 6); // flip above the header if there's no room below
  box.style.left = left + "px";
  box.style.top = top + "px";
}

function hideTooltip() {
  document.getElementById("tooltipBox").style.display = "none";
}

function syncLimitSelect(requestedLimit) {
  // Keeps the dropdown honest about what was ACTUALLY last scanned — it
  // used to just sit on its default ("First 50") forever regardless of
  // what scan actually ran (e.g. a smaller --limit passed on the CLI at
  // startup), which is exactly what caused the "why only 8 of 50" confusion.
  const select = document.getElementById("limitSelect");
  const value = requestedLimit === null ? "all" : String(requestedLimit);
  const matchingOption = Array.from(select.options).some(o => o.value === value);
  if (matchingOption) {
    select.value = value;
  } else {
    // A limit that doesn't match any fixed dropdown option (e.g. a custom
    // CLI --limit) — add a temporary option so the dropdown still tells
    // the truth instead of silently reverting to its default.
    let custom = select.querySelector("option[data-custom]");
    if (!custom) {
      custom = document.createElement("option");
      custom.setAttribute("data-custom", "1");
      select.insertBefore(custom, select.firstChild);
    }
    custom.value = value;
    custom.textContent = `Last scan: ${value}`;
    select.value = value;
  }
}

function checkMark(v) {
  if (v === null || v === undefined) return '<span class="check-u">?</span>';
  return v ? '<span class="check-y">&#10003;</span>' : '<span class="check-n">&#10007;</span>';
}

// Columns rendered as their raw number (colored by pass/fail) instead of a checkmark.
const NUMERIC_CHECK_KEYS = new Set(["pe_ok"]);

function checkCell(c) {
  if (!c || c.value === null || c.value === undefined) return '<span class="check-u">?</span>';
  if (c.raw === null || c.raw === undefined) return checkMark(c.value);
  const cls = c.value ? "check-y" : "check-n";
  return `<span class="${cls}">${c.raw.toFixed(1)}</span>`;
}

function epsBeatCell(c) {
  if (!c || c.value === null || c.value === undefined) return '<span class="check-u">?</span>';
  const raw = c.raw;
  const tip = (raw && raw.actual !== null && raw.actual !== undefined && raw.estimate !== null && raw.estimate !== undefined)
    ? `Last report: earned $${raw.actual.toFixed(2)} per share vs $${raw.estimate.toFixed(2)} expected` : "";
  const cls = c.value ? "check-y" : "check-n";
  return `<span class="${cls}" title="${tip}">${c.value ? "&#10003; beat" : "&#10007; missed"}</span>`;
}

function changePctCell(pct) {
  if (pct === null || pct === undefined) return '<span class="check-u">?</span>';
  const cls = pct > 0 ? "check-y" : pct < 0 ? "check-n" : "";
  const sign = pct > 0 ? "+" : "";
  return `<span class="${cls}">${sign}${(pct * 100).toFixed(2)}%</span>`;
}

// The 6 put-related columns (Strike, DTE, Delta, Premium, RoR%, OI) as one
// block, in one of four states: never fetched (Get Option button),
// fetched-but-failed (error + Retry button), fetched-but-nothing-qualified
// (dashes + a refresh icon), or a real result (dashes + a refresh icon).
// Every state past "never fetched" has some way to re-fetch.
function renderOptionCells(r) {
  if (!r.options_fetched) {
    return `<td colspan="6"><button class="get-option-btn" onclick="getOption('${r.symbol}', this)">Get price</button></td>`;
  }
  if (r.options_error) {
    return `<td colspan="6"><span class="check-n" title="${r.options_error}">${r.options_error}</span> <button class="get-option-btn" onclick="getOption('${r.symbol}', this)">Retry</button></td>`;
  }
  if (!r.put) {
    // Fetched cleanly, but nothing in the chain qualified (e.g. Yahoo's
    // free feed only returned a sparse/thin chain for this symbol — a
    // known data-completeness gap, not a bug). Still needs a way to
    // retry: the chain can fill in later, or a wider band might match
    // next time. No `force` needed — the backend's cache-skip guard only
    // ever triggers when a real put already exists.
    if (r.options_note === "no_live_quotes" || (!marketOpenNow() && r.options_note !== "no_chain")) {
      return `<td colspan="6" class="check-u" title="Yahoo shows no live put prices while the US market is closed. Press Update put prices during market hours (9:30am to 4pm Eastern, Monday to Friday).">no live price (market closed)</td>`;
    }
    if (r.options_note === "no_chain") {
      return `<td colspan="6" class="check-u" title="Yahoo lists no put contracts for this stock in the 30 to 90 day window.">no puts listed</td>`;
    }
    return `<td colspan="6" class="check-u" title="Prices were checked, but no put landed in the 20% to 30% chance range. Often a thin options listing on Yahoo's free data.">none found</td>`;
  }
  const p = r.put;
  let fallbackNote = "";
  if (p.is_fallback_pick) {
    fallbackNote = '<div class="fallback" title="No expiration in the search window had a clean, close-to-target match — this is the best validated (real bid/ask, sane IV) contract found regardless, which can be far from -0.20 delta.">no clean match — best available</div>';
  } else if (p.at_or_above_target_delta === false) {
    fallbackNote = '<div class="fallback">below target Δ</div>';
  }
  const expDate = p.expiration ? new Date(p.expiration + "T00:00:00").toLocaleDateString(undefined, {month: "short", day: "numeric"}) : "";
  const cash = Math.round(p.strike * 100).toLocaleString();
  const stale = marketOpenNow() ? "" : '<span class="sub" title="The market is closed, so this is the last quote Yahoo had">last known</span>';
  return `<td>$${p.premium.toFixed(0)}${stale}</td>` +
    `<td class="ror-cell">${(p.return_on_risk * 100).toFixed(2)}%</td>` +
    `<td>${p.strike.toFixed(2)}<span class="sub" title="Cash you must keep ready: strike x 100">$${cash} cash</span></td>` +
    `<td>${p.dte}<span class="sub" title="Expiry date">${expDate}</span></td>` +
    `<td>${(p.delta_abs * 100).toFixed(0)}%${fallbackNote}</td>` +
    (p.open_interest < 10
      ? `<td class="low-oi" title="Only ${p.open_interest} of this contract exist, so a fair price can be hard to get"><span class="check-n">&#10007;</span> ${p.open_interest}</td>`
      : `<td title="${p.open_interest} of this contract exist"><span class="check-y">&#10003;</span> ${p.open_interest}</td>`);
}

// Shared busy-state feedback for every button in the page: the instant an
// action starts (before any network round trip), disable it, dim it, and
// show a spinner — so a click always visibly registers as "doing
// something," even for an action that turns out to be fast or a no-op.
// Per user request, after "Fetch options for all shown" silently did
// nothing when every row was already fetched — there was no way to tell
// a real no-op from a broken button.
function setBusy(btn, busyText) {
  if (!btn) return;
  if (btn.dataset.origText === undefined) btn.dataset.origText = btn.textContent;
  btn.disabled = true;
  btn.classList.add("busy");
  btn.innerHTML = `<span class="spinner"></span>${busyText}`;
}
function clearBusy(btn, text) {
  if (!btn) return;
  btn.disabled = false;
  btn.classList.remove("busy");
  btn.textContent = text !== undefined ? text : (btn.dataset.origText || btn.textContent);
}

// One Cancel button covers two independent long-running things: a
// server-side scan/create-universe (tracked in _state, stopped via
// POST /api/cancel) and the client-side sequential options-fetch loop
// (tracked by fetchingAllOptions, stopped via optionsFetchCancelled — the
// server has no idea that loop is even happening, it's just repeated
// /api/option calls). Shown whenever either is active.
function updateCancelButtonVisibility() {
  const cancelBtn = document.getElementById("cancelBtn");
  if (!cancelBtn) return;
  const serverBusy = ["scanning", "creating_universe"].includes(window.__lastKnownStatus);
  cancelBtn.style.display = serverBusy ? "" : "none";
  const stopBtn = document.getElementById("cancelFetchBtn");
  if (stopBtn) stopBtn.style.display = fetchingAllOptions ? "" : "none";
}

async function cancelRun() {
  if (fetchingAllOptions) optionsFetchCancelled = true;
  await fetch("/api/cancel", {method: "POST"});
}

async function getOption(symbol, btnEl, force) {
  if (btnEl) setBusy(btnEl, "Fetching...");
  try {
    const url = `/api/option?symbol=${encodeURIComponent(symbol)}${force ? "&force=1" : ""}`;
    const res = await fetch(url, {method: "POST"});
    const data = await res.json();
    const row = currentRows.find(r => r.symbol === symbol);
    if (!row) return;
    if (!data.ok) {
      row.options_fetched = true;
      row.options_error = data.message || "fetch failed";
    } else {
      row.options_fetched = true;
      row.options_error = data.options_error || null;
      row.options_note = data.options_note || null;
      row.put = data.put;
      if (data.spot !== undefined) {
        row.spot = data.spot;
        row.day_change_pct = data.day_change_pct;
      }
      if (data.quote_as_of) row.quote_as_of = data.quote_as_of;
      updateAsOf();
    }
    renderTable();  // replaces btnEl entirely (new "Retry" button, or real data) — no clearBusy needed on success
  } catch (e) {
    if (btnEl) clearBusy(btnEl);  // no explicit text: falls back to the button's own pre-busy text (dataset.origText) — "Get Option", "Retry", or the refresh icon, whichever this button actually was
  }
}

// Cap on how many shown rows will trigger AUTOMATIC option-fetching after a
// scan completes — a small, well-filtered result (like the ~106/503 the
// current uptrend+vol filters typically produce) is fine to auto-fetch;
// a large one (e.g. an unfiltered full Russell 3000 scan) should NOT
// auto-fetch, since that's exactly the rate-limit-prone bulk options
// fetching this whole on-demand design was built to avoid. Manual "Fetch
// options for all shown" / per-row "Get Option" always remain available
// regardless of this cap.
const AUTO_FETCH_OPTIONS_MAX_ROWS = 150;
let fetchingAllOptions = false;
let optionsFetchCancelled = false;  // checked between symbols in the loop below — can't interrupt a single in-flight fetch, but stops the next one from starting

// forceOverride lets a caller pin the force-refresh behavior instead of
// reading the checkbox — used by the automatic post-scan/page-load
// auto-fetch below, which must NOT force-refetch everything just because
// the checkbox defaults to checked. Only an explicit user click of "Fetch
// options for all shown" (forceOverride left undefined) reads the checkbox.
async function fetchAllOptions(forceOverride) {
  const btn = document.getElementById("fetchAllBtn");
  if (fetchingAllOptions) return;
  setBusy(btn, "Checking...");  // immediate feedback even before we know if there's anything to do
  // Targets = never attempted, OR attempted but failed (rate-limited, no
  // match, cancelled mid-run, etc.) — a symbol with a REAL put already
  // (options_fetched && no error) is normally excluded, so clicking this
  // again after a partial/interrupted run naturally continues through
  // whatever's left rather than needing a separate "resume" button. The
  // "Force refresh" toggle overrides that exclusion so already-fetched
  // rows get re-pulled too (premiums move during the day) — same escape
  // hatch as the per-row refresh button, just for everything shown at once.
  const forceRefresh = forceOverride !== undefined ? forceOverride : document.getElementById("forceRefreshToggle")?.checked;
  const targets = getDisplayRows().filter(r => !r.error && (forceRefresh || !r.options_fetched || r.options_error)).map(r => r.symbol);
  if (targets.length === 0) {
    // Previously silently did nothing here — indistinguishable from a
    // broken button. Every row already fetched is a real, common case
    // (e.g. right after a scan's own auto-fetch already ran), so say so.
    clearBusy(btn, "Nothing to do: every stock shown already has a price");
    setTimeout(() => clearBusy(btn), 2500);
    return;
  }
  fetchingAllOptions = true;
  optionsFetchCancelled = false;
  updateCancelButtonVisibility();
  for (let i = 0; i < targets.length; i++) {
    if (optionsFetchCancelled) {
      setFetchStatus(`Stopped: put prices loaded for ${i} of ${targets.length}. Press Update put prices to continue.`);
      clearBusy(btn, `Cancelled: got ${i} of ${targets.length} prices`);
      setTimeout(() => clearBusy(btn), 2500);
      fetchingAllOptions = false;
      updateCancelButtonVisibility();
      return;
    }
    setBusy(btn, `Getting prices... (${i + 1}/${targets.length})`);
    const left = Math.round((targets.length - i) * 1.5);
    setFetchStatus(`Loading put prices: ${i + 1} of ${targets.length} (about ${left < 50 ? Math.max(5, Math.round(left / 5) * 5) + " sec" : Math.max(1, Math.round(left / 60)) + " min"} left)...`);
    await getOption(targets[i], null, forceRefresh);
  }
  const got = targets.filter(sym => { const r = currentRows.find(x => x.symbol === sym); return r && r.put; }).length;
  const closedNote = (got < targets.length && !marketOpenNow()) ? " (the market is closed, so most have none right now)" : "";
  setFetchStatus(`Checked ${targets.length} stock${targets.length === 1 ? "" : "s"}: ${got} got a put price${closedNote}. All done.`);
  clearBusy(btn, fetchLabel());
  fetchingAllOptions = false;
  updateCancelButtonVisibility();
}

// Plain-English column headings (the technical term stays in the tooltip).
const HEADER_LABELS = {
  symbol: "Stock", sector: "Sector", spot: "Price ($)", change_pct: "Change today",
  strike: "Buy at\\n(strike, $)", dte: "Days\\nleft", delta: "Chance of\\nbuying", premium: "You collect\\n($ / 100 shares)", ror: "Return\\n(not per year)", oi: "Easy to trade?\\n(open contracts)",
  check_pe_ok: "Price vs profit (P/E)", check_fcf_positive: "Makes cash?", check_eps_beat: "Beat earnings?",
  check_debt_to_equity: "Debt OK?", check_roe: "Efficient? (ROE)", check_rev_growth: "Sales growing?",
  check_uptrend_legacy: "12-18 month trend?", check_csp_trigger: "Dip lately?",
  next_earnings: "Next earnings", beta: "Market swings (Beta)", realized_vol: "Volatility\\n(% / year)",
  vol_pctile: "Volatility rank", short_pct_float: "Shorted %",
};

function renderHeader() {
  COLUMNS = buildColumns();
  const row = document.getElementById("headerRow");
  row.innerHTML = "";
  COLUMNS.forEach(col => {
    const th = document.createElement("th");
    let label = col.label;
    if (label === null) {
      // pull check label from first available row
      const r = currentRows.find(r => r.checks && r.checks[col.key.replace("check_","")]);
      label = r ? r.checks[col.key.replace("check_","")].label : col.key;
    }
    th.textContent = (HEADER_LABELS[col.key] || label) + (col.key === sortKey ? (sortDir === -1 ? " \u25BC" : " \u25B2") : "");
    th.style.whiteSpace = "pre-line";
    if (COLUMN_TOOLTIPS[col.key]) {
      th.addEventListener("mouseenter", () => showTooltip(th, COLUMN_TOOLTIPS[col.key]));
      th.addEventListener("mouseleave", hideTooltip);
    }
    if (col.key === sortKey) th.classList.add("sorted");
    th.onclick = () => {
      if (sortKey === col.key) sortDir *= -1; else { sortKey = col.key; sortDir = -1; }
      renderTable();
    };
    row.appendChild(th);
  });
  const groupRow = document.getElementById("groupRow");
  if (groupRow) {
    const leading = 2 + (isColumnVisible("change_pct") ? 1 : 0);
    const healthCols = Math.max(0, COLUMNS.length - leading - 6);
    groupRow.innerHTML = `<th colspan="${leading}"></th><th colspan="6">The put (promise) you would sell</th>` +
      (healthCols > 0 ? `<th colspan="${healthCols}">About the company</th>` : "");
  }
}

function sortValue(row, key) {
  if (key === "symbol") return row.symbol;
  if (key === "sector") return row.sector || "";
  if (key === "spot") return row.spot;
  if (key === "change_pct") return row.day_change_pct === null || row.day_change_pct === undefined ? -Infinity : row.day_change_pct;
  if (key === "beta") return row.beta === null || row.beta === undefined ? -Infinity : row.beta;
  if (key === "realized_vol") return row.realized_vol === null || row.realized_vol === undefined ? -Infinity : row.realized_vol;
  if (key === "vol_pctile") return row.realized_vol_pctile === null || row.realized_vol_pctile === undefined ? -Infinity : row.realized_vol_pctile;
  if (key === "short_pct_float") return row.short_pct_float === null || row.short_pct_float === undefined ? -Infinity : row.short_pct_float;
  if (key.startsWith("check_")) {
    const checkKey = key.replace("check_","");
    const c = row.checks[checkKey];
    if (!c) return 0;
    if (NUMERIC_CHECK_KEYS.has(checkKey) && c.raw !== null && c.raw !== undefined) return c.raw;
    return c.value === true ? 1 : c.value === false ? -1 : 0;
  }
  if (key === "next_earnings") return row.next_earnings_date || "";
  if (key === "strike") return row.put ? row.put.strike : -Infinity;
  if (key === "dte") return row.put ? row.put.dte : -Infinity;
  if (key === "delta") return row.put ? row.put.delta_abs : -Infinity;
  if (key === "premium") return row.put ? row.put.premium : -Infinity;
  if (key === "ror") return row.put ? row.put.return_on_risk : -Infinity;
  if (key === "oi") return row.put ? row.put.open_interest : -Infinity;
  return 0;
}

// There's no official/standardized abbreviation scheme for these sector
// names (unlike, say, ISO country or currency codes) — this is just
// common financial-site shorthand (Finviz, Yahoo Finance, etc. all
// abbreviate similarly), applied purely for column width. Full name is
// always still available via the title attribute (hover) and unaffected
// in sorting/filtering, which use the untouched r.sector value.
const SECTOR_ABBREVIATIONS = {
  "Technology": "Tech",
  "Healthcare": "Health",
  "Financial Services": "Fin. Svcs",
  "Consumer Cyclical": "Cons. Cyc.",
  "Consumer Defensive": "Cons. Def.",
  "Communication Services": "Comm. Svcs",
  "Industrials": "Industrials",
  "Basic Materials": "Materials",
  "Energy": "Energy",
  "Utilities": "Utilities",
  "Real Estate": "Real Estate",
};
function sectorCell(sector) {
  if (!sector) return "";
  const abbrev = SECTOR_ABBREVIATIONS[sector] || sector;
  return `<span title="${sector}">${abbrev}</span>`;
}

// DTE/delta ceiling for the "Hide DTE > 45 or Δ > 0.30" toggle. The delta
// half is mostly a safety net — the dashboard's own put search already
// never returns anything outside CSP_DASHBOARD_ACCEPTABLE_DELTA_RANGE
// (0.20-0.30) — but DTE is NOT otherwise capped below the 90-day search
// ceiling, so a row can legitimately show a 60-90 DTE contract when the
// nearer expirations didn't have an in-band strike. Per user request:
// hide those when only near-term trades are of interest right now.
const FAR_CONTRACT_MAX_DTE = 45;
const FAR_CONTRACT_MAX_DELTA = 0.30;
const MIN_ROR_PCT = 0.025;  // 2.5% — for the "Hide RoR% < 2.5%" toggle

function styleFilteredRows(sourceRows) {
  const mode = getEffectiveMode();
  let rows = sourceRows;
  if (mode === "options") {
    rows = rows.filter(r =>
      r.error || (r.realized_vol !== null && r.realized_vol !== undefined && r.realized_vol >= OPTIONS_MODE_MIN_REALIZED_VOL)
    );
  } else if (mode === "curated") {
    rows = rows.filter(r =>
      r.error || (
        r.realized_vol !== null && r.realized_vol !== undefined && r.realized_vol >= CURATED_MODE_MIN_REALIZED_VOL
        && r.curated_gate_passed !== false
      )
    );
  }
  return rows;
}

function getDisplayRows() {
  const mode = getEffectiveMode();
  // Error rows stay visible regardless of mode/toggle (they're a data
  // problem, not a "didn't qualify" result), in every branch below.
  // Rows the user typed in themselves (r.forced) skip every filter here too
  // — they're added back untouched at the end — since those are exactly the
  // stocks they asked to see.
  let rows = currentRows.filter(r => !r.forced);
  const breakdown = [];
  const apply = (label, predicate) => {
    const before = rows.length;
    rows = rows.filter(predicate);
    breakdown.push([label, before - rows.length]);
  };
  const styled = styleFilteredRows(rows);
  breakdown.push(["by the strictness level", rows.length - styled.length]);
  rows = styled;
  // A row with no fetched put yet (or none found) has nothing to judge
  // against — leave it visible rather than treating "unknown" as "far".
  const hideFarDte = document.getElementById("hideFarDteToggle");
  if (hideFarDte && hideFarDte.checked) {
    apply("puts lasting over 45 days", r => r.error || !r.put || r.put.dte <= FAR_CONTRACT_MAX_DTE);
  }
  const hideRiskyDelta = document.getElementById("hideRiskyDeltaToggle");
  if (hideRiskyDelta && hideRiskyDelta.checked) {
    apply("puts with over a 30% chance of buying", r => r.error || !r.put || r.put.delta_abs <= FAR_CONTRACT_MAX_DELTA);
  }
  const hideNonUptrend = document.getElementById("hideNonUptrendToggle");
  if (hideNonUptrend && hideNonUptrend.checked) {
    // check_uptrend_legacy is deliberately informational-only everywhere
    // else (the 30-week MA check is what actually gates the dashboard) —
    // this is an opt-in view for when you specifically want to cross-
    // reference against it too, not a change to the default filter.
    apply("without a 12-18 month trend", r =>
      r.error || (r.checks && r.checks.uptrend_legacy && r.checks.uptrend_legacy.value === true)
    );
  }
  const hideLowRor = document.getElementById("hideLowRorToggle");
  if (hideLowRor && hideLowRor.checked) {
    // "-" means unknown (no put fetched yet, or genuinely none found) —
    // NOT the same as "bad," so it's always kept regardless of this
    // toggle, same treatment as the DTE/delta toggle above.
    apply("puts paying under 2.5%", r => r.error || !r.put || r.put.return_on_risk >= MIN_ROR_PCT);
  }
  const hideUnknown = document.getElementById("hideUnknownToggle");
  if (hideUnknown && hideUnknown.checked) {
    // Only the fetched-but-nothing-qualified state (real "-" cells) —
    // not-yet-fetched rows still show their Get Option button and stay
    // visible, since hiding those would just look like they vanished.
    apply("stocks with no put found", r => r.error || !r.options_fetched || r.options_error || r.put);
  }
  window.__hideBreakdown = breakdown;
  return rows.concat(currentRows.filter(r => r.forced));
}

function renderTable() {
  renderHeader();
  const displayRows = getDisplayRows();
  const countEl = document.getElementById("rowCount");
  if (countEl) {
    const hidden = currentRows.length - displayRows.length;
    const mine = currentRows.filter(r => r.forced).length;
    const reasonTarget = {"puts lasting over 45 days": "hideFarDteToggle", "puts with over a 30% chance of buying": "hideRiskyDeltaToggle", "puts paying under 2.5%": "hideLowRorToggle", "stocks with no put found": "hideUnknownToggle", "without a 12-18 month trend": "hideNonUptrendToggle", "by the strictness level": "modeSelect"};
    const singular = t => t.replace(/^puts /, "put ").replace(/^stocks /, "stock ");
    const parts = (window.__hideBreakdown || []).filter(x => x[1] > 0).map(x => `<a class="reason" data-for="${reasonTarget[x[0]] || ""}" title="Show me the choice that does this">${x[1]} ${x[1] === 1 ? singular(x[0]) : x[0]}</a>`);
    countEl.innerHTML = currentRows.length === 0 ? "" :
      `Showing ${displayRows.length} of ${currentRows.length} stocks` + (mine ? ` (${currentRows.length - mine} passed the scan + ${mine} you added)` : "") +
      (hidden > 0 ? `. ${hidden} hidden by your filters and strictness: ${parts.join(", ")}.` : ".");
    const sc = document.getElementById("styleCount");
    if (sc) {
      const st = (window.__hideBreakdown || []).find(x => x[0] === "by the strictness level");
      sc.textContent = (currentRows.length && st && st[1] > 0) ? `Right now it hides ${st[1]} of the scanned stocks.` : "";
    }
  }
  // How many stocks each filter would hide on its own (counts can overlap).
  const ownRows = currentRows.filter(r => !r.forced && !r.error);
  const countFor = window.__countFor = {
    hideFarDteToggle: r => r.put && r.put.dte > FAR_CONTRACT_MAX_DTE,
    hideRiskyDeltaToggle: r => r.put && r.put.delta_abs > FAR_CONTRACT_MAX_DELTA,
    hideLowRorToggle: r => r.put && r.put.return_on_risk < MIN_ROR_PCT,
    hideUnknownToggle: r => r.options_fetched && !r.options_error && !r.put,
    hideNonUptrendToggle: r => !(r.checks && r.checks.uptrend_legacy && r.checks.uptrend_legacy.value === true),
  };
  document.querySelectorAll(".n[data-count]").forEach(el => {
    const fn = countFor[el.getAttribute("data-count")];
    const n = fn ? ownRows.filter(fn).length : 0;
    const forcedHit = fn ? currentRows.filter(r => r.forced && !r.error && fn(r)).length : 0;
    el.textContent = currentRows.length ? (n === 0 ? "(nothing to hide now)" : `(would hide ${n}${forcedHit ? `; your ${forcedHit} stay` : ""})`) : "";
    const lab = el.closest("label");
    if (lab) lab.classList.toggle("dim", currentRows.length > 0 && n === 0);
  });
  shownCount = displayRows.filter(r => !r.error).length;
  const fetchBtn = document.getElementById("fetchAllBtn");
  if (fetchBtn && !fetchBtn.classList.contains("busy") && !fetchingAllOptions && !fetchBtn.disabled) {
    fetchBtn.textContent = fetchLabel();
    fetchBtn.dataset.origText = fetchBtn.textContent;
  }
  const lc = document.getElementById("legendChecks");
  if (lc) lc.style.display = CHECK_KEYS.some(isColumnVisible) ? "" : "none";
  const wrapEl = document.querySelector(".table-wrap");
  const hintEl = document.getElementById("scrollHint");
  if (wrapEl && hintEl) setTimeout(() => { hintEl.style.display = wrapEl.scrollWidth > wrapEl.clientWidth + 4 ? "" : "none"; }, 0);
  const sinks = r => (!r.error && !r.put) ? 1 : 0;
  const sorted = [...displayRows].sort((a, b) => {
    if (sinks(a) !== sinks(b)) return sinks(a) - sinks(b);
    const va = sortValue(a, sortKey), vb = sortValue(b, sortKey);
    if (va < vb) return -1 * sortDir;
    if (va > vb) return 1 * sortDir;
    return 0;
  });

  const tbody = document.getElementById("tbody");
  tbody.innerHTML = "";
  const noPut = r => !r.error && !r.put;
  const noPutCount = sorted.filter(noPut).length;
  let dividerShown = false;
  sorted.forEach(r => {
    if (!dividerShown && noPut(r)) {
      dividerShown = true;
      const dv = document.createElement("tr");
      dv.className = "divider";
      const noPutRows = sorted.filter(noPut);
      const noChain = noPutRows.filter(x => x.options_note === "no_chain").length;
      const notYet = noPutRows.filter(x => !x.options_fetched).length;
      const closedCount = noPutRows.filter(x => x.options_note === "no_live_quotes").length;
      const rest = noPutCount - noChain - notYet;
      const parts = [];
      if (rest > 0) parts.push((!marketOpenNow() || closedCount >= Math.max(1, rest / 2))
        ? `${rest} ${rest === 1 ? "has" : "have"} no live put price because the US market is closed (open 9:30am to 4pm Eastern, Monday to Friday); press Update put prices during market hours`
        : `${rest} had no put in the 20% to 30% chance range (often a thin listing on Yahoo's free data)`);
      if (noChain > 0) parts.push(`${noChain} ${noChain === 1 ? "has" : "have"} no puts listed at all`);
      if (notYet > 0) parts.push(`${notYet} ${notYet === 1 ? "is" : "are"} not looked up yet`);
      dv.innerHTML = `<td colspan="${COLUMNS.length}">${noPutCount} stock${noPutCount === 1 ? "" : "s"} below without a put: ${parts.join("; ")}.</td>`;
      tbody.appendChild(dv);
    }
    const tr = document.createElement("tr");
    tr.dataset.sym = r.symbol;
    if (r.error) {
      tr.innerHTML = `<td>${r.symbol}</td><td colspan="${COLUMNS.length - 1}" class="err">${r.error}</td>`;
      tbody.appendChild(tr);
      return;
    }
    const upCheck = r.checks && r.checks.uptrend;
    const forcedTip = r.forced
      ? `Kept because you added it. Filters and the screening style do not apply to your own tickers.${upCheck && upCheck.value === false ? " Note: its price is NOT rising (it is below its 30-week average), so it would normally be left out." : ""}`
      : "";
    const refreshBtn = (r.options_fetched && !r.options_error)
      ? ` <button class="refresh-option-btn" title="Look this stock up again (prices change during the day)" onclick="getOption('${r.symbol}', this, true)">&#8635;</button>` : "";
    const co = r.name ? `<span class="co" title="${r.name.replace(/"/g, "&quot;")}">${r.name}</span>` : "";
    const symbolCell = (r.forced ? `<span title="${forcedTip}"><b>${r.symbol}</b> &#9733; <span class="mine">yours</span></span>` : `<b>${r.symbol}</b>`) + refreshBtn + co;
    let cells = `<td>${symbolCell}</td><td>${r.spot.toFixed(2)}</td>` + (isColumnVisible("change_pct") ? `<td>${changePctCell(r.day_change_pct)}</td>` : "");
    cells += renderOptionCells(r);
    if (isColumnVisible("sector")) cells += `<td>${sectorCell(r.sector)}</td>`;
    CHECK_KEYS.filter(isColumnVisible).forEach(k => {
      const c = r.checks[k];
      let content;
      if (k === "eps_beat") content = epsBeatCell(c);
      else if (NUMERIC_CHECK_KEYS.has(k)) content = checkCell(c);
      else if (k === "csp_trigger" && c && c.value === false) content = '<span class="check-u">&ndash;</span>';
      else content = checkMark(c ? c.value : null);
      cells += `<td title="${c ? c.reason : ''}">${content}</td>`;
    });
    if (isColumnVisible("next_earnings")) {
      const ne = r.next_earnings_date;
      const today = new Date().toISOString().slice(0, 10);
      const beforeExpiry = ne && r.put && r.put.expiration && ne >= today && ne <= r.put.expiration;
      cells += `<td class="${beforeExpiry ? "warn-cell" : ""}" title="${beforeExpiry ? "Earnings come before this put expires. Big news can move the price sharply." : ""}">${ne && ne >= today ? new Date(ne + "T00:00:00").toLocaleDateString() + (beforeExpiry ? '<span class="sub">before expiry</span>' : "") : '<span class="check-u" title="No upcoming earnings date is listed">n/a</span>'}</td>`;
    }
    if (isColumnVisible("beta")) {
      cells += `<td>${r.beta === null || r.beta === undefined ? '<span class="check-u">?</span>' : r.beta.toFixed(2)}</td>`;
    }
    if (isColumnVisible("realized_vol")) {
      cells += `<td>${r.realized_vol === null || r.realized_vol === undefined ? '<span class="check-u">?</span>' : (r.realized_vol * 100).toFixed(1) + '%'}</td>`;
    }
    if (isColumnVisible("vol_pctile")) {
      cells += `<td>${r.realized_vol_pctile === null || r.realized_vol_pctile === undefined ? '<span class="check-u">?</span>' : r.realized_vol_pctile.toFixed(0)}</td>`;
    }
    if (isColumnVisible("short_pct_float")) {
      cells += `<td>${r.short_pct_float === null || r.short_pct_float === undefined ? '<span class="check-u">?</span>' : (r.short_pct_float * 100).toFixed(1) + '%'}</td>`;
    }
    tr.innerHTML = cells;
    tbody.appendChild(tr);
  });
}

let lastGeneratedAt = null;
function updateAsOf(generatedAt) {
  if (generatedAt !== undefined) lastGeneratedAt = generatedAt;
  const el = document.getElementById("asOf");
  const fmt = iso => { const d = new Date(iso); return `${d.toLocaleDateString()}, ${d.toLocaleTimeString()}`; };
  const quoteTimes = currentRows.map(r => r.quote_as_of).filter(Boolean).sort();
  const latest = quoteTimes.length ? fmt(quoteTimes[quoteTimes.length - 1]) : null;
  el.textContent = latest
    ? (marketOpenNow() ? `Put prices as of ${latest}.` : `Last known put prices, from ${latest} (the market is closed now).`)
    : (currentRows.length ? "Put prices: loading." : "Put prices: \u2014");
}

// Maps the backend's internal mode string ("general"/"options"/"curated")
// back onto the UI's (mode dropdown, strict checkbox) pair — the inverse
// of getEffectiveMode(). Needed because the dropdown itself only ever has
// "general"/"options" as real values; "curated" is represented as
// Options + the Strict checkbox.
function syncModeControlsFrom(backendMode) {
  const mSel = document.getElementById("modeSelect");
  if (Array.from(mSel.options).some(o => o.value === backendMode)) {
    mSel.value = backendMode;
  }
}

async function whereIs() {
  const out = document.getElementById("whereAnswer");
  const sym = document.getElementById("whereInput").value.trim().toUpperCase();
  if (!sym) { out.textContent = ""; return; }
  const row = currentRows.find(r => r.symbol === sym);
  if (row) {
    if (row.error) { out.textContent = `${sym}: could not be loaded (${row.error}).`; return; }
    const shownIdx = getDisplayRows().findIndex(r => r.symbol === sym);
    if (shownIdx >= 0) {
      out.textContent = `${sym} is in the table (highlighted below).`;
      const tr = document.querySelector(`tbody tr[data-sym="${sym}"]`);
      if (tr) { tr.scrollIntoView({behavior: "smooth", block: "center"}); tr.classList.remove("flashrow"); void tr.offsetWidth; tr.classList.add("flashrow"); }
      return;
    }
    const styleHid = styleFilteredRows([row]).length === 0;
    const toggles = [["hideFarDteToggle", "Hide puts lasting over 45 days"], ["hideRiskyDeltaToggle", "Hide over a 30% chance of buying"], ["hideLowRorToggle", "Hide paying under 2.5%"], ["hideUnknownToggle", "Hide no put found"], ["hideNonUptrendToggle", "Hide without a 12-18 month trend"]];
    const countFor = window.__countFor || {};
    const hitBy = toggles.filter(([id]) => { const cb = document.getElementById(id); return cb && cb.checked && countFor[id] && countFor[id](row); }).map(x => x[1]);
    if (styleHid) { out.textContent = `${sym} is hidden by the strictness level (it barely moves). Choose Low strictness to see it, or type it in "Or look up just these tickers".`; return; }
    out.textContent = hitBy.length ? `${sym} is hidden by the filter "${hitBy[0]}". Untick that filter to see it.` : `${sym} is loaded but hidden. Try unticking the filters.`;
    return;
  }
  out.textContent = "Checking...";
  try {
    const res = await fetch(`/api/why?symbol=${encodeURIComponent(sym)}`);
    const d = await res.json();
    if (d.status === "removed") out.textContent = `${sym} was removed by the scan because ${d.reason}. To see it anyway, type it in "Or look up just these tickers" above.`;
    else if (d.status === "not_in_list") out.textContent = `${sym} is not in the stock list that was scanned. Pick a bigger list, or type it in "Or look up just these tickers" above.`;
    else if (d.status === "in_table") out.textContent = `${sym} is in the table.`;
    else out.textContent = `${sym} was not part of the last scan. Type it in "Or look up just these tickers" above to check it.`;
  } catch (e) {
    out.textContent = "Could not check right now.";
  }
}

// US stock market hours: 9:30am to 4:00pm Eastern, Monday to Friday (holidays ignored).
function marketOpenNow() {
  try {
    const parts = new Intl.DateTimeFormat("en-US", {timeZone: "America/New_York", weekday: "short", hour: "numeric", minute: "numeric", hour12: false}).formatToParts(new Date());
    const get = t => (parts.find(p => p.type === t) || {}).value;
    const wd = get("weekday"), mins = parseInt(get("hour"), 10) * 60 + parseInt(get("minute"), 10);
    return ["Mon", "Tue", "Wed", "Thu", "Fri"].includes(wd) && mins >= 570 && mins < 960;
  } catch (e) { return true; }
}

function setFetchStatus(text) {
  const el = document.getElementById("fetchStatus");
  if (el) el.textContent = text;
}

let shownCount = 0;
function fetchLabel() {
  if (!shownCount) return "Update put prices";
  const secs = Math.round(shownCount * 1.5);
  const est = secs < 50 ? `about ${Math.max(5, Math.round(secs / 5) * 5)} sec` : `about ${Math.max(1, Math.round(secs / 60))} min`;
  return `Update put prices (${shownCount} stocks, ${est})`;
}

function updateListLabel() {
  const el = document.getElementById("listLabel");
  const s = window.__lastStatusObj;
  if (!el || !s || !s.generated_at) { if (el) el.textContent = ""; return; }
  let name;
  if (s.custom_symbols_count) {
    name = `your own list of ${s.custom_symbols_count} tickers`;
  } else {
    const sel = document.getElementById("universeSelect");
    const opt = Array.from(sel.options).find(o => o.value === s.universe);
    name = opt ? opt.textContent.split(" \u2014 ")[0] : "the chosen list";
    if (name.startsWith("Saved list")) name = name.replace(/ \(.*$/, "");
    if (s.requested_limit) name += ` (first ${s.requested_limit} only)`;
  }
  const d = new Date(s.generated_at);
  el.textContent = `Results from: ${name}, scanned ${d.toLocaleDateString()} ${d.toLocaleTimeString()}.`;
}

async function pollStatus() {
  const res = await fetch("/api/status");
  const s = await res.json();
  window.__lastStatusObj = s;
  updateListLabel();
  updateListChangeCue();
  window.__lastKnownStatus = s.status;
  const statusEl = document.getElementById("status");
  const btn = document.getElementById("scanBtn");
  const customBtn = document.getElementById("customRunBtn");
  const createBtn = document.getElementById("createUniverseBtn");
  if (s.status === "scanning" || s.status === "creating_universe") {
    const creating = s.status === "creating_universe";
    statusEl.textContent = creating
      ? `creating new universe (full Russell 3000 re-screen) — ${s.progress}/${s.total} (${s.current_symbol || ""})...`
      : s.append_mode
        ? ""
        : `scanning ${s.progress}/${s.total} (${s.current_symbol || ""})...`;
    if (s.append_mode) document.getElementById("customStatus").textContent = `Adding your tickers... ${s.progress} of ${s.total} (${s.current_symbol || ""})`;
    // Only show Create New Universe as busy when IT is actually what's
    // running — not for every ordinary scan (that used to make this
    // button look like it was running on its own any time you clicked
    // "Run scan" for something unrelated).
    if (creating) {
      setBusy(createBtn, `${s.progress}/${s.total}`);
      clearBusy(btn);
      clearBusy(customBtn);
    } else {
      setBusy(btn, `${s.progress}/${s.total}`);
      setBusy(customBtn, `${s.progress}/${s.total}`);
      clearBusy(createBtn);
    }
    updateCancelButtonVisibility();
    setTimeout(pollStatus, 800);
  } else {
    clearBusy(btn);
    clearBusy(customBtn);
    clearBusy(createBtn);
    updateCancelButtonVisibility();
    if (window.__awaitingAppend && s.append_mode && (s.status === "done" || s.status === "cancelled" || s.status === "error")) {
      window.__awaitingAppend = false;
      // A small merge-only scan — never touches Universe/limit/mode
      // selects or last-scan-config, so none of the normal done/cancelled
      // sync logic below applies. Just reload the (now-merged) rows.
      const customEl = document.getElementById("customStatus");
      if (s.status === "error") {
        customEl.textContent = `Could not add your tickers: ${s.error}`;
      } else {
        await loadData();
        const qualNote = s.last_append_added < s.total ? ` (${s.total - s.last_append_added} could not be loaded)` : "";
        customEl.textContent = s.status === "cancelled"
          ? `Stopped after ${s.progress} of ${s.total}. They stay until the next scan.`
          : `Added ${s.last_append_added} of ${s.total}${qualNote}. Look for the star.`;
        if (s.status === "done") fetchAllOptions(false);  // only fetches rows still missing options, i.e. the new ones
      }
      return;
    }
    if (s.status === "cancelled") {
      const source = s.custom_symbols_count ? `your list of ${s.custom_symbols_count}` : `${s.total} stocks`;
      statusEl.textContent = `Stopped at ${s.progress} of ${s.total} (you cancelled it, or Yahoo Finance briefly limited the requests, so wait a few minutes before scanning again). ${s.row_count} stocks are shown.`;
      await refreshUniverseDropdown();
      await loadData();
      return;
    }
    if (s.status === "done") {
      await refreshUniverseDropdown();
      if (s.last_created_universe) {
        // Just-created universe — auto-select it so the next "Run scan"
        // uses it immediately, and surface a clear confirmation message
        // instead of the generic scan-summary text below.
        const uSel = document.getElementById("universeSelect");
        uSel.value = `generated:${s.last_created_universe.id}`;
        onUniverseChange();
        statusEl.textContent = `created "${s.last_created_universe.name}" — selected as the current universe.`;
        return;
      }
      const source = s.custom_symbols_count ? `your list of ${s.custom_symbols_count}` : `${s.total} stocks`;
      const lowVolNote = s.removed_low_vol ? `, ${s.removed_low_vol} too calm` : "";
      const fundNote = s.removed_failed_fundamentals ? `, ${s.removed_failed_fundamentals} with weak finances` : "";
      statusEl.textContent = s.custom_symbols_count
        ? `Done: ${s.row_count} of your ${s.custom_symbols_count} tickers loaded.`
        : `Done: checked ${s.base_total || s.total} stocks and kept ${s.row_count - (s.forced_count || 0)}. Removed by the scan: ${s.removed_no_uptrend} whose price is not rising${lowVolNote}${fundNote}.`;
      syncModeControlsFrom(s.mode);
      if (!s.custom_symbols_count) {
        syncLimitSelect(s.requested_limit);
        const uSel = document.getElementById("universeSelect");
        if (Array.from(uSel.options).some(o => o.value === s.universe)) uSel.value = s.universe;
        onUniverseChange();
      }
      updateAsOf(s.generated_at);
      await loadData();
      const shown = getDisplayRows().length;
      if (shown > 0 && shown <= AUTO_FETCH_OPTIONS_MAX_ROWS) {
          fetchAllOptions(false);  // not awaited; forced false regardless of the checkbox — a page reload or scan-done poll must never trigger a full force-refresh on its own
      }
    } else if (s.status === "error") {
      statusEl.textContent = `error: ${s.error}`;
    } else {
      statusEl.textContent = "idle";
    }
  }
}

// Repopulates the Universe dropdown's "Generated" optgroup from the
// server's list of saved universes (see engine/generated_universes.py),
// preserving the current selection if it's still valid. Called on page
// load and after any scan or universe-create/delete completes.
async function refreshUniverseDropdown() {
  const group = document.getElementById("generatedUniverseGroup");
  if (!group) return;
  const uSel = document.getElementById("universeSelect");
  const previousValue = uSel.value;
  try {
    const res = await fetch("/api/universes");
    const universes = await res.json();
    group.innerHTML = "";
    universes.forEach(u => {
      const opt = document.createElement("option");
      opt.value = `generated:${u.id}`;
      const d = new Date(u.generated_at * 1000);
      const lvl = {general: "Low", options: "Medium", curated: "High"}[u.generated_under_mode] || "";
      opt.textContent = `Saved list (${u.size} stocks, made ${d.toLocaleDateString()}${lvl ? " under " + lvl + " strictness" : ""})`;
      group.appendChild(opt);
    });
    if (Array.from(uSel.options).some(o => o.value === previousValue)) {
      uSel.value = previousValue;
    }
    updateListLabel();
    onUniverseChange();
  } catch (e) {
    // leave whatever was there — a transient fetch failure shouldn't wipe the dropdown
  }
}

function updateListChangeCue() {
  const cue = document.getElementById("listChangeCue");
  const s = window.__lastStatusObj;
  if (!cue || !s || !s.generated_at || s.custom_symbols_count) { if (cue) cue.style.display = "none"; return; }
  const uni = document.getElementById("universeSelect").value;
  const lim = document.getElementById("limitSelect").value;
  const lastLim = s.requested_limit ? String(s.requested_limit) : "all";
  cue.style.display = (uni !== s.universe || lim !== lastLim) ? "" : "none";
}

function onUniverseChange() {
  updateListChangeCue();
  const note = document.getElementById("universeNote");
  const v = document.getElementById("universeSelect").value;
  if (note) note.textContent = v.startsWith("generated:")
    ? "This saved list is a shortcut: stocks that already passed a full scan, so scanning it is quick."
    : "";
  const uSel = document.getElementById("universeSelect");
  const delBtn = document.getElementById("deleteUniverseBtn");
  // Always visible, just grayed out (disabled) for universes that can't
  // be deleted — the built-in S&P 500/Russell 2000/3000 ones — rather
  // than disappearing, so its existence and purpose stay discoverable.
  if (delBtn) {
    const isGen = uSel.value.startsWith("generated:");
    delBtn.disabled = !isGen;
    delBtn.textContent = isGen ? `Delete "${uSel.options[uSel.selectedIndex].textContent}"` : "Delete the selected saved list";
    delBtn.title = isGen ? "Permanently deletes this saved list. You will be asked to confirm." : "Only lists you created can be deleted, not the built-in S&P 500 or Russell lists.";
  }
}

async function createNewUniverse() {
  if (!confirm("This scans about 2,600 stocks, takes about 3 minutes and makes thousands of requests. Continue?")) return;
  setBusy(document.getElementById("createUniverseBtn"), "Starting...");
  const mode = getEffectiveMode();
  const res = await fetch(`/api/universes?mode=${mode}`, {method: "POST"});
  if (res.status === 409) {
    clearBusy(document.getElementById("createUniverseBtn"));
    document.getElementById("status").textContent = "scan already running";
    return;
  }
  pollStatus();
}

async function deleteSelectedUniverse() {
  const uSel = document.getElementById("universeSelect");
  if (!uSel.value.startsWith("generated:")) return;
  const name = uSel.options[uSel.selectedIndex].textContent;
  if (!confirm(`Delete "${name}"? This can't be undone from the UI.`)) return;
  const uid = uSel.value.split(":")[1];
  const res = await fetch(`/api/universes/${uid}`, {method: "DELETE"});
  if (res.ok) {
    uSel.value = "russell3000";
    await refreshUniverseDropdown();
  }
}

async function loadData() {
  const res = await fetch("/api/data");
  const d = await res.json();
  currentRows = d.rows || [];
  renderTable();
  updateAsOf(d.generated_at);
}

async function startScan() {
  const limit = document.getElementById("limitSelect").value;
  const universe = document.getElementById("universeSelect").value;
  setFetchStatus("");
  const big = {russell2000: "2,000", russell3000: "2,600"}[universe];
  if (big && limit === "all" && !confirm(`This scans about ${big} stocks and can take several minutes. Continue?`)) return;
  const mode = getEffectiveMode();
  setBusy(document.getElementById("scanBtn"), "Starting...");
  document.getElementById("status").textContent = "starting...";
  const res = await fetch(`/api/scan?limit=${limit}&universe=${universe}&mode=${mode}`, {method: "POST"});
  if (res.status === 409) {
    document.getElementById("status").textContent = "scan already running";
  }
  pollStatus();
}

function getCustomMode() {
  return document.getElementById("customScanMode").value;
}

document.addEventListener("click", ev => {
  const a = ev.target.closest && ev.target.closest("a.reason");
  if (!a) return;
  const el = document.getElementById(a.getAttribute("data-for"));
  if (!el) return;
  const target = el.closest("label") || el;
  target.scrollIntoView({behavior: "smooth", block: "center"});
  target.classList.remove("flash"); void target.offsetWidth; target.classList.add("flash");
});

function updateCustomRunLabel() {
  const btn = document.getElementById("customRunBtn");
  const add = getCustomMode() === "add";
  btn.textContent = "Run";
  btn.classList.toggle("warn", !add);
  delete btn.dataset.origText;  // setBusy/clearBusy remember the label; force it to re-read the new one
}

async function startCustomRun() {
  const raw = document.getElementById("customSymbols").value.trim();
  if (!raw) return;
  const append = getCustomMode() === "add";
  if (!append && !confirm("This replaces the whole table with just your tickers until the next scan. Continue?")) return;
  const mode = document.getElementById("modeSelect").value;
  const customStatusEl = document.getElementById("customStatus");
  window.__awaitingAppend = append;
  setBusy(document.getElementById("customRunBtn"), append ? "Adding..." : "Starting...");
  (append ? customStatusEl : document.getElementById("status")).textContent = append ? "Adding your tickers..." : "starting...";
  const res = await fetch(`/api/scan?mode=${mode}${append ? "&append=1" : ""}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({symbols: raw}),
  });
  const msgEl = append ? customStatusEl : document.getElementById("status");
  if (res.status === 409) {
    msgEl.textContent = "Something is already running. Wait for it to finish.";
  } else if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    msgEl.textContent = data.message || "That list of tickers is not valid.";
  }
  pollStatus();  // resolves the busy state above either way (running -> progress spinner, or reverts to idle label)
}

document.querySelectorAll('#columnToggles input[data-col]').forEach(cb => {
  cb.addEventListener("change", () => {
    const col = cb.getAttribute("data-col");
    if (cb.checked) hiddenColumns.delete(col); else hiddenColumns.add(col);
    renderTable();
  });
});

// ---------------- Sector Signals tab ----------------
function showTab(name) {
  document.getElementById("tabPut").style.display = name === "put" ? "" : "none";
  document.getElementById("tabSectors").style.display = name === "sectors" ? "" : "none";
  document.getElementById("tabBtnPut").classList.toggle("active", name === "put");
  document.getElementById("tabBtnSectors").classList.toggle("active", name === "sectors");
  try { localStorage.setItem("activeTab", name); } catch (e) {}
  if (name === "sectors" && !window.__sectorsLoaded) loadSectors();
}

let sectorResults = [];
async function loadSectors() {
  try {
    const res = await fetch("/api/sectors");
    const d = await res.json();
    sectorResults = d.results || [];
    window.__sectorsLoaded = true;
    const asOf = document.getElementById("sectorAsOf");
    asOf.textContent = sectorAsOfText(d);
    renderSectors();
  } catch (e) {}
  pollSectorStatus();
}

async function sectorScan() {
  const btn = document.getElementById("sectorScanBtn");
  setBusy(btn, "Starting...");
  window.__sectorScanStarted = true;
  try {
    const res = await fetch("/api/sectors/scan", {method: "POST"});
    if (res.status === 409) document.getElementById("sectorStatus").textContent = "Already running.";
  } catch (e) {
    clearBusy(btn, "Scan sectors");
    document.getElementById("sectorStatus").textContent = "Could not reach the server.";
    return;
  }
  pollSectorStatus();
}

async function pollSectorStatus() {
  if (window.__sectorPolling) return;
  window.__sectorPolling = true;
  try { await pollSectorStatusOnce(); } finally { window.__sectorPolling = false; }
}

async function pollSectorStatusOnce() {
  let s;
  try {
    const res = await fetch("/api/sectors/status");
    s = await res.json();
  } catch (e) {
    document.getElementById("sectorStatus").textContent = "Could not reach the server.";
    clearBusy(document.getElementById("sectorScanBtn"), "Scan sectors");
    return;
  }
  const btn = document.getElementById("sectorScanBtn");
  const st = document.getElementById("sectorStatus");
  if (s.status === "scanning") {
    setBusy(btn, s.phase === "prices" ? "Loading prices..." : `${s.progress}/${s.total}`);
    st.textContent = s.phase === "prices" ? "Loading a year of daily prices for __N_ALL__ symbols..." : `Checking sector ${s.progress} of ${s.total} for patterns...`;
    setTimeout(pollSectorStatus, 700);
    return;
  }
  clearBusy(btn, "Scan sectors");
  if (s.status === "error") { st.textContent = `Error: ${s.error}`; return; }
  if (s.status === "done") {
    st.textContent = window.__sectorScanStarted ? "Done." : "Last scan loaded.";
    if (!window.__sectorsRendered || window.__lastSectorGen !== s.generated_at) {
      window.__lastSectorGen = s.generated_at;
      const r = await fetch("/api/sectors"); const d = await r.json();
      sectorResults = d.results || [];
      document.getElementById("sectorAsOf").textContent = sectorAsOfText(d);
      renderSectors();
    }
  } else {
    st.textContent = "idle";
  }
}

function sectorAsOfText(d) {
  if (!d.generated_at) return "Not scanned yet. Press Scan sectors.";
  const dates = (d.results || []).flatMap(g => [g.fund.as_of].concat(g.holdings.map(h => h.as_of))).filter(Boolean).sort();
  const through = dates.length ? new Date(dates[dates.length - 1] + "T00:00:00").toLocaleDateString() : "?";
  const g = new Date(d.generated_at);
  return `Prices through ${through} (end-of-day candles). Scanned ${g.toLocaleDateString()}, ${g.toLocaleTimeString()}.`;
}

function sigBadge(sig) {
  const st = sig.status === "confirmed" ? "confirmed" : (sig.status === "forming" ? "forming since" : "latest");
  const when = new Date(sig.date + "T00:00:00").toLocaleDateString(undefined, {month: "short", day: "numeric"});
  const safe = (sig.detail || "").replace(/"/g, "&quot;");
  const short = sig.name.replace("First red candle above 21-day EMA", "1st red above 21 EMA").replace("First green candle below 21-day EMA", "1st green below 21 EMA").replace("Inverse head and shoulders", "Inverse head &amp; shoulders").replace("Head and shoulders", "Head &amp; shoulders");
  const why = document.getElementById("sectorShowDetails").checked ? `<span class="why">${safe}</span>` : "";
  return `<span class="sig" title="${safe}"><span class="badge ${sig.direction}">${short}</span><span class="st">${st} ${when}</span>${why}</span>`;
}

function esc(t) { return String(t === null || t === undefined ? "" : t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }

function leanOf(sigs) {
  let score = 0, bull = 0, bear = 0;
  (sigs || []).forEach(sg => {
    const w = sg.status === "confirmed" ? 2 : (sg.status === "forming" ? 1 : 0);
    if (sg.direction === "bullish") { score += w; bull += w; }
    if (sg.direction === "bearish") { score -= w; bear += w; }
  });
  if (bull === 0 && bear === 0) return '<span class="check-u">&ndash;</span>';
  if (bull > 0 && bear > 0 && score === 0) return '<span class="check-u" title="Bullish and bearish signals cancel out">mixed</span>';
  return score > 0 ? '<span class="check-y" title="More bullish than bearish signals">bullish</span>' : '<span class="check-n" title="More bearish than bullish signals">bearish</span>';
}

function sectorRowCells(r, label, leanSigs) {
  const sigs = (r.signals || []);
  const price = r.last_close === null || r.last_close === undefined ? "?" : r.last_close.toFixed(2);
  const chg = r.change_pct === null || r.change_pct === undefined ? "?" : `<span class="${r.change_pct > 0 ? "check-y" : r.change_pct < 0 ? "check-n" : ""}">${r.change_pct > 0 ? "+" : ""}${(r.change_pct * 100).toFixed(2)}%</span>`;
  const ema = r.above_ema21 === null || r.above_ema21 === undefined ? "?" : (r.above_ema21 ? '<span class="check-y">above</span>' : '<span class="check-n">below</span>');
  const weight = r.weight === null || r.weight === undefined ? "" : `${(r.weight * 100).toFixed(1)}%`;
  const sigHtml = r.error ? `<span class="check-u">${esc(r.error)}</span>` : (sigs.length ? sigs.map(sigBadge).join("") : '<span class="check-u">none</span>');
  return `<td>${label}</td><td>${weight}</td><td>${price}</td><td>${chg}</td><td>${ema}</td><td>${leanOf(leanSigs || sigs)}</td><td class="sigs">${sigHtml}</td>`;
}

function sectorFilter(r) {
  const only = document.getElementById("sectorOnlySignals").checked;
  const dir = document.getElementById("sectorDirection").value;
  const hideForming = document.getElementById("sectorHideForming").checked;
  let sigs = r.signals || [];
  if (dir !== "all") sigs = sigs.filter(s => s.direction === dir);
  if (hideForming) sigs = sigs.filter(s => s.status !== "forming");
  return {keep: !only || sigs.length > 0 || !!r.error, sigs};
}

function renderSectors() {
  const body = document.getElementById("sectorBody");
  body.innerHTML = "";
  let shownStocks = 0, fundRows = 0, totalStocks = 0, withSig = 0, fundsWithSig = 0, unchecked = 0;
  sectorResults.forEach(group => {
    const fund = Object.assign({}, group.fund, {});
    const ff = sectorFilter(fund);
    const rows = group.holdings.map(h => ({h, f: sectorFilter(h)}));
    totalStocks += group.holdings.filter(h => !h.error).length;
    unchecked += group.holdings.filter(h => h.error).length;
    fundsWithSig += (fund.signals && fund.signals.length ? 1 : 0);
    withSig += group.holdings.filter(h => h.signals && h.signals.length).length;
    const anyShown = ff.keep || rows.some(x => x.f.keep);
    if (!anyShown) return;
    const tr = document.createElement("tr");
    tr.className = "etf-row";
    tr.innerHTML = sectorRowCells(Object.assign({}, fund, {signals: ff.sigs, weight: null}), `<b>${esc(group.etf)}</b> <span class="co">${esc(group.sector)} fund</span>`, fund.signals);
    body.appendChild(tr);
    fundRows++;
    rows.forEach(({h, f}) => {
      if (!f.keep) return;
      const r = document.createElement("tr");
      r.innerHTML = sectorRowCells(Object.assign({}, h, {signals: f.sigs}), `<b>${esc(h.symbol)}</b><span class="co" title="${esc(h.name)}">${esc(h.name)}</span>`, h.signals);
      body.appendChild(r);
      shownStocks++;
    });
  });
  const c = document.getElementById("sectorCount");
  c.textContent = sectorResults.length ? `${withSig} of ${totalStocks} stocks and ${fundsWithSig} of ${sectorResults.length} funds have at least one signal. Showing ${shownStocks} stocks under ${fundRows} fund headings.${unchecked ? ` ${unchecked} stock${unchecked === 1 ? "" : "s"} could not be checked (shown with the reason).` : ""}` : "";
  window.__sectorsRendered = true;
}

(function restoreTab() {
  let t = "put";
  try { t = localStorage.getItem("activeTab") || "put"; } catch (e) {}
  if (t === "sectors") showTab("sectors");
})();

// initial load
loadData();
pollStatus();
applyModePreset();
updateCustomRunLabel();
refreshUniverseDropdown();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return INDEX_HTML


INDEX_HTML = (INDEX_HTML.replace("__N_ETF__", str(len(SECTOR_ETFS)))
              .replace("__N_HOLD__", str(len(SECTOR_ETFS) * 10))
              .replace("__N_ALL__", str(len(SECTOR_ETFS) * 11)))


def run_dashboard(port: int = 8899, initial_limit: int | None = None):
    _load_sector_cache()
    # Resume the last scan you actually ran, if one's on record — takes
    # priority over --limit, which was really only ever a "keep my own
    # testing cheap" convenience, not something meant to override a real
    # remembered session. Falls back to --limit / hardcoded defaults only
    # on a genuinely first-ever run (no persisted config yet).
    last = _load_last_scan_config()
    if last:
        args = (last.get("limit"), last.get("custom_symbols"), last.get("universe", "russell3000"), last.get("mode", "options"))
    else:
        # First-ever run: a small S&P 500 scan so a fresh install finishes
        # in about a minute, not a full ~2,600-symbol Russell 3000 pull.
        args = (initial_limit if initial_limit is not None else 50, None, "sp500", "options")
    t = threading.Thread(target=_run_scan, args=args, daemon=True)
    t.start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    run_dashboard()
