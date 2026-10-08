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
    "last_append_added": 0,  # how many of the requested symbols actually passed the live gate and got merged in
}


_BUSY_STATUSES = {"scanning", "creating_universe"}

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

        def progress_cb(i, total, sym):
            with _state_lock:
                _state["progress"] = i
                _state["total"] = total
                _state["current_symbol"] = sym

        min_realized_vol = MODE_MIN_REALIZED_VOL.get(mode)
        require_curated_fundamentals = MODE_REQUIRE_CURATED_FUNDAMENTALS.get(mode, False)
        rows, removed, removed_low_vol, removed_failed_fundamentals = scan_universe(
            symbols, progress_cb=progress_cb, require_uptrend=True, min_realized_vol=min_realized_vol,
            require_curated_fundamentals=require_curated_fundamentals, cancel_event=_cancel_event,
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
            return jsonify({"ok": True, "put": row["put"], "options_error": row.get("options_error")})
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

    put, error = fetch_option_for_symbol(symbol, spot)
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
                r["spot"] = spot
                r["day_change_pct"] = day_change_pct
                if quote_as_of:
                    r["quote_as_of"] = quote_as_of
                break

    return jsonify({
        "ok": True, "put": put_dict, "options_error": error,
        "spot": spot, "day_change_pct": day_change_pct, "quote_as_of": quote_as_of,
    })


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
<title>CSP Candidates</title>
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
  .refresh-option-btn { padding: 0 4px; margin-left: 4px; border: none; background: transparent; color: #888; cursor: pointer; font-size: 12px; line-height: 1; }
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
  th, td { padding: 6px 7px; text-align: right; border-bottom: 1px solid #eee; white-space: nowrap; }
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
</style>
</head>
<body>
  <h1>Cash-Secured Put Candidates</h1>
  <div class="sub">Ranked by RoR% (premium ÷ net capital at risk, over the contract's actual DTE — same convention as thinkorswim, NOT normalized to a common DTE) of the put nearest-or-above -0.20 delta, per symbol. Because it's not DTE-normalized, a longer-dated contract's RoR% isn't directly comparable to a shorter-dated one's — see the RoR% column tooltip. Every stock shown here is already confirmed to be above a rising 30-week moving average (Weinstein Stage 2 — Section 12 criterion 1) — anything that isn't gets removed automatically before this list is built, so there's no separate Uptrend column. The scan itself never touches the options chain — click "Get Option" on a row (or "Fetch options for all shown") to look one up on demand. Suggest-only — nothing here places an order.</div>
  <div id="asOf" class="as-of">Data as of: —</div>

  <div class="controls">
    <label>Mode: </label>
    <select id="modeSelect" onchange="applyModePreset()">
      <option value="general">General</option>
      <option value="options" selected>Options Trading</option>
    </select>
    <span id="modeNote" class="sub" style="margin:0;"></span>
  </div>
  <div class="controls">
    <button id="createUniverseBtn" onclick="createNewUniverse()">Create New Universe (full re-screen, ~3 min)</button>
    <span class="sub" style="margin:0;">Screens the full Russell 3000 under the CURRENT mode's live criteria (above) and saves whatever survives as a new, named entry in the Universe dropdown below — a fast starting pool for future scans. Never overwrites an existing one; old ones stay until you delete them.</span>
  </div>
  <div class="controls">
    <label>Universe: </label>
    <select id="universeSelect" onchange="onUniverseChange()">
      <option value="sp500">S&amp;P 500 (~500)</option>
      <option value="russell2000">Russell 2000 (~2000, small-cap)</option>
      <option value="russell3000" selected>Russell 3000 (~2600, ~98% of US market)</option>
      <optgroup id="generatedUniverseGroup" label="Generated"></optgroup>
    </select>
    <button id="deleteUniverseBtn" onclick="deleteSelectedUniverse()" disabled title="Only universes you created with &quot;Create New Universe&quot; can be deleted — the built-in ones (S&amp;P 500, Russell 2000/3000) can't.">Delete this universe</button>
    <label>Scan: </label>
    <select id="limitSelect">
      <option value="25">First 25</option>
      <option value="50">First 50</option>
      <option value="100">First 100</option>
      <option value="all" selected>Full list</option>
    </select>
    <button id="scanBtn" onclick="startScan()">Run scan</button>
    <button id="fetchAllBtn" onclick="fetchAllOptions()">Fetch options for all shown</button>
    <label class="toggle" title="Re-fetch options even for rows that already have a result when you click &quot;Fetch options for all shown&quot; — premiums move during the day. Doesn't apply to the automatic fetch after a scan or on page load."><input type="checkbox" id="forceRefreshToggle" checked> Force refresh</label>
    <button id="cancelBtn" onclick="cancelRun()" style="display:none; background:#dc2626;">Cancel</button>
    <span id="status">idle</span>
  </div>
  <div class="controls">
    <label class="toggle"><input type="checkbox" id="hideFarContractsToggle" checked onchange="renderTable()"> Hide DTE &gt; 45 or Δ &gt; 0.30</label>
    <label class="toggle"><input type="checkbox" id="hideNonUptrendToggle" onchange="renderTable()"> Hide non-uptrend (12/18mo)</label>
    <label class="toggle" id="strictToggleLabel" title="Hides anything failing: profitable, FCF+ or 2yr+ cash runway, D/E <= 150%, ROE >= 10%, and a higher realized-vol floor"><input type="checkbox" id="strictToggle" onchange="applyModePreset()"> Hide non-Strict (fundamentals + higher vol floor)</label>
    <label class="toggle" title="A row showing &quot;-&quot; has no fetched put yet (or none found) — unknown, not bad — so it's kept regardless of this toggle. Use &quot;Hide unknown/-&quot; to hide those instead."><input type="checkbox" id="hideLowRorToggle" checked onchange="renderTable()"> Hide RoR% &lt; 2.5%</label>
    <label class="toggle" title="Hides rows where options were checked but nothing qualified (shown as &quot;-&quot;) — often a thin/incomplete options chain on Yahoo's free feed, not necessarily a bad stock. Rows not yet fetched still show their Get Option button regardless."><input type="checkbox" id="hideUnknownToggle" onchange="renderTable()"> Hide unknown/-</label>
  </div>

  <div class="controls">
    <label>Or paste your own list: </label>
    <textarea id="customSymbols" placeholder="AAPL, MSFT, KO"></textarea>
    <select id="customScanMode" title="Add: merges into the current list, doesn't touch the Universe, temporary — next scan drops it. Replace: normal scan of just this list.">
      <option value="add" selected>Add to current list</option>
      <option value="replace">Replace current list</option>
    </select>
    <button id="customRunBtn" onclick="startCustomRun()">Run</button>
  </div>

  <div class="controls" id="columnToggles">
    <label>Show columns:</label>
    <label class="toggle"><input type="checkbox" data-col="fcf_positive"> FCF+</label>
    <label class="toggle"><input type="checkbox" data-col="eps_beat" checked> EPS</label>
    <label class="toggle"><input type="checkbox" data-col="next_earnings" checked> Earnings</label>
    <label class="toggle"><input type="checkbox" data-col="rev_growth" checked> Rev Growth</label>
    <label class="toggle"><input type="checkbox" data-col="uptrend_legacy" checked> 12/18mo Uptrend</label>
    <label class="toggle"><input type="checkbox" data-col="roe"> ROE</label>
    <label class="toggle"><input type="checkbox" data-col="beta" checked> Beta</label>
    <label class="toggle"><input type="checkbox" data-col="realized_vol"> Vol</label>
    <label class="toggle"><input type="checkbox" data-col="vol_pctile"> Vol %ile</label>
    <label class="toggle"><input type="checkbox" data-col="short_pct_float"> Short % Float</label>
  </div>

  <div class="table-wrap">
  <table id="dataTable">
    <thead>
      <tr id="headerRow"></tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
  </div>

  <div id="tooltipBox"></div>

<script>
const CHECK_KEYS = ["pe_ok","fcf_positive","eps_beat","debt_to_equity","roe","rev_growth","uptrend_legacy","csp_trigger"];
// Which columns the checkboxes above the table can hide. Everything else
// (pe_ok, debt_to_equity, roe, csp_trigger, and all the put/option columns)
// is always shown.
const TOGGLEABLE_COLUMNS = new Set(["fcf_positive","eps_beat","next_earnings","rev_growth","roe","beta","realized_vol","vol_pctile","uptrend_legacy","short_pct_float"]);
// Must match which checkboxes above start unchecked in the HTML (fcf_positive, roe) —
// these are the two quality filters NOT in the source guidebook (see ROE tooltip),
// off by default so the table defaults closer to doc-only criteria. short_pct_float
// is off by default everywhere too — squeeze-screening signal, not part of the CSP
// strategy itself, and Yahoo's short interest data is always 1-2 weeks stale.
const hiddenColumns = new Set(["fcf_positive","roe","realized_vol","vol_pctile","short_pct_float"]);
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
  general: {fcf_positive: false, roe: false, rev_growth: true, eps_beat: true, next_earnings: true, beta: true, realized_vol: false, vol_pctile: false, uptrend_legacy: true, short_pct_float: false},
  options: {fcf_positive: false, roe: false, rev_growth: false, eps_beat: true, next_earnings: true, beta: true, realized_vol: true, vol_pctile: true, uptrend_legacy: true, short_pct_float: false},
  curated: {fcf_positive: true, roe: true, rev_growth: false, eps_beat: true, next_earnings: true, beta: true, realized_vol: true, vol_pctile: false, uptrend_legacy: false, short_pct_float: false},
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
  const mode = document.getElementById("modeSelect").value;
  if (mode !== "options") return mode;
  const strict = document.getElementById("strictToggle");
  return (strict && strict.checked) ? "curated" : "options";
}

function applyModePreset() {
  const rawMode = document.getElementById("modeSelect").value;
  const mode = getEffectiveMode();
  const strictLabel = document.getElementById("strictToggleLabel");
  if (strictLabel) strictLabel.style.display = (rawMode === "options") ? "" : "none";

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
    note.textContent = `Hides stocks below ${(OPTIONS_MODE_MIN_REALIZED_VOL * 100).toFixed(0)}% realized volatility (not enough to carry rich premium) — applies instantly to what's loaded, and to any new scan.`;
  } else if (mode === "curated") {
    note.textContent = `Hides stocks below ${(CURATED_MODE_MIN_REALIZED_VOL * 100).toFixed(0)}% realized volatility, AND anything failing hard fundamentals gates (must be profitable, FCF+ or 2yr+ cash runway, D/E <= 150%, ROE >= 10%) — a much smaller, stricter list meant to survive real due diligence, not just a volatility cut. Applies instantly to what's loaded, and to any new scan.`;
  } else {
    note.textContent = "";
  }
  renderTable();
}

function isColumnVisible(key) {
  return !hiddenColumns.has(key);
}

const COLUMN_TOOLTIPS = {
  symbol: "Stock ticker. Colored green if every currently-shown check passed, red if any failed (only counts columns you haven't hidden).",
  sector: "GICS sector classification.",
  spot: "Current stock price.",
  change_pct: "Price change vs. yesterday's close.",
  beta: "Volatility relative to the overall market: 1.0 moves with the market, >1 is more volatile (amplified moves both ways), <1 is less. Higher beta generally means richer option premium — a cheap way to spot 'wheel-friendly' names without checking the options chain at all. Not in the source guidebook. Note: beta only captures MARKET-correlated volatility — a stock can carry rich premium from purely idiosyncratic moves beta won't reflect. Realized Vol (next column) is the more direct signal.",
  realized_vol: "20-day annualized realized volatility, computed directly from price history (same number used internally to sanity-check option quotes). More direct than beta for 'will this carry rich premium,' since it captures ALL volatility, not just the market-correlated part. In Options Trading mode, anything below 25% is eliminated entirely, not just marked. Not in the source guidebook.",
  vol_pctile: "Where CURRENT realized vol ranks vs this stock's own trailing 1-year range (0-100). Different question from the Realized Vol column: that asks 'is this stock volatile enough in general,' this asks 'is NOW an elevated moment for THIS stock specifically' — closer to what IV Rank means for options traders (no free source gives true IV Rank for individual equities, on TradingView or elsewhere — this is a realized-vol-based proxy for it). A high-vol stock can still show a LOW percentile if it's currently calmer than its own norm, and vice versa. Not in the source guidebook.",
  short_pct_float: "Percent of the tradeable float currently sold short. A squeeze-screening signal, NOT part of the CSP income strategy — high short interest means a rising price can force short covering, accelerating the move. Yahoo's free data is ALWAYS 1-2 weeks stale (FINRA settles short interest twice a month, reported ~1-2 weeks later) and doesn't include cost-to-borrow or utilization, both of which mattered more than raw short% in 2021's GME squeeze. Treat as a slow-moving hint, not current data. Not in the source guidebook.",
  check_pe_ok: "Price-to-earnings ratio <= 100 — or, if higher, the company's free cash flow could pay off ALL its debt within about a year. A high P/E with debt that can't be covered is treated as too risky. Shown as the actual P/E number, colored green (pass) or red (fail).",
  check_fcf_positive: "Free cash flow (cash left over after running the business and capital spending) is positive over the trailing twelve months.",
  check_eps_beat: "Earnings per share (EPS = profit ÷ shares outstanding) for the most recently reported quarter: actual vs. what Wall Street analysts expected (consensus estimate). Green = beat, red = missed. This approximates the guide's 'double beat' rule; true revenue-beat data isn't freely available, so only the EPS half is checked.",
  check_debt_to_equity: "Debt-to-equity <= 200% (total debt no more than 2x shareholder equity), shown as the actual percentage. Financials and Utilities are exempted since those sectors structurally run higher debt as part of the business model.",
  check_roe: "Return on Equity (net profit ÷ shareholder equity) >= 15% — how efficiently the company turns shareholder equity into profit, shown as the actual percentage. This is a company-quality metric, NOT the same thing as the RoR% column on the right (that one is Return on Risk for the specific option contract) — the similar-looking abbreviations are unrelated to each other.",
  check_rev_growth: "Revenue grew year-over-year in the most recent quarter (not shrinking).",
  check_uptrend_legacy: "The project's ORIGINAL uptrend check, kept for comparison only — does NOT filter the list (the 30-week MA check does that). SMA50/100/200 stack + weekly higher-highs/higher-lows vs. 12 months ago OR 18 months ago, approximating the source doc's literal 'uptrend lasting 12 to 18 months' language. Shown so you can see how often it agrees or disagrees with the actual filter.",
  check_csp_trigger: "Section 17a entry trigger for selling a cash-secured put: 3 consecutive down days, or the stock dropped 5%+ in one day. Informational only right now — it does NOT remove rows, unlike the uptrend check.",
  next_earnings: "Date of the next scheduled earnings report, if known. Useful for CSPs since earnings can move the stock sharply before the option expires.",
  strike: "Strike price of the selected put. If assigned, this is the price per share you'd be obligated to buy at.",
  dte: "Days to expiration of the selected contract. Must be >= 30 (the doc's target); this is whichever expiration closest to 30 days actually had a contract reaching -0.20 delta — the search only moves further out if a nearer expiration doesn't reach it.",
  delta: "The put's delta, computed from the contract's own quoted implied volatility. Only shown if it lands between -0.20 and -0.30 — otherwise (e.g. no reliable quote anywhere close to that range) the row shows N/A rather than a low-confidence number.",
  premium: "Total premium collected for selling 1 contract (100 shares), at the bid/ask midpoint. Formula: mid price × 100.",
  ror: "Return on risk: premium ÷ NET capital at risk (strike × 100 − premium received), over the contract's actual DTE — matches thinkorswim's RoR calculation exactly (verified directly against a live ToS quote). Deliberately NOT normalized to a common DTE, so a 90-day contract's RoR% isn't apples-to-apples with a 30-day contract's — a longer-dated trade will tend to show a bigger raw number even if its annualized return is worse. The table is ranked by this, highest first. Unrelated to the ROE column on the left (company return on equity) despite the similar name.",
  oi: "Open interest — how many of this exact contract (same strike + expiration) are currently open across all traders. Higher usually means better liquidity (tighter bid/ask, easier to get filled near the quoted price). Very low OI (single digits) is a caution sign, not a disqualifier.",
};

function buildColumns() {
  return [
    {key: "symbol", label: "Symbol"},
    {key: "sector", label: "Sector"},
    {key: "spot", label: "Spot"},
    {key: "change_pct", label: "Chg %"},
    ...CHECK_KEYS.filter(isColumnVisible).map(k => ({key: "check_" + k, label: null})),
    ...(isColumnVisible("next_earnings") ? [{key: "next_earnings", label: "Earnings"}] : []),
    ...(isColumnVisible("beta") ? [{key: "beta", label: "Beta"}] : []),
    ...(isColumnVisible("realized_vol") ? [{key: "realized_vol", label: "Vol"}] : []),
    ...(isColumnVisible("vol_pctile") ? [{key: "vol_pctile", label: "Vol %ile"}] : []),
    ...(isColumnVisible("short_pct_float") ? [{key: "short_pct_float", label: "Short % Float"}] : []),
    {key: "strike", label: "Strike"},
    {key: "dte", label: "DTE"},
    {key: "delta", label: "Delta"},
    {key: "premium", label: "Premium"},
    {key: "ror", label: "RoR%"},
    {key: "oi", label: "OI"},
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
  if (!raw || raw.actual === null || raw.actual === undefined || raw.estimate === null || raw.estimate === undefined) {
    return checkMark(c.value);
  }
  const cls = c.value ? "check-y" : "check-n";
  return `<span class="${cls}">$${raw.actual.toFixed(2)} vs $${raw.estimate.toFixed(2)}</span>`;
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
    return `<td colspan="6"><button class="get-option-btn" onclick="getOption('${r.symbol}', this)">Get Option</button></td>`;
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
    return `<td colspan="5">-</td><td>- <button class="refresh-option-btn" title="Re-check for a matching contract" onclick="getOption('${r.symbol}', this)">&#8635;</button></td>`;
  }
  const p = r.put;
  let fallbackNote = "";
  if (p.is_fallback_pick) {
    fallbackNote = '<div class="fallback" title="No expiration in the search window had a clean, close-to-target match — this is the best validated (real bid/ask, sane IV) contract found regardless, which can be far from -0.20 delta.">no clean match — best available</div>';
  } else if (p.at_or_above_target_delta === false) {
    fallbackNote = '<div class="fallback">below target Δ</div>';
  }
  return `<td>${p.strike.toFixed(2)}</td><td>${p.dte}</td><td>${p.delta_abs.toFixed(2)}${fallbackNote}</td>` +
    `<td>$${p.premium.toFixed(0)}</td>` +
    `<td class="ror-cell">${(p.return_on_risk * 100).toFixed(2)}%</td>` +
    `<td>${p.open_interest} <button class="refresh-option-btn" title="Re-fetch this option — premiums move during the day" onclick="getOption('${r.symbol}', this, true)">&#8635;</button></td>`;
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
  cancelBtn.style.display = (serverBusy || fetchingAllOptions) ? "" : "none";
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
    clearBusy(btn, "Nothing to fetch — all shown rows already have options");
    setTimeout(() => clearBusy(btn), 2500);
    return;
  }
  fetchingAllOptions = true;
  optionsFetchCancelled = false;
  updateCancelButtonVisibility();
  for (let i = 0; i < targets.length; i++) {
    if (optionsFetchCancelled) {
      clearBusy(btn, `Cancelled — fetched ${i}/${targets.length}`);
      setTimeout(() => clearBusy(btn), 2500);
      fetchingAllOptions = false;
      updateCancelButtonVisibility();
      return;
    }
    setBusy(btn, `Fetching options... (${i + 1}/${targets.length})`);
    await getOption(targets[i], null, forceRefresh);
  }
  clearBusy(btn, "Fetch options for all shown");
  fetchingAllOptions = false;
  updateCancelButtonVisibility();
}

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
    th.textContent = label;
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

function getDisplayRows() {
  const mode = getEffectiveMode();
  // Error rows stay visible regardless of mode/toggle (they're a data
  // problem, not a "didn't qualify" result), in every branch below.
  let rows = currentRows;
  if (mode === "options") {
    rows = rows.filter(r =>
      r.error || (r.realized_vol !== null && r.realized_vol !== undefined && r.realized_vol >= OPTIONS_MODE_MIN_REALIZED_VOL)
    );
  } else if (mode === "curated") {
    // Mirrors the backend's min_realized_vol + require_curated_fundamentals
    // filters — instant on already-loaded data, same as Options mode above.
    rows = rows.filter(r =>
      r.error || (
        r.realized_vol !== null && r.realized_vol !== undefined && r.realized_vol >= CURATED_MODE_MIN_REALIZED_VOL
        && r.curated_gate_passed !== false
      )
    );
  }
  const hideFar = document.getElementById("hideFarContractsToggle");
  if (hideFar && hideFar.checked) {
    // A row with no fetched put yet (or none found) has nothing to judge
    // against — leave it visible rather than treating "unknown" as "far".
    rows = rows.filter(r =>
      r.error || !r.put || (r.put.dte <= FAR_CONTRACT_MAX_DTE && r.put.delta_abs <= FAR_CONTRACT_MAX_DELTA)
    );
  }
  const hideNonUptrend = document.getElementById("hideNonUptrendToggle");
  if (hideNonUptrend && hideNonUptrend.checked) {
    // check_uptrend_legacy is deliberately informational-only everywhere
    // else (the 30-week MA check is what actually gates the dashboard) —
    // this is an opt-in view for when you specifically want to cross-
    // reference against it too, not a change to the default filter.
    rows = rows.filter(r =>
      r.error || (r.checks && r.checks.uptrend_legacy && r.checks.uptrend_legacy.value === true)
    );
  }
  const hideLowRor = document.getElementById("hideLowRorToggle");
  if (hideLowRor && hideLowRor.checked) {
    // "-" means unknown (no put fetched yet, or genuinely none found) —
    // NOT the same as "bad," so it's always kept regardless of this
    // toggle, same treatment as the DTE/delta toggle above.
    rows = rows.filter(r => r.error || !r.put || r.put.return_on_risk >= MIN_ROR_PCT);
  }
  const hideUnknown = document.getElementById("hideUnknownToggle");
  if (hideUnknown && hideUnknown.checked) {
    // Only the fetched-but-nothing-qualified state (real "-" cells) —
    // not-yet-fetched rows still show their Get Option button and stay
    // visible, since hiding those would just look like they vanished.
    rows = rows.filter(r => r.error || !r.options_fetched || r.options_error || r.put);
  }
  return rows;
}

function renderTable() {
  renderHeader();
  const sorted = [...getDisplayRows()].sort((a, b) => {
    const va = sortValue(a, sortKey), vb = sortValue(b, sortKey);
    if (va < vb) return -1 * sortDir;
    if (va > vb) return 1 * sortDir;
    return 0;
  });

  const tbody = document.getElementById("tbody");
  tbody.innerHTML = "";
  sorted.forEach(r => {
    const tr = document.createElement("tr");
    if (r.error) {
      tr.innerHTML = `<td>${r.symbol}</td><td colspan="${COLUMNS.length - 1}" class="err">${r.error}</td>`;
      tbody.appendChild(tr);
      return;
    }
    let cells = `<td>${r.symbol}</td><td>${sectorCell(r.sector)}</td><td>${r.spot.toFixed(2)}</td><td>${changePctCell(r.day_change_pct)}</td>`;
    CHECK_KEYS.filter(isColumnVisible).forEach(k => {
      const c = r.checks[k];
      let content;
      if (k === "eps_beat") content = epsBeatCell(c);
      else if (NUMERIC_CHECK_KEYS.has(k)) content = checkCell(c);
      else content = checkMark(c ? c.value : null);
      cells += `<td title="${c ? c.reason : ''}">${content}</td>`;
    });
    if (isColumnVisible("next_earnings")) {
      cells += `<td>${r.next_earnings_date || "-"}</td>`;
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
    cells += renderOptionCells(r);
    tr.innerHTML = cells;
    tbody.appendChild(tr);
  });
}

let lastGeneratedAt = null;
function updateAsOf(generatedAt) {
  if (generatedAt !== undefined) lastGeneratedAt = generatedAt;
  const el = document.getElementById("asOf");
  if (!lastGeneratedAt) { el.textContent = "Data as of: —"; return; }
  const fmt = iso => { const d = new Date(iso); return `${d.toLocaleDateString()}, ${d.toLocaleTimeString()}`; };
  let text = `Data as of: ${fmt(lastGeneratedAt)}`;
  // Spot/Chg% get re-quoted live on each option fetch, so they can be newer than the scan itself.
  const quoteTimes = currentRows.map(r => r.quote_as_of).filter(Boolean).sort();
  if (quoteTimes.length) {
    text += ` · Spot/Chg% re-quoted for ${quoteTimes.length} row${quoteTimes.length === 1 ? "" : "s"}, latest ${fmt(quoteTimes[quoteTimes.length - 1])}`;
  }
  el.textContent = text;
}

// Maps the backend's internal mode string ("general"/"options"/"curated")
// back onto the UI's (mode dropdown, strict checkbox) pair — the inverse
// of getEffectiveMode(). Needed because the dropdown itself only ever has
// "general"/"options" as real values; "curated" is represented as
// Options + the Strict checkbox.
function syncModeControlsFrom(backendMode) {
  const mSel = document.getElementById("modeSelect");
  const strict = document.getElementById("strictToggle");
  if (backendMode === "curated") {
    mSel.value = "options";
    if (strict) strict.checked = true;
  } else if (Array.from(mSel.options).some(o => o.value === backendMode)) {
    mSel.value = backendMode;
    if (strict) strict.checked = false;
  }
}

async function pollStatus() {
  const res = await fetch("/api/status");
  const s = await res.json();
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
        ? `adding to current list — ${s.progress}/${s.total} (${s.current_symbol || ""})...`
        : `scanning ${s.progress}/${s.total} (${s.current_symbol || ""})...`;
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
    if (s.append_mode && (s.status === "done" || s.status === "cancelled" || s.status === "error")) {
      // A small merge-only scan — never touches Universe/limit/mode
      // selects or last-scan-config, so none of the normal done/cancelled
      // sync logic below applies. Just reload the (now-merged) rows.
      if (s.status === "error") {
        statusEl.textContent = `error adding to current list: ${s.error}`;
      } else {
        await loadData();
        const qualNote = s.last_append_added < s.total ? ` (${s.total - s.last_append_added} didn't pass the live filter)` : "";
        statusEl.textContent = s.status === "cancelled"
          ? `cancelled — added ${s.progress}/${s.total} before stopping (temporary — next scan drops them)`
          : `done — added ${s.last_append_added}/${s.total} symbol(s) to current list${qualNote} (temporary — next scan drops them)`;
      }
      return;
    }
    if (s.status === "cancelled") {
      const source = s.custom_symbols_count ? `custom list of ${s.custom_symbols_count}` : `scanned ${s.total}`;
      statusEl.textContent = `cancelled — ${source}, stopped at ${s.progress}/${s.total} (${s.row_count} rows kept from before the cancel)`;
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
      const source = s.custom_symbols_count ? `custom list of ${s.custom_symbols_count}` : `scanned ${s.total}`;
      const lowVolNote = s.removed_low_vol ? `, ${s.removed_low_vol} removed low-vol` : "";
      const fundNote = s.removed_failed_fundamentals ? `, ${s.removed_failed_fundamentals} removed fundamentals` : "";
      statusEl.textContent = `done — ${source}, ${s.row_count} shown (${s.removed_no_uptrend} removed, no uptrend${lowVolNote}${fundNote})`;
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
        statusEl.textContent += ` — auto-fetching options for ${shown} shown (≤${AUTO_FETCH_OPTIONS_MAX_ROWS})`;
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
      opt.textContent = `${u.name} (${d.toLocaleDateString()})`;
      group.appendChild(opt);
    });
    if (Array.from(uSel.options).some(o => o.value === previousValue)) {
      uSel.value = previousValue;
    }
    onUniverseChange();
  } catch (e) {
    // leave whatever was there — a transient fetch failure shouldn't wipe the dropdown
  }
}

function onUniverseChange() {
  const uSel = document.getElementById("universeSelect");
  const delBtn = document.getElementById("deleteUniverseBtn");
  // Always visible, just grayed out (disabled) for universes that can't
  // be deleted — the built-in S&P 500/Russell 2000/3000 ones — rather
  // than disappearing, so its existence and purpose stay discoverable.
  if (delBtn) delBtn.disabled = !uSel.value.startsWith("generated:");
}

async function createNewUniverse() {
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
  const mode = getEffectiveMode();
  setBusy(document.getElementById("scanBtn"), "Starting...");
  document.getElementById("status").textContent = "starting...";
  const res = await fetch(`/api/scan?limit=${limit}&universe=${universe}&mode=${mode}`, {method: "POST"});
  if (res.status === 409) {
    document.getElementById("status").textContent = "scan already running";
  }
  pollStatus();
}

async function startCustomRun() {
  const raw = document.getElementById("customSymbols").value.trim();
  if (!raw) return;
  const append = document.getElementById("customScanMode").value === "add";
  const mode = document.getElementById("modeSelect").value;
  setBusy(document.getElementById("customRunBtn"), append ? "Adding..." : "Starting...");
  document.getElementById("status").textContent = append ? "adding to current list..." : "starting...";
  const res = await fetch(`/api/scan?mode=${mode}${append ? "&append=1" : ""}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({symbols: raw}),
  });
  if (res.status === 409) {
    document.getElementById("status").textContent = "scan already running";
  } else if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    document.getElementById("status").textContent = data.message || "invalid symbol list";
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

// initial load
loadData();
pollStatus();
applyModePreset();
refreshUniverseDropdown();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return INDEX_HTML


def run_dashboard(port: int = 8899, initial_limit: int | None = None):
    # Resume the last scan you actually ran, if one's on record — takes
    # priority over --limit, which was really only ever a "keep my own
    # testing cheap" convenience, not something meant to override a real
    # remembered session. Falls back to --limit / hardcoded defaults only
    # on a genuinely first-ever run (no persisted config yet).
    last = _load_last_scan_config()
    if last:
        args = (last.get("limit"), last.get("custom_symbols"), last.get("universe", "russell3000"), last.get("mode", "options"))
    else:
        args = (initial_limit, None, "russell3000", "options")
    t = threading.Thread(target=_run_scan, args=args, daemon=True)
    t.start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    run_dashboard()
