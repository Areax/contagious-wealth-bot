"""The market pages (Overview, Sectors, Stock lookup, Signal combos), served
under /market/ by the bot's web server. Data lives in cache/market/ and is
kept fresh by market/refresh.py."""
from __future__ import annotations

import json
import threading

from flask import Blueprint, abort, jsonify, redirect, render_template, send_file

from market import refresh
from market.fetchers.paths import DATA_DIR
from market.fetchers.sector_view import sectors_view

bp = Blueprint("market", __name__, template_folder="templates", url_prefix="/market")


def _load(name, default):
    path = DATA_DIR / name
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return default


def build_context():
    parts = dict(
        fear_greed=_load("fear_greed.json", {}),
        vix=_load("vix.json", {}),
        heatmap=_load("heatmap.json", {"stocks": []}),
        fedwatch=_load("fedwatch.json", {"meetings": []}),
        econ=_load("econ_calendar.json", {"days": [], "fred_actuals": {}}),
        earnings=_load("earnings_calendar.json", {"next_week": {"range": "", "companies": []},
                                                  "last_week": {"range": "", "companies": []}}),
        briefing=_load("briefing.json", {"headlines": []}),
    )
    parts["oldest_ts"] = min([d.get("as_of", 0) for d in parts.values() if d.get("as_of")], default=None)
    parts["is_static"] = False
    return parts


def build_sectors_context():
    data = _load("sector_signals.json", {})
    combos = _load("sector_combos.json", {})
    return {"data": data, "view": sectors_view(data, combos), "combos": combos,
            "scan": _load("daily_scan.json", {}), "fund": _load("fundamentals.json", {}), "is_static": False}


# Pages link to each other with relative names ("sectors.html"), so the
# overview must be reached as /market/ (with the slash) or /market/index.html.
@bp.route("")
def market_root():
    return redirect("/market/")


@bp.route("/")
@bp.route("/index.html")
def dashboard():
    return render_template("dashboard.html", **build_context())


@bp.route("/sectors.html")
def sectors():
    return render_template("sectors.html", **build_sectors_context())


@bp.route("/stock.html")
def stock():
    return render_template("stock.html", **build_sectors_context())


@bp.route("/combos.html")
def combos_page():
    return render_template("combos.html", **build_sectors_context())


@bp.route("/<any(sector_charts, daily_scan_charts):name>.json")
def chart_data(name):
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="application/json", max_age=0)


@bp.route("/refresh", methods=["POST"])
def refresh_now():
    if refresh.status["running"]:
        return jsonify({"ok": True, "running": True, "log": "A refresh is already running."})
    threading.Thread(target=refresh.run_now, daemon=True).start()
    return jsonify({"ok": True, "running": True})


@bp.route("/refresh/status")
def refresh_status():
    return jsonify(refresh.status)
