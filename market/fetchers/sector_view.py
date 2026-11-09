"""Turns data/sector_signals.json into what the Sectors and Stock lookup
pages show: each signal's track record in plain words, this week's firings
grouped by signal, and a per-stock summary for the lookup page."""
from __future__ import annotations

# Ordered best to worst: a signal's track record is its best verdict across horizons.
TRACK = [
    ("works", "Held up in testing", "rise"),
    ("reverses", "Did the opposite in testing", "drop"),
    ("too_small", "Real, but too small to pay for trading costs", "quiet"),
    ("hint_works", "Unproven (weak hint, likely chance)", "quiet"),
    ("hint_backwards", "Unproven (weak hint of the opposite, likely chance)", "quiet"),
    ("none", "Unproven: no better than chance", "quiet"),
    ("too_few", "Not enough firings to test", "quiet"),
]


def _track(row: dict | None) -> dict:
    if not row:
        return {"status": "untested", "text": "Not tested yet", "tone": "quiet", "horizons": []}
    verdicts = {h: (st["verdict"] if st else "too_few") for h, st in row["horizons"].items()}
    for status, text, tone in TRACK:
        hs = [int(h) for h, v in verdicts.items() if v == status]
        if hs:
            return {"status": status, "text": text, "tone": tone, "horizons": sorted(hs)}
    return {"status": "untested", "text": "Not tested yet", "tone": "quiet", "horizons": []}


def _trade_word(st: dict | None) -> dict | None:
    """Plain verdict for the stop-and-target backtest of one signal."""
    if not st:
        return None
    sure = st.get("z") is not None and abs(st["z"]) >= 2
    if st["avg_r"] > 0:
        text, tone = ("Made money" if sure else "Made a little, could be luck"), "rise"
    else:
        text, tone = ("Lost money" if sure or st["avg_r"] < -0.1 else "Roughly broke even"), ("drop" if st["avg_r"] < -0.02 else "quiet")
    return {**st, "text": text, "tone": tone, "sure": sure}


def sectors_view(data: dict, combos: dict | None = None) -> dict:
    if not data.get("sectors"):
        return {}
    study = data.get("study")
    study_rows = {r["key"]: r for r in (study or {}).get("rows", [])}
    trades = (study or {}).get("trades") or {}
    signals = []
    for s in data["signals"]:
        signals.append({**s, "track": _track(study_rows.get(s["key"])),
                        "trade": _trade_word((trades.get("rows") or {}).get(s["key"]))})
    meta = {s["key"]: s for s in signals}
    proven = [s for s in signals if s["track"]["status"] in ("works", "reverses")]

    # This week's firings, grouped by signal; a stock held by two ETFs counts once.
    firing: dict[str, dict] = {}
    stocks: dict[str, dict] = {}
    for sec in data["sectors"]:
        for st in sec["stocks"]:
            sym = st["symbol"]
            if sym not in stocks:
                stocks[sym] = {**st, "sectors": []}
            stocks[sym]["sectors"].append({"etf": sec["etf"], "name": sec["name"],
                                           "score": sec["regime"].get("score")})
            if st.get("missing"):
                continue
            for ev in st.get("recent", []):
                group = firing.setdefault(ev["key"], {"signal": meta[ev["key"]], "stocks": {}})
                prev = group["stocks"].get(sym)
                if not prev or ev["date"] > prev["date"]:
                    group["stocks"][sym] = {"symbol": sym, "name": st.get("name"), "date": ev["date"],
                                            "chg_pct": st.get("chg_pct")}
    week = sorted(
        ({"signal": g["signal"], "stocks": sorted(g["stocks"].values(), key=lambda x: x["symbol"])}
         for g in firing.values()),
        # best backtest first, so the signals that made money lead
        key=lambda g: (-(g["signal"]["trade"]["avg_r"] if g["signal"]["trade"] else -9),
                       -len(g["stocks"]), g["signal"]["label"]),
    )
    n_week_stocks = len({x["symbol"] for g in week for x in g["stocks"]})

    for sec in data["sectors"]:
        sec["n_with_signals"] = sum(1 for st in sec["stocks"] if st.get("recent"))

    # How many checks (indicators plus recent signals) point each way today.
    combos = combos or {}
    today = combos.get("today") or {}
    n_checks = {d: len((combos.get(d) or {}).get("checks", [])) for d in ("long", "short")}
    labels = {d: {c["key"]: c["label"] for c in (combos.get(d) or {}).get("checks", [])} for d in ("long", "short")}
    for sym, st in stocks.items():
        up = (today.get("long") or {}).get(sym)
        down = (today.get("short") or {}).get(sym)
        st["checks"] = None if not (up and down) else {
            "up": up["score"], "down": down["score"], "n_up": n_checks["long"], "n_down": n_checks["short"],
            "up_on": [labels["long"].get(k, k) for k in up["on"]], "down_on": [labels["short"].get(k, k) for k in down["on"]],
        }
    for sec in data["sectors"]:
        for rank, st in enumerate(sec["stocks"], 1):
            st["rank"] = rank
            st["checks"] = stocks.get(st["symbol"], {}).get("checks")
    scored = [st for st in stocks.values() if st.get("checks")]
    leaders = {
        # share of checks agreeing, so the two sides compare fairly
        "up": sorted(scored, key=lambda x: (-x["checks"]["up"] / x["checks"]["n_up"], x["checks"]["down"]))[:3],
        "down": sorted(scored, key=lambda x: (-x["checks"]["down"] / x["checks"]["n_down"], x["checks"]["up"]))[:3],
    }

    tested = [s for s in signals if s["trade"]]
    consensus = {
        "n": len(tested),
        "made": [s for s in tested if s["trade"]["avg_r"] > 0],
        "sure": [s for s in tested if s["trade"]["avg_r"] > 0 and s["trade"]["sure"]],
        "best": max(tested, key=lambda s: s["trade"]["avg_r"]) if tested else None,
    }

    return {
        "signals": signals,
        "trades": trades,
        "consensus": consensus,
        "leaders": leaders,
        "combos": combos,
        "meta": meta,
        "proven": proven,
        "week": week,
        "n_week_firings": sum(len(g["stocks"]) for g in week),
        "n_week_stocks": n_week_stocks,
        "stocks": stocks,
    }
