"""Keeps the market pages' data up to date. Replaces the old GitHub Action:
the bot's own server runs this on a schedule while it is open.

  python -m market.refresh           refresh whatever is due
  python -m market.refresh --force   refresh everything now

Schedule: weekdays at 12am, 8am and 5pm Eastern (SLOT_HOURS).
- Every source refreshes at each of those times.
- The slow S&P 500 studies (sector signals, daily scan, fundamentals) only
  change after a close, so they rerun only when a new close is in (the 5pm
  run); at 12am and 8am they are skipped unless one was missed.
If the computer was asleep or the bot was closed at a scheduled time, the
missed refresh runs as soon as the bot is back.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
SLOT_HOURS = (0, 8, 17)         # 12am, 8am, 5pm ET, Monday to Friday
CLOSE_CUTOFF_HOUR = 17          # 5pm ET: Yahoo has the day's final bar by then
CHECK_SECONDS = 60
ROOT = Path(__file__).resolve().parent.parent


def _sources():
    from market.fetchers import (briefing, daily_scan, earnings_calendar, econ_calendar, fear_greed, fedwatch,
                                 fundamentals, heatmap, sector_signals, vix)
    # (name, module, "slot" = every scheduled time, "close" = once per close)
    return [
        ("Fear & Greed", fear_greed, "slot"),
        ("VIX / S&P", vix, "slot"),
        ("S&P heatmap", heatmap, "slot"),
        ("FedWatch (self-computed)", fedwatch, "slot"),
        ("Econ calendar (PPI/CPI)", econ_calendar, "slot"),
        ("Earnings calendar", earnings_calendar, "slot"),
        ("Briefing.com", briefing, "slot"),
        ("Sector signals + event study", sector_signals, "close"),
        ("Daily S&P 500 signal scan", daily_scan, "close"),
        # last: it reads the two lists above. Each stock is cached a week.
        ("Debt, cash flow and P/E", fundamentals, "close"),
    ]


def _last_weekday_time(now: datetime, hours) -> datetime:
    """The most recent Monday-to-Friday time at one of `hours` (ET), at or before now."""
    now = now.astimezone(ET)
    d = now.date()
    while True:
        if d.weekday() < 5:
            for h in sorted(hours, reverse=True):
                t = datetime(d.year, d.month, d.day, h, tzinfo=ET)
                if t <= now:
                    return t
        d -= timedelta(days=1)


def last_slot(now: datetime | None = None) -> datetime:
    return _last_weekday_time(now or datetime.now(ET), SLOT_HOURS)


def last_close_cutoff(now: datetime | None = None) -> datetime:
    return _last_weekday_time(now or datetime.now(ET), (CLOSE_CUTOFF_HOUR,))


def _built_at(path: Path) -> float:
    """When the data was fetched: its own "as_of" stamp, else the file time
    (a copied or restored file gets a new file time but keeps old data)."""
    try:
        as_of = json.loads(path.read_text()).get("as_of")
        if isinstance(as_of, (int, float)) and as_of > 0:
            return float(as_of)
    except (OSError, ValueError, AttributeError):
        pass
    return path.stat().st_mtime


def _is_due(path: Path, kind: str, now: datetime) -> bool:
    if not path.exists():
        return True
    since = last_close_cutoff(now) if kind == "close" else last_slot(now)
    return _built_at(path) < since.timestamp()


def main(force: bool = False) -> None:
    from market.fetchers.cache_utils import retry

    now = datetime.now(ET)
    for name, module, kind in _sources():
        if not force and not _is_due(module.CACHE_PATH, kind, now):
            print(f"[skip]  {name} (fresh)")
            continue
        print(f"[fetch] {name} ...", end=" ", flush=True)
        start = time.time()
        try:
            retry(module.fetch, attempts=3, base_delay=8)
            print(f"ok ({time.time() - start:.1f}s)", flush=True)
        except Exception as exc:
            print(f"FAILED: {exc}", flush=True)


# --- background runner used by the web server --------------------------------

_lock = threading.Lock()
status = {"running": False, "started": None, "finished": None, "log": "", "ok": None}


def run_now(force: bool = False) -> dict:
    """Runs one refresh in a child process (keeps the server's memory small and
    isolates the big downloads). Returns at once if one is already running."""
    if not _lock.acquire(blocking=False):
        return status
    try:
        status.update(running=True, started=time.time(), log="", ok=None)
        cmd = [sys.executable, "-m", "market.refresh"] + (["--force"] if force else [])
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        status.update(ok=proc.returncode == 0, log=(proc.stdout + proc.stderr)[-20000:])
    except Exception as exc:  # never kill the scheduler thread
        status.update(ok=False, log=f"{status['log']}\n{exc}")
    finally:
        status.update(running=False, finished=time.time())
        _lock.release()
    return status


def start_scheduler() -> None:
    """Refresh what is due now, then again at each scheduled time. Checks the
    clock every minute, so a time missed while the computer slept is caught
    up on waking."""
    def loop():
        done = None
        while True:
            slot = last_slot()
            if slot != done:
                run_now()
                done = slot
            time.sleep(CHECK_SECONDS)
    threading.Thread(target=loop, name="market-refresh", daemon=True).start()


if __name__ == "__main__":
    main(force="--force" in sys.argv)
