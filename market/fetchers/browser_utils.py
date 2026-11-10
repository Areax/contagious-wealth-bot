"""Shared Playwright helper for sources that block plain HTTP requests
(Briefing.com and the earnings calendar).

Sets itself up: if the Playwright package or its Chromium browser is missing
(a fresh download, a new computer, a new server), the first refresh that
needs it installs them into this Python environment and carries on.
"""
import subprocess
import sys
from contextlib import contextmanager

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _run(*args: str) -> None:
    print(f"[setup] {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], check=True)


def _sync_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        _run("-m", "pip", "install", "--quiet", "playwright>=1.40")
        from playwright.sync_api import sync_playwright
    return sync_playwright


def _launch(p):
    try:
        return p.chromium.launch(headless=True)
    except Exception as exc:
        # Playwright's message when the browser was never downloaded
        if "playwright install" not in str(exc) and "Executable doesn't exist" not in str(exc):
            raise
        _run("-m", "playwright", "install", "chromium")
        return p.chromium.launch(headless=True)


@contextmanager
def browser_page():
    with _sync_playwright()() as p:
        browser = _launch(p)
        try:
            page = browser.new_page(user_agent=USER_AGENT)
            yield page
        finally:
            browser.close()
