"""Shared Playwright helper for sources that block plain HTTP requests."""
from contextlib import contextmanager

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


@contextmanager
def browser_page():
    # Imported here so the rest of the market pages run without Playwright;
    # only the Briefing.com and earnings-calendar refreshes need it.
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(user_agent=USER_AGENT)
            yield page
        finally:
            browser.close()
