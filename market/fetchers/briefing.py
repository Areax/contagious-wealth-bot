"""Briefing.com's free homepage teaser headlines (most content is paywalled;
this only pulls the couple of free "Latest Comments" headlines plus a link
to the full site — the page is an Angular SPA so it needs a real browser).
"""
import json
import time
from pathlib import Path

from market.fetchers.browser_utils import browser_page
from market.fetchers.paths import DATA_DIR

URL = "https://www.briefing.com/"
CACHE_PATH = DATA_DIR / "briefing.json"


def fetch():
    with browser_page() as page:
        page.goto(URL, wait_until="domcontentloaded", timeout=30000)
        try:
            page.wait_for_selector(".latest-comments-container", timeout=10000)
        except Exception:
            headlines = []
        else:
            page.wait_for_timeout(500)
            headlines = page.eval_on_selector_all(
                ".latest-comments-container .headline",
                "els => els.map(e => e.textContent.trim())",
            )

    result = {"as_of": int(time.time()), "headlines": headlines, "url": URL}
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(fetch())
