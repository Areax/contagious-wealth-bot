"""CNN Fear & Greed Index — unofficial but stable public JSON endpoint."""
import json
import time
from pathlib import Path

import requests
from market.fetchers.paths import DATA_DIR

URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CACHE_PATH = DATA_DIR / "fear_greed.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://www.cnn.com/markets/fear-and-greed",
}


def fetch():
    resp = requests.get(URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    fg = data["fear_and_greed"]
    history = data.get("fear_and_greed_historical", {}).get("data", [])
    result = {
        "as_of": int(time.time()),
        "score": round(fg["score"], 1),
        "rating": fg["rating"],
        "previous_close": round(fg["previous_close"], 1),
        "previous_1_week": round(fg["previous_1_week"], 1),
        "previous_1_month": round(fg["previous_1_month"], 1),
        "previous_1_year": round(fg["previous_1_year"], 1),
        "history": [{"date": pt["x"], "score": round(pt["y"], 1)} for pt in history[-90:]],
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(fetch(), indent=2)[:1000])
