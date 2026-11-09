"""Where the market pages keep their cached data: cache/market/ in the bot's
folder (already gitignored, and deleting it just forces a fresh download)."""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "market"
