"""Small helpers shared by fetchers: retry-with-backoff and cache freshness."""
import time
from pathlib import Path


def is_fresh(path: Path, ttl_seconds: int) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < ttl_seconds


def retry(fn, attempts=3, base_delay=5, retry_on=(Exception,)):
    """Call fn(), retrying on the given exception types with linear backoff."""
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except retry_on as exc:
            last_exc = exc
            if i < attempts - 1:
                time.sleep(base_delay * (i + 1))
    raise last_exc
