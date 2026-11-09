"""Shared FRED (Federal Reserve Economic Data) access.

Uses curl via subprocess rather than the `requests` library: in this
environment, `requests`/urllib3 intermittently hangs talking to
fred.stlouisfed.org (a LibreSSL/urllib3-v2 interaction), while curl has been
100% reliable. FRED's fredgraph.csv endpoint needs no API key.
"""
import csv
import io
import subprocess

BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def fetch_series(series_id):
    """Returns a list of (date_str, value_str) rows for a FRED series, oldest first."""
    url = BASE_URL.format(series_id=series_id)
    result = subprocess.run(
        ["curl", "-s", "--max-time", "20", url],
        capture_output=True, text=True, check=True,
    )
    rows = list(csv.reader(io.StringIO(result.stdout)))
    return [(r[0], r[1]) for r in rows[1:] if len(r) == 2 and r[1] not in ("", ".")]


def latest_value(series_id):
    rows = fetch_series(series_id)
    if not rows:
        raise ValueError(f"No FRED data for {series_id}")
    date, value = rows[-1]
    return date, float(value)


def month_average(series_id, year, month):
    prefix = f"{year:04d}-{month:02d}"
    rows = fetch_series(series_id)
    vals = [float(v) for d, v in rows if d.startswith(prefix)]
    if not vals:
        raise ValueError(f"No FRED {series_id} data for {prefix}")
    return sum(vals) / len(vals)
