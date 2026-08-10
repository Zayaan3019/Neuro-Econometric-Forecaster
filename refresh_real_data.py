"""
Extend the real GSPC/macro datasets through the present day.

MODEL_EVALUATION_REPORT.md's data window ends 2024-12-31; this script was
written 2026-08-10, ~19 months later -- that gap is real, unused market data,
not a bug, but leaving it unused means every evaluation number in this repo
is nearly two years stale. This script closes the gap using the same
workaround MODEL_EVALUATION_REPORT.md documents was needed the first time:
`yfinance` itself is IP-rate-limited in this environment (reverified
2026-08-10, `yf.Ticker(...).history()` returns an empty frame here), but
Yahoo's own chart API endpoint, hit directly, is not.

Idempotent: only appends rows with dates strictly after the existing CSV's
last row, safe to rerun. Does not touch anything before the existing data's
end date -- the 2010-2024 real data and its provenance are untouched.
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

DATA_DIR = Path(__file__).parent / "data"
MACRO_DIR = DATA_DIR / "macro_cache"
GSPC_CSV = DATA_DIR / "GSPC_ohlcv.csv"

MACRO_TICKERS = {
    "VIX": "^VIX",
    "TNX": "^TNX",
    "FVX": "^FVX",
    "IRX": "^IRX",
    "DXY": "DX-Y.NYB",
    "GLD": "GLD",
    "TLT": "TLT",
}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def _fetch_chart(ticker: str, period1: int, period2: int) -> dict:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker, safe='')}"
        f"?period1={period1}&period2={period2}&interval=1d"
    )
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    result = data.get("chart", {}).get("result")
    if not result:
        err = data.get("chart", {}).get("error")
        raise RuntimeError(f"Yahoo chart API returned no result for {ticker}: {err}")
    return result[0]


def _last_date_in_csv(path: Path) -> str:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1]["Date"]


def _refresh_ohlcv(path: Path, ticker: str) -> int:
    last_date = _last_date_in_csv(path)
    period1 = int(datetime.strptime(last_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) + 1
    period2 = int(time.time())
    if period1 >= period2:
        print(f"  {path.name}: already current (last row {last_date})")
        return 0

    result = _fetch_chart(ticker, period1, period2)
    ts = result["timestamp"]
    quote_data = result["indicators"]["quote"][0]

    new_rows = []
    for i, t in enumerate(ts):
        o, h, l, c, v = (quote_data[k][i] for k in ("open", "high", "low", "close", "volume"))
        if None in (o, h, l, c, v):
            continue  # holiday/gap entries Yahoo sometimes includes with null fields
        date_str = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
        if date_str <= last_date:
            continue  # overlap guard: Yahoo's period1 boundary is inclusive-ish, don't trust it blindly
        new_rows.append([date_str, o, h, l, c, int(v)])

    if not new_rows:
        print(f"  {path.name}: no new rows past {last_date}")
        return 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

    print(f"  {path.name}: appended {len(new_rows)} rows ({new_rows[0][0]} -> {new_rows[-1][0]})")
    return len(new_rows)


def _refresh_macro(path: Path, ticker: str) -> int:
    last_date = _last_date_in_csv(path)
    period1 = int(datetime.strptime(last_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) + 1
    period2 = int(time.time())
    if period1 >= period2:
        print(f"  {path.name}: already current (last row {last_date})")
        return 0

    result = _fetch_chart(ticker, period1, period2)
    ts = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]

    new_rows = []
    for i, t in enumerate(ts):
        c = closes[i]
        if c is None:
            continue
        date_str = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
        if date_str <= last_date:
            continue
        new_rows.append([date_str, c])

    if not new_rows:
        print(f"  {path.name}: no new rows past {last_date}")
        return 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

    print(f"  {path.name}: appended {len(new_rows)} rows ({new_rows[0][0]} -> {new_rows[-1][0]})")
    return len(new_rows)


def main() -> None:
    print(f"GSPC_ohlcv.csv (ticker ^GSPC):")
    total = _refresh_ohlcv(GSPC_CSV, "^GSPC")

    print(f"\nmacro_cache/ ({len(MACRO_TICKERS)} tickers):")
    for name, ticker in MACRO_TICKERS.items():
        path = MACRO_DIR / f"{name}.csv"
        if not path.exists():
            print(f"  {path.name}: MISSING, skipping (not part of this refresh)", file=sys.stderr)
            continue
        total += _refresh_macro(path, ticker)
        time.sleep(1.0)  # be polite to Yahoo's endpoint between tickers

    print(f"\nTotal new rows appended across all files: {total}")


if __name__ == "__main__":
    main()
