"""Bring the bundled market history up to the latest completed trading session.

The deployed image ships the OHLCV and macro history the model was evaluated
on. Without this module, every forecast would be made from the last bundled
bar, however old, while reporting today's timestamp. Here the bundled history
is extended at request time with the daily bars published since, fetched from
the same Yahoo chart endpoint ``refresh_real_data.py`` uses.

Rules this module enforces:

* **Only completed sessions.** Yahoo's daily series includes today's bar while
  the market is open, with a "close" that is really the last trade. A daily
  model fed that partial bar would forecast from a price that does not exist
  yet, so a bar dated today (New York time) is dropped until 16:30 ET.
* **Bundled history is never rewritten.** New rows are appended only after the
  bundled data's last date, the same overlap guard the refresh script uses.
* **Degrade, don't fail.** If the fetch fails, the forecast is still served from
  the bundled data, and the response says so (``data_source="bundled"`` plus
  the reason) rather than presenting stale data as current.
* **Bounded upstream load.** Results are cached in-process for
  ``REFRESH_TTL_SECONDS``, so a warm Lambda calls Yahoo at most once per TTL.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional, Tuple
from urllib.parse import quote

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={p2}&interval=1d"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
REFRESH_TTL_SECONDS = 15 * 60
#: US equity close is 16:00 ET; the extra 30 minutes lets the official close settle.
SESSION_FINAL_AFTER = (16, 30)

try:  # zoneinfo needs the system tz database, present on Lambda's Amazon Linux
    from zoneinfo import ZoneInfo

    NEW_YORK = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - fallback for minimal images
    NEW_YORK = timezone(timedelta(hours=-5))


def fetch_chart(symbol: str, period1: int, period2: int, timeout: float = 10.0) -> dict:
    """One call to Yahoo's chart API. Raises on HTTP or payload errors."""
    url = CHART_URL.format(symbol=quote(symbol, safe=""), p1=period1, p2=period2)
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    result = payload.get("chart", {}).get("result")
    if not result:
        raise RuntimeError(f"no chart result for {symbol}: {payload.get('chart', {}).get('error')}")
    return result[0]


def _session_is_final(day: pd.Timestamp, now: datetime) -> bool:
    now_ny = now.astimezone(NEW_YORK)
    if day.date() < now_ny.date():
        return True
    if day.date() > now_ny.date():
        return False
    return (now_ny.hour, now_ny.minute) >= SESSION_FINAL_AFTER


def parse_bars(result: dict, *, after: pd.Timestamp, now: datetime, ohlcv: bool) -> pd.DataFrame:
    """Completed daily bars strictly after ``after``, as a Date-indexed frame."""
    stamps = result.get("timestamp") or []
    quote_data = (result.get("indicators", {}).get("quote") or [{}])[0]
    offset = int(result.get("meta", {}).get("gmtoffset", 0))
    keys = ("open", "high", "low", "close", "volume") if ohlcv else ("close",)
    rows = []
    for index, stamp in enumerate(stamps):
        values = [(quote_data.get(k) or [None] * len(stamps))[index] for k in keys]
        if any(v is None for v in values):
            continue  # holiday or gap entries arrive with null fields
        # Exchange-local date, not UTC: a 09:30 ET bar is still that trading day.
        day = pd.Timestamp(datetime.fromtimestamp(stamp + offset, tz=timezone.utc).date())
        if day <= after or not _session_is_final(day, now):
            continue
        rows.append([day, *values])
    columns = ["Date", "Open", "High", "Low", "Close", "Volume"] if ohlcv else ["Date", "Close"]
    frame = pd.DataFrame(rows, columns=columns).drop_duplicates("Date", keep="last")
    return frame.set_index("Date").sort_index()


@dataclass
class MarketSnapshot:
    ohlcv: pd.DataFrame
    macro: pd.DataFrame
    as_of: pd.Timestamp
    source: str  # "bundled+live" or "bundled"
    live_rows: int
    fetched_at: datetime
    note: Optional[str] = None
    macro_as_of: Dict[str, str] = field(default_factory=dict)


class LiveMarketData:
    """In-process cache of bundled history extended with live bars."""

    def __init__(
        self,
        load_bundled: Callable[[], Tuple[pd.DataFrame, pd.DataFrame]],
        *,
        ticker: str,
        macro_tickers: Dict[str, str],
        fetch: Callable[[str, int, int], dict] = fetch_chart,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        ttl_seconds: float = REFRESH_TTL_SECONDS,
    ) -> None:
        self._load_bundled = load_bundled
        self._ticker = ticker
        self._macro_tickers = macro_tickers
        self._fetch = fetch
        self._clock = clock
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._bundled: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None
        self._snapshot: Optional[MarketSnapshot] = None
        self._snapshot_at = 0.0

    def snapshot(self) -> MarketSnapshot:
        with self._lock:
            if self._snapshot is not None and time.monotonic() - self._snapshot_at < self._ttl:
                return self._snapshot
            if self._bundled is None:
                self._bundled = self._load_bundled()
            self._snapshot = self._build(*self._bundled)
            self._snapshot_at = time.monotonic()
            return self._snapshot

    def _build(self, ohlcv: pd.DataFrame, macro: pd.DataFrame) -> MarketSnapshot:
        now = self._clock()
        last = ohlcv.index[-1]
        period1 = int(pd.Timestamp(last).tz_localize("UTC").timestamp()) + 1
        period2 = int(now.timestamp())
        try:
            live = parse_bars(self._fetch(self._ticker, period1, period2), after=last, now=now, ohlcv=True)
        except Exception as exc:  # noqa: BLE001 - any upstream failure degrades to bundled data
            logger.warning("live OHLCV refresh failed, serving bundled data: %s", exc)
            return MarketSnapshot(
                ohlcv=ohlcv, macro=macro, as_of=last, source="bundled", live_rows=0, fetched_at=now,
                note=f"Live market data unavailable ({type(exc).__name__}); using bundled history.",
            )

        merged_ohlcv = pd.concat([ohlcv, live]) if len(live) else ohlcv
        merged_macro, macro_as_of, macro_errors = self._extend_macro(macro, now)
        note = None
        if macro_errors:
            note = (
                "Macro series not refreshed (forward-filled from last value): "
                + ", ".join(sorted(macro_errors))
            )
        return MarketSnapshot(
            ohlcv=merged_ohlcv,
            macro=merged_macro,
            as_of=merged_ohlcv.index[-1],
            source="bundled+live",
            live_rows=len(live),
            fetched_at=now,
            note=note,
            macro_as_of=macro_as_of,
        )

    def _extend_macro(self, macro: pd.DataFrame, now: datetime):
        if macro.empty:
            return macro, {}, []
        frames, as_of, errors = {}, {}, []
        for name, symbol in self._macro_tickers.items():
            if name not in macro.columns:
                continue
            series = macro[name].dropna()
            last = series.index[-1]
            try:
                result = self._fetch(
                    symbol, int(pd.Timestamp(last).tz_localize("UTC").timestamp()) + 1, int(now.timestamp())
                )
                live = parse_bars(result, after=last, now=now, ohlcv=False)["Close"].rename(name)
                series = pd.concat([series, live]) if len(live) else series
            except Exception as exc:  # noqa: BLE001
                logger.warning("macro %s refresh failed: %s", name, exc)
                errors.append(name)
            frames[name] = series
            as_of[name] = str(series.index[-1].date())
        if not frames:
            return macro, as_of, errors
        # Columns without a live source are kept as they are, forward-filled
        # onto the new dates like everything else.
        untouched = [c for c in macro.columns if c not in frames]
        parts = list(frames.values()) + [macro[c] for c in untouched]
        merged = pd.concat(parts, axis=1)[list(macro.columns)].sort_index().ffill()
        return merged, as_of, errors


def next_session_after(day: pd.Timestamp) -> pd.Timestamp:
    """Next weekday. Exchange holidays are not modelled; the response says so."""
    nxt = day + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return nxt
