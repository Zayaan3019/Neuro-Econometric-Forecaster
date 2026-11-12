"""Live market-data extension: completed sessions only, no rewrites, honest fallback."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from app.data.live_market import LiveMarketData, next_session_after, parse_bars

# 2026-11-10 and 2026-11-11 are a Tuesday and a Wednesday.
NY_OFFSET = -18000  # gmtoffset Yahoo reports for ^GSPC in winter (seconds)


def _stamp(day: str, hour_utc: int = 14, minute: int = 30) -> int:
    return int(datetime.fromisoformat(f"{day}T{hour_utc:02d}:{minute:02d}:00+00:00").timestamp())


def _chart(days, closes, ohlcv=True):
    quote = {"close": closes}
    if ohlcv:
        quote.update(open=closes, high=closes, low=closes, volume=[1] * len(closes))
    return {"timestamp": [_stamp(d) for d in days], "meta": {"gmtoffset": NY_OFFSET},
            "indicators": {"quote": [quote]}}


def _bundled(last="2026-11-06"):
    idx = pd.to_datetime(["2026-11-05", last])
    ohlcv = pd.DataFrame({"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0],
                          "Close": [1.0, 2.0], "Volume": [1, 1]}, index=idx)
    macro = pd.DataFrame({"VIX": [15.0, 16.0]}, index=idx)
    return ohlcv, macro


def test_only_rows_after_the_bundled_end_are_appended():
    result = _chart(["2026-11-06", "2026-11-09", "2026-11-10"], [9.0, 3.0, 4.0])
    now = datetime(2026, 11, 12, 15, tzinfo=timezone.utc)
    bars = parse_bars(result, after=pd.Timestamp("2026-11-06"), now=now, ohlcv=True)
    assert list(bars.index.strftime("%Y-%m-%d")) == ["2026-11-09", "2026-11-10"]


def test_todays_bar_is_excluded_until_the_session_is_final():
    result = _chart(["2026-11-10", "2026-11-11"], [4.0, 5.0])
    during = datetime(2026, 11, 11, 18, tzinfo=timezone.utc)  # 13:00 New York
    after = datetime(2026, 11, 11, 22, tzinfo=timezone.utc)   # 17:00 New York
    first = parse_bars(result, after=pd.Timestamp("2026-11-09"), now=during, ohlcv=True)
    second = parse_bars(result, after=pd.Timestamp("2026-11-09"), now=after, ohlcv=True)
    assert first.index[-1] == pd.Timestamp("2026-11-10")
    assert second.index[-1] == pd.Timestamp("2026-11-11")


def test_null_bars_are_skipped():
    result = _chart(["2026-11-09", "2026-11-10"], [None, 4.0])
    now = datetime(2026, 11, 12, tzinfo=timezone.utc)
    bars = parse_bars(result, after=pd.Timestamp("2026-11-06"), now=now, ohlcv=True)
    assert list(bars["Close"]) == [4.0]


def test_snapshot_extends_bundled_history_and_reports_its_date():
    def fetch(symbol, p1, p2):
        return _chart(["2026-11-09", "2026-11-10"], [3.0, 4.0], ohlcv=symbol == "^GSPC")

    data = LiveMarketData(_bundled, ticker="^GSPC", macro_tickers={"VIX": "^VIX"}, fetch=fetch,
                          clock=lambda: datetime(2026, 11, 11, 18, tzinfo=timezone.utc))
    snap = data.snapshot()
    assert snap.source == "bundled+live"
    assert snap.as_of == pd.Timestamp("2026-11-10")
    assert snap.live_rows == 2
    assert snap.ohlcv["Close"].iloc[-1] == 4.0
    assert snap.macro["VIX"].iloc[-1] == 4.0
    assert snap.ohlcv.loc["2026-11-06", "Close"] == 2.0, "bundled rows must not change"


def test_upstream_failure_serves_bundled_data_and_says_so():
    def fetch(symbol, p1, p2):
        raise ConnectionError("blocked")

    data = LiveMarketData(_bundled, ticker="^GSPC", macro_tickers={"VIX": "^VIX"}, fetch=fetch,
                          clock=lambda: datetime(2026, 11, 11, tzinfo=timezone.utc))
    snap = data.snapshot()
    assert snap.source == "bundled"
    assert snap.as_of == pd.Timestamp("2026-11-06")
    assert "unavailable" in snap.note


def test_a_failed_macro_series_is_named_not_hidden():
    def fetch(symbol, p1, p2):
        if symbol == "^VIX":
            raise TimeoutError
        return _chart(["2026-11-09"], [3.0])

    data = LiveMarketData(_bundled, ticker="^GSPC", macro_tickers={"VIX": "^VIX"}, fetch=fetch,
                          clock=lambda: datetime(2026, 11, 11, tzinfo=timezone.utc))
    snap = data.snapshot()
    assert snap.source == "bundled+live"
    assert "VIX" in snap.note
    assert snap.macro["VIX"].iloc[-1] == 16.0  # forward-filled, as the note says


def test_snapshot_is_cached_within_its_ttl():
    calls = []

    def fetch(symbol, p1, p2):
        calls.append(symbol)
        return _chart([], [])

    data = LiveMarketData(_bundled, ticker="^GSPC", macro_tickers={}, fetch=fetch,
                          clock=lambda: datetime(2026, 11, 11, tzinfo=timezone.utc))
    data.snapshot()
    data.snapshot()
    assert calls == ["^GSPC"]


@pytest.mark.parametrize("day,expected", [("2026-11-06", "2026-11-09"), ("2026-11-10", "2026-11-11")])
def test_next_session_skips_weekends(day, expected):
    assert next_session_after(pd.Timestamp(day)) == pd.Timestamp(expected)
