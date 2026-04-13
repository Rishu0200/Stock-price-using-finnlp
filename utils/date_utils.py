# ──────────────────────────────────────────────────────────────
#  utils/date_utils.py
#  Date / trading-calendar helper functions.
#  Used by data_ingestion and data_transformation to align news
#  headlines to the correct trading day (Phase 5 logic).
# ──────────────────────────────────────────────────────────────

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from config.constants import MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE, MARKET_CLOSE_HOUR
from utils.logging_utils import get_logger

log = get_logger(__name__)

# US Eastern UTC offsets (approximate — ignores DST edge cases)
ET_OFFSET_STANDARD = -5   # UTC-5 in winter
ET_OFFSET_DST      = -4   # UTC-4 in summer


def get_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """
    Return all business days (Mon–Fri) between start and end.
    We approximate US trading days with pandas BDay.
    For production, replace with pandas_market_calendars (NYSE).

    Parameters
    ----------
    start : "YYYY-MM-DD"
    end   : "YYYY-MM-DD"
    """
    return pd.bdate_range(start=start, end=end, freq="B")


def get_next_trading_day(date: datetime, trading_days_index: pd.DatetimeIndex) -> datetime | None:
    """
    Given an arbitrary datetime, return the next date that is in
    trading_days_index (handles weekends and holidays).

    Returns None if no trading day is found within 7 calendar days.
    """
    trading_days_set = {d.date() for d in trading_days_index}
    check_date = date.date() if isinstance(date, datetime) else date

    for offset in range(1, 8):
        candidate = check_date + timedelta(days=offset)
        if candidate in trading_days_set:
            return datetime.combine(candidate, datetime.min.time())

    log.warning("No trading day found within 7 days of %s", date)
    return None


def assign_trade_date(
    pub_datetime: datetime,
    trading_days_index: pd.DatetimeIndex,
    cutoff_hour: int = MARKET_CLOSE_HOUR,
) -> datetime | None:
    """
    Assign the correct trading day to a news headline:
      - Published before market close on day T  →  day T
      - Published after  market close on day T  →  day T+1 (next trading day)

    This prevents look-ahead bias: after-hours news cannot affect the same day's
    closing price because the close has already occurred.

    Parameters
    ----------
    pub_datetime       : timezone-aware or naive UTC datetime of publication
    trading_days_index : pd.DatetimeIndex of valid trading days
    cutoff_hour        : hour (24h) of market close in Eastern time (default 16)

    Returns
    -------
    datetime of the assigned trading day (midnight UTC)
    """
    if pub_datetime is None:
        return None

    trading_set = {d.date() for d in trading_days_index}
    pub_date    = pub_datetime.date()

    # Convert UTC hour to approximate Eastern (EST)
    et_hour = pub_datetime.hour + ET_OFFSET_STANDARD
    if et_hour < 0:
        et_hour += 24

    same_day_is_trading = pub_date in trading_set
    before_close = et_hour < cutoff_hour

    if same_day_is_trading and before_close:
        return datetime.combine(pub_date, datetime.min.time())

    # After close or weekend/holiday: push to next trading day
    return get_next_trading_day(datetime.combine(pub_date, datetime.min.time()), trading_days_index)


def parse_date_flexible(date_str: str) -> datetime | None:
    """
    Try a handful of common date formats and return a datetime.
    Returns None if none match.
    """
    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S %z",   # RSS RFC 2822
        "%a, %d %b %Y %H:%M:%S GMT",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    log.debug("Could not parse date string: %s", date_str)
    return None


def trading_days_between(start: str, end: str) -> int:
    """Return the count of business days between two date strings."""
    return len(get_trading_days(start, end))
