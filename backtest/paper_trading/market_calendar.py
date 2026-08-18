"""
paper_trading/market_calendar.py — NYSE calendar, skip non-trading days.
"""

import logging
from datetime import datetime, date

import pandas as pd

logger = logging.getLogger(__name__)

# NYSE holidays (2024-2025) — extend as needed
NYSE_HOLIDAYS = [
    # 2024
    date(2024, 1, 1),   # New Year's Day
    date(2024, 1, 15),  # MLK Day
    date(2024, 2, 19),  # Presidents Day
    date(2024, 3, 29),  # Good Friday
    date(2024, 5, 27),  # Memorial Day
    date(2024, 6, 19),  # Juneteenth
    date(2024, 7, 4),   # Independence Day
    date(2024, 9, 2),   # Labor Day
    date(2024, 11, 28), # Thanksgiving
    date(2024, 12, 25), # Christmas
    # 2025
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 20),  # MLK Day
    date(2025, 2, 17),  # Presidents Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Christmas
    # 2026
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # MLK Day
    date(2026, 2, 16),  # Presidents Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth
    date(2026, 7, 3),   # Independence Day (observed)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas
]


def is_trading_day(d: date | None = None) -> bool:
    """Check if a given date is a NYSE trading day."""
    if d is None:
        d = datetime.now().date()

    # Weekend check
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    # Holiday check
    if d in NYSE_HOLIDAYS:
        return False

    return True


def next_trading_day(d: date | None = None) -> date:
    """Get the next trading day after the given date."""
    if d is None:
        d = datetime.now().date()

    from datetime import timedelta
    d = d + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def market_is_open(dt: datetime | None = None) -> bool:
    """
    Check if the US stock market is currently open.
    Market hours: 9:30 AM - 4:00 PM ET, Mon-Fri, non-holidays.
    """
    if dt is None:
        dt = datetime.now()

    import pytz
    et = pytz.timezone("US/Eastern")
    dt_et = dt.astimezone(et) if dt.tzinfo else et.localize(dt)

    if not is_trading_day(dt_et.date()):
        return False

    market_open = dt_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = dt_et.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= dt_et < market_close


def get_market_hours(dt: datetime | None = None) -> dict:
    """Get market open/close times for a given date in ET."""
    import pytz
    et = pytz.timezone("US/Eastern")

    if dt is None:
        dt = datetime.now()

    dt_et = dt.astimezone(et) if dt.tzinfo else et.localize(dt)

    market_open = dt_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = dt_et.replace(hour=16, minute=0, second=0, microsecond=0)

    return {
        "is_trading_day": is_trading_day(dt_et.date()),
        "market_open": market_open.isoformat(),
        "market_close": market_close.isoformat(),
        "is_open": market_is_open(dt),
    }
