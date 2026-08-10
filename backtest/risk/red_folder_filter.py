"""
News Sentiment & Red Folder Day Filter
=======================================
Based on research on news sentiment and trading

Key Insight: Red folder days (high-impact economic events) cause major market
volatility. Filtering trades around these events can significantly improve
risk-adjusted returns.

Implementation:
1. Parse economic calendar for red folder events
2. Filter out trades 30 min before/after events
3. Reduce position size on event days
4. Optional: Add news sentiment scoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")


class RedFolderEvent:
    """Represents a high-impact economic event."""
    
    def __init__(
        self,
        event_name: str,
        event_time: datetime,
        currency: str,
        impact: str = "red",
        forecast: Optional[float] = None,
        previous: Optional[float] = None,
    ):
        self.event_name = event_name
        self.event_time = event_time
        self.currency = currency
        self.impact = impact
        self.forecast = forecast
        self.previous = previous
    
    def __repr__(self):
        return f"RedFolderEvent({self.event_name}, {self.event_time}, {self.currency})"


# Known high-impact events (red folder)
RED_FOLDER_EVENTS = {
    "NFP": {
        "name": "Non-Farm Payrolls",
        "schedule": "First Friday of every month, 8:30 AM ET",
        "impact": "USD pairs, gold, indices",
        "typical_move": "50-200 pips (USD), 1-3% (indices)",
    },
    "FOMC": {
        "name": "Federal Open Market Committee",
        "schedule": "8 times per year, 2:00 PM ET",
        "impact": "All USD pairs, indices, bonds, gold",
        "typical_move": "100-500 pips (USD), 2-5% (bonds)",
    },
    "CPI": {
        "name": "Consumer Price Index",
        "schedule": "Monthly, 8:30 AM ET",
        "impact": "USD pairs, bonds, gold, indices",
        "typical_move": "50-150 pips (USD), 1-3% (gold)",
    },
    "GDP": {
        "name": "Gross Domestic Product",
        "schedule": "Quarterly, 8:30 AM ET",
        "impact": "USD pairs, indices",
        "typical_move": "30-100 pips (USD), 1-2% (indices)",
    },
    "ECB": {
        "name": "European Central Bank Rate Decision",
        "schedule": "Every 6 weeks, 7:45 AM ET",
        "impact": "EUR pairs",
        "typical_move": "100-300 pips (EUR)",
    },
    "BOE": {
        "name": "Bank of England Rate Decision",
        "schedule": "Monthly, 7:00 AM ET",
        "impact": "GBP pairs",
        "typical_move": "100-200 pips (GBP)",
    },
    "BOJ": {
        "name": "Bank of Japan Rate Decision",
        "schedule": "8 times per year, 11:00 PM ET",
        "impact": "JPY pairs",
        "typical_move": "100-300 pips (JPY)",
    },
    "RETAIL_SALES": {
        "name": "Retail Sales",
        "schedule": "Monthly, 8:30 AM ET",
        "impact": "USD pairs, retail stocks",
        "typical_move": "30-80 pips (USD)",
    },
    "EMPLOYMENT": {
        "name": "Employment Data",
        "schedule": "Monthly, 8:30 AM ET",
        "impact": "USD pairs, indices",
        "typical_move": "50-150 pips (USD)",
    },
}


def get_hardcoded_red_folder_events(
    start_date: str = "2022-01-01",
    end_date: str = "2025-12-31",
) -> List[RedFolderEvent]:
    """
    Get hardcoded red folder events for backtesting.
    This is a fallback when live calendar data is not available.
    
    Parameters
    ----------
    start_date : str
        Start date for events
    end_date : str
        End date for events
    
    Returns
    -------
    List of RedFolderEvent objects
    """
    events = []
    
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    # Generate NFP events (first Friday of every month)
    current = start
    while current <= end:
        # Find first Friday of month
        first_day = current.replace(day=1)
        # Friday is weekday 4
        days_until_friday = (4 - first_day.weekday()) % 7
        if days_until_friday == 0 and first_day.weekday() != 4:
            days_until_friday = 7
        first_friday = first_day + timedelta(days=days_until_friday)
        
        if start <= first_friday <= end:
            event_time = first_friday.replace(hour=8, minute=30, second=0)
            events.append(RedFolderEvent(
                event_name="NFP",
                event_time=event_time,
                currency="USD",
                impact="red",
            ))
        
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    # Generate FOMC events (8 times per year)
    fomc_dates = [
        # 2022
        "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
        "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
        # 2023
        "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
        "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
        # 2024
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        # 2025
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    ]
    
    for date_str in fomc_dates:
        event_date = pd.Timestamp(date_str)
        if start <= event_date <= end:
            event_time = event_date.replace(hour=14, minute=0, second=0)
            events.append(RedFolderEvent(
                event_name="FOMC",
                event_time=event_time,
                currency="USD",
                impact="red",
            ))
    
    # Generate CPI events (monthly, usually second Wednesday)
    current = start
    while current <= end:
        # Find second Wednesday of month
        first_day = current.replace(day=1)
        # Wednesday is weekday 2
        days_until_wednesday = (2 - first_day.weekday()) % 7
        if days_until_wednesday == 0 and first_day.weekday() != 2:
            days_until_wednesday = 7
        first_wednesday = first_day + timedelta(days=days_until_wednesday)
        second_wednesday = first_wednesday + timedelta(days=7)
        
        if start <= second_wednesday <= end:
            event_time = second_wednesday.replace(hour=8, minute=30, second=0)
            events.append(RedFolderEvent(
                event_name="CPI",
                event_time=event_time,
                currency="USD",
                impact="red",
            ))
        
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    # Sort events by time
    events.sort(key=lambda x: x.event_time)
    
    return events


def is_red_folder_day(
    current_time: datetime,
    red_events: List[RedFolderEvent],
    buffer_minutes: int = 30,
) -> bool:
    """
    Check if current time is near a red folder event.
    
    Parameters
    ----------
    current_time : datetime
        Current time to check
    red_events : List[RedFolderEvent]
        List of red folder events
    buffer_minutes : int
        Minutes before/after event to consider "near" (default: 30)
    
    Returns
    -------
    bool: True if near a red folder event
    """
    for event in red_events:
        time_diff = abs((current_time - event.event_time).total_seconds()) / 60
        if time_diff < buffer_minutes:
            return True
    return False


def get_red_folder_events_in_range(
    current_time: datetime,
    red_events: List[RedFolderEvent],
    range_minutes: int = 60,
) -> List[RedFolderEvent]:
    """
    Get red folder events within a time range.
    
    Parameters
    ----------
    current_time : datetime
        Current time
    red_events : List[RedFolderEvent]
        List of red folder events
    range_minutes : int
        Minutes before/after to search (default: 60)
    
    Returns
    -------
    List of RedFolderEvent objects within range
    """
    events_in_range = []
    
    for event in red_events:
        time_diff = abs((current_time - event.event_time).total_seconds()) / 60
        if time_diff < range_minutes:
            events_in_range.append(event)
    
    return events_in_range


def calculate_event_risk(
    current_time: datetime,
    red_events: List[RedFolderEvent],
    buffer_minutes: int = 30,
) -> float:
    """
    Calculate risk multiplier based on proximity to red folder events.
    
    Parameters
    ----------
    current_time : datetime
        Current time
    red_events : List[RedFolderEvent]
        List of red folder events
    buffer_minutes : int
        Minutes before/after event to reduce risk (default: 30)
    
    Returns
    -------
    float: Risk multiplier (0.0 to 1.0)
        - 1.0: No event nearby (full risk)
        - 0.5: Event within 30 min (half risk)
        - 0.0: Event within 5 min (no trading)
    """
    min_distance = float('inf')
    
    for event in red_events:
        time_diff = abs((current_time - event.event_time).total_seconds()) / 60
        min_distance = min(min_distance, time_diff)
    
    if min_distance < 5:
        return 0.0  # No trading within 5 min of event
    elif min_distance < buffer_minutes:
        return 0.5  # Half risk within 30 min of event
    else:
        return 1.0  # Full risk


def apply_red_folder_filter(
    long_entries: pd.Series,
    long_exits: pd.Series,
    short_entries: pd.Series,
    short_exits: pd.Series,
    df: pd.DataFrame,
    red_events: List[RedFolderEvent],
    buffer_minutes: int = 30,
) -> tuple:
    """
    Apply red folder day filter to trading signals.
    
    Parameters
    ----------
    long_entries, long_exits, short_entries, short_exits : pd.Series (bool)
        Original trading signals
    df : DataFrame with OHLCV data
    red_events : List[RedFolderEvent]
        List of red folder events
    buffer_minutes : int
        Minutes before/after event to filter trades (default: 30)
    
    Returns
    -------
    Tuple of filtered signals (long_entries, long_exits, short_entries, short_exits)
    """
    # Create mask for non-event times
    non_event_mask = pd.Series(True, index=df.index)
    
    for idx in df.index:
        if isinstance(idx, pd.Timestamp):
            current_time = idx.to_pydatetime()
        else:
            current_time = idx
        
        if is_red_folder_day(current_time, red_events, buffer_minutes):
            non_event_mask[idx] = False
    
    # Apply filter: only allow entries when not near red folder events
    filtered_long_entries = long_entries & non_event_mask
    filtered_short_entries = short_entries & non_event_mask
    
    # Exits remain unchanged (exit regardless of events)
    filtered_long_exits = long_exits
    filtered_short_exits = short_exits
    
    return filtered_long_entries, filtered_long_exits, filtered_short_entries, filtered_short_exits


def red_folder_summary(
    df: pd.DataFrame,
    red_events: List[RedFolderEvent],
    buffer_minutes: int = 30,
) -> dict:
    """
    Generate summary statistics for red folder days.
    
    Parameters
    ----------
    df : DataFrame with OHLCV data
    red_events : List[RedFolderEvent]
        List of red folder events
    buffer_minutes : int
        Minutes before/after event to consider "near event"
    
    Returns
    -------
    dict with red folder statistics
    """
    total_bars = len(df)
    event_bars = 0
    
    for idx in df.index:
        if isinstance(idx, pd.Timestamp):
            current_time = idx.to_pydatetime()
        else:
            current_time = idx
        
        if is_red_folder_day(current_time, red_events, buffer_minutes):
            event_bars += 1
    
    non_event_bars = total_bars - event_bars
    
    summary = {
        "total_bars": total_bars,
        "event_bars": event_bars,
        "non_event_bars": non_event_bars,
        "event_pct": event_bars / total_bars * 100,
        "non_event_pct": non_event_bars / total_bars * 100,
        "total_events": len(red_events),
    }
    
    return summary


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/admin1/project9/backtest")
    
    # Load sample data
    DATA = "/mnt/c/Users/Admin/project9/data"
    df = pd.read_parquet(f"{DATA}/NVDA_5min.parquet")
    
    print("=" * 80)
    print("RED FOLDER DAY FILTER ANALYSIS")
    print("=" * 80)
    
    # Get hardcoded red folder events
    red_events = get_hardcoded_red_folder_events("2022-01-01", "2024-12-31")
    
    print(f"\nTotal red folder events: {len(red_events)}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Show first few events
    print(f"\nFirst 10 events:")
    for event in red_events[:10]:
        print(f"  {event.event_name}: {event.event_time} ({event.currency})")
    
    # Calculate summary
    summary = red_folder_summary(df, red_events, buffer_minutes=30)
    
    print(f"\nRed Folder Summary:")
    print(f"  Total bars: {summary['total_bars']}")
    print(f"  Event bars (within 30 min): {summary['event_bars']} ({summary['event_pct']:.1f}%)")
    print(f"  Non-event bars: {summary['non_event_bars']} ({summary['non_event_pct']:.1f}%)")
    
    # Test risk multiplier
    print(f"\nRisk Multiplier Examples:")
    test_times = [
        datetime(2023, 1, 6, 8, 0, 0),   # 30 min before NFP
        datetime(2023, 1, 6, 8, 30, 0),   # At NFP
        datetime(2023, 1, 6, 9, 0, 0),    # 30 min after NFP
        datetime(2023, 1, 6, 10, 0, 0),   # 1.5 hours after NFP
    ]
    
    for test_time in test_times:
        risk = calculate_event_risk(test_time, red_events, buffer_minutes=30)
        print(f"  {test_time}: risk={risk:.1f}")
    
    print("\n" + "=" * 80)
    print("RED FOLDER FILTER IMPLEMENTATION COMPLETE")
    print("=" * 80)
