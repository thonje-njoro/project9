#!/usr/bin/env python3
"""
scripts/health_check.py — Check paper trader heartbeat.
Run via cron to detect if the paper trader has stopped.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

STATE_DIR = Path(__file__).parent.parent / "backtest" / "paper_trading" / "state"
HEARTBEAT_FILE = STATE_DIR / "heartbeat.json"
MAX_AGE_MINUTES = 5


def check():
    """Check if paper trader heartbeat is recent."""
    if not HEARTBEAT_FILE.exists():
        print("CRITICAL: No heartbeat file found")
        return 1

    try:
        with open(HEARTBEAT_FILE) as f:
            data = json.load(f)

        timestamp = datetime.fromisoformat(data["timestamp"])
        age = datetime.utcnow() - timestamp

        if age > timedelta(minutes=MAX_AGE_MINUTES):
            print(f"WARNING: Heartbeat is {age.total_seconds():.0f}s old (max {MAX_AGE_MINUTES * 60}s)")
            return 1

        print(f"OK: Heartbeat {age.total_seconds():.0f}s ago, status={data.get('status')}")
        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(check())
