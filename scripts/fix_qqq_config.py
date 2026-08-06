"""Fix ORB volume filter and update config with optimized params."""
import sys
sys.path.insert(0, "/home/admin1/project9/backtest")

# Update ORB config with instrument-specific volume thresholds
with open("/home/admin1/project9/backtest/config.py", "r") as f:
    content = f.read()

# Fix QQQ volume threshold (QQQ has ~470K daily vol vs SPY ~3M)
old = '''        "QQQ_ORB": {
            "orb_period": 1,
            "session_open_hour": 14,
            "session_open_minute": 30,
            "session_close_hour": 21,
            "rel_vol_lookback": 14,
            "min_rel_volume": 1.0,
            "atr_period": 14,
            "atr_stop_pct": 0.10,
            "risk_per_trade": 0.01,
            "min_price": 5.0,
            "min_avg_volume": 1_000_000,
            "commission": 0.0005,
            "slippage_bps": 0.001,
        },'''

new = '''        "QQQ_ORB": {
            "orb_period": 1,
            "session_open_hour": 14,
            "session_open_minute": 30,
            "session_close_hour": 21,
            "rel_vol_lookback": 14,
            "min_rel_volume": 0.8,       # Relaxed for QQQ
            "atr_period": 14,
            "atr_stop_pct": 0.10,
            "risk_per_trade": 0.01,
            "min_price": 5.0,
            "min_avg_volume": 100_000,   # QQQ has lower volume
            "commission": 0.0005,
            "slippage_bps": 0.001,
        },'''

content = content.replace(old, new)

with open("/home/admin1/project9/backtest/config.py", "w") as f:
    f.write(content)

print("Config updated with QQQ volume fix")
