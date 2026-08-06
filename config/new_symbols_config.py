# Config additions for 4 new symbols (TSLA, NVDA, AMD, GLD)
# Add these to /home/admin1/project9/backtest/config.py

# New ORB instruments
NEW_INSTRUMENTS = {
    "TSLA_ORB": {"asset_class": "stock", "strategy": "orb", "base_tf": "15Min", "target_tf": "15Min"},
    "NVDA_ORB": {"asset_class": "stock", "strategy": "orb", "base_tf": "15Min", "target_tf": "15Min"},
    "AMD_ORB": {"asset_class": "stock", "strategy": "orb", "base_tf": "15Min", "target_tf": "15Min"},
    "GLD_ORB": {"asset_class": "stock", "strategy": "orb", "base_tf": "15Min", "target_tf": "15Min"},
}

# New strategy params (will be populated by MiMo Claw optimization)
NEW_STRATEGY_PARAMS = {
    "orb": {
        "TSLA_ORB": {
            "orb_period": 1, "session_open_hour": 14, "session_open_minute": 30,
            "session_close_hour": 21, "rel_vol_lookback": 14,
            "min_rel_volume": 1.0,  # Will be optimized
            "atr_period": 14, "atr_stop_pct": 0.10,  # Will be optimized
            "risk_per_trade": 0.01, "min_price": 5.0, "min_avg_volume": 100_000,
            "commission": 0.0005, "slippage_bps": 0.001,
        },
        "NVDA_ORB": {
            "orb_period": 1, "session_open_hour": 14, "session_open_minute": 30,
            "session_close_hour": 21, "rel_vol_lookback": 14,
            "min_rel_volume": 1.0,
            "atr_period": 14, "atr_stop_pct": 0.10,
            "risk_per_trade": 0.01, "min_price": 5.0, "min_avg_volume": 100_000,
            "commission": 0.0005, "slippage_bps": 0.001,
        },
        "AMD_ORB": {
            "orb_period": 1, "session_open_hour": 14, "session_open_minute": 30,
            "session_close_hour": 21, "rel_vol_lookback": 14,
            "min_rel_volume": 1.0,
            "atr_period": 14, "atr_stop_pct": 0.10,
            "risk_per_trade": 0.01, "min_price": 5.0, "min_avg_volume": 100_000,
            "commission": 0.0005, "slippage_bps": 0.001,
        },
        "GLD_ORB": {
            "orb_period": 1, "session_open_hour": 14, "session_open_minute": 30,
            "session_close_hour": 21, "rel_vol_lookback": 14,
            "min_rel_volume": 0.5,  # GLD has lower volume
            "atr_period": 14, "atr_stop_pct": 0.10,
            "risk_per_trade": 0.01, "min_price": 5.0, "min_avg_volume": 50_000,
            "commission": 0.0005, "slippage_bps": 0.001,
        },
    },
}
