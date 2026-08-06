# Read config.py and add new instruments/strategies
with open('/home/admin1/project9/backtest/config.py', 'r') as f:
    content = f.read()

# Add new instruments after the existing ones
old_instruments_end = '''    # --- Short Volatility on SPY ---
    "SPY_VOL": {"asset_class": "stock",  "strategy": "short_volatility",   "base_tf": "1Min", "target_tf": "1D"},
}'''

new_instruments_end = '''    # --- Short Volatility on SPY ---
    "SPY_VOL": {"asset_class": "stock",  "strategy": "short_volatility",   "base_tf": "1Min", "target_tf": "1D"},
    # --- ORB Strategy (15-min Opening Range Breakout) ---
    "SPY_ORB": {"asset_class": "stock", "strategy": "orb", "base_tf": "15Min", "target_tf": "15Min"},
    "QQQ_ORB": {"asset_class": "stock", "strategy": "orb", "base_tf": "15Min", "target_tf": "15Min"},
    # --- XAUUSD Session Mean Reversion ---
    "XAUUSD_MR": {"asset_class": "forex", "strategy": "xauusd_session_mr", "base_tf": "1H", "target_tf": "1H"},
}'''

content = content.replace(old_instruments_end, new_instruments_end)

# Add strategy params for ORB
old_strategy_params_end = '''    # ── Short Volatility (SPY vol-risk premium) ──
    "short_volatility": {'''

new_strategy_params_end = '''    # ── ORB Strategy (Opening Range Breakout) ──
    "orb": {
        "SPY_ORB": {
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
        },
        "QQQ_ORB": {
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
        },
    },
    # ── XAUUSD Session Mean Reversion ──
    "xauusd_session_mr": {
        "XAUUSD_MR": {
            "london_start": 8,
            "london_end": 12,
            "ny_start": 14,
            "ny_end": 20,
            "z_entry": 2.0,
            "z_exit": 0.5,
            "z_stop": 3.0,
            "vwap_window": 20,
            "regime_lookback": 168,
            "hurst_threshold": 0.45,
            "vr_threshold": 0.85,
            "half_life_min": 4,
            "half_life_max": 48,
            "atr_period": 14,
            "atr_stop_mult": 2.0,
            "trail_atr_mult": 1.5,
            "risk_per_trade": 0.005,
            "max_hold_bars": 12,
            "block_nfp": True,
            "block_fomc": True,
            "block_cpi": True,
            "news_buffer_before": 30,
            "news_buffer_after": 120,
            "min_bars_after_open": 2,
            "close_before_rollover": 20,
            "commission": 0.00002,
            "slippage_bps": 0.001,
        },
    },
    # ── Short Volatility (SPY vol-risk premium) ──
    "short_volatility": {'''

content = content.replace(old_strategy_params_end, new_strategy_params_end)

# Add entry filters
old_entry_filters = '''    "short_volatility": [],
}'''

new_entry_filters = '''    "short_volatility": [],
    "orb": ["volatility_regime"],
    "xauusd_session_mr": ["volatility_regime"],
}'''

content = content.replace(old_entry_filters, new_entry_filters)

with open('/home/admin1/project9/backtest/config.py', 'w') as f:
    f.write(content)

print("config.py updated")
