# Config additions for ORB and XAUUSD Session Mean Reversion strategies
# These need to be merged into backtest/config.py

# === INSTRUMENTS additions ===
# ORB strategy for SPY and QQQ (15-min data)
INSTRUMENTS_ORB = {
    "SPY_ORB": {"asset_class": "stock", "strategy": "orb", "base_tf": "15Min", "target_tf": "15Min"},
    "QQQ_ORB": {"asset_class": "stock", "strategy": "orb", "base_tf": "15Min", "target_tf": "15Min"},
}

# XAUUSD session mean reversion (1-hour data)
INSTRUMENTS_XAUUSD_MR = {
    "XAUUSD_MR": {"asset_class": "forex", "strategy": "xauusd_session_mr", "base_tf": "1H", "target_tf": "1H"},
}

# === STRATEGY_PARAMS additions ===
STRATEGY_PARAMS_ORB = {
    "orb": {
        "SPY_ORB": {
            # Opening Range
            "orb_period": 1,           # Number of bars for opening range (1 = first 15-min bar)
            "session_open_hour": 14,   # 14:30 UTC = 9:30 AM ET
            "session_open_minute": 30,
            "session_close_hour": 21,  # 21:00 UTC = 4:00 PM ET
            
            # Relative Volume
            "rel_vol_lookback": 14,    # 14-day lookback for average OR volume
            "min_rel_volume": 1.0,     # Minimum 100% relative volume
            
            # Risk
            "atr_period": 14,          # 14-day ATR
            "atr_stop_pct": 0.10,      # Stop at 10% of ATR
            "risk_per_trade": 0.01,    # 1% risk per trade
            
            # Filters
            "min_price": 5.0,          # Minimum price filter
            "min_avg_volume": 1_000_000,  # Minimum 14-day avg volume
            
            # Execution
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
}

STRATEGY_PARAMS_XAUUSD_MR = {
    "xauusd_session_mr": {
        "XAUUSD_MR": {
            # Session definitions (UTC)
            "london_start": 8,
            "london_end": 12,
            "ny_start": 14,
            "ny_end": 20,
            
            # Mean reversion parameters
            "z_entry": 2.0,            # Enter when 2σ from session mean
            "z_exit": 0.5,             # Exit when 0.5σ from mean
            "z_stop": 3.0,             # Stop at 3σ from mean
            "vwap_window": 20,         # Rolling VWAP window
            
            # Regime detection
            "regime_lookback": 168,    # 1 week of hourly bars
            "hurst_threshold": 0.45,   # Below = mean-reverting
            "vr_threshold": 0.85,      # Below = mean-reverting
            "half_life_min": 4,        # Minimum half-life (hours)
            "half_life_max": 48,       # Maximum half-life (hours)
            
            # Risk management
            "atr_period": 14,
            "atr_stop_mult": 2.0,      # Stop at 2x ATR
            "trail_atr_mult": 1.5,     # Trailing stop at 1.5x ATR
            "risk_per_trade": 0.005,   # 0.5% risk per trade (conservative)
            "max_hold_bars": 12,       # Max 12 hours hold
            
            # News filter
            "block_nfp": True,
            "block_fomc": True,
            "block_cpi": True,
            "news_buffer_before": 30,  # Minutes before news
            "news_buffer_after": 120,  # Minutes after news
            
            # Session filter
            "min_bars_after_open": 2,  # Wait 2 hours after session open
            "close_before_rollover": 20,  # Close by 20:00 UTC
            
            # Execution
            "commission": 0.00002,     # Forex commission
            "slippage_bps": 0.001,
        },
    },
}

# === ENTRY_FILTERS additions ===
ENTRY_FILTERS_ORB = {
    "orb": ["volatility_regime"],
}

ENTRY_FILTERS_XAUUSD_MR = {
    "xauusd_session_mr": ["volatility_regime"],
}
