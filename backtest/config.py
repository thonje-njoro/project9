BACKTEST_CONFIG = {
    "start_date": "2022-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 50_000,
    "commission": 0.0005,
    "risk_free_rate": 0.045,
    "commission_stock": 0.0005,
    "commission_forex": 0.00002,
    "commission_crypto": 0.0010,
    "min_ticket_fee": 1.00,
    "slippage_bps": 0.001,
    "max_position_value_pct": 0.15,
}

INSTRUMENTS = {
    # --- Existing best performers ---
    "GLD":     {"asset_class": "stock",  "strategy": "kalman_trend",      "base_tf": "1Min", "target_tf": "4H"},
    "XAU/USD": {"asset_class": "forex",  "strategy": "kalman_trend",      "base_tf": "1D",   "target_tf": "1D"},
    # --- Copper/Gold ratio mean-reversion ---
    "CPER":    {"asset_class": "stock",  "strategy": "kalman_trend",      "base_tf": "1Min", "target_tf": "4H"},
    "CPER_GLD_RATIO": {"asset_class": "synthetic", "strategy": "cper_gld_ratio", "base_tf": "4H",  "target_tf": "4H"},
    # --- Equity diversification ---
    "IWM":     {"asset_class": "stock",  "strategy": "kalman_trend",      "base_tf": "1Min", "target_tf": "4H"},
    # --- Bond trend with short-side (captures bond bear moves) ---
    "TLT":     {"asset_class": "stock",  "strategy": "kalman_trend",      "base_tf": "1Min", "target_tf": "4H"},
    # --- Intraday VWAP Mean Reversion ---
    "SPY":     {"asset_class": "stock",  "strategy": "vwap_mean_reversion", "base_tf": "1Min", "target_tf": "15Min"},
    "IWM_VWAP": {"asset_class": "stock", "strategy": "vwap_mean_reversion", "base_tf": "1Min", "target_tf": "15Min"},
    # --- Short Volatility on SPY ---
    "SPY_VOL": {"asset_class": "stock",  "strategy": "short_volatility",   "base_tf": "1Min", "target_tf": "1D"},
    # --- ORB Strategy (15-min Opening Range Breakout) ---
    "SPY_ORB": {"asset_class": "stock", "strategy": "orb", "base_tf": "15Min", "target_tf": "15Min"},
    "QQQ_ORB": {"asset_class": "stock", "strategy": "orb", "base_tf": "15Min", "target_tf": "15Min"},
    # --- Momentum ORB (60-min OR, optimized by MiMo Claw) ---
    "NVDA_MORB": {"asset_class": "stock", "strategy": "momentum_orb", "base_tf": "5Min", "target_tf": "5Min"},
    "AMD_MORB":  {"asset_class": "stock", "strategy": "momentum_orb", "base_tf": "5Min", "target_tf": "5Min"},
    "PLTR_MORB": {"asset_class": "stock", "strategy": "momentum_orb", "base_tf": "5Min", "target_tf": "5Min"},
    "MRVL_MORB": {"asset_class": "stock", "strategy": "momentum_orb", "base_tf": "5Min", "target_tf": "5Min"},
    # --- XAUUSD Session Mean Reversion ---
    "XAUUSD_MR": {"asset_class": "forex", "strategy": "xauusd_session_mr", "base_tf": "1H", "target_tf": "1H"},
}

STRATEGY_PARAMS = {
    "kalman_trend": {
        "GLD": {
            "Q": 0.01, "R": 1.0, "use_adaptive_noise": False,
            "velocity_threshold_pct": 0.10,
            "mean_revert": False, "use_trailing_stop": True,
            "trail_atr_mult": 2.5, "long_only": True, "use_vwap_exit": True,
            "use_pullback_entry": False,
        },
        "XAU/USD": {
            "Q": 0.02, "R": 1.0, "use_adaptive_noise": False,
            "velocity_threshold_pct": 0.10,
            "mean_revert": True, "mr_deviation": 1.5,
            "use_trailing_stop": True, "trail_atr_mult": 2.5,
            "long_only": True, "use_vwap_exit": False,
        },
        "CPER": {
            "Q": 0.02, "R": 1.0, "use_adaptive_noise": False,
            "velocity_threshold_pct": 0.15,
            "mean_revert": False, "use_trailing_stop": True,
            "trail_atr_mult": 3.0, "long_only": True, "use_vwap_exit": True,
            "use_pullback_entry": True, "pullback_max_candles": 2, "entry_window_periods": 4,
        },
        "IWM": {
            "Q": 0.01, "R": 1.0, "use_adaptive_noise": False,
            "velocity_threshold_pct": 0.10,
            "mean_revert": False, "use_trailing_stop": True,
            "trail_atr_mult": 2.5, "long_only": True, "use_vwap_exit": True,
            "use_pullback_entry": False,
        },
        "TLT": {
            "Q": 0.01, "R": 1.0, "use_adaptive_noise": False,
            "velocity_threshold_pct": 0.10,
            "mean_revert": False, "use_trailing_stop": True,
            "trail_atr_mult": 2.5, "long_only": False, "use_vwap_exit": True,
            "use_pullback_entry": True, "pullback_max_candles": 2, "entry_window_periods": 3,
        },
    },
    "cper_gld_ratio": {
        "CPER_GLD_RATIO": {
            "z_entry": 1.8,
            "z_exit": 0.0,
            "z_take_profit": 0.0,
            "window": 20,
            "use_trailing_stop": True,
            "trail_atr_mult": 2.0,
        },
    },
    "gold_oil_ratio": {},
    "crypto_momentum": {
    },
    "prop_firm_sprint": {
        "SPY": {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 2,
            "rsi_oversold": 10.0,
            "rsi_overbought": 90.0,
            "ema_trend_period": 200,
            "atr_period": 14,
            "risk_per_trade": 0.0085,
            "stop_loss_r_mult": 1.0,
            "partial_tp_ratio": 0.6,
            "partial_tp_r_mult": 0.5,
            "trail_atr_mult": 2.0,
            "max_hold_bars": 8,
            "long_only": True,
        },
    },
    # ── VWAP Mean Reversion (intraday, 15Min track) ──
    "vwap_mean_reversion": {
        "SPY": {
            "use_daily_vwap": True,
            "vwap_window": 20,
            "z_score_lookback": 20,
            "z_entry": 2.2,
            "z_exit": 0.0,
            "require_reversal": True,
            "reversal_lookback": 3,
            "use_trailing_stop": True,
            "trail_atr_mult": 2.0,
            "max_hold_bars": 48,
            "use_time_stop": True,
            "long_only": True,
            "min_volume_ratio": 0.3,
            "skip_trend_days": True,
            "adx_threshold": 25.0,
            "use_relative_volume": True,
            "max_relative_volume": 1.5,
            "rel_volume_period": 14,
            "use_atr_change_filter": True,
            "max_atr_change": 0.05,
            "use_time_filter": True,
            "entry_start_hour_et": 9,
            "entry_end_hour_et": 16,
            "use_take_profit": True,
            "atr_tp_mult": 3.0,
        },
        "IWM_VWAP": {
            "use_daily_vwap": True,
            "vwap_window": 20,
            "z_score_lookback": 20,
            "z_entry": 2.2,
            "z_exit": 0.0,
            "require_reversal": True,
            "reversal_lookback": 3,
            "use_trailing_stop": True,
            "trail_atr_mult": 2.0,
            "max_hold_bars": 48,
            "use_time_stop": True,
            "long_only": True,
            "min_volume_ratio": 0.3,
            "skip_trend_days": True,
            "adx_threshold": 25.0,
            "use_relative_volume": True,
            "max_relative_volume": 1.5,
            "rel_volume_period": 14,
            "use_atr_change_filter": True,
            "max_atr_change": 0.05,
            "use_time_filter": True,
            "entry_start_hour_et": 9,
            "entry_end_hour_et": 16,
            "use_take_profit": True,
            "atr_tp_mult": 3.0,
        },
    },
    # ── ORB Strategy (Opening Range Breakout) ──
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
            "min_rel_volume": 0.8,       # Relaxed for QQQ
            "atr_period": 14,
            "atr_stop_pct": 0.10,
            "risk_per_trade": 0.01,
            "min_price": 5.0,
            "min_avg_volume": 100_000,   # QQQ has lower volume
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
    # ── Momentum ORB (60-min OR, optimized by MiMo Claw) ──
    "momentum_orb": {
        "NVDA_MORB": {
            "or_minutes": 60, "atr_mult_stop": 1.0, "trail_mult": 1.0,
            "min_or_range_atr": 0.0, "min_price": 5.0, "min_volume": 100_000,
            "risk_per_trade": 0.01,
        },
        "AMD_MORB": {
            "or_minutes": 60, "atr_mult_stop": 1.5, "trail_mult": 1.0,
            "min_or_range_atr": 0.0, "min_price": 5.0, "min_volume": 100_000,
            "risk_per_trade": 0.01,
        },
        "PLTR_MORB": {
            "or_minutes": 60, "atr_mult_stop": 3.0, "trail_mult": 1.0,
            "min_or_range_atr": 0.0, "min_price": 5.0, "min_volume": 100_000,
            "risk_per_trade": 0.01,
        },
        "MRVL_MORB": {
            "or_minutes": 5, "atr_mult_stop": 0.5, "trail_mult": 1.0,
            "min_or_range_atr": 0.0, "min_price": 5.0, "min_volume": 100_000,
            "risk_per_trade": 0.01,
        },
    },
    # ── Short Volatility (SPY vol-risk premium) ──
    "short_volatility": {
        "SPY_VOL": {
            "vol_period": 20,
            "vol_lookback": 252,
            "entry_percentile": 0.25,
            "exit_percentile": 0.75,
            "use_trailing_stop": True,
            "trail_atr_mult": 3.0,
            "long_only": True,
        },
    },
}

ENGINE_CONFIG = {
    "use_regime_filter": False,
    "hmm_states": 3, "use_garch": True,
    "use_risk_parity": True,
    "risk_parity_lookback": 60,
    "risk_parity_max_weight": 0.40,
    "risk_parity_min_weight": 0.05,
}

RISK_CONFIG = {
    "max_risk_per_trade_pct": 0.02,
    "max_exposure_pct": 0.06,
    "atr_period": 14,
    "max_concurrent_positions": 4,
    "use_kelly_sizing": True,
    "kelly_fraction": 0.5,
    "kelly_min_trades": 5,
}

PROP_FIRM_RULES = {
    "daily_drawdown_pct": 0.04,
    "max_drawdown_pct": 0.10,
    "consistency_warn_pct": 0.30,
    "consistency_block_pct": 0.40,
}

ENTRY_FILTERS = {
    "kalman_trend": ["volatility_regime"],
    "mean_reversion": ["volatility_regime"],
    "gold_oil_ratio": [],
    "crypto_momentum": [],
    "cper_gld_ratio": [],
    "vwap_mean_reversion": ["volatility_regime"],
    "short_volatility": [],
    "orb": ["volatility_regime"],
    "momentum_orb": ["volatility_regime"],
    "xauusd_session_mr": ["volatility_regime"],
}

SYSTEM_HEALTH_CONFIG = {
    "historical_max_dd": 0.10,
    "sharpe_trigger": 0.3,
    "profit_factor_trigger": 1.1,
    "lookback_months": 6,
    "cusum_h": 5.0, "cusum_target": 0.0,
    "use_kl_enhanced_regime": False,
}

LLM_DATA_CONFIG = {"enabled": False}
REFLECTION_CONFIG = {
    "enabled": True,
    "max_entries": 200,
    "write_trade_memory": True,
    "log_path": "~/.project9/trade_memory/trade_memory.md",
}
AGENT_VALIDATOR_CONFIG = {"enabled": False}

# ── Data vendor routing ───────────────────────────────────────────────────
# Controls which data sources are used and their fallback order.
# Each key names a data category; the value is an ordered list of vendors.
# Available: alpaca, yfinance, alpha_vantage
DATA_VENDOR_CONFIG = {
    "core_stock_apis": ["alpaca", "yfinance", "alpha_vantage"],
    "forex_data": ["yfinance", "alpha_vantage"],
    "fundamental_data": ["alpha_vantage"],     # uses the API key below
    "crypto_data": ["alpaca", "yfinance"],
}
# Alpha Vantage is set via ALPHA_VANTAGE_API_KEY env var or the hardcoded
# default below.  Free tier: 5 calls/min, 500 calls/day.
# ALPHA_VANTAGE_API_KEY = "QYLL5W42OLQBBNSY"  # set in .env instead

# ════════════════════════════════════════════════════════════════
# PROP FIRM CHALLENGE CONFIG (separate sprint-mode system)
# ════════════════════════════════════════════════════════════════
# Activated by passing challenge_mode=True in main.py.
# Uses SPY 15-min enhanced mean reversion (RSI(2)+BB filter).
# Designed for $50k / 10% profit in 22 trading days / 4% daily DD.

CHALLENGE_MODE = False  # Toggle in main.py entry point

CHALLENGE_CONFIG = {
    "initial_capital": 50_000,
    "profit_target_pct": 0.10,
    "profit_target_dollars": 5_000,
    "max_trading_days": 22,
    "daily_dd_limit_pct": 0.035,   # 3.5% buffer below FTMO's 4%
    "max_dd_limit_pct": 0.09,       # 9% buffer below FTMO's 10%
    "min_trading_days": 10,
    "consistency_max_day_pct": 0.30,
    "instrument": "SPY",
    "timeframe": "15Min",
}

SPRINT_STRATEGY_PARAMS = {
    "SPY": {
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 2,
        "rsi_oversold": 10.0,
        "rsi_overbought": 90.0,
        "ema_trend_period": 200,
        "atr_period": 14,
        "risk_per_trade": 0.0085,       # 0.85% of equity
        "stop_loss_r_mult": 1.0,        # 1× ATR hard stop
        "partial_tp_ratio": 0.6,        # 60% partial take-profit
        "partial_tp_r_mult": 0.5,       # Exit partial at 0.5R
        "trail_atr_mult": 2.0,          # Trail remaining at 2× ATR
        "max_hold_bars": 8,             # 2-hour time stop
        "long_only": True,
    },
}

CHALLENGE_RISK_PARAMS = {
    "max_risk_per_trade_pct": 0.0085,
    "max_exposure_pct": 2.0,           # 200% notional (intraday leverage OK)
    "max_concurrent_positions": 1,
    "max_trades_per_day": 4,
    "daily_loss_limit_pct": 0.035,
    "progressive_loss_reduction": [
        (2, 0.75),    # 2 consecutive losses → 75% risk
        (3, 0.50),    # 3 → 50%
        (4, 0.25),    # 4 → 25%
    ],
}

CHALLENGE_LIFECYCLE = {
    "probing": {"days": (1, 3),     "risk_multiplier": 0.5,  "max_trades_day": 3},
    "acceleration": {"days": (4, 20), "risk_multiplier": 1.0, "max_trades_day": 4},
    "preservation": {"days": (21, 30), "risk_multiplier": 0.5, "max_trades_day": 2},
}
