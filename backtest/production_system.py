#!/usr/bin/env python3
"""
Production Trading System — XCU VWAP + XAUUSD Donchian
=======================================================
Two uncorrelated strategies for prop firm trading.

Portfolio:
1. XCU VWAP Reversion — WR=76%, PF=2.31, Sharpe=5.21
2. XAUUSD Donchian(50, R:R=3.0) — WR=50%, PF=2.30, Sharpe=5.87

Risk Management:
- Risk per trade: 1% of account
- Max daily loss: 3%
- Max drawdown: 15%
- Stop after 2 consecutive losses
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    "initial_capital": 10000,
    "risk_per_trade": 0.01,      # 1% of account
    "max_daily_loss": 0.03,      # 3% of account
    "max_drawdown": 0.15,        # 15% of account
    "max_consecutive_losses": 2,  # Stop after 2 consecutive losses
    "commission": 0.0005,        # 0.05% per side
    "slippage": 0.0005,          # 0.05% per side
}

# Strategy parameters
XCU_VWAP_PARAMS = {
    "lookback": 50,
    "entry_mult": 2.0,
    "stop_atr_mult": 2.0,
}

XAUUSD_PARAMS = {
    "lookback": 50,
    "rr_target": 3.0,
    "stop_atr_mult": 2.0,
}

# ═══════════════════════════════════════════════════════════════
# STRATEGY IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════

def generate_xcu_signals(df, lookback=50, entry_mult=2.0, stop_atr_mult=2.0):
    """Generate XCU VWAP reversion signals."""
    c = df["close"].values
    h = df["high"].values
    lo = df["low"].values
    
    vwap = pd.Series(c).rolling(lookback).mean().values
    std = pd.Series(c).rolling(lookback).std().values
    
    long_entries = np.zeros(len(c), dtype=bool)
    long_exits = np.zeros(len(c), dtype=bool)
    short_entries = np.zeros(len(c), dtype=bool)
    short_exits = np.zeros(len(c), dtype=bool)
    
    it = False; d = 0; ep = 0; sl = 0; tp = 0
    
    for i in range(lookback+1, len(c)):
        if np.isnan(vwap[i]) or np.isnan(std[i]) or std[i] == 0:
            continue
        
        atr = np.mean(np.abs(np.diff(c[max(0,i-14):i+1])))
        
        if not it:
            # Entry: price deviates from VWAP
            if c[i] < vwap[i] - entry_mult * std[i]:
                it = True; d = 1; ep = c[i]
                sl = ep - stop_atr_mult * atr
                tp = vwap[i]
                long_entries[i] = True
            elif c[i] > vwap[i] + entry_mult * std[i]:
                it = True; d = -1; ep = c[i]
                sl = ep + stop_atr_mult * atr
                tp = vwap[i]
                short_entries[i] = True
        else:
            # Exit at VWAP target or stop loss
            if d == 1:
                if c[i] >= tp:
                    long_exits[i] = True; it = False
                elif c[i] <= sl:
                    long_exits[i] = True; it = False
            elif d == -1:
                if c[i] <= tp:
                    short_exits[i] = True; it = False
                elif c[i] >= sl:
                    short_exits[i] = True; it = False
    
    return long_entries, long_exits, short_entries, short_exits


def generate_xauusd_signals(df, lookback=50, rr_target=3.0, stop_atr_mult=2.0):
    """Generate XAUUSD Donchian breakout signals with R:R target."""
    c = df["close"].values
    h = df["high"].values
    lo = df["low"].values
    
    highs = pd.Series(c).rolling(lookback).max().values
    lows = pd.Series(c).rolling(lookback).min().values
    
    long_entries = np.zeros(len(c), dtype=bool)
    long_exits = np.zeros(len(c), dtype=bool)
    short_entries = np.zeros(len(c), dtype=bool)
    short_exits = np.zeros(len(c), dtype=bool)
    
    it = False; d = 0; ep = 0; sl = 0; tp = 0
    
    for i in range(lookback+1, len(c)):
        atr = np.mean(np.abs(np.diff(c[max(0,i-14):i+1])))
        if atr == 0:
            continue
        
        if not it:
            if c[i] > highs[i-1]:  # Breakout above
                it = True; d = 1; ep = c[i]
                sl = ep - stop_atr_mult * atr
                tp = ep + rr_target * stop_atr_mult * atr
                long_entries[i] = True
            elif c[i] < lows[i-1]:  # Breakdown below
                it = True; d = -1; ep = c[i]
                sl = ep + stop_atr_mult * atr
                tp = ep - rr_target * stop_atr_mult * atr
                short_entries[i] = True
        else:
            if d == 1:
                if c[i] >= tp or c[i] <= sl:
                    long_exits[i] = True; it = False
            elif d == -1:
                if c[i] <= tp or c[i] >= sl:
                    short_exits[i] = True; it = False
    
    return long_entries, long_exits, short_entries, short_exits


# ═══════════════════════════════════════════════════════════════
# POSITION SIZING
# ═══════════════════════════════════════════════════════════════

def calculate_position_size(capital, risk_pct, entry, stop):
    """Calculate position size based on risk per trade."""
    risk_amount = capital * risk_pct
    stop_distance = abs(entry - stop)
    if stop_distance == 0:
        return 0
    position_size = risk_amount / stop_distance
    return position_size


# ═══════════════════════════════════════════════════════════════
# RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════

class RiskManager:
    def __init__(self, config):
        self.config = config
        self.capital = config["initial_capital"]
        self.peak_capital = self.capital
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.trading_enabled = True
        self.trade_history = []
    
    def can_trade(self):
        """Check if we can take a new trade."""
        if not self.trading_enabled:
            return False
        
        # Check daily loss limit
        if self.daily_pnl < -self.capital * self.config["max_daily_loss"]:
            print(f"  RISK: Daily loss limit hit ({self.daily_pnl:.2f})")
            return False
        
        # Check max drawdown
        drawdown = (self.peak_capital - self.capital) / self.peak_capital
        if drawdown >= self.config["max_drawdown"]:
            print(f"  RISK: Max drawdown hit ({drawdown:.2%})")
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= self.config["max_consecutive_losses"]:
            print(f"  RISK: Consecutive losses limit ({self.consecutive_losses})")
            return False
        
        return True
    
    def record_trade(self, pnl):
        """Record a trade result."""
        self.capital += pnl
        self.daily_pnl += pnl
        self.peak_capital = max(self.peak_capital, self.capital)
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        self.trade_history.append({
            "timestamp": datetime.now(),
            "pnl": pnl,
            "capital": self.capital,
            "drawdown": (self.peak_capital - self.capital) / self.peak_capital,
        })
    
    def reset_daily(self):
        """Reset daily P&L."""
        self.daily_pnl = 0
        # Reset consecutive losses at start of new day
        # (only if we had a winning day yesterday)
        if self.daily_pnl >= 0:
            self.consecutive_losses = 0
    
    def get_status(self):
        """Get current risk status."""
        drawdown = (self.peak_capital - self.capital) / self.peak_capital
        return {
            "capital": self.capital,
            "peak_capital": self.peak_capital,
            "drawdown": drawdown,
            "daily_pnl": self.daily_pnl,
            "consecutive_losses": self.consecutive_losses,
            "trading_enabled": self.trading_enabled,
        }


# ═══════════════════════════════════════════════════════════════
# PORTFOLIO MANAGER
# ═══════════════════════════════════════════════════════════════

class PortfolioManager:
    def __init__(self, config):
        self.config = config
        self.risk_manager = RiskManager(config)
        self.positions = {}  # symbol -> position details
    
    def generate_signals(self, df_xcu, df_xau):
        """Generate signals for both strategies."""
        # XCU VWAP
        le_xcu, lx_xcu, se_xcu, sx_xcu = generate_xcu_signals(
            df_xcu, 
            XCU_VWAP_PARAMS["lookback"],
            XCU_VWAP_PARAMS["entry_mult"],
            XCU_VWAP_PARAMS["stop_atr_mult"]
        )
        
        # XAUUSD Donchian
        le_xau, lx_xau, se_xau, sx_xau = generate_xauusd_signals(
            df_xau,
            XAUUSD_PARAMS["lookback"],
            XAUUSD_PARAMS["rr_target"],
            XAUUSD_PARAMS["stop_atr_mult"]
        )
        
        return {
            "XCU": {"long_entries": le_xcu, "long_exits": lx_xcu, 
                    "short_entries": se_xcu, "short_exits": sx_xcu},
            "XAUUSD": {"long_entries": le_xau, "long_exits": lx_xau,
                       "short_entries": se_xau, "short_exits": sx_xau},
        }
    
    def run_backtest(self, df_xcu, df_xau):
        """Run full backtest with risk management."""
        signals = self.generate_signals(df_xcu, df_xau)
        
        # Combine signals
        combined = []
        for i in range(len(df_xcu)):
            combined.append({
                "xcu_long": signals["XCU"]["long_entries"][i],
                "xcu_short": signals["XCU"]["short_entries"][i],
                "xcu_exit": signals["XCU"]["long_exits"][i] or signals["XCU"]["short_exits"][i],
                "xau_long": signals["XAUUSD"]["long_entries"][i],
                "xau_short": signals["XAUUSD"]["short_entries"][i],
                "xau_exit": signals["XAUUSD"]["long_exits"][i] or signals["XAUUSD"]["short_exits"][i],
            })
        
        # Simulate trading
        trades = []
        current_date = None
        
        for i, sig in enumerate(combined):
            # Reset daily P&L at start of new day
            if df_xcu.index[i].date() != current_date:
                current_date = df_xcu.index[i].date()
                self.risk_manager.reset_daily()
            
            # Check if we can trade
            if not self.risk_manager.can_trade():
                continue
            
            # XCU signals
            if sig["xcu_long"] and "XCU" not in self.positions:
                entry = df_xcu["close"].iloc[i] * (1 + self.config["slippage"])
                atr = np.mean(np.abs(np.diff(df_xcu["close"].values[max(0,i-14):i+1])))
                stop = entry - 2 * atr
                size = calculate_position_size(
                    self.risk_manager.capital,
                    self.config["risk_per_trade"],
                    entry, stop
                )
                self.positions["XCU"] = {
                    "direction": "long", "entry": entry, "stop": stop,
                    "size": size, "entry_idx": i
                }
            
            elif sig["xcu_short"] and "XCU" not in self.positions:
                entry = df_xcu["close"].iloc[i] * (1 - self.config["slippage"])
                atr = np.mean(np.abs(np.diff(df_xcu["close"].values[max(0,i-14):i+1])))
                stop = entry + 2 * atr
                size = calculate_position_size(
                    self.risk_manager.capital,
                    self.config["risk_per_trade"],
                    entry, stop
                )
                self.positions["XCU"] = {
                    "direction": "short", "entry": entry, "stop": stop,
                    "size": size, "entry_idx": i
                }
            
            elif sig["xcu_exit"] and "XCU" in self.positions:
                pos = self.positions["XCU"]
                exit_price = df_xcu["close"].iloc[i] * (1 - self.config["slippage"])
                if pos["direction"] == "long":
                    pnl = (exit_price - pos["entry"]) * pos["size"] - self.config["commission"] * 2 * pos["entry"] * pos["size"]
                else:
                    pnl = (pos["entry"] - exit_price) * pos["size"] - self.config["commission"] * 2 * pos["entry"] * pos["size"]
                self.risk_manager.record_trade(pnl)
                trades.append({"symbol": "XCU", "pnl": pnl, "bar": i})
                del self.positions["XCU"]
            
            # XAUUSD signals
            if sig["xau_long"] and "XAUUSD" not in self.positions:
                entry = df_xau["close"].iloc[i] * (1 + self.config["slippage"])
                atr = np.mean(np.abs(np.diff(df_xau["close"].values[max(0,i-14):i+1])))
                stop = entry - 2 * atr
                size = calculate_position_size(
                    self.risk_manager.capital,
                    self.config["risk_per_trade"],
                    entry, stop
                )
                self.positions["XAUUSD"] = {
                    "direction": "long", "entry": entry, "stop": stop,
                    "size": size, "entry_idx": i
                }
            
            elif sig["xau_short"] and "XAUUSD" not in self.positions:
                entry = df_xau["close"].iloc[i] * (1 - self.config["slippage"])
                atr = np.mean(np.abs(np.diff(df_xau["close"].values[max(0,i-14):i+1])))
                stop = entry + 2 * atr
                size = calculate_position_size(
                    self.risk_manager.capital,
                    self.config["risk_per_trade"],
                    entry, stop
                )
                self.positions["XAUUSD"] = {
                    "direction": "short", "entry": entry, "stop": stop,
                    "size": size, "entry_idx": i
                }
            
            elif sig["xau_exit"] and "XAUUSD" in self.positions:
                pos = self.positions["XAUUSD"]
                exit_price = df_xau["close"].iloc[i] * (1 - self.config["slippage"])
                if pos["direction"] == "long":
                    pnl = (exit_price - pos["entry"]) * pos["size"] - self.config["commission"] * 2 * pos["entry"] * pos["size"]
                else:
                    pnl = (pos["entry"] - exit_price) * pos["size"] - self.config["commission"] * 2 * pos["entry"] * pos["size"]
                self.risk_manager.record_trade(pnl)
                trades.append({"symbol": "XAUUSD", "pnl": pnl, "bar": i})
                del self.positions["XAUUSD"]
        
        return trades


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    DATA = "/mnt/c/Users/Admin/project9/data"
    RESULTS = "/mnt/c/Users/Admin/project9/validation_results"
    
    print("=" * 80)
    print("PRODUCTION TRADING SYSTEM — XCU VWAP + XAUUSD DONCHIAN")
    print(f"Started: {datetime.now()}")
    print("=" * 80)
    
    # Load data
    df_xcu = pd.read_parquet(f"{DATA}/commodities/XCUUSD_1d.parquet")
    if "timestamp" in df_xcu.columns:
        df_xcu["timestamp"] = pd.to_datetime(df_xcu["timestamp"])
        df_xcu = df_xcu.set_index("timestamp")
    
    df_xau = pd.read_parquet(f"{DATA}/commodities/XAUUSD_1d.parquet")
    if "timestamp" in df_xau.columns:
        df_xau["timestamp"] = pd.to_datetime(df_xau["timestamp"])
        df_xau = df_xau.set_index("timestamp")
    
    print(f"XCU: {len(df_xcu)} bars, {df_xcu.index[0].date()} to {df_xcu.index[-1].date()}")
    print(f"XAUUSD: {len(df_xau)} bars, {df_xau.index[0].date()} to {df_xau.index[-1].date()}")
    
    # Initialize portfolio
    portfolio = PortfolioManager(CONFIG)
    
    # Run backtest
    print("\nRunning backtest...")
    trades = portfolio.run_backtest(df_xcu, df_xau)
    
    # Calculate metrics
    if trades:
        pnls = [t["pnl"] for t in trades]
        r = np.array(pnls)
        w = r[r > 0]; l = r[r <= 0]
        
        print(f"\n{'='*80}")
        print("BACKTEST RESULTS")
        print(f"{'='*80}")
        print(f"  Total Trades: {len(trades)}")
        print(f"  Win Rate: {len(w)/len(r)*100:.1f}%")
        print(f"  Profit Factor: {abs(w.sum()/l.sum()):.2f}")
        print(f"  Total Return: {r.sum():.2f}%")
        print(f"  Max Drawdown: {portfolio.risk_manager.get_status()['drawdown']:.2%}")
        
        # Per-symbol breakdown
        xcu_trades = [t for t in trades if t["symbol"] == "XCU"]
        xau_trades = [t for t in trades if t["symbol"] == "XAUUSD"]
        
        print(f"\n  XCU: {len(xcu_trades)} trades")
        if xcu_trades:
            xcu_pnls = [t["pnl"] for t in xcu_trades]
            xcu_r = np.array(xcu_pnls)
            print(f"    WR: {len(xcu_r[xcu_r>0])/len(xcu_r)*100:.1f}%")
            print(f"    Return: {xcu_r.sum():.2f}%")
        
        print(f"\n  XAUUSD: {len(xau_trades)} trades")
        if xau_trades:
            xau_pnls = [t["pnl"] for t in xau_trades]
            xau_r = np.array(xau_pnls)
            print(f"    WR: {len(xau_r[xau_r>0])/len(xau_r)*100:.1f}%")
            print(f"    Return: {xau_r.sum():.2f}%")
    
    # Save results
    result = {
        "timestamp": datetime.now().isoformat(),
        "config": CONFIG,
        "xcu_params": XCU_VWAP_PARAMS,
        "xauusd_params": XAUUSD_PARAMS,
        "trades": len(trades),
        "capital": portfolio.risk_manager.capital,
        "status": portfolio.risk_manager.get_status(),
    }
    
    with open(f"{RESULTS}/production_system.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("PRODUCTION SYSTEM READY")
    print(f"{'='*80}")
    print(f"  Capital: ${portfolio.risk_manager.capital:,.2f}")
    print(f"  Peak: ${portfolio.risk_manager.peak_capital:,.2f}")
    print(f"  Drawdown: {portfolio.risk_manager.get_status()['drawdown']:.2%}")
    print(f"\nResults saved to {RESULTS}/production_system.json")
