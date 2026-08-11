#!/usr/bin/env python3
"""
XCU/USD VWAP Reversion Execution Bot.
Fetches live data, computes signals, and generates trade alerts.
"""

import sys, os, json, time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class VWAPBot:
    """
    Live trading bot for XCU/USD VWAP Reversion strategy.
    
    Parameters:
        lookback: VWAP lookback period (default 20)
        entry_mult: Entry multiplier for VWAP std dev (default 1.0)
        risk_per_trade: Risk per trade as % of capital (default 1%)
        max_positions: Max concurrent positions (default 1)
    """
    
    def __init__(self, lookback=20, entry_mult=1.0, risk_per_trade=1.0, max_positions=1,
                 symbol="XCU/USD", api_key=None):
        self.lookback = lookback
        self.entry_mult = entry_mult
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.symbol = symbol
        self.api_key = api_key or os.environ.get("LSE_API_KEY", "lse_live_f4c9a7419371ecdd9365e146247b0289")
        
        self.position = 0  # 0=flat, 1=long, -1=short
        self.entry_price = None
        self.entry_time = None
        self.capital = 100000.0
        self.trade_log = []
        self.state_file = "results/vwap_bot_state.json"
        
        self._load_state()
    
    def _load_state(self):
        """Load bot state from disk."""
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                state = json.load(f)
                self.position = state.get("position", 0)
                self.entry_price = state.get("entry_price")
                self.entry_time = state.get("entry_time")
                self.capital = state.get("capital", 100000.0)
                self.trade_log = state.get("trade_log", [])
    
    def _save_state(self):
        """Save bot state to disk."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump({
                "position": self.position,
                "entry_price": self.entry_price,
                "entry_time": self.entry_time,
                "capital": self.capital,
                "trade_log": self.trade_log[-100:],  # Keep last 100 trades
            }, f, indent=2)
    
    def fetch_latest_data(self, days=60):
        """Fetch recent daily candles from LSE API."""
        from lse import LSE
        client = LSE(api_key=self.api_key, timeout=60)
        
        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        candles = client.candles(self.symbol, "1d", start=start, end=end)
        
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df.get('timestamp', df.get('ts', '')))
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['volume'] = pd.to_numeric(df.get('volume', 0), errors='coerce').fillna(0)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    def compute_signals(self, df):
        """Compute VWAP reversion signals on latest data."""
        if len(df) < self.lookback + 5:
            return None, None, None
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(self.lookback).sum() / df['volume'].rolling(self.lookback).sum()
        vwap_std = typical_price.rolling(self.lookback).std()
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        current_vwap = vwap.iloc[-1]
        current_std = vwap_std.iloc[-1]
        price = latest['close']
        
        upper_band = current_vwap + self.entry_mult * current_std
        lower_band = current_vwap - self.entry_mult * current_std
        
        # Signal logic
        signal = 0
        reason = ""
        
        # Entry conditions
        if self.position == 0:
            if price < lower_band:
                signal = 1
                reason = f"LONG entry: price {price:.2f} < lower band {lower_band:.2f} (VWAP={current_vwap:.2f}, σ={current_std:.2f})"
            elif price > upper_band:
                signal = -1
                reason = f"SHORT entry: price {price:.2f} > upper band {upper_band:.2f} (VWAP={current_vwap:.2f}, σ={current_std:.2f})"
        
        # Exit conditions
        elif self.position == 1:
            if price > current_vwap:
                signal = 0
                reason = f"EXIT long: price {price:.2f} > VWAP {current_vwap:.2f}"
            else:
                signal = 1  # Hold
                reason = f"HOLD long: price {price:.2f} < VWAP {current_vwap:.2f}"
        
        elif self.position == -1:
            if price < current_vwap:
                signal = 0
                reason = f"EXIT short: price {price:.2f} < VWAP {current_vwap:.2f}"
            else:
                signal = -1  # Hold
                reason = f"HOLD short: price {price:.2f} > VWAP {current_vwap:.2f}"
        
        levels = {
            "price": round(price, 4),
            "vwap": round(current_vwap, 4),
            "upper_band": round(upper_band, 4),
            "lower_band": round(lower_band, 4),
            "vwap_std": round(current_std, 4),
        }
        
        return signal, reason, levels
    
    def calculate_position_size(self, price, stop_loss_price):
        """Calculate position size based on risk management."""
        risk_amount = self.capital * (self.risk_per_trade / 100)
        risk_per_unit = abs(price - stop_loss_price)
        if risk_per_unit == 0:
            return 0
        size = risk_amount / risk_per_unit
        return round(size, 2)
    
    def execute_signal(self, signal, reason, levels):
        """Execute a trading signal."""
        now = datetime.utcnow().isoformat()
        price = levels["price"]
        vwap = levels["vwap"]
        
        action = None
        trade_record = None
        
        # Close existing position if signal changes
        if self.position != 0 and (signal == 0 or signal != self.position):
            pnl_pct = (price / self.entry_price - 1) * self.position * 100
            pnl_dollar = self.capital * (self.risk_per_trade / 100) * (1 if pnl_pct > 0 else -1)
            self.capital += pnl_dollar
            
            trade_record = {
                "type": "close",
                "direction": "long" if self.position == 1 else "short",
                "entry_price": self.entry_price,
                "exit_price": price,
                "entry_time": self.entry_time,
                "exit_time": now,
                "pnl_pct": round(pnl_pct, 4),
                "pnl_dollar": round(pnl_dollar, 2),
                "capital_after": round(self.capital, 2),
                "reason": reason,
            }
            self.trade_log.append(trade_record)
            
            action = "CLOSE"
            self.position = 0
            self.entry_price = None
            self.entry_time = None
        
        # Open new position
        if signal != 0 and self.position == 0:
            # Stop loss at 2x VWAP std dev
            if signal == 1:
                stop_loss = price - 2 * levels["vwap_std"]
            else:
                stop_loss = price + 2 * levels["vwap_std"]
            
            size = self.calculate_position_size(price, stop_loss)
            
            self.position = signal
            self.entry_price = price
            self.entry_time = now
            
            trade_record = {
                "type": "open",
                "direction": "long" if signal == 1 else "short",
                "entry_price": price,
                "stop_loss": round(stop_loss, 4),
                "take_profit": round(vwap, 4),
                "size": size,
                "time": now,
                "reason": reason,
            }
            action = "OPEN"
        
        self._save_state()
        
        return action, trade_record
    
    def get_status(self):
        """Get current bot status."""
        return {
            "symbol": self.symbol,
            "position": "LONG" if self.position == 1 else "SHORT" if self.position == -1 else "FLAT",
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "capital": round(self.capital, 2),
            "total_trades": len(self.trade_log),
            "last_trade": self.trade_log[-1] if self.trade_log else None,
        }
    
    def run_once(self):
        """Single iteration: fetch data, compute signal, execute."""
        print(f"\n{'='*60}")
        print(f"  VWAP Bot — {self.symbol} — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*60}")
        
        # Fetch data
        df = self.fetch_latest_data(days=60)
        if df.empty:
            print("  ERROR: Could not fetch data")
            return None
        
        print(f"  Data: {len(df)} candles, latest: {df.iloc[-1]['timestamp']}")
        
        # Compute signal
        signal, reason, levels = self.compute_signals(df)
        if signal is None:
            print("  ERROR: Could not compute signals")
            return None
        
        # Display
        print(f"\n  Current price: {levels['price']}")
        print(f"  VWAP({self.lookback}): {levels['vwap']}")
        print(f"  Upper band: {levels['upper_band']}")
        print(f"  Lower band: {levels['lower_band']}")
        print(f"\n  Signal: {signal} — {reason}")
        
        # Execute
        action, trade = self.execute_signal(signal, reason, levels)
        
        if action:
            print(f"\n  ⚡ ACTION: {action}")
            if trade:
                print(f"  {json.dumps(trade, indent=4)}")
        else:
            print(f"\n  ⏸  No action needed")
        
        # Status
        status = self.get_status()
        print(f"\n  Position: {status['position']} | Capital: ${status['capital']:,.2f} | Trades: {status['total_trades']}")
        
        return {
            "action": action,
            "signal": signal,
            "reason": reason,
            "levels": levels,
            "trade": trade,
            "status": status,
        }
    
    def run_loop(self, check_interval_hours=4):
        """Run bot in a loop, checking every N hours."""
        print(f"  Starting VWAP Bot loop (check every {check_interval_hours}h)")
        print(f"  Press Ctrl+C to stop")
        
        while True:
            try:
                result = self.run_once()
                if result:
                    # Save latest result
                    with open("results/vwap_bot_latest.json", "w") as f:
                        json.dump(result, f, indent=2, default=str)
                
                print(f"\n  Next check in {check_interval_hours} hours...")
                time.sleep(check_interval_hours * 3600)
                
            except KeyboardInterrupt:
                print("\n  Bot stopped by user")
                break
            except Exception as e:
                print(f"\n  ERROR: {e}")
                time.sleep(60)  # Retry in 1 minute


def main():
    """Run the bot in single-check mode (for cron/scheduled execution)."""
    bot = VWAPBot(
        lookback=20,
        entry_mult=1.0,
        risk_per_trade=1.0,
        symbol="XCU/USD",
    )
    result = bot.run_once()
    
    # Output for cron integration
    if result and result.get("action"):
        print(f"\n🔔 ALERT: {result['action']} signal for {bot.symbol}")
        print(f"   {result['reason']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="XCU/USD VWAP Reversion Bot")
    parser.add_argument("--loop", action="store_true", help="Run in continuous loop")
    parser.add_argument("--interval", type=int, default=4, help="Check interval in hours (default: 4)")
    parser.add_argument("--status", action="store_true", help="Show current bot status")
    args = parser.parse_args()
    
    bot = VWAPBot()
    
    if args.status:
        status = bot.get_status()
        print(json.dumps(status, indent=2))
    elif args.loop:
        bot.run_loop(check_interval_hours=args.interval)
    else:
        main()
