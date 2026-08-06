# Professional Pitfalls of 5-Minute ORB Strategy on SPY/QQQ

## Research Document: Implementation Pitfalls & Code-Level Mitigations

**Strategy**: Opening Range Breakout (ORB) on SPY/QQQ using 1-minute data  
**Source Paper**: Zarattini et al. "A Profitable Day Trading Strategy For The U.S. Equity Market"  
**Backtesting Stack**: backtesting.py + vectorbt + pandas  

---

## 1. Look-Ahead Bias in Signal Generation

### The Problem

Look-ahead bias occurs when your backtest uses information that wouldn't be available at the time of the trading decision. In ORB, there are **three critical look-ahead traps**:

#### Trap 1: Using the Close of the 5th Minute Bar to Enter on the 6th Bar
The 5-minute opening range candle closes at 9:35:00.000. The signal is known only **after** 9:35:00. But the entry must happen at the open of the next 1-minute bar (9:36:00). If your code uses `close` of the 9:35 bar to determine direction and enters at `close` of the same bar, you have look-ahead bias.

```python
# WRONG: Entry on same bar as signal (uses close which isn't known until 9:35:00)
if bar.time == "09:35":
    if bar.close > bar.open:  # Signal
        entry_price = bar.close  # LOOK-AHEAD: close not known until bar completes
```

#### Trap 2: ATR Calculation Using Future Data
The 14-day ATR for stop placement must be calculated using data **before** the current day. If your ATR includes bars from the current day (which are being discovered in real-time), you contaminate the stop level.

```python
# WRONG: ATR includes current day's bars
atr_14d = df['high'].rolling(14*390).max() - df['low'].rolling(14*390).min()
# This includes the current day's intraday bars, which you don't have at 9:30 AM
```

#### Trap 3: Relative Volume Uses Intraday Volume That Hasn't Accumulated Yet
The relative volume filter compares today's opening range volume to the 14-day average. If you calculate this using the **full day's** volume (which you don't have at 9:35 AM), you have look-ahead bias.

```python
# WRONG: Uses full day volume for comparison
today_volume = df[df.date == today]['volume'].sum()  # Can't know this at 9:35
```

### Code-Level Mitigation

```python
def compute_orb_signal(df_1min: pd.DataFrame, today: datetime.date) -> dict:
    """
    Compute ORB signal with strict no-look-ahead enforcement.
    
    All data access is time-gated: we only use bars whose timestamps
    are <= the decision time.
    """
    # Get today's bars up to and including 9:35
    today_start = pd.Timestamp(f"{today} 09:30:00", tz="US/Eastern")
    today_end = pd.Timestamp(f"{today} 09:35:00", tz="US/Eastern")
    
    # STRICT: Only use bars where timestamp <= 09:35:00
    # The 09:35 bar's CLOSE is known at 09:36:00
    orb_bars = df_1min[
        (df_1min.index >= today_start) & 
        (df_1min.index <= today_end)
    ]
    
    if len(orb_bars) < 5:
        return None  # Missing bars - skip this day
    
    # Signal: direction of 5-min ORB candle
    orb_open = orb_bars.iloc[0]['open']   # Known at 09:30:00
    orb_close = orb_bars.iloc[-1]['close'] # Known at 09:36:00 (next bar)
    orb_high = orb_bars['high'].max()
    orb_low = orb_bars['low'].min()
    
    # ATR: Use PRIOR 14 trading days ONLY (no current day data)
    prior_days = df_1min[df_1min.index < today_start]
    daily_groups = prior_days.groupby(prior_days.index.date)
    
    # Daily true range
    daily_tr = []
    for date, group in daily_groups:
        if len(group) < 300:  # Incomplete day
            continue
        d_high = group['high'].max()
        d_low = group['low'].min()
        daily_tr.append(d_high - d_low)
    
    if len(daily_tr) < 14:
        return None  # Insufficient history
    
    atr_14d = np.mean(daily_tr[-14:])  # Last 14 COMPLETE prior days only
    
    # Relative Volume: Compare today's OR volume to prior 14-day avg OR volume
    prior_or_volumes = []
    for date, group in daily_groups:
        or_bars = group.between_time('09:30', '09:35')
        if len(or_bars) >= 5:
            prior_or_volumes.append(or_bars['volume'].sum())
    
    if len(prior_or_volumes) < 14:
        return None
    
    today_or_volume = orb_bars['volume'].sum()
    avg_or_volume = np.mean(prior_or_volumes[-14:])
    relative_volume = today_or_volume / avg_or_volume if avg_or_volume > 0 else 0
    
    return {
        'signal': 'LONG' if orb_close > orb_open else 'SHORT',
        'entry_price': orb_close,  # Use as reference, actual fill is next bar open
        'stop_distance': 0.10 * atr_14d,
        'orb_high': orb_high,
        'orb_low': orb_low,
        'relative_volume': relative_volume,
        'signal_time': pd.Timestamp(f"{today} 09:36:00", tz="US/Eastern"),
        # NOTE: Entry can only happen at 09:36:00 or later
    }
```

### Vectorized Checkpoint (VectorBT)
```python
def add_no_lookahead_checkpoint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column that flags bars where data is 'safe' to use.
    Every signal must reference only safe bars.
    """
    df = df.copy()
    df['safe_for_signal'] = False
    
    # A bar is safe for signal generation only AFTER it completes
    # For 1-min bars, bar[i] is safe starting at bar[i+1]
    df['safe_for_signal'] = True
    df['safe_for_signal'] = df['safe_for_signal'].shift(1).fillna(False)
    
    # For ORB: the 9:35 bar is safe at 9:36
    # Entry is at 9:36 bar's OPEN (which IS known at 9:36)
    return df
```

### Validation Test
```python
def test_no_lookahead():
    """Verify no future data leaks into signals."""
    signals = []
    for day in trading_days:
        # Simulate real-time: feed bars one at a time
        for bar_idx in range(len(df[df.date == day])):
            current_bar = df[df.date == day].iloc[bar_idx]
            available_data = df[df.index <= current_bar.name]
            signal = compute_orb_signal(available_data, day)
            # Signal should be None until at least bar 5 (9:35) completes
            if bar_idx < 5:
                assert signal is None, f"Signal generated too early at bar {bar_idx}"
    print("PASSED: No look-ahead bias detected")
```

---

## 2. Transaction Cost Modeling

### The Problem

The Zarattini paper reports returns **before** transaction costs. Real-world costs for ORB are severe because:

1. **High trade frequency**: 1 trade/day × 252 days/year = 252 round trips
2. **Small average winner**: ORB has ~48% hit rate with moderate R-multiples
3. **Costs compound**: A 5 bps round-trip cost on 252 trades = 12.6% annual drag

#### Cost Components

| Component | SPY/QQQ Range | Typical Assumption |
|-----------|---------------|-------------------|
| Commission | $0 (most brokers) | $0 |
| SEC Fee (sells) | $0.0000278 × value | ~$0.28 per $10K sold |
| FINRA TAF | $0.000166/share | $0.166 per 1000 shares |
| Spread (SPY) | $0.01 (1 cent) | 0.005% at $500 |
| Spread (QQQ) | $0.01 (1 cent) | 0.004% at $500 |
| Market Impact | $0.01-$0.05 | Size-dependent |
| Slippage (1-min bar) | $0.02-$0.10 | Volatility-dependent |

**Total realistic round-trip cost**: 10-30 bps per trade (not the 2-5 bps many backtests assume)

### Code-Level Mitigation

```python
class RealisticCostModel:
    """
    Model all transaction costs for ORB strategy.
    
    Costs are parameterized so you can stress-test different assumptions.
    """
    
    def __init__(
        self,
        commission_per_share: float = 0.0,      # $0 for most brokers
        sec_fee_rate: float = 0.0000278,          # SEC fee on sells
        finra_taf_per_share: float = 0.000166,    # FINRA TAF
        spread_multiplier: float = 1.0,           # 1.0 = half-spread, 2.0 = full spread
        market_impact_bps: float = 2.0,           # Additional impact beyond spread
        slippage_model: str = 'bar_pct',          # 'fixed', 'bar_pct', 'volatility'
        slippage_pct: float = 0.05,               # % of bar range for slippage
        min_slippage: float = 0.01,               # Minimum $0.01 slippage
    ):
        self.commission_per_share = commission_per_share
        self.sec_fee_rate = sec_fee_rate
        self.finra_taf_per_share = finra_taf_per_share
        self.spread_multiplier = spread_multiplier
        self.market_impact_bps = market_impact_bps
        self.slippage_model = slippage_model
        self.slippage_pct = slippage_pct
        self.min_slippage = min_slippage
    
    def compute_entry_cost(
        self,
        shares: int,
        price: float,
        bid: float,
        ask: float,
        bar_high: float,
        bar_low: float,
    ) -> dict:
        """Compute total cost for a BUY entry."""
        # 1. Commission
        commission = shares * self.commission_per_share
        
        # 2. Spread cost (you buy at ask, not mid)
        spread = ask - bid
        spread_cost = shares * spread * self.spread_multiplier / 2
        
        # 3. Market impact (beyond spread)
        notional = shares * price
        impact_cost = notional * self.market_impact_bps / 10000
        
        # 4. Slippage (you don't get the open price exactly)
        slippage = self._compute_slippage(price, bar_high, bar_low)
        slippage_cost = shares * slippage
        
        # 5. FINRA TAF (on sells only, but counted per round trip)
        finra = 0  # Entry is a buy, no TAF
        
        total = commission + spread_cost + impact_cost + slippage_cost + finra
        
        return {
            'commission': commission,
            'spread_cost': spread_cost,
            'impact_cost': impact_cost,
            'slippage_cost': slippage_cost,
            'total': total,
            'bps': total / notional * 10000 if notional > 0 else 0,
        }
    
    def compute_exit_cost(
        self,
        shares: int,
        price: float,
        bid: float,
        ask: float,
        bar_high: float,
        bar_low: float,
        is_stop_exit: bool = False,
    ) -> dict:
        """Compute total cost for a SELL exit."""
        # 1. Commission
        commission = shares * self.commission_per_share
        
        # 2. Spread cost (you sell at bid, not mid)
        spread = ask - bid
        spread_cost = shares * spread * self.spread_multiplier / 2
        
        # 3. Market impact
        notional = shares * price
        impact_cost = notional * self.market_impact_bps / 10000
        
        # 4. Slippage (worse for stop exits)
        slippage_mult = 1.5 if is_stop_exit else 1.0
        slippage = self._compute_slippage(price, bar_high, bar_low) * slippage_mult
        slippage_cost = shares * slippage
        
        # 5. SEC Fee (on sells)
        sec_fee = notional * self.sec_fee_rate
        
        # 6. FINRA TAF
        finra = shares * self.finra_taf_per_share
        
        total = commission + spread_cost + impact_cost + slippage_cost + sec_fee + finra
        
        return {
            'commission': commission,
            'spread_cost': spread_cost,
            'impact_cost': impact_cost,
            'slippage_cost': slippage_cost,
            'sec_fee': sec_fee,
            'finra_taf': finra,
            'total': total,
            'bps': total / notional * 10000 if notional > 0 else 0,
        }
    
    def _compute_slippage(self, price, bar_high, bar_low):
        """Model slippage as a function of bar volatility."""
        if self.slippage_model == 'fixed':
            return max(self.min_slippage, price * 0.0001)
        elif self.slippage_model == 'bar_pct':
            bar_range = bar_high - bar_low
            return max(self.min_slippage, bar_range * self.slippage_pct)
        elif self.slippage_model == 'volatility':
            # Scale slippage with recent volatility
            bar_range = bar_high - bar_low
            return max(self.min_slippage, bar_range * 0.1)
        else:
            return self.min_slippage


def apply_costs_to_backtest(
    trades_df: pd.DataFrame,
    cost_model: RealisticCostModel,
    market_data: pd.DataFrame,  # Must have bid/ask columns
) -> pd.DataFrame:
    """
    Apply realistic costs to each trade in the backtest.
    
    Args:
        trades_df: DataFrame with columns [entry_time, exit_time, entry_price, 
                   exit_price, shares, is_stop_exit]
        cost_model: RealisticCostModel instance
        market_data: 1-min data with bid/ask/high/low
    
    Returns:
        trades_df with cost columns added
    """
    costs = []
    for _, trade in trades_df.iterrows():
        # Get market data at entry time
        entry_bar = market_data.loc[trade['entry_time']]
        exit_bar = market_data.loc[trade['exit_time']]
        
        entry_cost = cost_model.compute_entry_cost(
            shares=trade['shares'],
            price=trade['entry_price'],
            bid=entry_bar.get('bid', trade['entry_price'] - 0.005),
            ask=entry_bar.get('ask', trade['entry_price'] + 0.005),
            bar_high=entry_bar['high'],
            bar_low=entry_bar['low'],
        )
        
        exit_cost = cost_model.compute_exit_cost(
            shares=trade['shares'],
            price=trade['exit_price'],
            bid=exit_bar.get('bid', trade['exit_price'] - 0.005),
            ask=exit_bar.get('ask', trade['exit_price'] + 0.005),
            bar_high=exit_bar['high'],
            bar_low=exit_bar['low'],
            is_stop_exit=trade.get('is_stop_exit', False),
        )
        
        costs.append({
            'entry_cost': entry_cost['total'],
            'exit_cost': exit_cost['total'],
            'total_cost': entry_cost['total'] + exit_cost['total'],
            'entry_bps': entry_cost['bps'],
            'exit_bps': exit_cost['bps'],
            'round_trip_bps': entry_cost['bps'] + exit_cost['bps'],
        })
    
    costs_df = pd.DataFrame(costs)
    return pd.concat([trades_df.reset_index(drop=True), costs_df], axis=1)


def stress_test_costs(trades_df, market_data):
    """
    Run backtest under multiple cost scenarios to find breakeven.
    """
    scenarios = {
        'optimistic': {'spread_multiplier': 0.5, 'market_impact_bps': 1.0, 'slippage_pct': 0.02},
        'realistic': {'spread_multiplier': 1.0, 'market_impact_bps': 2.0, 'slippage_pct': 0.05},
        'pessimistic': {'spread_multiplier': 1.5, 'market_impact_bps': 5.0, 'slippage_pct': 0.10},
        'worst_case': {'spread_multiplier': 2.0, 'market_impact_bps': 10.0, 'slippage_pct': 0.15},
    }
    
    results = {}
    for name, params in scenarios.items():
        model = RealisticCostModel(**params)
        costed = apply_costs_to_backtest(trades_df, model, market_data)
        total_pnl = costed['pnl'].sum() - costed['total_cost'].sum()
        avg_cost_bps = costed['round_trip_bps'].mean()
        results[name] = {
            'net_pnl': total_pnl,
            'avg_cost_bps': avg_cost_bps,
            'total_cost': costed['total_cost'].sum(),
            'trade_count': len(costed),
        }
    
    return pd.DataFrame(results).T
```

---

## 3. Market Microstructure Effects

### The Problem

#### Bid-Ask Spread Dynamics
- SPY has a 1-cent spread most of the time ($0.01/$500 = 0.002%)
- But during the **first 5 minutes** (9:30-9:35), spreads can widen to $0.02-$0.05
- This is exactly when ORB generates signals
- After hours/earnings, spreads can be $0.10+

#### Partial Fills
- If you place a market order for 1000 shares, you might only get 200 filled at the best price
- The rest fills at progressively worse prices
- This is invisible in 1-minute OHLCV data (you only see the VWAP of fills)

#### Order Type Selection
- **Market orders**: Guaranteed fill, worst price
- **Limit orders**: Best price, no fill guarantee
- **Stop-market**: Triggers at stop, fills at market (slippage risk)
- **Stop-limit**: Triggers at stop, fills at limit (may not fill)

### Code-Level Mitigation

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"

@dataclass
class OrderBook:
    """Simulated order book for realistic fill modeling."""
    bids: list  # [(price, size), ...] descending
    asks: list  # [(price, size), ...] ascending
    timestamp: datetime

@dataclass
class FillResult:
    filled: bool
    fill_price: float
    fill_size: int
    partial: bool
    slippage: float  # vs. intended price
    time_to_fill_ms: int

class MicrostructureAwareExecutor:
    """
    Simulate realistic order execution accounting for:
    - Bid-ask spread at time of order
    - Partial fills
    - Order queue position
    - Spread widening during high volatility
    """
    
    def __init__(
        self,
        base_spread: float = 0.01,
        spread_widen_factor_open: float = 3.0,  # Spreads 3x wider at open
        spread_widen_factor_volatility: float = 2.0,
        partial_fill_prob: float = 0.15,
        queue_position_slippage_bps: float = 0.5,
    ):
        self.base_spread = base_spread
        self.spread_widen_factor_open = spread_widen_factor_open
        self.spread_widen_factor_volatility = spread_widen_factor_volatility
        self.partial_fill_prob = partial_fill_prob
        self.queue_position_slippage_bps = queue_position_slippage_bps
    
    def estimate_spread(
        self,
        time: datetime,
        atr_14d: float,
        current_volatility: float,
    ) -> float:
        """
        Estimate bid-ask spread based on time of day and volatility.
        
        Spreads are widest at:
        - Market open (9:30-9:35) - up to 3x normal
        - High volatility periods
        - Lunch hour (12:00-13:00) - thinner book
        - Close (15:50-16:00) - institutional rebalancing
        """
        spread = self.base_spread
        
        # Time-of-day adjustment
        hour, minute = time.hour, time.minute
        if hour == 9 and minute < 35:
            spread *= self.spread_widen_factor_open  # Open volatility
        elif hour == 12:
            spread *= 1.3  # Lunch thinning
        elif hour == 15 and minute >= 50:
            spread *= 1.5  # Close rebalancing
        
        # Volatility adjustment
        vol_ratio = current_volatility / atr_14d if atr_14d > 0 else 1.0
        spread *= max(1.0, vol_ratio * self.spread_widen_factor_volatility)
        
        return spread
    
    def simulate_fill(
        self,
        order_type: OrderType,
        side: Literal['buy', 'sell'],
        size: int,
        intended_price: float,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        spread: float,
        bar_volume: int,
    ) -> FillResult:
        """
        Simulate order fill with microstructure effects.
        
        Key insight: 1-minute OHLCV data tells you the RANGE of prices
        but not the SEQUENCE. A bar that goes Low->High behaves differently
        than High->Low for stop orders.
        """
        half_spread = spread / 2
        
        if order_type == OrderType.MARKET:
            # Market order: buy at ask, sell at bid + queue slippage
            if side == 'buy':
                fill_price = bar_open + half_spread
            else:
                fill_price = bar_open - half_spread
            
            # Queue position slippage (you're not first in queue)
            queue_slippage = intended_price * self.queue_position_slippage_bps / 10000
            fill_price += queue_slippage if side == 'buy' else -queue_slippage
            
            # Partial fills (more likely for large orders relative to bar volume)
            fill_ratio = min(1.0, bar_volume / (size * 10))  # Rough heuristic
            if np.random.random() < self.partial_fill_prob * (1 - fill_ratio):
                filled_size = int(size * np.random.uniform(0.3, 0.8))
                return FillResult(
                    filled=True,
                    fill_price=fill_price,
                    fill_size=filled_size,
                    partial=True,
                    slippage=abs(fill_price - intended_price),
                    time_to_fill_ms=100,
                )
            
            return FillResult(
                filled=True,
                fill_price=fill_price,
                fill_size=size,
                partial=False,
                slippage=abs(fill_price - intended_price),
                time_to_fill_ms=50,
            )
        
        elif order_type in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT):
            # Stop order: triggered when price crosses stop level
            # CRITICAL: We don't know if price hit the stop first or the open first
            
            stop_price = intended_price
            
            # Check if stop was triggered during the bar
            if side == 'buy':  # Buy stop (above current price)
                triggered = bar_high >= stop_price
            else:  # Sell stop (below current price) - used for stop loss
                triggered = bar_low <= stop_price
            
            if not triggered:
                return FillResult(
                    filled=False,
                    fill_price=0,
                    fill_size=0,
                    partial=False,
                    slippage=0,
                    time_to_fill_ms=0,
                )
            
            # If triggered, fill at stop price + slippage
            # Slippage is worse in fast markets (gap through stop)
            if side == 'sell':  # Stop loss exit
                # Check if price gapped through stop
                if bar_low < stop_price - spread:
                    # Gapped through: fill at worse price
                    fill_price = min(bar_open, stop_price - spread)
                else:
                    fill_price = stop_price - half_spread
            else:
                if bar_high > stop_price + spread:
                    fill_price = max(bar_open, stop_price + spread)
                else:
                    fill_price = stop_price + half_spread
            
            return FillResult(
                filled=True,
                fill_price=fill_price,
                fill_size=size,
                partial=False,
                slippage=abs(fill_price - stop_price),
                time_to_fill_ms=200,
            )


def apply_microstructure_to_backtest(
    trades_df: pd.DataFrame,
    market_data: pd.DataFrame,
    executor: MicrostructureAwareExecutor,
) -> pd.DataFrame:
    """
    Re-simulate each trade with microstructure-aware fills.
    
    This catches cases where:
    1. Stop losses don't fill at the exact stop price
    2. Entries have spread costs that compound
    3. Partial fills reduce position size (and thus P&L)
    """
    adjusted_trades = []
    
    for _, trade in trades_df.iterrows():
        entry_bar = market_data.loc[trade['entry_time']]
        exit_bar = market_data.loc[trade['exit_time']]
        
        # Estimate spread at entry (wider at open)
        spread = executor.estimate_spread(
            time=trade['entry_time'],
            atr_14d=trade.get('atr_14d', 1.0),
            current_volatility=entry_bar['high'] - entry_bar['low'],
        )
        
        # Simulate entry fill
        entry_fill = executor.simulate_fill(
            order_type=OrderType.MARKET,
            side='buy' if trade['signal'] == 'LONG' else 'sell',
            size=trade['shares'],
            intended_price=trade['entry_price'],
            bar_open=entry_bar['open'],
            bar_high=entry_bar['high'],
            bar_low=entry_bar['low'],
            bar_close=entry_bar['close'],
            spread=spread,
            bar_volume=entry_bar['volume'],
        )
        
        # Simulate exit fill (stop or EOD)
        if trade.get('is_stop_exit', False):
            exit_fill = executor.simulate_fill(
                order_type=OrderType.STOP_MARKET,
                side='sell' if trade['signal'] == 'LONG' else 'buy',
                size=entry_fill.fill_size,
                intended_price=trade['stop_price'],
                bar_open=exit_bar['open'],
                bar_high=exit_bar['high'],
                bar_low=exit_bar['low'],
                bar_close=exit_bar['close'],
                spread=spread,
                bar_volume=exit_bar['volume'],
            )
        else:
            # EOD exit: market order at close
            exit_fill = executor.simulate_fill(
                order_type=OrderType.MARKET,
                side='sell' if trade['signal'] == 'LONG' else 'buy',
                size=entry_fill.fill_size,
                intended_price=trade['exit_price'],
                bar_open=exit_bar['open'],
                bar_high=exit_bar['high'],
                bar_low=exit_bar['low'],
                bar_close=exit_bar['close'],
                spread=spread,
                bar_volume=exit_bar['volume'],
            )
        
        # Recalculate P&L with actual fill prices
        if entry_fill.filled and exit_fill.filled:
            if trade['signal'] == 'LONG':
                pnl = (exit_fill.fill_price - entry_fill.fill_price) * exit_fill.fill_size
            else:
                pnl = (entry_fill.fill_price - exit_fill.fill_price) * exit_fill.fill_size
        else:
            pnl = 0  # Unfilled orders
        
        adjusted_trades.append({
            **trade.to_dict(),
            'entry_fill_price': entry_fill.fill_price,
            'exit_fill_price': exit_fill.fill_price,
            'entry_slippage': entry_fill.slippage,
            'exit_slippage': exit_fill.slippage,
            'partial_fill': entry_fill.partial,
            'actual_shares': exit_fill.fill_size,
            'adjusted_pnl': pnl,
            'spread_at_entry': spread,
        })
    
    return pd.DataFrame(adjusted_trades)
```

---

## 4. Overfitting Risks with Relative Volume Filter

### The Problem

The Relative Volume filter is the **most dangerous parameter** in the strategy for overfitting because:

1. **Threshold selection**: Why top 20 stocks? Why not 10 or 30?
2. **Lookback period**: Why 14 days? Why not 7 or 21?
3. **Volume definition**: Does "OR Volume" mean 9:30-9:35 or 9:30-9:34?
4. **Selection bias**: Trading only the "best" stocks each day is curve-fitting
5. **In-sample optimization**: The paper optimized on 2016-2023 data

#### The Multiple Testing Problem
If you test 100 different Relative Volume thresholds, one will appear to work by chance at the 95% confidence level. The paper doesn't report how many thresholds they tested.

### Code-Level Mitigation

```python
from scipy import stats
import numpy as np
from itertools import product

def multiple_testing_correction(
    all_results: list[dict],  # List of {'threshold': X, 'sharpe': Y, 'trades': N}
    correction_method: str = 'bonferroni',
) -> pd.DataFrame:
    """
    Apply multiple testing correction to strategy optimization results.
    
    This is MANDATORY when you've tested multiple parameter values.
    Without it, you're data-mining.
    """
    df = pd.DataFrame(all_results)
    
    # Convert Sharpe to p-value (H0: strategy has zero alpha)
    df['p_value'] = df['sharpe'].apply(
        lambda s: 1 - stats.norm.cdf(s) if s > 0 else stats.norm.cdf(s)
    )
    
    n_tests = len(df)
    
    if correction_method == 'bonferroni':
        # Most conservative: multiply p-values by number of tests
        df['adjusted_p'] = df['p_value'] * n_tests
        df['adjusted_p'] = df['adjusted_p'].clip(upper=1.0)
    elif correction_method == 'holm':
        # Step-down procedure: less conservative than Bonferroni
        df = df.sort_values('p_value')
        df['rank'] = range(1, n_tests + 1)
        df['adjusted_p'] = df['p_value'] * (n_tests - df['rank'] + 1)
        df['adjusted_p'] = df['adjusted_p'].clip(upper=1.0)
    elif correction_method == 'bh':
        # Benjamini-Hochberg: controls FDR, not FWER
        df = df.sort_values('p_value')
        df['rank'] = range(1, n_tests + 1)
        df['adjusted_p'] = df['p_value'] * n_tests / df['rank']
        df['adjusted_p'] = df['adjusted_p'].clip(upper=1.0)
    
    # Only parameters surviving correction are statistically significant
    df['significant'] = df['adjusted_p'] < 0.05
    
    return df.sort_values('sharpe', ascending=False)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0,
    kurtosis: float = 3,
) -> float:
    """
    Compute the Deflated Sharpe Ratio (Bailey & López de Prado, 2014).
    
    This accounts for the fact that you selected the BEST strategy
    out of N trials. The observed Sharpe is biased upward.
    
    Args:
        observed_sharpe: Sharpe ratio of the best strategy found
        n_trials: Number of strategies tested (including the winner)
        n_observations: Number of trades or periods
        skewness: Return skewness (default 0 = normal)
        kurtosis: Return kurtosis (default 3 = normal)
    
    Returns:
        Deflated Sharpe Ratio (probability that true SR >= 0)
    """
    # Expected maximum Sharpe under null (all strategies have SR=0)
    euler_mascheroni = 0.5772
    e_max_sr = (1 - euler_mascheroni) * stats.norm.ppf(1 - 1/n_trials) + \
               euler_mascheroni * stats.norm.ppf(1 - 1/(n_trials * np.e))
    
    # Standard error of Sharpe ratio
    se_sr = np.sqrt(
        (1 - skewness * observed_sharpe + (kurtosis - 1)/4 * observed_sharpe**2) /
        (n_observations - 1)
    )
    
    # Deflated Sharpe Ratio
    dsr = stats.norm.cdf((observed_sharpe - e_max_sr) / se_sr)
    
    return dsr


def walk_forward_validation(
    df: pd.DataFrame,
    strategy_fn,
    param_grid: dict,
    train_days: int = 252,  # 1 year
    test_days: int = 63,    # 3 months
    step_days: int = 21,    # 1 month
) -> pd.DataFrame:
    """
    Walk-forward analysis to prevent overfitting.
    
    Train on past data, test on future data, roll forward.
    Only strategies that perform well in OOS periods are valid.
    """
    results = []
    dates = df.index.date.unique()
    
    for start_idx in range(0, len(dates) - train_days - test_days, step_days):
        train_dates = dates[start_idx:start_idx + train_days]
        test_dates = dates[start_idx + train_days:start_idx + train_days + test_days]
        
        train_data = df[df.index.date.isin(train_dates)]
        test_data = df[df.index.date.isin(test_dates)]
        
        # Find best parameters on training data
        best_sharpe = -np.inf
        best_params = None
        
        for params in _generate_param_combos(param_grid):
            result = strategy_fn(train_data, **params)
            if result['sharpe'] > best_sharpe:
                best_sharpe = result['sharpe']
                best_params = params
        
        # Test best parameters on OOS data
        oos_result = strategy_fn(test_data, **best_params)
        
        results.append({
            'train_start': train_dates[0],
            'train_end': train_dates[-1],
            'test_start': test_dates[0],
            'test_end': test_dates[-1],
            'is_sharpe': best_sharpe,
            'oos_sharpe': oos_result['sharpe'],
            'is_trades': best_params,
            'oos_trades': oos_result.get('n_trades', 0),
            'degradation': (best_sharpe - oos_result['sharpe']) / best_sharpe if best_sharpe > 0 else 0,
        })
    
    return pd.DataFrame(results)


def permutation_test(
    actual_pnl: np.ndarray,
    n_permutations: int = 10000,
) -> dict:
    """
    Permutation test: shuffle trade outcomes and see if actual performance
    is better than random.
    
    This is the simplest, most robust check against overfitting.
    If your strategy can't beat shuffled returns, it's not real.
    """
    actual_sharpe = actual_pnl.mean() / actual_pnl.std() * np.sqrt(252)
    
    null_sharpes = []
    for _ in range(n_permutations):
        shuffled = np.random.permutation(actual_pnl)
        null_sharpe = shuffled.mean() / shuffled.std() * np.sqrt(252)
        null_sharpes.append(null_sharpe)
    
    null_sharpes = np.array(null_sharpes)
    p_value = (null_sharpes >= actual_sharpe).mean()
    
    return {
        'actual_sharpe': actual_sharpe,
        'null_mean_sharpe': null_sharpes.mean(),
        'null_std_sharpe': null_sharpes.std(),
        'p_value': p_value,
        'significant': p_value < 0.05,
        'percentile': (null_sharpes < actual_sharpe).mean() * 100,
    }


def parameter_sensitivity_heatmap(
    df: pd.DataFrame,
    strategy_fn,
    param1_name: str,
    param1_values: list,
    param2_name: str,
    param2_values: list,
) -> pd.DataFrame:
    """
    Test parameter sensitivity. Robust strategies work across a RANGE
    of parameters, not just at a single point.
    
    A strategy that only works at exactly (top_n=20, lookback=14) is overfit.
    A strategy that works across (top_n=15-25, lookback=10-20) is robust.
    """
    results = []
    for p1, p2 in product(param1_values, param2_values):
        params = {param1_name: p1, param2_name: p2}
        result = strategy_fn(df, **params)
        results.append({
            param1_name: p1,
            param2_name: p2,
            'sharpe': result['sharpe'],
            'return': result.get('total_return', 0),
            'trades': result.get('n_trades', 0),
            'hit_rate': result.get('hit_rate', 0),
        })
    
    results_df = pd.DataFrame(results)
    
    # Check for plateau: if only 1 cell is good, it's overfit
    sharpe_matrix = results_df.pivot(param1_name, param2_name, 'sharpe')
    top_5pct_threshold = sharpe_matrix.quantile(0.95).quantile(0.95)
    robust_params = results_df[results_df['sharpe'] >= top_5pct_threshold * 0.8]
    
    return results_df, robust_params
```

---

## 5. Data Quality Issues

### The Problem

#### Missing Bars
1-minute data frequently has gaps due to:
- Trading halts (news, volatility)
- Exchange technical issues
- Pre-market/post-market gaps
- Low liquidity periods (no trades for 1+ minutes)

If your ORB has only 4 bars instead of 5 (9:30-9:34 instead of 9:30-9:35), the signal is corrupted.

#### Corporate Actions
- **Stock splits**: TSLA did a 3:1 split on Aug 25, 2022. Pre-split data must be adjusted.
- **Dividends**: SPY pays quarterly dividends (~$1.50-1.80). Unadjusted data shows price drops.
- **Symbol changes**: Some stocks change tickers (FB → META).

#### Survivorship Bias in Data
- If you're trading a universe of stocks, dead stocks are missing from your data
- SPY/QQQ are ETFs (no survivorship issue), but the paper trades individual stocks

### Code-Level Mitigation

```python
import pandas as pd
import numpy as np
from typing import Optional

class DataQualityChecker:
    """
    Comprehensive data quality checks for 1-minute ORB backtesting.
    
    Run this BEFORE any backtesting. Garbage in = garbage out.
    """
    
    def __init__(self, expected_bar_interval: str = '1min'):
        self.expected_interval = pd.Timedelta(expected_bar_interval)
        self.issues = []
    
    def run_all_checks(self, df: pd.DataFrame, symbol: str) -> dict:
        """Run all quality checks and return report."""
        self.issues = []
        
        self.check_missing_bars(df, symbol)
        self.check_price_anomalies(df, symbol)
        self.check_volume_anomalies(df, symbol)
        self.check_corporate_actions(df, symbol)
        self.check_ohlc_integrity(df, symbol)
        self.check_timestamp_alignment(df, symbol)
        self.check_trading_hours(df, symbol)
        
        return {
            'symbol': symbol,
            'total_bars': len(df),
            'issues': self.issues,
            'issue_count': len(self.issues),
            'clean': len(self.issues) == 0,
        }
    
    def check_missing_bars(self, df: pd.DataFrame, symbol: str):
        """Detect missing 1-minute bars during trading hours."""
        # Generate expected trading minutes
        dates = df.index.date.unique()
        
        for date in dates:
            day_data = df[df.index.date == date]
            
            # Expected: 390 bars (9:30-15:59) for regular trading hours
            # But ORB only needs 9:30-9:35 (5 bars)
            orb_bars = day_data.between_time('09:30', '09:35')
            
            if len(orb_bars) < 5:
                self.issues.append({
                    'type': 'MISSING_ORB_BARS',
                    'severity': 'CRITICAL',
                    'date': str(date),
                    'symbol': symbol,
                    'expected': 5,
                    'actual': len(orb_bars),
                    'action': 'SKIP this trading day - signal will be unreliable',
                })
            
            # Check for gaps in the full day (affects exit timing)
            expected_minutes = pd.date_range(
                start=f"{date} 09:30:00",
                end=f"{date} 15:59:00",
                freq='1min',
            )
            missing = expected_minutes.difference(day_data.index)
            
            if len(missing) > 10:  # More than 10 missing bars
                self.issues.append({
                    'type': 'EXCESSIVE_MISSING_BARS',
                    'severity': 'WARNING',
                    'date': str(date),
                    'symbol': symbol,
                    'missing_count': len(missing),
                    'action': 'Investigate - may affect exit timing and P&L',
                })
    
    def check_price_anomalies(self, df: pd.DataFrame, symbol: str):
        """Detect price spikes, gaps, and impossible moves."""
        df = df.copy()
        df['return'] = df['close'].pct_change()
        
        # Extreme returns (>5% in 1 minute for SPY is suspicious)
        extreme = df[abs(df['return']) > 0.05]
        for idx, row in extreme.iterrows():
            self.issues.append({
                'type': 'EXTREME_RETURN',
                'severity': 'WARNING',
                'timestamp': str(idx),
                'symbol': symbol,
                'return': f"{row['return']:.4%}",
                'action': 'Verify against news - could be data error or halt',
            })
        
        # Zero prices
        zero_prices = df[(df['open'] == 0) | (df['close'] == 0)]
        if len(zero_prices) > 0:
            self.issues.append({
                'type': 'ZERO_PRICE',
                'severity': 'CRITICAL',
                'count': len(zero_prices),
                'symbol': symbol,
                'action': 'Remove or forward-fill these bars',
            })
        
        # Negative prices
        neg_prices = df[(df['open'] < 0) | (df['close'] < 0)]
        if len(neg_prices) > 0:
            self.issues.append({
                'type': 'NEGATIVE_PRICE',
                'severity': 'CRITICAL',
                'count': len(neg_prices),
                'symbol': symbol,
                'action': 'Data corruption - remove these bars',
            })
    
    def check_volume_anomalies(self, df: pd.DataFrame, symbol: str):
        """Detect suspicious volume patterns."""
        # Zero volume bars (shouldn't have trades if volume = 0)
        zero_vol = df[df['volume'] == 0]
        if len(zero_vol) > 0:
            self.issues.append({
                'type': 'ZERO_VOLUME',
                'severity': 'WARNING',
                'count': len(zero_vol),
                'symbol': symbol,
                'action': 'May indicate missing data or halts',
            })
        
        # Extreme volume (>10x average)
        avg_vol = df['volume'].mean()
        extreme_vol = df[df['volume'] > avg_vol * 10]
        if len(extreme_vol) > 0:
            self.issues.append({
                'type': 'EXTREME_VOLUME',
                'severity': 'INFO',
                'count': len(extreme_vol),
                'symbol': symbol,
                'action': 'Verify against news/events',
            })
    
    def check_corporate_actions(self, df: pd.DataFrame, symbol: str):
        """
        Detect unadjusted corporate actions.
        
        For SPY/QQQ: mainly dividends and rebalances.
        For individual stocks: splits, special dividends, M&A.
        """
        df = df.copy()
        df['return'] = df['close'].pct_change()
        
        # Detect potential unadjusted splits (price drops >40% in one day)
        daily_returns = df.groupby(df.index.date).apply(
            lambda x: (x['close'].iloc[-1] / x['open'].iloc[0]) - 1
        )
        
        for date, ret in daily_returns.items():
            if ret < -0.40:
                self.issues.append({
                    'type': 'POTENTIAL_UNADJUSTED_SPLIT',
                    'severity': 'CRITICAL',
                    'date': str(date),
                    'symbol': symbol,
                    'daily_return': f"{ret:.2%}",
                    'action': 'Check for stock split - use split-adjusted data',
                })
            elif ret < -0.03 and ret > -0.40:
                # Could be dividend ex-date (SPY ~0.3-0.4% per quarter)
                if symbol in ('SPY', 'QQQ'):
                    self.issues.append({
                        'type': 'POTENTIAL_EXDIV',
                        'severity': 'INFO',
                        'date': str(date),
                        'symbol': symbol,
                        'return': f"{ret:.2%}",
                        'action': 'Likely ex-dividend date - verify adjustment',
                    })
    
    def check_ohlc_integrity(self, df: pd.DataFrame, symbol: str):
        """Verify OHLC relationships are valid."""
        # High must be >= Open, Close, Low
        invalid_high = df[df['high'] < df[['open', 'close']].max(axis=1)]
        if len(invalid_high) > 0:
            self.issues.append({
                'type': 'INVALID_HIGH',
                'severity': 'CRITICAL',
                'count': len(invalid_high),
                'symbol': symbol,
                'action': 'Data integrity error - source data is corrupt',
            })
        
        # Low must be <= Open, Close, High
        invalid_low = df[df['low'] > df[['open', 'close']].min(axis=1)]
        if len(invalid_low) > 0:
            self.issues.append({
                'type': 'INVALID_LOW',
                'severity': 'CRITICAL',
                'count': len(invalid_low),
                'symbol': symbol,
                'action': 'Data integrity error - source data is corrupt',
            })
    
    def check_timestamp_alignment(self, df: pd.DataFrame, symbol: str):
        """Verify timestamps are on exact minute boundaries."""
        # Check if timestamps are aligned to minute
        misaligned = df.index[df.index.second != 0]
        if len(misaligned) > 0:
            self.issues.append({
                'type': 'MISALIGNED_TIMESTAMPS',
                'severity': 'WARNING',
                'count': len(misaligned),
                'symbol': symbol,
                'action': 'Round timestamps to nearest minute',
            })
    
    def check_trading_hours(self, df: pd.DataFrame, symbol: str):
        """Verify data is during regular trading hours."""
        pre_market = df.between_time('04:00', '09:29')
        post_market = df.between_time('16:01', '20:00')
        
        if len(pre_market) > 0 or len(post_market) > 0:
            self.issues.append({
                'type': 'EXTENDED_HOURS_DATA',
                'severity': 'INFO',
                'pre_market_bars': len(pre_market),
                'post_market_bars': len(post_market),
                'symbol': symbol,
                'action': 'Filter to RTH only (9:30-16:00) for ORB strategy',
            })


def clean_data_for_orb(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Apply all necessary cleaning for ORB backtesting.
    
    Returns cleaned DataFrame and logs all modifications.
    """
    original_len = len(df)
    modifications = []
    
    # 1. Filter to regular trading hours only
    df = df.between_time('09:30', '15:59')
    modifications.append(f"Filtered to RTH: {original_len} -> {len(df)} bars")
    
    # 2. Remove zero/negative prices
    bad_prices = (df['open'] <= 0) | (df['close'] <= 0) | (df['high'] <= 0) | (df['low'] <= 0)
    df = df[~bad_prices]
    modifications.append(f"Removed {bad_prices.sum()} bars with invalid prices")
    
    # 3. Remove extreme outliers (>10% in 1 minute for SPY/QQQ)
    df['return'] = df['close'].pct_change()
    extreme = abs(df['return']) > 0.10
    df = df[~extreme]
    modifications.append(f"Removed {extreme.sum()} extreme return bars")
    df = df.drop(columns=['return'])
    
    # 4. Ensure timestamps are timezone-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize('US/Eastern')
        modifications.append("Localized timestamps to US/Eastern")
    
    # 5. Sort by timestamp
    df = df.sort_index()
    modifications.append("Sorted by timestamp")
    
    # 6. Remove duplicate timestamps
    dupes = df.index.duplicated(keep='first')
    df = df[~dupes]
    modifications.append(f"Removed {dupes.sum()} duplicate timestamps")
    
    return df, modifications
```

---

## 6. Survivorship Bias

### The Problem

**For SPY/QQQ ETFs**: Survivorship bias is **minimal** because:
- SPY and QQQ are ETFs that track indices
- They don't go bankrupt (unlike individual stocks)
- The indices themselves have survivorship bias in their constituents

**For the paper's stock universe**: Survivorship bias is **severe** because:
- The paper trades "top 20 stocks by relative volume" from a universe
- If that universe only includes currently-listing stocks, dead stocks are missing
- Stocks that went bankrupt, were acquired, or delisted had different behavior
- This biases the backtest upward (surviving stocks tend to be winners)

**Key insight**: Even for SPY/QQQ, if you're comparing to "stocks" from the paper, you need to account for this.

### Code-Level Mitigation

```python
class SurvivorshipBiasAnalyzer:
    """
    Detect and mitigate survivorship bias in ORB backtesting.
    
    For SPY/QQQ: bias is minimal but check index composition changes.
    For stock universe: bias is severe - use point-in-time data.
    """
    
    def __init__(self, universe_type: str = 'etf'):
        self.universe_type = universe_type  # 'etf' or 'stocks'
    
    def check_sp500_composition_changes(self, date_range: tuple) -> dict:
        """
        Check how many S&P 500 constituents changed during backtest period.
        
        This matters because ORB selects "top 20 stocks" - if the universe
        changes, the strategy is implicitly selecting from a different pool.
        """
        # S&P 500 changes are published by S&P Dow Jones
        # For a proper backtest, you need point-in-time constituent data
        
        return {
            'recommendation': 'Use point-in-time S&P 500 constituent data',
            'data_sources': [
                'Compustat (academic)',
                'CRSP (academic)',
                'S&P Dow Jones (commercial)',
                'Sharadar (affordable)',
            ],
            'bias_magnitude': '1-3% annual return overstatement typical',
        }
    
    def compute_survivorship_adjusted_returns(
        self,
        live_returns: np.ndarray,
        dead_stock_return: float = -0.50,  # Assume dead stocks lost 50%
        dead_stock_fraction: float = 0.05,  # ~5% of universe dies per decade
        years: float = 7.0,
    ) -> dict:
        """
        Estimate bias from missing dead stocks.
        
        This is a rough correction. For precise work, use point-in-time data.
        """
        n_dead = int(dead_stock_fraction * years * len(live_returns) / 10)
        
        # Add assumed dead stock returns
        dead_returns = np.full(n_dead, dead_stock_return / 252)  # Daily return
        all_returns = np.concatenate([live_returns, dead_returns])
        
        live_sharpe = live_returns.mean() / live_returns.std() * np.sqrt(252)
        adjusted_sharpe = all_returns.mean() / all_returns.std() * np.sqrt(252)
        
        return {
            'live_sharpe': live_sharpe,
            'adjusted_sharpe': adjusted_sharpe,
            'bias_estimate': live_sharpe - adjusted_sharpe,
            'dead_stocks_added': n_dead,
        }
    
    def verify_etf_data_integrity(self, df: pd.DataFrame, symbol: str) -> dict:
        """
        For ETFs: verify no survivorship bias but check for:
        - Ticker changes (rare for SPY/QQQ)
        - Fund reorganizations
        - Data vendor adjustments
        """
        checks = {
            'ticker_consistent': True,
            'data_gaps': 0,
            'suspicious_returns': 0,
        }
        
        # Check for ticker changes (should be none for SPY/QQQ)
        if 'ticker' in df.columns:
            unique_tickers = df['ticker'].unique()
            if len(unique_tickers) > 1:
                checks['ticker_consistent'] = False
                checks['tickers_found'] = list(unique_tickers)
        
        # Check for data gaps (could indicate missing data)
        df_copy = df.copy()
        df_copy['date'] = df_copy.index.date
        trading_days = df_copy['date'].nunique()
        expected_days = len(pd.bdate_range(df.index.min(), df.index.max()))
        checks['data_gaps'] = expected_days - trading_days
        checks['coverage'] = trading_days / expected_days if expected_days > 0 else 0
        
        # Check for suspicious returns
        df_copy['return'] = df_copy['close'].pct_change()
        suspicious = (df_copy['return'].abs() > 0.05).sum()
        checks['suspicious_returns'] = suspicious
        
        return checks
```

---

## 7. Execution Timing (Latency, Order Queue Position)

### The Problem

#### Signal-to-Order Latency
The ORB signal is determined at 9:35:00 (close of 5th bar). But:
1. **Data transmission**: 1-min bar data arrives 100-500ms after bar close
2. **Signal computation**: Python processing takes 10-100ms
3. **Order transmission**: Network round-trip to broker: 5-50ms
4. **Broker processing**: 10-100ms

**Total latency**: 125-750ms. During this time, the price has moved.

#### Order Queue Position
- If you place a limit order, your position in the queue determines fill probability
- At 9:36 AM, thousands of orders are queued ahead of yours
- For market orders, you get filled but at progressively worse prices

#### Bar Boundary Timing
- 1-minute bars close at :00 seconds
- Your order arrives at :00.500 (500ms later)
- The next bar has already started

### Code-Level Mitigation

```python
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class LatencyModel:
    """Model all latency components in the order flow."""
    
    data_feed_latency_ms: int = 200    # Time for bar data to reach you
    signal_compute_ms: int = 50        # Time to compute signal
    order_transmit_ms: int = 30        # Time to send order to broker
    broker_process_ms: int = 50        # Broker processing time
    
    @property
    def total_latency_ms(self) -> int:
        return (self.data_feed_latency_ms + self.signal_compute_ms + 
                self.order_transmit_ms + self.broker_process_ms)
    
    @property
    def price_drift_per_ms(self) -> float:
        """Estimate price drift per millisecond for SPY at market open."""
        # SPY moves ~$0.01 per second at open (rough estimate)
        return 0.01 / 1000  # $0.00001 per ms
    
    def estimate_entry_slippage(self) -> float:
        """Estimate slippage due to latency."""
        return self.total_latency_ms * self.price_drift_per_ms


class ExecutionTimingOptimizer:
    """
    Optimize execution timing for ORB strategy.
    
    Key insight: You can't eliminate latency, but you can:
    1. Pre-compute what you can before bar close
    2. Use smart order types
    3. Model the cost of latency and include it in P&L
    """
    
    def __init__(self, latency_model: LatencyModel):
        self.latency = latency_model
    
    def pre_computation_schedule(self, current_time: datetime) -> dict:
        """
        Define what to pre-compute before the signal bar closes.
        
        At 9:34:00 (1 minute before signal):
        - Compute ATR (already known from prior days)
        - Compute relative volume (known from 4 of 5 bars)
        - Pre-calculate position size for both long and short scenarios
        
        At 9:35:00 (signal bar closes):
        - Only need to determine direction (1 comparison)
        - Submit order immediately
        """
        schedule = {
            '09:29:00': [
                'Verify data feed is active',
                'Load prior 14-day ATR',
                'Load prior 14-day OR volume averages',
                'Pre-compute position size templates',
            ],
            '09:30:00': [
                'Start accumulating OR volume',
                'Track OR high/low in real-time',
            ],
            '09:34:00': [
                'Pre-compute relative volume (4/5 bars available)',
                'Pre-calculate stop levels for both long and short',
                'Prepare order parameters for both scenarios',
            ],
            '09:35:00': [
                'Determine direction (close vs open of 5th bar)',
                'Submit order immediately',
                'Log execution timestamp for latency tracking',
            ],
        }
        return schedule
    
    def smart_order_selection(
        self,
        signal: str,
        current_price: float,
        spread: float,
        urgency: str = 'high',  # ORB entries are time-sensitive
    ) -> dict:
        """
        Select optimal order type for ORB entry.
        
        For ORB:
        - Entry: Market order (speed > price, you need to be in NOW)
        - Stop loss: Stop-market (guarantees trigger, accept slippage)
        - EOD exit: Market order at 15:55 (5 min before close)
        """
        if urgency == 'high':
            # Market order: fastest fill, worst price
            return {
                'order_type': 'MARKET',
                'expected_fill_price': current_price + spread/2 if signal == 'LONG' else current_price - spread/2,
                'expected_slippage': spread/2,
                'fill_probability': 1.0,
                'fill_time_ms': 100,
            }
        elif urgency == 'medium':
            # Aggressive limit: slightly inside the spread
            offset = spread * 0.25
            limit_price = current_price + offset if signal == 'LONG' else current_price - offset
            return {
                'order_type': 'LIMIT',
                'limit_price': limit_price,
                'expected_fill_probability': 0.70,  # 70% chance of fill
                'expected_slippage': 0,
                'fill_time_ms': 500,
                'risk': 'May not fill - price moves away',
            }
        else:
            # Passive limit: at the bid/ask
            limit_price = current_price if signal == 'LONG' else current_price
            return {
                'order_type': 'LIMIT',
                'limit_price': limit_price,
                'expected_fill_probability': 0.40,
                'expected_slippage': 0,
                'fill_time_ms': 2000,
                'risk': 'High probability of missing the trade',
            }
    
    def simulate_latency_impact(
        self,
        trades_df: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add latency-based slippage to each trade.
        
        For ORB at 9:35-9:36, latency is most costly because:
        - Price is moving fastest (opening range breakout)
        - Spread is widest
        - Volume is highest (many competing orders)
        """
        latency_cost = []
        
        for _, trade in trades_df.iterrows():
            entry_time = trade['entry_time']
            entry_bar = market_data.loc[entry_time]
            
            # Price movement during latency period
            bar_range = entry_bar['high'] - entry_bar['low']
            bar_duration_ms = 60000  # 1 minute
            
            # Latency slippage as fraction of bar range
            latency_fraction = self.latency.total_latency_ms / bar_duration_ms
            latency_slippage = bar_range * latency_fraction
            
            # Adjust for time of day (worse at open)
            if entry_time.hour == 9 and entry_time.minute < 40:
                latency_slippage *= 2.0  # 2x worse at open
            
            latency_cost.append({
                'latency_ms': self.latency.total_latency_ms,
                'latency_slippage': latency_slippage,
                'latency_cost_per_share': latency_slippage,
            })
        
        latency_df = pd.DataFrame(latency_cost)
        return pd.concat([trades_df.reset_index(drop=True), latency_df], axis=1)


def backtest_with_realistic_timing(
    df: pd.DataFrame,
    strategy_fn,
    latency_model: LatencyModel,
) -> dict:
    """
    Run backtest with realistic execution timing.
    
    Key modifications:
    1. Signal computed at 9:35:00 + latency
    2. Entry at 9:36:00 bar open (not 9:35:00 close)
    3. Stop exits include queue position slippage
    4. EOD exits at 15:55 (not 16:00) to avoid close auction
    """
    # Adjust all entry times by latency
    df_adjusted = df.copy()
    
    # Signal bar is 9:35, but entry is delayed by latency
    # If latency > 60 seconds, entry shifts to next bar
    latency_bars = max(1, latency_model.total_latency_ms // 60000 + 1)
    
    # Entry price should be the OPEN of the bar AFTER signal + latency
    # Not the CLOSE of the signal bar
    
    return {
        'latency_bars': latency_bars,
        'recommendation': f'Entry should be at bar[signal_bar + {latency_bars}] open',
        'note': 'Most backtests use signal bar close - this is optimistic',
    }
```

---

## 8. Regime Changes (Volatility Clusters, Market Hours)

### The Problem

#### Volatility Regimes
ORB performance varies dramatically by volatility regime:
- **Low vol (VIX < 15)**: Small ORB ranges, small stops, small wins
- **Normal vol (VIX 15-25)**: Paper's sweet spot
- **High vol (VIX 25-35)**: Wide ORB ranges, stops get hit, whipsaws
- **Crisis (VIX > 35)**: Strategy breaks down, gap risk extreme

#### Market Hours Effects
- **9:30-10:00**: Highest volume, widest spreads, most ORB activity
- **10:00-11:30**: Trending period, good for ORB continuation
- **11:30-13:30**: Lunch lull, lower volume, mean-reverting
- **13:30-15:00**: Afternoon trend, institutional activity
- **15:00-16:00**: Close auction preparation, MOC orders

#### Structural Breaks
- 2020 COVID crash: VIX went from 15 to 80 in weeks
- 2022 rate hikes: Changed correlation structure
- 2023 AI boom: Changed sector dynamics
- Market microstructure changes (more HFT, fewer human market makers)

### Code-Level Mitigation

```python
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

@dataclass
class RegimeState:
    volatility_regime: str  # 'low', 'normal', 'high', 'crisis'
    vix_level: float
    trend_regime: str  # 'trending', 'mean_reverting', 'random'
    market_phase: str  # 'open', 'morning', 'lunch', 'afternoon', 'close'
    confidence: float  # 0-1

class RegimeDetector:
    """
    Detect market regime to condition ORB strategy.
    
    Key insight: A strategy that works in all regimes is suspicious.
    A strategy that knows when NOT to trade is more realistic.
    """
    
    def __init__(
        self,
        vix_lookback: int = 20,
        vol_regime_thresholds: dict = None,
    ):
        self.vix_lookback = vix_lookback
        self.vol_regime_thresholds = vol_regime_thresholds or {
            'low': 15,
            'normal': 25,
            'high': 35,
        }
    
    def detect_volatility_regime(
        self,
        vix_series: pd.Series,
        current_date: datetime.date,
    ) -> str:
        """
        Classify current volatility regime using VIX.
        
        Uses both current level and recent trend (rising/falling VIX).
        """
        current_vix = vix_series.loc[current_date] if current_date in vix_series.index else None
        
        if current_vix is None:
            return 'unknown'
        
        if current_vix < self.vol_regime_thresholds['low']:
            return 'low'
        elif current_vix < self.vol_regime_thresholds['normal']:
            return 'normal'
        elif current_vix < self.vol_regime_thresholds['high']:
            return 'high'
        else:
            return 'crisis'
    
    def detect_trend_regime(
        self,
        returns: pd.Series,
        lookback: int = 20,
    ) -> str:
        """
        Detect if market is trending or mean-reverting.
        
        ORB works better in trending markets (breakouts follow through).
        In mean-reverting markets, ORB breakouts reverse (whipsaws).
        """
        if len(returns) < lookback:
            return 'unknown'
        
        recent = returns.tail(lookback)
        
        # Hurst exponent approximation
        # H > 0.5: trending, H < 0.5: mean-reverting, H = 0.5: random
        lags = range(2, min(20, len(recent) // 2))
        tau = [np.sqrt(np.std(np.subtract(recent[lag:], recent[:-lag]))) for lag in lags]
        
        if len(tau) < 2:
            return 'unknown'
        
        # Linear fit to log(tau) vs log(lag)
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        hurst = poly[0]
        
        if hurst > 0.55:
            return 'trending'
        elif hurst < 0.45:
            return 'mean_reverting'
        else:
            return 'random'
    
    def detect_market_phase(self, time: datetime) -> str:
        """Classify current market phase."""
        hour, minute = time.hour, time.minute
        
        if hour == 9 and minute < 45:
            return 'open'
        elif hour < 11 or (hour == 11 and minute < 30):
            return 'morning'
        elif hour < 13 or (hour == 13 and minute < 30):
            return 'lunch'
        elif hour < 15:
            return 'afternoon'
        else:
            return 'close'
    
    def get_regime_state(
        self,
        current_date: datetime.date,
        current_time: datetime,
        vix_series: pd.Series,
        returns: pd.Series,
    ) -> RegimeState:
        """Get complete regime state for trading decision."""
        vol_regime = self.detect_volatility_regime(vix_series, current_date)
        trend_regime = self.detect_trend_regime(returns)
        market_phase = self.detect_market_phase(current_time)
        
        # Confidence is higher in normal conditions
        confidence = 1.0
        if vol_regime == 'crisis':
            confidence *= 0.3
        elif vol_regime == 'high':
            confidence *= 0.6
        if trend_regime == 'mean_reverting':
            confidence *= 0.7
        if market_phase == 'lunch':
            confidence *= 0.8
        
        return RegimeState(
            volatility_regime=vol_regime,
            vix_level=vix_series.get(current_date, 0),
            trend_regime=trend_regime,
            market_phase=market_phase,
            confidence=confidence,
        )


class RegimeAwareORB:
    """
    ORB strategy conditioned on market regime.
    
    Different parameters for different regimes:
    - Low vol: Tighter stops, smaller positions
    - Normal vol: Paper's default parameters
    - High vol: Wider stops, smaller positions, fewer trades
    - Crisis: No trading or very small positions
    """
    
    def __init__(self):
        self.regime_params = {
            'low': {
                'stop_atr_pct': 0.08,      # Tighter stops
                'position_risk_pct': 0.01,  # 1% risk
                'max_positions': 4,
                'min_relative_volume': 1.5, # Higher vol filter
            },
            'normal': {
                'stop_atr_pct': 0.10,      # Paper's default
                'position_risk_pct': 0.01,
                'max_positions': 4,
                'min_relative_volume': 1.0,
            },
            'high': {
                'stop_atr_pct': 0.15,      # Wider stops
                'position_risk_pct': 0.005, # Half risk
                'max_positions': 2,
                'min_relative_volume': 2.0,
            },
            'crisis': {
                'stop_atr_pct': 0.20,
                'position_risk_pct': 0.002, # Minimal risk
                'max_positions': 1,
                'min_relative_volume': 3.0,
            },
        }
    
    def get_trade_parameters(
        self,
        regime: RegimeState,
        atr_14d: float,
        account_size: float,
    ) -> dict:
        """Get position sizing and stop parameters for current regime."""
        params = self.regime_params[regime.volatility_regime]
        
        stop_distance = params['stop_atr_pct'] * atr_14d
        risk_amount = account_size * params['position_risk_pct']
        shares = int(risk_amount / stop_distance) if stop_distance > 0 else 0
        
        return {
            'stop_distance': stop_distance,
            'shares': shares,
            'risk_amount': risk_amount,
            'max_positions': params['max_positions'],
            'min_relative_volume': params['min_relative_volume'],
            'regime_confidence': regime.confidence,
            'should_trade': regime.confidence > 0.5,
        }
    
    def analyze_regime_performance(
        self,
        trades_df: pd.DataFrame,
        regime_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Analyze ORB performance by regime.
        
        If strategy only works in one regime, it's regime-dependent
        and will fail when regime changes.
        """
        trades_df = trades_df.copy()
        trades_df['regime'] = trades_df['entry_time'].map(
            lambda t: regime_series.get(t.date(), 'unknown')
        )
        
        regime_stats = trades_df.groupby('regime').agg({
            'pnl': ['mean', 'std', 'count', 'sum'],
            'is_win': 'mean',
        }).round(4)
        
        regime_stats.columns = ['avg_pnl', 'std_pnl', 'trade_count', 'total_pnl', 'win_rate']
        regime_stats['sharpe'] = regime_stats['avg_pnl'] / regime_stats['std_pnl'] * np.sqrt(252)
        
        return regime_stats


def detect_structural_breaks(
    returns: pd.Series,
    window: int = 63,  # 3 months
) -> pd.DataFrame:
    """
    Detect structural breaks in return distribution.
    
    Uses rolling statistics to identify when the data-generating process changed.
    """
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
    
    # Detect breaks using CUSUM
    cumsum = (returns - returns.mean()).cumsum()
    
    # Find points where cumsum changes direction sharply
    breaks = []
    threshold = cumsum.std() * 2
    
    for i in range(window, len(cumsum) - window):
        before = cumsum.iloc[i-window:i].mean()
        after = cumsum.iloc[i:i+window].mean()
        
        if abs(after - before) > threshold:
            breaks.append({
                'date': cumsum.index[i],
                'before_mean': before,
                'after_mean': after,
                'change': after - before,
                'rolling_sharpe': rolling_sharpe.iloc[i],
            })
    
    return pd.DataFrame(breaks)
```

---

## 9. Capital Requirements and Leverage Constraints

### The Problem

#### Minimum Capital Requirements
For ORB on SPY/QQQ:
- SPY at $500: 100 shares = $50,000 per position
- QQQ at $450: 100 shares = $45,000 per position
- If max 4 positions: $180,000-$200,000 minimum

#### Leverage Constraints
- **Reg T margin**: 50% initial, 25% maintenance (2x leverage max)
- **Portfolio margin**: Up to 6x leverage (requires $125K+ and approval)
- **Pattern Day Trader**: 4x leverage for day traders (requires $25K+)

#### Capital Efficiency
- If you have $25K (PDT minimum), you can only hold 1 SPY position
- The paper assumes 4x leverage across 20 stocks - requires $500K+
- This is not realistic for retail traders

### Code-Level Mitigation

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class AccountConstraints:
    """Model real-world account constraints."""
    
    account_size: float
    margin_type: str  # 'cash', 'reg_t', 'portfolio'
    is_pdt: bool  # Pattern Day Trader status
    buying_power_multiplier: float = 1.0
    
    @property
    def max_buying_power(self) -> float:
        if self.margin_type == 'cash':
            return self.account_size
        elif self.margin_type == 'reg_t':
            return self.account_size * 2  # 50% initial margin
        elif self.margin_type == 'portfolio':
            return self.account_size * 6  # Up to 6x
        return self.account_size
    
    @property
    def max_day_trade_leverage(self) -> float:
        if self.is_pdt:
            return 4.0  # 4x for PDT
        else:
            return 1.0  # Cash account, no leverage for day trades
    
    @property
    def min_account_for_pdt(self) -> float:
        return 25000.0


class CapitalManager:
    """
    Manage capital allocation for ORB strategy.
    
    Enforces real-world constraints that the paper ignores.
    """
    
    def __init__(self, account: AccountConstraints):
        self.account = account
        self.open_positions = []
        self.day_trades_count = 0  # For non-PDT accounts
    
    def can_open_position(
        self,
        symbol: str,
        price: float,
        shares: int,
        is_day_trade: bool = True,
    ) -> tuple[bool, str]:
        """
        Check if position can be opened given account constraints.
        
        Returns (can_trade, reason).
        """
        # Check PDT rule
        if is_day_trade and not self.account.is_pdt:
            if self.day_trades_count >= 3:
                return False, "PDT: Already used 3 day trades in 5 business days"
        
        # Check buying power
        position_value = price * shares
        required_bp = position_value / self.account.max_day_trade_leverage
        
        available_bp = self.account.max_buying_power - self.get_used_buying_power()
        
        if required_bp > available_bp:
            return False, f"Insufficient buying power: need ${required_bp:,.0f}, have ${available_bp:,.0f}"
        
        # Check minimum position size
        if position_value < 2000:  # Not worth the overhead
            return False, f"Position too small: ${position_value:,.0f}"
        
        # Check maximum positions
        if len(self.open_positions) >= 4:  # Paper's max
            return False, "Maximum 4 concurrent positions"
        
        return True, "OK"
    
    def get_used_buying_power(self) -> float:
        """Calculate buying power used by open positions."""
        return sum(
            pos['value'] / self.account.max_day_trade_leverage
            for pos in self.open_positions
        )
    
    def calculate_position_size(
        self,
        price: float,
        stop_distance: float,
        risk_pct: float = 0.01,  # 1% risk per trade
        max_leverage: float = None,
    ) -> dict:
        """
        Calculate position size respecting all constraints.
        
        Returns position size that satisfies:
        1. Risk limit (1% of account)
        2. Buying power limit
        3. Leverage limit
        4. Minimum viable size
        """
        # Risk-based sizing
        risk_amount = self.account.account_size * risk_pct
        risk_based_shares = int(risk_amount / stop_distance) if stop_distance > 0 else 0
        
        # Buying power constraint
        available_bp = self.account.max_buying_power - self.get_used_buying_power()
        bp_based_shares = int(available_bp * self.account.max_day_trade_leverage / price)
        
        # Leverage constraint
        max_lev = max_leverage or self.account.max_day_trade_leverage
        max_position_value = self.account.account_size * max_lev
        lev_based_shares = int(max_position_value / price)
        
        # Take the minimum
        shares = min(risk_based_shares, bp_based_shares, lev_based_shares)
        
        # Enforce minimum
        if shares < 1:
            shares = 0
        
        return {
            'shares': shares,
            'position_value': shares * price,
            'risk_amount': shares * stop_distance,
            'risk_pct': (shares * stop_distance) / self.account.account_size if self.account.account_size > 0 else 0,
            'leverage': (shares * price) / self.account.account_size if self.account.account_size > 0 else 0,
            'buying_power_used': (shares * price) / self.account.max_day_trade_leverage,
            'binding_constraint': self._get_binding_constraint(
                risk_based_shares, bp_based_shares, lev_based_shares
            ),
        }
    
    def _get_binding_constraint(
        self,
        risk_shares: int,
        bp_shares: int,
        lev_shares: int,
    ) -> str:
        """Identify which constraint is binding."""
        min_shares = min(risk_shares, bp_shares, lev_shares)
        if min_shares == risk_shares:
            return 'risk_limit'
        elif min_shares == bp_shares:
            return 'buying_power'
        else:
            return 'leverage_limit'


def simulate_capital_constraints(
    trades_df: pd.DataFrame,
    initial_capital: float = 25000,
    margin_type: str = 'reg_t',
) -> pd.DataFrame:
    """
    Simulate ORB strategy with real capital constraints.
    
    Shows how performance degrades with realistic account sizes.
    """
    account = AccountConstraints(
        account_size=initial_capital,
        margin_type=margin_type,
        is_pdt=initial_capital >= 25000,
    )
    
    manager = CapitalManager(account)
    capital = initial_capital
    results = []
    
    for _, trade in trades_df.iterrows():
        # Check if we can trade
        can_trade, reason = manager.can_open_position(
            symbol=trade.get('symbol', 'SPY'),
            price=trade['entry_price'],
            shares=trade['shares'],
        )
        
        if not can_trade:
            results.append({
                **trade.to_dict(),
                'skipped': True,
                'skip_reason': reason,
                'capital': capital,
            })
            continue
        
        # Recalculate position size for our account
        sizing = manager.calculate_position_size(
            price=trade['entry_price'],
            stop_distance=trade['stop_distance'],
            risk_pct=0.01,
        )
        
        if sizing['shares'] == 0:
            results.append({
                **trade.to_dict(),
                'skipped': True,
                'skip_reason': 'Position too small for account',
                'capital': capital,
            })
            continue
        
        # Execute with our size
        actual_pnl = trade['pnl'] * (sizing['shares'] / trade['shares'])
        actual_cost = trade.get('total_cost', 0) * (sizing['shares'] / trade['shares'])
        
        capital += actual_pnl - actual_cost
        
        results.append({
            **trade.to_dict(),
            'skipped': False,
            'actual_shares': sizing['shares'],
            'actual_pnl': actual_pnl,
            'actual_cost': actual_cost,
            'capital': capital,
            'binding_constraint': sizing['binding_constraint'],
        })
    
    return pd.DataFrame(results)
```

---

## 10. Regulatory Considerations (PDT Rule, Margin Requirements)

### The Problem

#### Pattern Day Trader (PDT) Rule
- **Definition**: 4+ day trades in 5 business days = PDT
- **Consequence**: Must maintain $25,000 minimum equity
- **ORB impact**: 1 trade/day × 5 days = 5 day trades = PTD status

#### Margin Requirements
- **Reg T**: 50% initial margin for stocks
- **Maintenance margin**: 25% minimum (broker may require more)
- **Day trading margin**: 25% for PDT accounts (4x leverage)

#### Wash Sale Rule
- **Does NOT apply to day trading** (positions closed same day)
- **But applies if you re-enter the same stock** within 30 days at a loss

#### Tax Implications
- **Day trading profits**: Short-term capital gains (ordinary income rates)
- **Mark-to-market election**: Section 475(f) - beneficial for active traders
- **Wash sale disallowed losses**: Can accumulate across accounts

### Code-Level Mitigation

```python
from datetime import datetime, timedelta
from collections import deque
from typing import Optional

class RegulatoryCompliance:
    """
    Ensure ORB strategy complies with all regulatory requirements.
    
    Violations can result in:
    - Account restriction (PDT violation)
    - Margin calls (insufficient equity)
    - Tax penalties (wash sale, mark-to-market)
    """
    
    def __init__(
        self,
        account_size: float,
        broker: str = 'generic',
        margin_type: str = 'reg_t',
    ):
        self.account_size = account_size
        self.broker = broker
        self.margin_type = margin_type
        
        # PDT tracking
        self.day_trades = deque(maxlen=100)  # Store (date, symbol) tuples
        self.is_pdt = False
        
        # Margin tracking
        self.open_positions = []
        self.margin_used = 0
        
        # Tax tracking
        self.realized_gains = []  # (date, symbol, gain, is_wash_sale)
        self.disallowed_losses = 0
    
    def check_pdt_before_trade(
        self,
        symbol: str,
        trade_date: datetime.date,
        is_day_trade: bool = True,
    ) -> tuple[bool, str]:
        """
        Check if trade would trigger PDT rule violation.
        
        PDT Rule: 4+ day trades in 5 business days = must maintain $25K.
        
        A "day trade" is buying and selling the SAME security on the SAME day.
        """
        if not is_day_trade:
            return True, "Not a day trade"
        
        if self.account_size >= 25000:
            return True, "Account above $25K PDT threshold"
        
        # Count day trades in last 5 business days
        five_days_ago = trade_date - timedelta(days=7)  # Conservative: 7 calendar days
        recent_trades = [
            dt for dt, sym in self.day_trades
            if dt > five_days_ago and dt <= trade_date
        ]
        
        if len(recent_trades) >= 3:
            return False, f"PDT: Would be trade #{len(recent_trades)+1} in 5 days. Need $25K minimum."
        
        return True, f"OK: {len(recent_trades)+1}/3 day trades used"
    
    def record_day_trade(
        self,
        symbol: str,
        trade_date: datetime.date,
    ):
        """Record a completed day trade."""
        self.day_trades.append((trade_date, symbol))
        
        # Check if now PDT
        five_days_ago = trade_date - timedelta(days=7)
        recent_count = sum(1 for dt, _ in self.day_trades if dt > five_days_ago)
        
        if recent_count >= 4 and self.account_size < 25000:
            self.is_pdt = True
    
    def check_margin_requirement(
        self,
        position_value: float,
        side: str = 'long',
    ) -> tuple[bool, str, float]:
        """
        Check if position meets margin requirements.
        
        Reg T: 50% initial margin
        Maintenance: 25% (broker may require 30-40%)
        """
        if self.margin_type == 'cash':
            if position_value > self.account_size:
                return False, "Cash account: cannot buy on margin", 0
            return True, "Cash account", position_value
        
        # Initial margin requirement
        initial_margin_pct = 0.50  # Reg T
        required_initial = position_value * initial_margin_pct
        
        # Current margin usage
        current_margin = sum(pos['value'] * initial_margin_pct for pos in self.open_positions)
        
        available = self.account_size - current_margin
        
        if required_initial > available:
            return False, f"Insufficient margin: need ${required_initial:,.0f}, available ${available:,.0f}", 0
        
        return True, "OK", available - required_initial
    
    def calculate_maintenance_margin(self) -> dict:
        """
        Calculate current maintenance margin requirement.
        
        If equity falls below maintenance margin, you get a margin call.
        """
        total_position_value = sum(pos['value'] for pos in self.open_positions)
        
        # Maintenance margin: 25% for long, 30% for short
        long_value = sum(pos['value'] for pos in self.open_positions if pos['side'] == 'long')
        short_value = sum(pos['value'] for pos in self.open_positions if pos['side'] == 'short')
        
        maintenance_req = long_value * 0.25 + short_value * 0.30
        
        # Current equity
        unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in self.open_positions)
        equity = self.account_size + unrealized_pnl
        
        excess = equity - maintenance_req
        
        return {
            'equity': equity,
            'maintenance_requirement': maintenance_req,
            'excess_equity': excess,
            'margin_call': excess < 0,
            'margin_call_amount': abs(excess) if excess < 0 else 0,
        }
    
    def check_wash_sale(
        self,
        symbol: str,
        trade_date: datetime.date,
        loss: float,
    ) -> tuple[bool, float]:
        """
        Check wash sale rule.
        
        Wash sale: If you sell at a loss and buy the same security
        within 30 days before or after, the loss is disallowed.
        
        NOTE: This primarily affects end-of-day traders, not day traders.
        But if you close a loss position and re-enter same day, it applies.
        """
        if loss >= 0:
            return False, 0  # No loss, no wash sale concern
        
        # Check if we bought the same symbol in last 30 days
        thirty_days_ago = trade_date - timedelta(days=30)
        recent_buys = [
            g for g in self.realized_gains
            if g['symbol'] == symbol and g['date'] > thirty_days_ago and g['date'] < trade_date
        ]
        
        if len(recent_buys) > 0:
            # Wash sale: loss is disallowed
            self.disallowed_losses += abs(loss)
            return True, loss
        
        return False, 0
    
    def mark_to_market_election_check(self) -> dict:
        """
        Section 475(f) Mark-to-Market election.
        
        For qualifying traders:
        - All gains/losses treated as ordinary income
        - No wash sale rule
        - No $3,000 capital loss limitation
        - Must elect by April 15 of prior year
        """
        return {
            'eligible': True,  # Must meet "trader" status
            'requirements': [
                'Must be in the business of trading',
                'Must seek to profit from daily market movements',
                'Must carry on activity with continuity and regularity',
                'Must elect by April 15 of the tax year (or extension)',
            ],
            'benefits': [
                'All gains taxed as ordinary income (no preferential rates)',
                'Losses fully deductible (no $3K limit)',
                'No wash sale rule',
                'Unrealized gains/losses at year-end treated as realized',
            ],
            'drawbacks': [
                'All gains taxed at ordinary income rates (higher than capital gains)',
                'Must file Form 3115 to change accounting method',
                'Cannot hold investments (must be pure trading)',
            ],
            'recommendation': 'Consult tax advisor - beneficial if net profitable',
        }


def apply_regulatory_constraints_to_backtest(
    trades_df: pd.DataFrame,
    initial_capital: float = 25000,
    margin_type: str = 'reg_t',
) -> pd.DataFrame:
    """
    Apply all regulatory constraints to backtest.
    
    This shows the TRUE performance after regulatory friction.
    """
    compliance = RegulatoryCompliance(
        account_size=initial_capital,
        margin_type=margin_type,
    )
    
    results = []
    
    for _, trade in trades_df.iterrows():
        trade_date = trade['entry_time'].date() if hasattr(trade['entry_time'], 'date') else trade['entry_time']
        
        # Check PDT
        can_trade_pdt, pdt_msg = compliance.check_pdt_before_trade(
            symbol=trade.get('symbol', 'SPY'),
            trade_date=trade_date,
            is_day_trade=True,
        )
        
        if not can_trade_pdt:
            results.append({
                **trade.to_dict(),
                'skipped': True,
                'skip_reason': pdt_msg,
                'regulatory_issue': 'PDT',
            })
            continue
        
        # Check margin
        can_trade_margin, margin_msg, _ = compliance.check_margin_requirement(
            position_value=trade['entry_price'] * trade['shares'],
        )
        
        if not can_trade_margin:
            results.append({
                **trade.to_dict(),
                'skipped': True,
                'skip_reason': margin_msg,
                'regulatory_issue': 'MARGIN',
            })
            continue
        
        # Record the trade
        compliance.record_day_trade(
            symbol=trade.get('symbol', 'SPY'),
            trade_date=trade_date,
        )
        
        # Check wash sale if there's a loss
        pnl = trade.get('pnl', 0)
        if pnl < 0:
            is_wash, disallowed = compliance.check_wash_sale(
                symbol=trade.get('symbol', 'SPY'),
                trade_date=trade_date,
                loss=pnl,
            )
        else:
            is_wash, disallowed = False, 0
        
        results.append({
            **trade.to_dict(),
            'skipped': False,
            'pdt_trades_used': len(compliance.day_trades),
            'is_wash_sale': is_wash,
            'disallowed_loss': disallowed,
            'regulatory_issue': None,
        })
    
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    skipped = results_df[results_df['skipped'] == True]
    if len(skipped) > 0:
        print(f"WARNING: {len(skipped)} trades skipped due to regulatory constraints")
        print(f"  PDT violations: {(skipped['regulatory_issue'] == 'PDT').sum()}")
        print(f"  Margin violations: {(skipped['regulatory_issue'] == 'MARGIN').sum()}")
    
    return results_df
```

---

## Summary: Pitfall Impact Matrix

| Pitfall | Impact | Mitigation Difficulty | Priority |
|---------|--------|----------------------|----------|
| 1. Look-ahead bias | **CRITICAL** - Invalidates entire backtest | Medium | P0 |
| 2. Transaction costs | **HIGH** - 10-30 bps per trade | Easy | P0 |
| 3. Microstructure | **HIGH** - Stop fills are wrong | Medium | P1 |
| 4. Overfitting (Rel Vol) | **HIGH** - False confidence | Hard | P1 |
| 5. Data quality | **MEDIUM** - Corrupts signals | Easy | P1 |
| 6. Survivorship bias | **LOW** for SPY/QQQ, **HIGH** for stocks | Medium | P2 |
| 7. Execution timing | **MEDIUM** - Latency slippage | Medium | P2 |
| 8. Regime changes | **HIGH** - Strategy breaks in crisis | Hard | P1 |
| 9. Capital requirements | **HIGH** - Can't actually trade | Easy | P0 |
| 10. Regulatory (PDT) | **CRITICAL** - Account restriction | Easy | P0 |

## Implementation Priority

### Phase 1: Data Integrity (Must Do First)
1. Implement `DataQualityChecker` and validate all data
2. Implement `clean_data_for_orb()` preprocessing
3. Verify no look-ahead bias with `test_no_lookahead()`

### Phase 2: Realistic Backtesting
4. Apply `RealisticCostModel` with stress tests
5. Apply `MicrostructureAwareExecutor` for fill simulation
6. Apply `RegulatoryCompliance` for PDT/margin checks
7. Apply `CapitalManager` for realistic position sizing

### Phase 3: Robustness Testing
8. Run `walk_forward_validation()` for out-of-sample testing
9. Run `permutation_test()` for statistical significance
10. Run `parameter_sensitivity_heatmap()` for robustness
11. Compute `deflated_sharpe_ratio()` for multiple testing correction

### Phase 4: Production Readiness
12. Implement `RegimeDetector` for regime conditioning
13. Implement `ExecutionTimingOptimizer` for live execution
14. Set up monitoring for data quality and regime changes

---

## References

- Zarattini, C., Barbon, A., & Aziz, A. (2024). "A Profitable Day Trading Strategy For The U.S. Equity Market." SSRN.
- Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio." Journal of Portfolio Management.
- López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley.
- Harris, L. (2003). "Trading and Exchanges." Oxford University Press.
- SEC Rule 15c3-1 (Net Capital Rule)
- FINRA Rule 4210 (Margin Requirements)
- SEC Pattern Day Trader Rule (Rule 2520)
