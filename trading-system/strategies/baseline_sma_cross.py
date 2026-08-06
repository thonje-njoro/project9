import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

SYMBOL_FILE = "data/XAUUSD_15m_train.parquet"

df = pd.read_parquet(SYMBOL_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

# backtesting.py expects these exact column names, capitalized
df = df.rename(columns={
    "open": "Open", "high": "High", "low": "Low",
    "close": "Close", "volume": "Volume"
})

class SmaCross(Strategy):
    fast = 10
    slow = 30
    risk_pct = 0.01       # risk 1% of equity per trade
    sl_atr_mult = 2.0     # stop-loss = 2x ATR away from entry

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        self.sma_fast = self.I(lambda x: x.rolling(self.fast).mean(), close)
        self.sma_slow = self.I(lambda x: x.rolling(self.slow).mean(), close)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        self.atr = self.I(lambda: tr.rolling(14).mean())

    def next(self):
        price = self.data.Close[-1]
        atr = self.atr[-1]
        if atr != atr or atr == 0:
            return

        sl_distance = atr * self.sl_atr_mult
        margin_ratio = 1/30  # must match the Backtest(margin=...) value below

        # size_fraction is cash committed as margin; leverage multiplies real exposure
        size_fraction = (self.risk_pct * price * margin_ratio) / sl_distance
        size_fraction = max(0.001, min(0.1, size_fraction))  # hard safety cap: never >10% of equity as margin

        buy_signal = crossover(self.sma_fast, self.sma_slow)
        sell_signal = crossover(self.sma_slow, self.sma_fast)

        if buy_signal:
            if self.position.is_short:
                self.position.close()
            if not self.position:
                self.buy(size=size_fraction, sl=price - sl_distance)
        elif sell_signal:
            if self.position.is_long:
                self.position.close()
            if not self.position:
                self.sell(size=size_fraction, sl=price + sl_distance)

if __name__ == "__main__":
    bt = Backtest(df, SmaCross, cash=10_000, commission=0.0002, margin=1/30, finalize_trades=True)
    stats = bt.run()
    print(stats)
    bt.plot(filename="runs/baseline_sma_cross.html", open_browser=False)
