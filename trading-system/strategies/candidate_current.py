import pandas as pd
from backtesting import Strategy

class CandidateCurrent(Strategy):
    bb_period = 20
    bb_std = 2.5
    rsi_period = 14
    atr_period = 14
    trend_period = 480
    risk_pct = 0.015
    sl_atr_mult = 2.0
    tp_atr_mult = 4.0      # take-profit = 1.5x the stop distance (3.0 * 1.5)
    atr_regime_period = 50        # longer ATR for regime detection
    atr_regime_mult   = 1.5       # if current ATR > 1.5x the 50-period ATR, market is trending — skip entries

    def init(self):
        close = pd.Series(self.data.Close)
        high  = pd.Series(self.data.High)
        low   = pd.Series(self.data.Low)

        # Assign to self BEFORE registering indicators that use them
        self._close = close
        self._high  = high
        self._low   = low

        self.bb_mid  = self.I(lambda: close.rolling(self.bb_period).mean(), name="BB_MID")
        self.bb_std  = self.I(lambda: close.rolling(self.bb_period).std(),  name="BB_STD")
        self.rsi     = self.I(self._rsi, name="RSI")
        self.atr     = self.I(self._atr, name="ATR")
        self.trend   = self.I(lambda: close.rolling(self.trend_period).mean(), name="TREND")
        # Regime detection ATR (50-period)
        self._atr_regime = self.I(
            lambda: pd.concat([
                self._high - self._low,
                (self._high - self._close.shift()).abs(),
                (self._low - self._close.shift()).abs(),
            ], axis=1).max(axis=1).rolling(self.atr_regime_period).mean(),
            name="ATR_REGIME"
        )

    def _rsi(self):
        delta = self._close.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta).clip(lower=0).rolling(self.rsi_period).mean()
        return (100 - 100 / (1 + gain / loss)).values

    def _atr(self):
        tr = pd.concat([
            self._high - self._low,
            (self._high - self._close.shift()).abs(),
            (self._low - self._close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean().values

    def next(self):
        close   = self.data.Close[-1]
        bb_mid  = self.bb_mid[-1]
        bb_std  = self.bb_std[-1]
        bb_upper = bb_mid + self.bb_std * bb_std
        bb_lower = bb_mid - self.bb_std * bb_std
        rsi     = self.rsi[-1]
        atr     = self.atr[-1]
        trend   = self.trend[-1]

        # Guard: skip until all indicators have warmed up
        for val in [bb_mid, bb_std, rsi, atr, trend]:
            if val != val or val == 0:
                return

        # --- 1. Circuit breaker: reset at start of new day ---
        current_date = self.data.index[-1].date()
        if not hasattr(self, '_last_day') or self._last_day != current_date:
            self._last_day = current_date
            self._day_start_equity = self.equity
            self._entry_allowed = True

        day_pnl = (self.equity - self._day_start_equity) / self._day_start_equity
        if day_pnl < -0.03:
            self._entry_allowed = False

        # --- 2. Exits: always run, even when circuit breaker active ---
        if self.position.is_long:
            if close >= bb_mid or rsi > 70:
                self.position.close()
        elif self.position.is_short:
            if close <= bb_mid or rsi < 30:
                self.position.close()

        # --- 3. Entries: only when flat and circuit breaker allows ---
        if not self._entry_allowed:
            return
        if self.position:
            return

        sl_distance = atr * self.sl_atr_mult
        size_fraction = (self.risk_pct * close * (1/30)) / sl_distance
        size_fraction = max(0.001, min(0.10, size_fraction))

        if close < bb_lower and rsi < 30 and close > trend:
            tp_distance = sl_distance * (self.tp_atr_mult / self.sl_atr_mult)
            self.buy(size=size_fraction, sl=close - sl_distance, tp=close + tp_distance)
        elif close > bb_upper and rsi > 70 and close < trend:
            tp_distance = sl_distance * (self.tp_atr_mult / self.sl_atr_mult)
            self.sell(size=size_fraction, sl=close + sl_distance, tp=close - tp_distance)
        if __name__ == "__main__": pass
