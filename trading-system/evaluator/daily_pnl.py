import pandas as pd


def compute_daily_pnl_pct(equity_curve, initial_cash):
    """
    equity_curve: the '_equity_curve' DataFrame from backtesting.py's stats object.
                  Has a datetime index and an 'Equity' column.
    initial_cash: the starting cash used in Backtest(cash=...)

    Returns a pandas Series indexed by date, with each day's P&L as a
    percentage of the *previous day's* closing equity (this matches how
    prop firms compute daily loss: relative to prior day's balance, not
    the account's original starting balance).
    """
    eq = equity_curve["Equity"].copy()
    eq.index = pd.to_datetime(eq.index)

    daily_close = eq.resample("D").last().dropna()

    daily_pnl_pct = daily_close.pct_change() * 100
    daily_pnl_pct.iloc[0] = ((daily_close.iloc[0] - initial_cash) / initial_cash) * 100

    return daily_pnl_pct
