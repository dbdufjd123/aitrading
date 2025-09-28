import numpy as np
import pandas as pd

def realized_vol(series: pd.Series, window: int = 60) -> pd.Series:
    r = series.pct_change()
    return r.rolling(window).std()

def max_drawdown(equity_curve: pd.Series) -> float:
    cummax = equity_curve.cummax()
    dd = (equity_curve / cummax) - 1.0
    return dd.min()

def annualize_sharpe(returns: pd.Series, periods_per_year: int) -> float:
    mu = returns.mean()
    sd = returns.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return (mu / sd) * np.sqrt(periods_per_year)