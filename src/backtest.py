import numpy as np
import pandas as pd
from .utils import annualize_sharpe, max_drawdown

def simple_backtest(df: pd.DataFrame, preds: pd.Series, cost_bp: float = 1.5, periods_per_year: int = 60*24*365):
    out = df.copy()
    out["pred"] = preds.reindex(out.index).fillna(0).replace({-1:-1, 0:0, 1:1})
    out["ret_mid"] = out["mid"].pct_change().fillna(0.0)

    # Gross pnl from holding previous bar's position
    out["strategy_gross"] = out["pred"].shift(1).fillna(0.0) * out["ret_mid"]

    # Costs: crossing spread/fees when changing position; apply per change + holding cost if desired
    bp = cost_bp * 1e-4
    trade = (out["pred"].diff().abs() > 0).astype(int)
    out["strategy_net"] = out["strategy_gross"] - bp*trade

    equity = (1.0 + out["strategy_net"]).cumprod()
    sharpe = annualize_sharpe(out["strategy_net"], periods_per_year=periods_per_year)
    mdd = max_drawdown(equity)

    summary = {
        "Gross mean (per step)": float(out["strategy_gross"].mean()),
        "Net mean (per step)": float(out["strategy_net"].mean()),
        "Sharpe (annualized)": float(sharpe),
        "Max Drawdown": float(mdd),
        "Final Equity": float(equity.iloc[-1])
    }
    return summary, out[["timestamp","mid","pred","ret_mid","strategy_net"]], equity