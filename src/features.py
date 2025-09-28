import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    out = df.copy()
    out["mid"] = (out["ask1"] + out["bid1"]) / 2.0
    # Spread relative to mid
    out["spread"] = (out["ask1"] - out["bid1"]) / out["mid"]
    bid_cols = [f"bidSize{i}" for i in range(1, levels+1)]
    ask_cols = [f"askSize{i}" for i in range(1, levels+1)]
    bsum = out[bid_cols].sum(axis=1)
    asum = out[ask_cols].sum(axis=1)
    out["imb"] = (bsum - asum) / (bsum + asum)
    # Rolling stats (short)
    out["rv_short"] = out["mid"].pct_change().rolling(60).std()
    out["imb_roll"] = out["imb"].rolling(30).mean()
    out = out.dropna().reset_index(drop=True)
    feature_cols = ["spread","imb","rv_short","imb_roll"] + bid_cols + ask_cols
    return out, feature_cols