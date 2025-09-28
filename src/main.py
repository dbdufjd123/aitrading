import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .data_loader import load_orderbook_csv
from .features import build_features
from .model import QuoteDirectionModel
from .backtest import simple_backtest
from .utils import realized_vol

def make_labels(df: pd.DataFrame, horizon: int = 5):
    df = df.copy()
    df["mid_fwd"] = df["mid"].shift(-horizon)
    ret = (df["mid_fwd"] - df["mid"]) / df["mid"]

    vol = df["mid"].pct_change().rolling(300).std().fillna(method="bfill")
    th = (vol.mean() * 0.2)

    # 0=하락, 1=중립, 2=상승  (XGBoost는 0부터 시작하는 정수 라벨 기대)
    y = np.ones(len(df), dtype=int)
    y[ret > +th] = 2
    y[ret < -th] = 0

    df["y"] = y
    df = df.dropna().reset_index(drop=True)
    return df

def train_eval(df: pd.DataFrame, feature_cols, n_splits=5):
    X = df[feature_cols].values
    y = df["y"].values
    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof = np.zeros_like(y)
    last_model = None
    for tr, te in tscv.split(X):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        model = QuoteDirectionModel().fit(Xtr, ytr)
        oof[te] = model.predict(Xte)
        last_model = model
    return oof, last_model

def main(args):
    raw = load_orderbook_csv(args.data)
    feat, feature_cols = build_features(raw, levels=10)
    labeled = make_labels(feat, horizon=args.h)

    preds, model = train_eval(labeled, feature_cols, n_splits=5)
    labeled["pred"] = preds

    summary, trades, equity = simple_backtest(labeled, labeled["pred"], cost_bp=args.cost_bp)
    print("==== Backtest Summary ====")
    final_model = QuoteDirectionModel().fit(labeled[feature_cols].values, labeled["y"].values)
    final_model.save(dirpath="models", name="xgb_quote")

    for k,v in summary.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    # Save outputs
    trades.to_csv(args.out_trades, index=False)
    equity.to_csv(args.out_equity, header=["equity"], index=False)
    print(f"\nSaved trades to: {args.out_trades}")
    print(f"Saved equity curve to: {args.out_equity}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/sample_orderbook.csv")
    parser.add_argument("--h", type=int, default=5, help="prediction horizon (steps)")
    parser.add_argument("--cost_bp", type=float, default=1.5, help="per-trade cost in basis points")
    parser.add_argument("--out_trades", type=str, default="./trades.csv")
    parser.add_argument("--out_equity", type=str, default="./equity.csv")
    args = parser.parse_args()
    main(args)