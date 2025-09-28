# AI Trading Starter (Quote Prediction + Risk Management)

This is a minimal, *single-program* skeleton you can run locally or in Colab.
It demonstrates:
- Loading (or simulating) order book snapshots
- Feature engineering (mid, spread, imbalance, rolling stats)
- XGBoost classifier for short-horizon mid-price direction
- Simple backtest with trading costs and a risk module (vol targeting, loss limits)

## Quick start (local Python 3.10+)
```bash
pip install -r requirements.txt
python -m src.main --data ./data/sample_orderbook.csv --h 5 --cost_bp 1.5
```

## Notes
- Replace `data/sample_orderbook.csv` with your own order book snapshots (timestamp, bid1, ask1, bidSize1..askSize10).
- Start here, then iterate: add LSTM/Transformer, execution model (limit fill prob), walk-forward retraining.