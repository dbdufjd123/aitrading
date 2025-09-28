# src/live.py
import time, collections, csv, os
import pandas as pd, numpy as np
from .features import build_features
from .model import QuoteDirectionModel
from .exchange import ExchangeWrapper
from .risk import RiskManager, RiskConfig, realized_vol

def _row_from_orderbook(ob):
    bid = ob["bids"][0][0] if ob["bids"] else np.nan
    ask = ob["asks"][0][0] if ob["asks"] else np.nan
    row = {"bid1": bid, "ask1": ask}
    for i in range(10):
        row[f"bidSize{i+1}"] = ob["bids"][i][1] if len(ob["bids"])>i else 0.0
        row[f"askSize{i+1}"] = ob["asks"][i][1] if len(ob["asks"])>i else 0.0
    row["timestamp"] = pd.Timestamp.utcnow()
    return row

def live_run(symbol="BTC/USDT", dry_run=True, cool_secs=1.0,
             entry_tau=0.08, exit_tau=0.04, log_path="./live_log.csv"):
    """
    - entry_tau: 강한 신호만 진입 (p_up - max(p_down,p_flat) > entry_tau)
    - exit_tau : 반대 신호가 약해도 청산 (max(p_down,p_flat) - p_up > exit_tau) 등
    """
    ex = ExchangeWrapper(dry_run=dry_run)
    model = QuoteDirectionModel.load(dirpath="models", name="xgb_quote")

    buf = collections.deque(maxlen=600)  # 최근 10분 버퍼(1초 간격 가정)
    pos_units = 0.0                      # 현재 보유 "단위" (정책 단위)
    qty_per_unit = 0.001                 # 1단위 = 0.001 BTC (원하면 조정)
    rm = RiskManager(RiskConfig())

    # 로깅 헤더
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts","mid","p_down","p_flat","p_up","pred","pos_units","action","price"])

    print(f"[live] {symbol} dry_run={dry_run}")
    while True:
        try:
            ob = ex.fetch_orderbook(symbol, limit=10)
            row = _row_from_orderbook(ob)
            buf.append(row)

            if len(buf) < 180:
                time.sleep(cool_secs); continue

            df = pd.DataFrame(list(buf))
            feats, cols = build_features(df, levels=10)
            X = feats[cols].values[-1:].copy()

            # --- 예측 확률/신호 ---
            proba = model.predict_proba(X)[0]   # order: [down, flat, up]
            p_down, p_flat, p_up = float(proba[0]), float(proba[1]), float(proba[2])
            pred = int(np.argmax(proba))        # 0=down,1=flat,2=up

            # --- 리스크/사이징 ---
            mid = float((row["bid1"] + row["ask1"]) / 2.0)
            rv = realized_vol(pd.Series(feats["mid"]), window=120)
            max_units = rm.position_size_from_vol(rv)

            # 하드스톱 체크
            if rm.account_hard_stops():
                if pos_units > 0:
                    # 강제 청산
                    ex.create_market_order(symbol, "sell", qty_per_unit*pos_units)
                    pos_units = 0.0
                    rm.mark_trade()
                print("[STOP] day/MDD limit hit. Pausing trades.")
                time.sleep(max(5.0, rm.cfg.cooldown_sec)); continue

            action = "hold"

            # --- 진입/청산 로직 (임계치 + 쿨다운) ---
            can_trade = rm.can_trade_now()

            # 강한 매수 시그널
            if can_trade and (p_up - max(p_down, p_flat) > entry_tau) and (pos_units < max_units):
                ex.create_market_order(symbol, "buy", qty_per_unit)
                pos_units += 1.0
                rm.mark_trade()
                action = "buy"

            # 강한 매도/청산 시그널
            elif can_trade and pos_units > 0 and (max(p_down, p_flat) - p_up > exit_tau):
                ex.create_market_order(symbol, "sell", qty_per_unit)
                pos_units = max(0.0, pos_units - 1.0)
                rm.mark_trade()
                action = "sell"

            # --- 비용 가정에 따른 “미세 진동” 억제 ---
            # 신호가 약할 땐 거래 안 함 (entry_tau/exit_tau가 그 역할)

            # --- 로깅 ---
            with open(log_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([row["timestamp"], f"{mid:.2f}", f"{p_down:.3f}", f"{p_flat:.3f}", f"{p_up:.3f}",
                            pred, pos_units, action, f"{mid:.2f}"])

            print(f"{row['timestamp']} mid={mid:.2f} pred={pred} p=[{p_down:.2f},{p_flat:.2f},{p_up:.2f}] "
                  f"pos={pos_units:.1f} act={action}")

            time.sleep(cool_secs)

        except KeyboardInterrupt:
            print("stopped by user"); break
        except Exception as e:
            print("error:", e); time.sleep(2.0)
