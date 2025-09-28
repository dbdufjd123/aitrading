# app.py — Live Quant Trading Dashboard + Auto Trading
import os, time, collections, csv
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

from src.exchange import ExchangeWrapper
from src.features import build_features
from src.model import QuoteDirectionModel

load_dotenv()

# -----------------------
# 기본 설정
# -----------------------
SYMBOL   = os.getenv("SYMBOL", "BTC/USDT")
DRY_RUN  = os.getenv("DRY_RUN", "true").lower() == "true"
POLL_SEC = float(os.getenv("COOL_SECS", "1.0"))
LEVELS   = 10
BUF_LEN  = 900  # 15분 버퍼(1초 샘플 가정)

st.set_page_config(page_title="Live Quant Trader", layout="wide")
st.title(f"Live Quant Trading — {SYMBOL} {'(DRY-RUN)' if DRY_RUN else '(LIVE)'}")

# -----------------------
# 세션 상태 초기화
# -----------------------
ss = st.session_state
ss.buf = ss.get("buf", collections.deque(maxlen=BUF_LEN))
ss.model = ss.get("model", QuoteDirectionModel.load(dirpath="models", name="xgb_quote"))
ss.ex = ss.get("ex", ExchangeWrapper(dry_run=DRY_RUN))
ss.pos_units = ss.get("pos_units", 0.0)          # 현재 보유 단위(스텝 수량 단위)
ss.qty_per_unit = ss.get("qty_per_unit", 0.001)  # 1단위 수량(BTC)
ss.last_trade_ts = ss.get("last_trade_ts", 0.0)
ss.auto_on = ss.get("auto_on", False)

# -----------------------
# 컨트롤(상단)
# -----------------------
cA, cB, cC, cD = st.columns([1.2, 1, 1, 0.8])
with cA:
    SYMBOL = st.text_input("Symbol", SYMBOL)
with cB:
    POLL_SEC = st.number_input("Poll(sec)", 0.5, 5.0, POLL_SEC, 0.5)
with cC:
    ss.qty_per_unit = st.number_input("Qty per unit (BTC)", 0.0001, 0.01, ss.qty_per_unit, 0.0001, format="%.4f")
with cD:
    if st.button("⟳ Refresh now"):
        st.rerun()

# 전략/리스크 파라미터
st.divider()
s1, s2, s3, s4, s5 = st.columns(5)
with s1:
    entry_tau = st.number_input("Entry τ (strong up gap)", 0.00, 0.50, 0.08, 0.01)
with s2:
    exit_tau  = st.number_input("Exit τ (down/flat gap)", 0.00, 0.50, 0.04, 0.01)
with s3:
    max_units = st.number_input("Max units", 0.0, 10.0, 2.0, 0.5)
with s4:
    cooldown_sec = st.number_input("Cooldown (sec)", 0.0, 60.0, 3.0, 0.5)
with s5:
    ss.auto_on = st.toggle("Auto trading", value=ss.auto_on, help="예측 확률 임계치 만족 시 자동으로 매수/매도")

log_path = "./live_log.csv"

# -----------------------
# 데이터 폴링
# -----------------------
def row_from_orderbook(ob: dict) -> dict:
    bid = ob["bids"][0][0] if ob["bids"] else np.nan
    ask = ob["asks"][0][0] if ob["asks"] else np.nan
    row = {"bid1": bid, "ask1": ask, "timestamp": pd.Timestamp.utcnow()}
    for i in range(LEVELS):
        row[f"bidSize{i+1}"] = ob["bids"][i][1] if len(ob["bids"])>i else 0.0
        row[f"askSize{i+1}"] = ob["asks"][i][1] if len(ob["asks"])>i else 0.0
    return row

def poll_once():
    ob = ss.ex.fetch_orderbook(SYMBOL, limit=LEVELS)
    row = row_from_orderbook(ob)
    ss.buf.append(row)

try:
    poll_once()
except Exception as e:
    st.error(f"Data fetch error: {e}")

df = pd.DataFrame(list(ss.buf))
if not df.empty:
    df["mid"] = (df["bid1"] + df["ask1"]) / 2.0

# -----------------------
# 모델 예측
# -----------------------
p_down = p_flat = p_up = np.nan
pred = None
if len(df) >= 180:
    feats, feat_cols = build_features(df, levels=LEVELS)
    X = feats[feat_cols].values[-1:].copy()
    try:
        proba = ss.model.predict_proba(X)[0]     # [down, flat, up]
        p_down, p_flat, p_up = map(float, proba)
        pred = int(np.argmax(proba))             # 0,1,2
    except Exception as e:
        st.warning(f"Model error: {e}")

# -----------------------
# KPI
# -----------------------
k1, k2, k3, k4 = st.columns(4)
mid_last = float(df["mid"].iloc[-1]) if not df.empty else float("nan")
spread_bp = ((df["ask1"].iloc[-1] - df["bid1"].iloc[-1]) / mid_last * 1e4) if not df.empty else float("nan")
k1.metric("Mid Price", f"{mid_last:,.2f}")
k2.metric("Spread (bp)", f"{spread_bp:,.1f}")
k3.metric("Pred (0↓/1·/2↑)", "-" if pred is None else str(pred))
k4.metric("P(up) / P(flat) / P(down)", "-" if pred is None else f"{p_up:.2f} / {p_flat:.2f} / {p_down:.2f}")

# -----------------------
# 차트
# -----------------------
st.subheader("Live Mid Price (last 15 min)")
fig = go.Figure()
if not df.empty:
    tmp = df[["timestamp","mid"]].set_index("timestamp")
    o = tmp["mid"].resample("5S").first()
    h = tmp["mid"].resample("5S").max()
    l = tmp["mid"].resample("5S").min()
    c = tmp["mid"].resample("5S").last()
    can = pd.DataFrame({"open":o,"high":h,"low":l,"close":c}).dropna()
    if len(can) >= 2:
        fig.add_trace(go.Candlestick(x=can.index, open=can["open"], high=can["high"], low=can["low"], close=can["close"], name="mid"))
fig.update_layout(height=420, margin=dict(l=10,r=10,b=10,t=10))
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# 트레이드(수동/자동)
# -----------------------
st.subheader("Trade Controls")

c1, c2, c3 = st.columns(3)

def log_trade(ts, mid, pdn, pfl, pup, pred, pos_units, action):
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["ts","mid","p_down","p_flat","p_up","pred","pos_units","action"])
        w.writerow([ts, f"{mid:.2f}", f"{pdn:.3f}", f"{pfl:.3f}", f"{pup:.3f}", pred, pos_units, action])

with c1:
    if st.button("BUY 1 unit"):
        try:
            res = ss.ex.create_market_order(SYMBOL, "buy", ss.qty_per_unit)
            ss.pos_units += 1.0
            ss.last_trade_ts = time.time()
            st.success(f"BUY OK: {res}")
            log_trade(df["timestamp"].iloc[-1] if not df.empty else pd.Timestamp.utcnow(),
                      mid_last, p_down, p_flat, p_up, pred, ss.pos_units, "manual_buy")
        except Exception as e:
            st.error(f"BUY ERROR: {e}")

with c2:
    if st.button("SELL 1 unit"):
        try:
            res = ss.ex.create_market_order(SYMBOL, "sell", ss.qty_per_unit)
            ss.pos_units = max(0.0, ss.pos_units - 1.0)
            ss.last_trade_ts = time.time()
            st.success(f"SELL OK: {res}")
            log_trade(df["timestamp"].iloc[-1] if not df.empty else pd.Timestamp.utcnow(),
                      mid_last, p_down, p_flat, p_up, pred, ss.pos_units, "manual_sell")
        except Exception as e:
            st.error(f"SELL ERROR: {e}")

with c3:
    st.info(f"Units: {ss.pos_units:.1f} | Mode: {'DRY-RUN' if DRY_RUN else 'LIVE'}")

# ------ 자동매매 로직 ------
def can_trade_now():
    return (time.time() - ss.last_trade_ts) >= cooldown_sec

if ss.auto_on and pred is not None and not np.isnan(p_up):
    # 강한 매수 신호: p_up가 다른 클래스보다 entry_tau만큼 우위
    strong_up = (p_up - max(p_down, p_flat)) > entry_tau
    # 청산/매도 신호: down/flat가 up보다 exit_tau만큼 우위
    strong_down = (max(p_down, p_flat) - p_up) > exit_tau

    action = None
    if strong_up and ss.pos_units < max_units and can_trade_now():
        try:
            res = ss.ex.create_market_order(SYMBOL, "buy", ss.qty_per_unit)
            ss.pos_units += 1.0
            ss.last_trade_ts = time.time()
            action = "auto_buy"
        except Exception as e:
            st.error(f"AUTO BUY ERROR: {e}")

    elif strong_down and ss.pos_units > 0 and can_trade_now():
        try:
            res = ss.ex.create_market_order(SYMBOL, "sell", ss.qty_per_unit)
            ss.pos_units = max(0.0, ss.pos_units - 1.0)
            ss.last_trade_ts = time.time()
            action = "auto_sell"
        except Exception as e:
            st.error(f"AUTO SELL ERROR: {e}")

    if action:
        st.write(f"**Auto action:** {action} @ {mid_last:,.2f}  |  units={ss.pos_units:.1f}")
        log_trade(df["timestamp"].iloc[-1] if not df.empty else pd.Timestamp.utcnow(),
                  mid_last, p_down, p_flat, p_up, pred, ss.pos_units, action)

# -----------------------
# 자동 갱신
# -----------------------
time.sleep(POLL_SEC)
st.rerun()
