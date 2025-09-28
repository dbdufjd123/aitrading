# src/risk.py
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd

def realized_vol(series: pd.Series, window: int = 120) -> float:
    r = series.pct_change()
    return float(r.rolling(window).std().iloc[-1])

@dataclass
class RiskConfig:
    target_vol: float = 0.0010     # 목표 일간이 아니라 "루프 주기 기준" 변동성 타겟 (대략값)
    max_units: float = 2.0         # 최대 포지션 단위 (여기서는 '스텝 수량' 기준)
    min_units: float = 0.0
    cooldown_sec: float = 3.0      # 체결 후 N초 재진입 금지
    day_loss_limit: float = -0.04  # 일중 손실 제한 ( -4% )
    mdd_limit: float = -0.15       # 최대드로다운 제한 ( -15% )
    taker_fee_bp: float = 2.0      # 시장가 수수료 bp 가정
    slippage_bp: float = 1.0       # 슬리피지 가정 bp

@dataclass
class RiskState:
    start_ts: float
    start_equity: float = 1.0
    equity: float = 1.0
    peak_equity: float = 1.0
    last_trade_ts: float = 0.0

class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.state = RiskState(start_ts=time.time())

    def can_trade_now(self) -> bool:
        return (time.time() - self.state.last_trade_ts) >= self.cfg.cooldown_sec

    def account_hard_stops(self) -> bool:
        # 일중 손실/ MDD 스톱
        ret_day = (self.state.equity - self.state.start_equity) / self.state.start_equity
        mdd = (self.state.equity / self.state.peak_equity) - 1.0
        return (ret_day <= self.cfg.day_loss_limit) or (mdd <= self.cfg.mdd_limit)

    def update_equity(self, ret_step: float):
        self.state.equity *= (1.0 + ret_step)
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)

    def position_size_from_vol(self, rv: float) -> float:
        # 간단 볼타겟: rv가 높을수록 사이즈 축소
        if np.isnan(rv) or rv <= 0:
            return self.cfg.min_units
        units = min(self.cfg.max_units, max(self.cfg.min_units, self.cfg.target_vol / rv))
        return float(units)

    def mark_trade(self):
        self.state.last_trade_ts = time.time()

    def est_round_trip_cost(self) -> float:
        # 한 번 포지션 변경 시 비용(bp): 수수료 + 슬리피지
        return (self.cfg.taker_fee_bp + self.cfg.slippage_bp) * 1e-4
