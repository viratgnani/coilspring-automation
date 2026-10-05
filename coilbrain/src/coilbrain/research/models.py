"""Pydantic models for the Research Engine."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class SignalFeatureRecord(BaseModel):
    """Signal-level data and T0 baseline microstructure."""
    model_config = ConfigDict(from_attributes=True)

    signal_id: str
    symbol: str
    t0_timestamp: float
    t0_price: float
    score: float
    grade: str

    mfe_pct: Optional[float] = None
    mae_pct: Optional[float] = None
    time_to_mfe_s: Optional[float] = None
    secs_to_1pct: Optional[float] = None
    secs_to_5pct: Optional[float] = None
    secs_to_10pct: Optional[float] = None

    t0_snapshot_ts: Optional[float] = None
    t0_oi_usdt: Optional[float] = None
    t0_oi_delta_pct: Optional[float] = None
    t0_funding_rate: Optional[float] = None
    t0_spread_abs: Optional[float] = None
    t0_spread_pct: Optional[float] = None
    t0_ob_imbalance: Optional[float] = None
    t0_trade_count: Optional[int] = None
    t0_traded_vol_usdt: Optional[float] = None
    t0_buy_vol_usdt: Optional[float] = None
    t0_sell_vol_usdt: Optional[float] = None
    t0_trade_imbalance: Optional[float] = None


class HorizonOutcomeRecord(BaseModel):
    """Outcomes and microstructure data for a specific horizon."""
    model_config = ConfigDict(from_attributes=True)

    signal_id: str
    horizon_s: int

    target_elapsed_s: float
    actual_elapsed_s: Optional[float] = None
    alignment_error_s: Optional[float] = None

    observation_ts: Optional[float] = None
    price: Optional[float] = None
    return_pct: Optional[float] = None

    snapshot_ts: Optional[float] = None
    oi_usdt: Optional[float] = None
    oi_delta_pct: Optional[float] = None
    funding_rate: Optional[float] = None
    spread_abs: Optional[float] = None
    spread_pct: Optional[float] = None
    ob_imbalance: Optional[float] = None
    trade_count: Optional[int] = None
    traded_vol_usdt: Optional[float] = None
    buy_vol_usdt: Optional[float] = None
    sell_vol_usdt: Optional[float] = None
    trade_imbalance: Optional[float] = None

    oi_delta_from_t0_pct: Optional[float] = None
    spread_delta_from_t0_pct: Optional[float] = None
    ob_imbalance_delta_from_t0: Optional[float] = None
    buy_volume_delta: Optional[float] = None
    sell_volume_delta: Optional[float] = None
    trade_imbalance_delta: Optional[float] = None
