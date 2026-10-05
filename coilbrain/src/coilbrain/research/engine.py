"""Analytical Engine for generating research outcomes."""

import logging
from typing import Any

import aiosqlite

from coilbrain.research.models import SignalFeatureRecord, HorizonOutcomeRecord
from coilbrain.research import repository

logger = logging.getLogger(__name__)

HORIZONS_S = [30, 60, 120, 300, 600, 1800, 3600]
T0_ALIGN_LOWER_S = -5.0
T0_ALIGN_UPPER_S = 10.0


def _get_horizon_tolerance(horizon_s: int) -> float:
    """Return the ± tolerance (in seconds) for a given horizon.
    
    Uses max(60s, 10% of horizon) to ensure we always capture at least one
    Phase 2 poll (which occurs every 300s).
    """
    return max(60.0, horizon_s * 0.1)


def _safe_pct_change(old_val: float | None, new_val: float | None) -> float | None:
    if old_val is None or new_val is None:
        return None
    if old_val == 0:
        return None  # Cannot compute % change from 0
    return (new_val - old_val) / old_val * 100.0


def _safe_delta(old_val: float | None, new_val: float | None) -> float | None:
    if old_val is None or new_val is None:
        return None
    return new_val - old_val


async def process_signal(db: aiosqlite.Connection, signal_row: dict[str, Any]) -> None:
    """Generate and store research outcomes for a completed signal."""
    signal_id = signal_row["signal_id"]
    symbol = signal_row["symbol"]
    binance_symbol = signal_row["binance_symbol"]
    t0_timestamp = signal_row["t0_timestamp"]

    # 1. T0 Alignment
    # Find nearest snapshot within [-5s, +10s]
    t0_snap = None
    if binance_symbol:
        t0_snap = await repository.get_closest_snapshot(
            db,
            binance_symbol,
            target_ts=t0_timestamp,
            lower_bound=t0_timestamp + T0_ALIGN_LOWER_S,
            upper_bound=t0_timestamp + T0_ALIGN_UPPER_S,
            prefer_pre_signal=True,
        )

    # Build SignalFeatureRecord
    feature_rec = SignalFeatureRecord(
        signal_id=signal_id,
        symbol=symbol,
        t0_timestamp=t0_timestamp,
        t0_price=signal_row["t0_price"],
        score=signal_row["score"],
        grade=signal_row["grade"],
        mfe_pct=signal_row["mfe_pct"],
        mae_pct=signal_row["mae_pct"],
        time_to_mfe_s=signal_row["time_to_max_price_s"],
        secs_to_1pct=signal_row["secs_to_1pct"],
        secs_to_5pct=signal_row["secs_to_5pct"],
        secs_to_10pct=signal_row["secs_to_10pct"],
    )

    if t0_snap:
        feature_rec.t0_snapshot_ts = t0_snap["snapshot_ts"]
        feature_rec.t0_oi_usdt = t0_snap["oi_usdt"]
        feature_rec.t0_oi_delta_pct = t0_snap["oi_delta_pct"]
        feature_rec.t0_funding_rate = t0_snap["funding_rate"]
        feature_rec.t0_spread_abs = t0_snap["spread_abs"]
        feature_rec.t0_spread_pct = t0_snap["spread_pct"]
        feature_rec.t0_ob_imbalance = t0_snap["ob_imbalance"]
        feature_rec.t0_trade_count = t0_snap["trade_count"]
        feature_rec.t0_traded_vol_usdt = t0_snap["traded_vol_usdt"]
        feature_rec.t0_buy_vol_usdt = t0_snap["buy_vol_usdt"]
        feature_rec.t0_sell_vol_usdt = t0_snap["sell_vol_usdt"]
        feature_rec.t0_trade_imbalance = t0_snap["trade_imbalance"]

    await repository.insert_signal_features(db, feature_rec)

    # 2. Horizon Alignment
    for horizon in HORIZONS_S:
        tol = _get_horizon_tolerance(horizon)
        
        target_elapsed = float(horizon)
        target_ts = t0_timestamp + target_elapsed

        # Price outcome
        price_obs = await repository.get_closest_price_observation(
            db,
            signal_id,
            target_elapsed_s=target_elapsed,
            lower_bound_elapsed_s=target_elapsed - tol,
            upper_bound_elapsed_s=target_elapsed + tol,
        )

        # Microstructure outcome
        ms_snap = None
        if binance_symbol:
            ms_snap = await repository.get_closest_snapshot(
                db,
                binance_symbol,
                target_ts=target_ts,
                lower_bound=target_ts - tol,
                upper_bound=target_ts + tol,
                prefer_pre_signal=False,
            )

        # Alignment metadata
        # If we have both price and MS, we record the max alignment error.
        actual_elapsed = None
        align_err = None
        if price_obs:
            actual_elapsed = price_obs["elapsed_seconds"]
            align_err = price_obs["elapsed_seconds"] - target_elapsed

        if ms_snap and (actual_elapsed is None):
            # Fallback alignment if no price observation exists
            actual_elapsed = ms_snap["snapshot_ts"] - t0_timestamp
            align_err = actual_elapsed - target_elapsed

        outcome_rec = HorizonOutcomeRecord(
            signal_id=signal_id,
            horizon_s=horizon,
            target_elapsed_s=target_elapsed,
            actual_elapsed_s=actual_elapsed,
            alignment_error_s=align_err,
        )

        if price_obs:
            outcome_rec.observation_ts = price_obs["obs_timestamp"]
            outcome_rec.price = price_obs["price"]
            outcome_rec.return_pct = price_obs["pct_from_t0"]

        if ms_snap:
            outcome_rec.snapshot_ts = ms_snap["snapshot_ts"]
            outcome_rec.oi_usdt = ms_snap["oi_usdt"]
            outcome_rec.oi_delta_pct = ms_snap["oi_delta_pct"]
            outcome_rec.funding_rate = ms_snap["funding_rate"]
            outcome_rec.spread_abs = ms_snap["spread_abs"]
            outcome_rec.spread_pct = ms_snap["spread_pct"]
            outcome_rec.ob_imbalance = ms_snap["ob_imbalance"]
            outcome_rec.trade_count = ms_snap["trade_count"]
            outcome_rec.traded_vol_usdt = ms_snap["traded_vol_usdt"]
            outcome_rec.buy_vol_usdt = ms_snap["buy_vol_usdt"]
            outcome_rec.sell_vol_usdt = ms_snap["sell_vol_usdt"]
            outcome_rec.trade_imbalance = ms_snap["trade_imbalance"]

        # 3. Microstructure Deltas (only if we have T0 and Horizon MS)
        if t0_snap and ms_snap:
            outcome_rec.oi_delta_from_t0_pct = _safe_pct_change(
                t0_snap["oi_usdt"], ms_snap["oi_usdt"]
            )
            outcome_rec.spread_delta_from_t0_pct = _safe_pct_change(
                t0_snap["spread_pct"], ms_snap["spread_pct"]
            )
            outcome_rec.ob_imbalance_delta_from_t0 = _safe_delta(
                t0_snap["ob_imbalance"], ms_snap["ob_imbalance"]
            )
            outcome_rec.buy_volume_delta = _safe_delta(
                t0_snap["buy_vol_usdt"], ms_snap["buy_vol_usdt"]
            )
            outcome_rec.sell_volume_delta = _safe_delta(
                t0_snap["sell_vol_usdt"], ms_snap["sell_vol_usdt"]
            )
            outcome_rec.trade_imbalance_delta = _safe_delta(
                t0_snap["trade_imbalance"], ms_snap["trade_imbalance"]
            )

        await repository.insert_horizon_outcome(db, outcome_rec)

    logger.debug("Research Engine: Processed signal %s", signal_id)


async def run_generation(db: aiosqlite.Connection) -> int:
    """Find all complete un-analyzed signals and process them."""
    signals = await repository.fetch_completed_signals_without_research(db)
    for sig in signals:
        await process_signal(db, sig)
    return len(signals)
