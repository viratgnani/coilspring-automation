"""Database access for the Research Engine."""

from __future__ import annotations

import aiosqlite
from typing import Any

from coilbrain.research.models import SignalFeatureRecord, HorizonOutcomeRecord


async def fetch_completed_signals_without_research(
    db: aiosqlite.Connection,
) -> list[dict[str, Any]]:
    """Return COMPLETE signals that don't yet have a row in research_signal_features."""
    async with db.execute(
        """
        SELECT s.* 
        FROM signals s
        LEFT JOIN research_signal_features rsf ON s.signal_id = rsf.signal_id
        WHERE s.obs_status = 'COMPLETE' 
          AND rsf.signal_id IS NULL
        ORDER BY s.t0_timestamp ASC
        """
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def insert_signal_features(
    db: aiosqlite.Connection, record: SignalFeatureRecord
) -> None:
    await db.execute(
        """
        INSERT INTO research_signal_features (
            signal_id, symbol, t0_timestamp, t0_price, score, grade,
            mfe_pct, mae_pct, time_to_mfe_s, secs_to_1pct, secs_to_5pct, secs_to_10pct,
            t0_snapshot_ts, t0_oi_usdt, t0_oi_delta_pct, t0_funding_rate,
            t0_spread_abs, t0_spread_pct, t0_ob_imbalance, t0_trade_count,
            t0_traded_vol_usdt, t0_buy_vol_usdt, t0_sell_vol_usdt, t0_trade_imbalance
        ) VALUES (
            :signal_id, :symbol, :t0_timestamp, :t0_price, :score, :grade,
            :mfe_pct, :mae_pct, :time_to_mfe_s, :secs_to_1pct, :secs_to_5pct, :secs_to_10pct,
            :t0_snapshot_ts, :t0_oi_usdt, :t0_oi_delta_pct, :t0_funding_rate,
            :t0_spread_abs, :t0_spread_pct, :t0_ob_imbalance, :t0_trade_count,
            :t0_traded_vol_usdt, :t0_buy_vol_usdt, :t0_sell_vol_usdt, :t0_trade_imbalance
        )
        """,
        record.model_dump(),
    )
    await db.commit()


async def insert_horizon_outcome(
    db: aiosqlite.Connection, record: HorizonOutcomeRecord
) -> None:
    await db.execute(
        """
        INSERT INTO research_horizon_outcomes (
            signal_id, horizon_s, target_elapsed_s, actual_elapsed_s, alignment_error_s,
            observation_ts, price, return_pct,
            snapshot_ts, oi_usdt, oi_delta_pct, funding_rate, spread_abs, spread_pct,
            ob_imbalance, trade_count, traded_vol_usdt, buy_vol_usdt, sell_vol_usdt, trade_imbalance,
            oi_delta_from_t0_pct, spread_delta_from_t0_pct, ob_imbalance_delta_from_t0,
            buy_volume_delta, sell_volume_delta, trade_imbalance_delta
        ) VALUES (
            :signal_id, :horizon_s, :target_elapsed_s, :actual_elapsed_s, :alignment_error_s,
            :observation_ts, :price, :return_pct,
            :snapshot_ts, :oi_usdt, :oi_delta_pct, :funding_rate, :spread_abs, :spread_pct,
            :ob_imbalance, :trade_count, :traded_vol_usdt, :buy_vol_usdt, :sell_vol_usdt, :trade_imbalance,
            :oi_delta_from_t0_pct, :spread_delta_from_t0_pct, :ob_imbalance_delta_from_t0,
            :buy_volume_delta, :sell_volume_delta, :trade_imbalance_delta
        )
        """,
        record.model_dump(),
    )
    await db.commit()


async def get_closest_snapshot(
    db: aiosqlite.Connection,
    binance_symbol: str,
    target_ts: float,
    lower_bound: float,
    upper_bound: float,
    prefer_pre_signal: bool = False,
) -> dict[str, Any] | None:
    """Find the closest snapshot to `target_ts` within the window [lower, upper]."""
    # If prefer_pre_signal is true, we sort by absolute difference ASC, then by snapshot_ts ASC
    # so earlier snapshots break ties.
    # If prefer_pre_signal is false, we sort by absolute difference ASC, then by snapshot_ts DESC
    # so later snapshots break ties.
    tiebreaker = "ASC" if prefer_pre_signal else "DESC"

    query = f"""
        SELECT *
        FROM microstructure_snapshots
        WHERE binance_symbol = ?
          AND snapshot_ts >= ?
          AND snapshot_ts <= ?
        ORDER BY ABS(snapshot_ts - ?) ASC, snapshot_ts {tiebreaker}
        LIMIT 1
    """
    async with db.execute(
        query, (binance_symbol, lower_bound, upper_bound, target_ts)
    ) as cur:
        row = await cur.fetchone()
    return dict(row) if row else None


async def get_closest_price_observation(
    db: aiosqlite.Connection,
    signal_id: str,
    target_elapsed_s: float,
    lower_bound_elapsed_s: float,
    upper_bound_elapsed_s: float,
) -> dict[str, Any] | None:
    """Find the price observation closest to `target_elapsed_s` within bounds."""
    # We always prefer the later observation on a tie.
    query = """
        SELECT *
        FROM price_observations
        WHERE signal_id = ?
          AND elapsed_seconds >= ?
          AND elapsed_seconds <= ?
        ORDER BY ABS(elapsed_seconds - ?) ASC, elapsed_seconds DESC
        LIMIT 1
    """
    async with db.execute(
        query, (signal_id, lower_bound_elapsed_s, upper_bound_elapsed_s, target_elapsed_s)
    ) as cur:
        row = await cur.fetchone()
    return dict(row) if row else None
