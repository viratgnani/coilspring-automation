"""Typed async repository functions for the Microstructure Recorder tables.

Pattern mirrors repository.py: pure functions that accept an open
aiosqlite.Connection.  No business logic lives here.

Tables written:
    microstructure_snapshots          -- one row per symbol per tick
    microstructure_signal_snapshots   -- junction: (signal_id, snapshot_id)
    microstructure_errors             -- recorder-level error log

Table read:
    signals                           -- to find active signals per symbol
"""

from __future__ import annotations

import time
from typing import Any

import aiosqlite

from coilbrain.microstructure_fetcher import MicrostructureSnapshot


# ---------------------------------------------------------------------------
# microstructure_snapshots
# ---------------------------------------------------------------------------

async def insert_snapshot(
    db: aiosqlite.Connection,
    snap: MicrostructureSnapshot,
) -> int:
    """Insert one snapshot row and return its auto-incremented id."""
    await db.execute(
        """
        INSERT INTO microstructure_snapshots (
            binance_symbol, snapshot_ts,
            oi_usdt, oi_delta_pct,
            funding_rate, mark_price, next_funding_ts,
            best_bid, best_ask, spread_abs, spread_pct,
            bid_depth_usdt, ask_depth_usdt, ob_imbalance,
            trade_count, traded_vol_usdt, buy_vol_usdt, sell_vol_usdt,
            trade_imbalance
        ) VALUES (
            :binance_symbol, :snapshot_ts,
            :oi_usdt, :oi_delta_pct,
            :funding_rate, :mark_price, :next_funding_ts,
            :best_bid, :best_ask, :spread_abs, :spread_pct,
            :bid_depth_usdt, :ask_depth_usdt, :ob_imbalance,
            :trade_count, :traded_vol_usdt, :buy_vol_usdt, :sell_vol_usdt,
            :trade_imbalance
        )
        """,
        {
            "binance_symbol":  snap.binance_symbol,
            "snapshot_ts":     snap.snapshot_ts,
            "oi_usdt":         snap.oi_usdt,
            "oi_delta_pct":    snap.oi_delta_pct,
            "funding_rate":    snap.funding_rate,
            "mark_price":      snap.mark_price,
            "next_funding_ts": snap.next_funding_ts,
            "best_bid":        snap.best_bid,
            "best_ask":        snap.best_ask,
            "spread_abs":      snap.spread_abs,
            "spread_pct":      snap.spread_pct,
            "bid_depth_usdt":  snap.bid_depth_usdt,
            "ask_depth_usdt":  snap.ask_depth_usdt,
            "ob_imbalance":    snap.ob_imbalance,
            "trade_count":     snap.trade_count,
            "traded_vol_usdt": snap.traded_vol_usdt,
            "buy_vol_usdt":    snap.buy_vol_usdt,
            "sell_vol_usdt":   snap.sell_vol_usdt,
            "trade_imbalance": snap.trade_imbalance,
        },
    )
    await db.commit()
    # lastrowid is set by aiosqlite after an INSERT
    cursor = await db.execute("SELECT last_insert_rowid()")
    row = await cursor.fetchone()
    return row[0]


async def fetch_latest_snapshot_ts(
    db: aiosqlite.Connection, binance_symbol: str
) -> float | None:
    """Return the snapshot_ts of the most recent snapshot for a symbol."""
    async with db.execute(
        """
        SELECT snapshot_ts FROM microstructure_snapshots
        WHERE binance_symbol = ?
        ORDER BY id DESC LIMIT 1
        """,
        (binance_symbol,),
    ) as cur:
        row = await cur.fetchone()
    return row[0] if row else None


async def fetch_snapshots_for_signal(
    db: aiosqlite.Connection, signal_id: str
) -> list[dict[str, Any]]:
    """Return all snapshots linked to a signal via the junction table."""
    async with db.execute(
        """
        SELECT ms.*, mss.elapsed_s
        FROM microstructure_snapshots ms
        JOIN microstructure_signal_snapshots mss ON ms.id = mss.snapshot_id
        WHERE mss.signal_id = ?
        ORDER BY ms.snapshot_ts
        """,
        (signal_id,),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# microstructure_signal_snapshots (junction)
# ---------------------------------------------------------------------------

async def insert_signal_snapshot(
    db: aiosqlite.Connection,
    signal_id: str,
    snapshot_id: int,
    elapsed_s: float,
) -> None:
    """Link a signal to a snapshot row."""
    await db.execute(
        """
        INSERT INTO microstructure_signal_snapshots
            (signal_id, snapshot_id, elapsed_s)
        VALUES (?, ?, ?)
        """,
        (signal_id, snapshot_id, elapsed_s),
    )
    await db.commit()


# ---------------------------------------------------------------------------
# signals queries used by the recorder
# ---------------------------------------------------------------------------

async def fetch_active_binance_symbols(db: aiosqlite.Connection) -> list[str]:
    """Return distinct binance_symbols for all OBSERVING signals.

    Ordered by the most recent signal t0_timestamp DESC so that the
    ms_max_symbols cap (applied by the picker loop) preferentially
    retains the newest symbols.  Only returns symbols that already
    have binance_symbol populated (set by the observatory picker).
    """
    async with db.execute(
        """
        SELECT binance_symbol
        FROM signals
        WHERE obs_status = 'OBSERVING' AND binance_symbol IS NOT NULL
        GROUP BY binance_symbol
        ORDER BY MAX(t0_timestamp) DESC
        """
    ) as cur:
        rows = await cur.fetchall()
    return [r[0] for r in rows]


async def fetch_observing_without_binance_symbol(
    db: aiosqlite.Connection,
) -> list[dict[str, Any]]:
    """Return OBSERVING signals where binance_symbol has not yet been set.

    Used for backward-compatibility recovery: the recorder's picker
    resolves these symbols itself and writes binance_symbol to the DB.
    """
    async with db.execute(
        "SELECT signal_id, symbol FROM signals WHERE obs_status = 'OBSERVING' AND binance_symbol IS NULL"
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def fetch_active_signals_for_symbol(
    db: aiosqlite.Connection, binance_symbol: str
) -> list[dict[str, Any]]:
    """Return (signal_id, t0_timestamp) for all OBSERVING signals for a symbol."""
    async with db.execute(
        """
        SELECT signal_id, t0_timestamp FROM signals
        WHERE obs_status = 'OBSERVING' AND binance_symbol = ?
        """,
        (binance_symbol,),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# microstructure_errors
# ---------------------------------------------------------------------------

async def insert_micro_error(
    db: aiosqlite.Connection,
    binance_symbol: str,
    data_type: str,
    error_msg: str,
) -> None:
    """Log a recorder-level error (separate from observatory errors)."""
    await db.execute(
        """
        INSERT INTO microstructure_errors
            (binance_symbol, error_ts, data_type, error_msg)
        VALUES (?, ?, ?, ?)
        """,
        (binance_symbol, time.time(), data_type, error_msg),
    )
    await db.commit()


async def purge_old_snapshots(
    db: aiosqlite.Connection,
    older_than_ts: float,
) -> int:
    """Delete microstructure snapshots older than ``older_than_ts``.

    Deletes junction rows (microstructure_signal_snapshots) first to
    satisfy FK constraints, then the snapshot rows themselves.

    Returns the number of snapshot rows deleted.
    """
    # Delete junction rows whose snapshot is being purged.
    await db.execute(
        """
        DELETE FROM microstructure_signal_snapshots
        WHERE snapshot_id IN (
            SELECT id FROM microstructure_snapshots WHERE snapshot_ts < ?
        )
        """,
        (older_than_ts,),
    )
    # Delete the stale snapshot rows.
    cursor = await db.execute(
        "DELETE FROM microstructure_snapshots WHERE snapshot_ts < ?",
        (older_than_ts,),
    )
    await db.commit()
    return cursor.rowcount
