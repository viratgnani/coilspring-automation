"""Typed async repository functions for all three CoilBrain tables.

All functions accept an open ``aiosqlite.Connection`` and are fully
composable -- callers (main.py, observatory.py) obtain the connection via
``database.get_db()`` and pass it here.  No business logic lives here.
"""

from __future__ import annotations

import time
from typing import Any

import aiosqlite

from coilbrain.models import SignalRecord


# ---------------------------------------------------------------------------
# signals table
# ---------------------------------------------------------------------------

async def insert_signal(db: aiosqlite.Connection, record: SignalRecord) -> None:
    """Persist a newly received signal.  obs_status defaults to ''OPEN''."""
    await db.execute(
        """
        INSERT INTO signals (
            signal_id, symbol, exchange, trigger, grade, score, signal,
            oi_status, t0_price, t0_timestamp,
            change_24h_pct, volume_24h_usd, breakout_pct,
            body_volume_spike, volume_spike, flat_period_days,
            received_timestamp, transport_latency_ms
        ) VALUES (
            :signal_id, :symbol, :exchange, :trigger, :grade, :score, :signal,
            :oi_status, :t0_price, :t0_timestamp,
            :change_24h_pct, :volume_24h_usd, :breakout_pct,
            :body_volume_spike, :volume_spike, :flat_period_days,
            :received_timestamp, :transport_latency_ms
        )
        """,
        {
            "signal_id": record.signal_id,
            "symbol": record.symbol,
            "exchange": record.exchange,
            "trigger": record.trigger,
            "grade": record.grade,
            "score": record.score,
            "signal": record.signal,
            "oi_status": record.oi_status,
            "t0_price": record.price,
            "t0_timestamp": record.signal_timestamp,
            "change_24h_pct": record.change_24h_pct,
            "volume_24h_usd": record.volume_24h_usd,
            "breakout_pct": record.breakout_pct,
            "body_volume_spike": record.body_volume_spike,
            "volume_spike": record.volume_spike,
            "flat_period_days": record.flat_period_days,
            "received_timestamp": record.received_timestamp,
            "transport_latency_ms": record.transport_latency_ms,
        },
    )
    await db.commit()


async def fetch_signal(
    db: aiosqlite.Connection, signal_id: str
) -> dict[str, Any] | None:
    """Return a signal row as a dict, or None if not found."""
    async with db.execute(
        "SELECT * FROM signals WHERE signal_id = ?", (signal_id,)
    ) as cur:
        row = await cur.fetchone()
    return dict(row) if row else None


async def fetch_open_signals(db: aiosqlite.Connection) -> list[dict[str, Any]]:
    """Return all signals with obs_status = ''OPEN''."""
    async with db.execute(
        "SELECT * FROM signals WHERE obs_status = 'OPEN'"
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def fetch_recovering_signals(db: aiosqlite.Connection) -> list[dict[str, Any]]:
    """Return signals that were OBSERVING when the process last died."""
    async with db.execute(
        "SELECT * FROM signals WHERE obs_status = 'OBSERVING'"
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def mark_observing(db: aiosqlite.Connection, signal_id: str) -> None:
    await db.execute(
        "UPDATE signals SET obs_status='OBSERVING', obs_started_at=? WHERE signal_id=?",
        (time.time(), signal_id),
    )
    await db.commit()


async def mark_complete(db: aiosqlite.Connection, signal_id: str) -> None:
    await db.execute(
        "UPDATE signals SET obs_status='COMPLETE', obs_completed_at=? WHERE signal_id=?",
        (time.time(), signal_id),
    )
    await db.commit()


async def mark_error(db: aiosqlite.Connection, signal_id: str) -> None:
    await db.execute(
        "UPDATE signals SET obs_status='ERROR', obs_completed_at=? WHERE signal_id=?",
        (time.time(), signal_id),
    )
    await db.commit()


async def set_binance_symbol(
    db: aiosqlite.Connection, signal_id: str, binance_symbol: str
) -> None:
    """Store the resolved Binance REST symbol on a signal row."""
    await db.execute(
        "UPDATE signals SET binance_symbol=? WHERE signal_id=?",
        (binance_symbol, signal_id),
    )
    await db.commit()


async def update_outcome(
    db: aiosqlite.Connection,
    signal_id: str,
    mfe_pct: float,
    mae_pct: float,
    max_price: float,
    min_price: float,
    time_to_max_price_s: float | None,
    secs_to_1pct: float | None,
    secs_to_5pct: float | None,
    secs_to_10pct: float | None,
) -> None:
    """Update the running outcome summary columns (called after each poll)."""
    await db.execute(
        """
        UPDATE signals SET
            mfe_pct = :mfe_pct,
            mae_pct = :mae_pct,
            max_price = :max_price,
            min_price = :min_price,
            time_to_max_price_s = :time_to_max_price_s,
            secs_to_1pct = :secs_to_1pct,
            secs_to_5pct = :secs_to_5pct,
            secs_to_10pct = :secs_to_10pct
        WHERE signal_id = :signal_id
        """,
        {
            "signal_id": signal_id,
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            "max_price": max_price,
            "min_price": min_price,
            "time_to_max_price_s": time_to_max_price_s,
            "secs_to_1pct": secs_to_1pct,
            "secs_to_5pct": secs_to_5pct,
            "secs_to_10pct": secs_to_10pct,
        },
    )
    await db.commit()


# ---------------------------------------------------------------------------
# price_observations table
# ---------------------------------------------------------------------------

async def insert_observation(
    db: aiosqlite.Connection,
    signal_id: str,
    obs_timestamp: float,
    elapsed_seconds: float,
    price: float,
    pct_from_t0: float,
) -> None:
    await db.execute(
        """
        INSERT INTO price_observations
            (signal_id, obs_timestamp, elapsed_seconds, price, pct_from_t0)
        VALUES (?, ?, ?, ?, ?)
        """,
        (signal_id, obs_timestamp, elapsed_seconds, price, pct_from_t0),
    )
    await db.commit()


async def fetch_latest_observation_timestamp(
    db: aiosqlite.Connection, signal_id: str
) -> float | None:
    """Return the obs_timestamp of the most recent observation, or None."""
    async with db.execute(
        """
        SELECT obs_timestamp FROM price_observations
        WHERE signal_id = ?
        ORDER BY id DESC LIMIT 1
        """,
        (signal_id,),
    ) as cur:
        row = await cur.fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# observation_errors table
# ---------------------------------------------------------------------------

async def insert_error(
    db: aiosqlite.Connection,
    signal_id: str,
    error_msg: str,
) -> None:
    await db.execute(
        "INSERT INTO observation_errors (signal_id, error_time, error_msg) VALUES (?, ?, ?)",
        (signal_id, time.time(), error_msg),
    )
    await db.commit()
