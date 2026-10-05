"""CoilBrain V1 Decision Journal (Repository).

Records every signal evaluation with full detail — whether TRADE or IGNORE.
This is the primary data source for later research and model calibration.

Schema
------
decision_journal
    One row per signal evaluation.
    All input features, classification, decision, and latency timestamps.

trade_journal
    One row per executed demo trade.
    Complete latency timeline from signal receipt to SL/TP confirmation.
    Entry, SL, TP, fill price, PnL, MFE, MAE, exit reason.

Both tables are append-only; nothing is ever deleted.

LATENCY TIMELINE (trade_journal)
---------------------------------
signal_timestamp              ← from CoilSpring (original signal fire time)
signal_received_timestamp     ← when CoilBrain received the POST
decision_start_timestamp      ← when decision_engine.decide() started
market_data_start_timestamp   ← when microstructure + ATR fetch started
market_data_complete_timestamp← when all market data was ready
decision_timestamp            ← when TRADE/IGNORE was determined
risk_start_timestamp          ← when risk engine calculate() started
risk_complete_timestamp       ← when SL/TP/sizing was computed
order_submission_timestamp    ← when place_market_order() was called
exchange_ack_timestamp        ← when Bybit returned orderId
fill_timestamp                ← when fill confirmed (order status query)
sl_submission_timestamp       ← when place_stop_loss() was called
sl_confirmed_timestamp        ← when SL was confirmed by exchange
tp_submission_timestamp       ← when place_take_profit() was called
tp_confirmed_timestamp        ← when TP was confirmed by exchange

DERIVED LATENCIES
-----------------
signal_to_decision_ms         = decision_timestamp - signal_timestamp  (ms)
signal_to_order_ms            = order_submission_timestamp - signal_timestamp (ms)
signal_to_fill_ms             = fill_timestamp - signal_timestamp (ms)
signal_to_protection_ms       = max(sl_confirmed_timestamp, tp_confirmed_timestamp) - signal_timestamp (ms)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decision_journal (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id                   TEXT NOT NULL,
    symbol                      TEXT NOT NULL,
    signal_timestamp            REAL NOT NULL,

    -- Decision output
    decision                    TEXT NOT NULL,
    decision_score              REAL NOT NULL,
    reasons_json                TEXT NOT NULL,

    -- Decision latency timestamps (Unix seconds)
    decision_start_ts           REAL,
    market_data_start_ts        REAL,
    market_data_complete_ts     REAL,
    decision_ts                 REAL,
    signal_to_decision_ms       REAL,
    data_fetch_latency_ms       REAL,

    -- Signal features
    cs_score                    REAL,
    cs_grade                    TEXT,
    body_spike                  REAL,
    volume_spike                REAL,
    volume_24h_usd              REAL,
    change_24h_pct              REAL,
    breakout_pct                REAL,
    flat_period_days            REAL,

    -- Feature classification
    intensity_regime            TEXT,
    oi_state                    TEXT,
    flow_state                  TEXT,
    ob_state                    TEXT,
    liquidity_state             TEXT,

    -- Microstructure snapshot
    snapshot_ts                 REAL,
    snapshot_age_ms             REAL,
    mark_price                  REAL,
    oi_usdt                     REAL,
    oi_delta_pct                REAL,
    funding_rate                REAL,
    best_bid                    REAL,
    best_ask                    REAL,
    spread_abs                  REAL,
    spread_pct                  REAL,
    bid_depth_usdt              REAL,
    ask_depth_usdt              REAL,
    ob_imbalance                REAL,
    trade_count                 INTEGER,
    traded_vol_usdt             REAL,
    buy_vol_usdt                REAL,
    sell_vol_usdt               REAL,
    trade_imbalance             REAL,

    -- ATR data used in risk sizing
    atr_value                   REAL,
    atr_source                  TEXT,

    -- Bybit
    bybit_symbol                TEXT,

    created_at                  REAL NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_dj_signal      ON decision_journal(signal_id);
CREATE INDEX IF NOT EXISTS idx_dj_symbol      ON decision_journal(symbol);
CREATE INDEX IF NOT EXISTS idx_dj_decision    ON decision_journal(decision);
CREATE INDEX IF NOT EXISTS idx_dj_created_at  ON decision_journal(created_at DESC);

CREATE TABLE IF NOT EXISTS trade_journal (
    id                              INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_journal_id             INTEGER REFERENCES decision_journal(id),
    signal_id                       TEXT NOT NULL,
    position_id                     TEXT NOT NULL,
    symbol                          TEXT NOT NULL,
    bybit_symbol                    TEXT NOT NULL,

    -- Risk plan
    planned_entry                   REAL,
    stop_loss                       REAL,
    take_profit                     REAL,
    position_size_usd               REAL,
    quantity                        REAL,
    risk_usd                        REAL,
    rr_ratio                        REAL,
    leverage                        REAL,
    atr_value                       REAL,
    atr_source                      TEXT,

    -- Bybit order confirmation
    bybit_order_id                  TEXT,
    bybit_sl_order_id               TEXT,
    bybit_tp_order_id               TEXT,
    actual_entry_price              REAL,
    slippage_pct                    REAL,

    -- Full latency timeline (all Unix seconds)
    signal_timestamp                REAL,
    signal_received_timestamp       REAL,
    decision_start_timestamp        REAL,
    market_data_start_timestamp     REAL,
    market_data_complete_timestamp  REAL,
    decision_timestamp              REAL,
    risk_start_timestamp            REAL,
    risk_complete_timestamp         REAL,
    order_submission_timestamp      REAL,
    exchange_ack_timestamp          REAL,
    fill_timestamp                  REAL,
    sl_submission_timestamp         REAL,
    sl_confirmed_timestamp          REAL,
    tp_submission_timestamp         REAL,
    tp_confirmed_timestamp          REAL,

    -- Derived latencies (ms)
    signal_to_decision_ms           REAL,
    signal_to_order_ms              REAL,
    signal_to_fill_ms               REAL,
    signal_to_protection_ms         REAL,

    -- Outcome (filled in when position closes)
    actual_exit_price               REAL,
    exit_reason                     TEXT,
    pnl_usd                         REAL,
    mfe_pct                         REAL,
    mae_pct                         REAL,
    duration_s                      REAL,

    created_at                      REAL NOT NULL,
    closed_at                       REAL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_tj_signal    ON trade_journal(signal_id);
CREATE INDEX IF NOT EXISTS idx_tj_position  ON trade_journal(position_id);
CREATE INDEX IF NOT EXISTS idx_tj_symbol    ON trade_journal(symbol);
"""


# ---------------------------------------------------------------------------
# Schema initialiser
# ---------------------------------------------------------------------------

async def ensure_schema(db: aiosqlite.Connection) -> None:
    for stmt in _SCHEMA.split(";"):
        stmt = stmt.strip()
        if stmt:
            await db.execute(stmt)
    await db.commit()


# ---------------------------------------------------------------------------
# Latency helpers
# ---------------------------------------------------------------------------

def compute_latencies(
    signal_timestamp: float | None,
    decision_timestamp: float | None,
    order_submission_timestamp: float | None,
    fill_timestamp: float | None,
    sl_confirmed_timestamp: float | None,
    tp_confirmed_timestamp: float | None,
) -> dict[str, float | None]:
    """Compute all derived latency fields from raw timestamps.

    Returns a dict with keys:
        signal_to_decision_ms
        signal_to_order_ms
        signal_to_fill_ms
        signal_to_protection_ms

    Any value is None if required timestamps are missing.
    """
    def _ms(t_start: float | None, t_end: float | None) -> float | None:
        if t_start is None or t_end is None:
            return None
        return (t_end - t_start) * 1000.0

    # Protection timestamp = when both SL and TP are confirmed
    protection_ts: float | None = None
    if sl_confirmed_timestamp is not None and tp_confirmed_timestamp is not None:
        protection_ts = max(sl_confirmed_timestamp, tp_confirmed_timestamp)
    elif sl_confirmed_timestamp is not None:
        protection_ts = sl_confirmed_timestamp
    elif tp_confirmed_timestamp is not None:
        protection_ts = tp_confirmed_timestamp

    return {
        "signal_to_decision_ms": _ms(signal_timestamp, decision_timestamp),
        "signal_to_order_ms": _ms(signal_timestamp, order_submission_timestamp),
        "signal_to_fill_ms": _ms(signal_timestamp, fill_timestamp),
        "signal_to_protection_ms": _ms(signal_timestamp, protection_ts),
    }


# ---------------------------------------------------------------------------
# Journal writers
# ---------------------------------------------------------------------------

async def record_decision(
    db: aiosqlite.Connection,
    *,
    signal_id: str,
    symbol: str,
    signal_timestamp: float,
    decision: str,
    decision_score: float,
    reasons: list[str],
    decision_start_ts: float | None = None,
    market_data_start_ts: float | None = None,
    market_data_complete_ts: float | None = None,
    decision_ts: float | None = None,
    signal_to_decision_ms: float | None = None,
    data_fetch_latency_ms: float | None = None,
    cs_score: float | None = None,
    cs_grade: str | None = None,
    body_spike: float | None = None,
    volume_spike: float | None = None,
    volume_24h_usd: float | None = None,
    change_24h_pct: float | None = None,
    breakout_pct: float | None = None,
    flat_period_days: float | None = None,
    intensity_regime: str | None = None,
    oi_state: str | None = None,
    flow_state: str | None = None,
    ob_state: str | None = None,
    liquidity_state: str | None = None,
    snapshot_ts: float | None = None,
    snapshot_age_ms: float | None = None,
    mark_price: float | None = None,
    oi_usdt: float | None = None,
    oi_delta_pct: float | None = None,
    funding_rate: float | None = None,
    best_bid: float | None = None,
    best_ask: float | None = None,
    spread_abs: float | None = None,
    spread_pct: float | None = None,
    bid_depth_usdt: float | None = None,
    ask_depth_usdt: float | None = None,
    ob_imbalance: float | None = None,
    trade_count: int | None = None,
    traded_vol_usdt: float | None = None,
    buy_vol_usdt: float | None = None,
    sell_vol_usdt: float | None = None,
    trade_imbalance: float | None = None,
    atr_value: float | None = None,
    atr_source: str | None = None,
    bybit_symbol: str | None = None,
) -> int:
    """Insert a decision record. Returns the new row id."""
    cur = await db.execute(
        """
        INSERT INTO decision_journal (
            signal_id, symbol, signal_timestamp,
            decision, decision_score, reasons_json,
            decision_start_ts, market_data_start_ts, market_data_complete_ts,
            decision_ts, signal_to_decision_ms, data_fetch_latency_ms,
            cs_score, cs_grade, body_spike, volume_spike, volume_24h_usd,
            change_24h_pct, breakout_pct, flat_period_days,
            intensity_regime, oi_state, flow_state, ob_state, liquidity_state,
            snapshot_ts, snapshot_age_ms, mark_price,
            oi_usdt, oi_delta_pct, funding_rate,
            best_bid, best_ask, spread_abs, spread_pct,
            bid_depth_usdt, ask_depth_usdt, ob_imbalance,
            trade_count, traded_vol_usdt, buy_vol_usdt, sell_vol_usdt,
            trade_imbalance, atr_value, atr_source,
            bybit_symbol, created_at
        ) VALUES (
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?
        )
        """,
        (
            signal_id, symbol, signal_timestamp,
            decision, decision_score, json.dumps(reasons),
            decision_start_ts, market_data_start_ts, market_data_complete_ts,
            decision_ts, signal_to_decision_ms, data_fetch_latency_ms,
            cs_score, cs_grade, body_spike, volume_spike, volume_24h_usd,
            change_24h_pct, breakout_pct, flat_period_days,
            intensity_regime, oi_state, flow_state, ob_state, liquidity_state,
            snapshot_ts, snapshot_age_ms, mark_price,
            oi_usdt, oi_delta_pct, funding_rate,
            best_bid, best_ask, spread_abs, spread_pct,
            bid_depth_usdt, ask_depth_usdt, ob_imbalance,
            trade_count, traded_vol_usdt, buy_vol_usdt, sell_vol_usdt,
            trade_imbalance, atr_value, atr_source,
            bybit_symbol, time.time(),
        ),
    )
    await db.commit()
    return cur.lastrowid


async def record_trade(
    db: aiosqlite.Connection,
    *,
    decision_journal_id: int | None,
    signal_id: str,
    position_id: str,
    symbol: str,
    bybit_symbol: str,
    planned_entry: float,
    stop_loss: float,
    take_profit: float,
    position_size_usd: float,
    quantity: float,
    risk_usd: float,
    rr_ratio: float,
    leverage: float,
    atr_value: float | None = None,
    atr_source: str | None = None,
    bybit_order_id: str | None = None,
    bybit_sl_order_id: str | None = None,
    bybit_tp_order_id: str | None = None,
    actual_entry_price: float | None = None,
    slippage_pct: float | None = None,
    # Full latency timeline
    signal_timestamp: float | None = None,
    signal_received_timestamp: float | None = None,
    decision_start_timestamp: float | None = None,
    market_data_start_timestamp: float | None = None,
    market_data_complete_timestamp: float | None = None,
    decision_timestamp: float | None = None,
    risk_start_timestamp: float | None = None,
    risk_complete_timestamp: float | None = None,
    order_submission_timestamp: float | None = None,
    exchange_ack_timestamp: float | None = None,
    fill_timestamp: float | None = None,
    sl_submission_timestamp: float | None = None,
    sl_confirmed_timestamp: float | None = None,
    tp_submission_timestamp: float | None = None,
    tp_confirmed_timestamp: float | None = None,
) -> int:
    """Insert an initial trade journal row. Returns the row id.

    Derived latencies are computed automatically from the timestamps provided.
    """
    latencies = compute_latencies(
        signal_timestamp=signal_timestamp,
        decision_timestamp=decision_timestamp,
        order_submission_timestamp=order_submission_timestamp,
        fill_timestamp=fill_timestamp,
        sl_confirmed_timestamp=sl_confirmed_timestamp,
        tp_confirmed_timestamp=tp_confirmed_timestamp,
    )

    cur = await db.execute(
        """
        INSERT INTO trade_journal (
            decision_journal_id, signal_id, position_id, symbol, bybit_symbol,
            planned_entry, stop_loss, take_profit, position_size_usd, quantity,
            risk_usd, rr_ratio, leverage, atr_value, atr_source,
            bybit_order_id, bybit_sl_order_id, bybit_tp_order_id,
            actual_entry_price, slippage_pct,
            signal_timestamp, signal_received_timestamp,
            decision_start_timestamp, market_data_start_timestamp,
            market_data_complete_timestamp, decision_timestamp,
            risk_start_timestamp, risk_complete_timestamp,
            order_submission_timestamp, exchange_ack_timestamp,
            fill_timestamp,
            sl_submission_timestamp, sl_confirmed_timestamp,
            tp_submission_timestamp, tp_confirmed_timestamp,
            signal_to_decision_ms, signal_to_order_ms,
            signal_to_fill_ms, signal_to_protection_ms,
            created_at
        ) VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?
        )
        """,
        (
            decision_journal_id, signal_id, position_id, symbol, bybit_symbol,
            planned_entry, stop_loss, take_profit, position_size_usd, quantity,
            risk_usd, rr_ratio, leverage, atr_value, atr_source,
            bybit_order_id, bybit_sl_order_id, bybit_tp_order_id,
            actual_entry_price, slippage_pct,
            signal_timestamp, signal_received_timestamp,
            decision_start_timestamp, market_data_start_timestamp,
            market_data_complete_timestamp, decision_timestamp,
            risk_start_timestamp, risk_complete_timestamp,
            order_submission_timestamp, exchange_ack_timestamp,
            fill_timestamp,
            sl_submission_timestamp, sl_confirmed_timestamp,
            tp_submission_timestamp, tp_confirmed_timestamp,
            latencies["signal_to_decision_ms"],
            latencies["signal_to_order_ms"],
            latencies["signal_to_fill_ms"],
            latencies["signal_to_protection_ms"],
            time.time(),
        ),
    )
    await db.commit()
    return cur.lastrowid


async def update_trade_latency(
    db: aiosqlite.Connection,
    trade_journal_id: int,
    *,
    fill_timestamp: float | None = None,
    sl_submission_timestamp: float | None = None,
    sl_confirmed_timestamp: float | None = None,
    tp_submission_timestamp: float | None = None,
    tp_confirmed_timestamp: float | None = None,
    exchange_ack_timestamp: float | None = None,
) -> None:
    """Update latency fields as they become available during execution."""
    await db.execute(
        """
        UPDATE trade_journal
        SET fill_timestamp=COALESCE(?, fill_timestamp),
            sl_submission_timestamp=COALESCE(?, sl_submission_timestamp),
            sl_confirmed_timestamp=COALESCE(?, sl_confirmed_timestamp),
            tp_submission_timestamp=COALESCE(?, tp_submission_timestamp),
            tp_confirmed_timestamp=COALESCE(?, tp_confirmed_timestamp),
            exchange_ack_timestamp=COALESCE(?, exchange_ack_timestamp)
        WHERE id=?
        """,
        (
            fill_timestamp, sl_submission_timestamp, sl_confirmed_timestamp,
            tp_submission_timestamp, tp_confirmed_timestamp,
            exchange_ack_timestamp, trade_journal_id,
        ),
    )
    await db.commit()


async def update_trade_outcome(
    db: aiosqlite.Connection,
    trade_journal_id: int,
    *,
    actual_exit_price: float,
    exit_reason: str,
    pnl_usd: float | None = None,
    mfe_pct: float | None = None,
    mae_pct: float | None = None,
    duration_s: float | None = None,
) -> None:
    """Update the exit data when a position closes."""
    now = time.time()
    await db.execute(
        """
        UPDATE trade_journal
        SET actual_exit_price=?, exit_reason=?, pnl_usd=?,
            mfe_pct=?, mae_pct=?, duration_s=?, closed_at=?
        WHERE id=?
        """,
        (
            actual_exit_price, exit_reason, pnl_usd,
            mfe_pct, mae_pct, duration_s, now,
            trade_journal_id,
        ),
    )
    await db.commit()


async def fetch_decisions(
    db: aiosqlite.Connection,
    symbol: str | None = None,
    decision: str | None = None,
    limit: int = 100,
) -> list[dict]:
    clauses = []
    params: list[Any] = []
    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol)
    if decision:
        clauses.append("decision = ?")
        params.append(decision)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    async with db.execute(
        f"SELECT * FROM decision_journal {where} ORDER BY created_at DESC LIMIT ?",
        params,
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]
