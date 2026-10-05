"""Async SQLite connection management and schema DDL for CoilBrain.

Responsibilities
----------------
* Open and configure the SQLite database (WAL mode, foreign keys).
* Create all three tables on first run (idempotent DDL).
* Expose a single ``get_db()`` context manager used by repository functions.

Nothing in this module touches business logic; it is purely infrastructure.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiosqlite

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS signals (
    signal_id            TEXT  PRIMARY KEY,
    symbol               TEXT  NOT NULL,
    exchange             TEXT  NOT NULL,
    trigger              TEXT  NOT NULL,
    grade                TEXT  NOT NULL,
    score                REAL  NOT NULL,
    signal               TEXT  NOT NULL,
    oi_status            TEXT  NOT NULL,
    t0_price             REAL  NOT NULL CHECK (t0_price > 0),
    t0_timestamp         REAL  NOT NULL,
    change_24h_pct       REAL,
    volume_24h_usd       REAL  CHECK (volume_24h_usd IS NULL OR volume_24h_usd >= 0),
    breakout_pct         REAL,
    body_volume_spike    REAL  CHECK (body_volume_spike IS NULL OR body_volume_spike >= 0),
    volume_spike         REAL  CHECK (volume_spike IS NULL OR volume_spike >= 0),
    flat_period_days     REAL  CHECK (flat_period_days IS NULL OR flat_period_days >= 0),
    received_timestamp   REAL  NOT NULL,
    transport_latency_ms REAL  NOT NULL,
    obs_status           TEXT  NOT NULL DEFAULT 'OPEN'
                              CHECK (obs_status IN ('OPEN','OBSERVING','COMPLETE','ERROR')),
    obs_started_at       REAL,
    obs_completed_at     REAL,
    binance_symbol       TEXT,
    mfe_pct              REAL,
    mae_pct              REAL,
    max_price            REAL,
    min_price            REAL,
    time_to_max_price_s  REAL,
    secs_to_1pct         REAL,
    secs_to_5pct         REAL,
    secs_to_10pct        REAL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_signals_symbol      ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_obs_status  ON signals(obs_status);
CREATE INDEX IF NOT EXISTS idx_signals_t0          ON signals(t0_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_grade       ON signals(grade);
CREATE INDEX IF NOT EXISTS idx_signals_score       ON signals(score DESC);

CREATE TABLE IF NOT EXISTS price_observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id       TEXT    NOT NULL REFERENCES signals(signal_id),
    obs_timestamp   REAL    NOT NULL,
    elapsed_seconds REAL    NOT NULL,
    price           REAL    NOT NULL CHECK (price > 0),
    pct_from_t0     REAL    NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_obs_signal_time
    ON price_observations(signal_id, obs_timestamp);
CREATE INDEX IF NOT EXISTS idx_obs_latest
    ON price_observations(signal_id, id DESC);

CREATE TABLE IF NOT EXISTS observation_errors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id   TEXT    NOT NULL REFERENCES signals(signal_id),
    error_time  REAL    NOT NULL,
    error_msg   TEXT    NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_err_signal ON observation_errors(signal_id);

-- ── Microstructure Recorder tables ─────────────────────────────────────────

-- One snapshot row per (symbol × poll tick).  All active signals for that
-- symbol share this row via microstructure_signal_snapshots.
CREATE TABLE IF NOT EXISTS microstructure_snapshots (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    binance_symbol   TEXT    NOT NULL,
    snapshot_ts      REAL    NOT NULL,
    -- Open Interest
    oi_usdt          REAL,
    oi_delta_pct     REAL,
    -- Funding / mark price
    funding_rate     REAL,
    mark_price       REAL,
    next_funding_ts  REAL,
    -- Order book
    best_bid         REAL,
    best_ask         REAL,
    spread_abs       REAL,
    spread_pct       REAL,
    bid_depth_usdt   REAL,
    ask_depth_usdt   REAL,
    ob_imbalance     REAL,
    -- Aggr trades (NULL when ms_enable_trades=false)
    trade_count      INTEGER,
    traded_vol_usdt  REAL,
    buy_vol_usdt     REAL,
    sell_vol_usdt    REAL,
    trade_imbalance  REAL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_ms_symbol_ts
    ON microstructure_snapshots(binance_symbol, snapshot_ts DESC);
CREATE INDEX IF NOT EXISTS idx_ms_ts
    ON microstructure_snapshots(snapshot_ts DESC);

-- Junction: one row per (signal × snapshot).
CREATE TABLE IF NOT EXISTS microstructure_signal_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id    TEXT    NOT NULL REFERENCES signals(signal_id),
    snapshot_id  INTEGER NOT NULL REFERENCES microstructure_snapshots(id),
    elapsed_s    REAL    NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_mss_signal
    ON microstructure_signal_snapshots(signal_id, snapshot_id);
CREATE INDEX IF NOT EXISTS idx_mss_snapshot
    ON microstructure_signal_snapshots(snapshot_id);

-- Recorder-level errors (isolated from observatory observation_errors).
CREATE TABLE IF NOT EXISTS microstructure_errors (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    binance_symbol TEXT    NOT NULL,
    error_ts       REAL    NOT NULL,
    data_type      TEXT    NOT NULL,
    error_msg      TEXT    NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_mse_symbol
    ON microstructure_errors(binance_symbol, error_ts DESC);

-- ── Research Engine tables ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS research_signal_features (
    signal_id              TEXT PRIMARY KEY REFERENCES signals(signal_id),
    symbol                 TEXT NOT NULL,
    t0_timestamp           REAL NOT NULL,
    t0_price               REAL NOT NULL,
    score                  REAL NOT NULL,
    grade                  TEXT NOT NULL,
    
    -- Overall signal performance up to 4h
    mfe_pct                REAL,
    mae_pct                REAL,
    time_to_mfe_s          REAL,
    secs_to_1pct           REAL,
    secs_to_5pct           REAL,
    secs_to_10pct          REAL,
    
    -- T0 Microstructure baseline (missing = NULL)
    t0_snapshot_ts         REAL,
    t0_oi_usdt             REAL,
    t0_oi_delta_pct        REAL,
    t0_funding_rate        REAL,
    t0_spread_abs          REAL,
    t0_spread_pct          REAL,
    t0_ob_imbalance        REAL,
    t0_trade_count         INTEGER,
    t0_traded_vol_usdt     REAL,
    t0_buy_vol_usdt        REAL,
    t0_sell_vol_usdt       REAL,
    t0_trade_imbalance     REAL
) STRICT;

CREATE TABLE IF NOT EXISTS research_horizon_outcomes (
    signal_id              TEXT NOT NULL REFERENCES research_signal_features(signal_id),
    horizon_s              INTEGER NOT NULL,
    
    -- Alignment metadata
    target_elapsed_s       REAL NOT NULL,
    actual_elapsed_s       REAL,
    alignment_error_s      REAL,

    -- Price Outcome
    observation_ts         REAL,
    price                  REAL,
    return_pct             REAL,
    
    -- Microstructure at Horizon
    snapshot_ts            REAL,
    oi_usdt                REAL,
    oi_delta_pct           REAL,
    funding_rate           REAL,
    spread_abs             REAL,
    spread_pct             REAL,
    ob_imbalance           REAL,
    trade_count            INTEGER,
    traded_vol_usdt        REAL,
    buy_vol_usdt           REAL,
    sell_vol_usdt          REAL,
    trade_imbalance        REAL,
    
    -- Deltas from T0
    oi_delta_from_t0_pct         REAL,
    spread_delta_from_t0_pct     REAL,
    ob_imbalance_delta_from_t0   REAL,
    buy_volume_delta             REAL,
    sell_volume_delta            REAL,
    trade_imbalance_delta        REAL,

    PRIMARY KEY (signal_id, horizon_s)
) STRICT;

-- ── Decision Engine tables ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS demo_positions (
    position_id         TEXT PRIMARY KEY,
    symbol              TEXT NOT NULL,
    bybit_symbol        TEXT NOT NULL,
    signal_id           TEXT NOT NULL,
    status              TEXT NOT NULL,
    entry_price         REAL,
    stop_loss           REAL,
    take_profit         REAL,
    position_size_usd   REAL,
    quantity            REAL,
    risk_usd            REAL,
    rr_ratio            REAL,
    leverage            REAL,
    bybit_order_id      TEXT,
    bybit_sl_order_id   TEXT,
    bybit_tp_order_id   TEXT,
    actual_entry_price  REAL,
    actual_exit_price   REAL,
    pnl_usd             REAL,
    mfe_pct             REAL,
    mae_pct             REAL,
    opened_at           REAL,
    closed_at           REAL,
    exit_reason         TEXT,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_pos_symbol_status
    ON demo_positions(symbol, status);
CREATE INDEX IF NOT EXISTS idx_pos_signal_id
    ON demo_positions(signal_id);

CREATE TABLE IF NOT EXISTS decision_journal (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id                   TEXT NOT NULL,
    symbol                      TEXT NOT NULL,
    signal_timestamp            REAL NOT NULL,
    decision                    TEXT NOT NULL,
    decision_score              REAL NOT NULL,
    reasons_json                TEXT NOT NULL,
    decision_start_ts           REAL,
    market_data_start_ts        REAL,
    market_data_complete_ts     REAL,
    decision_ts                 REAL,
    signal_to_decision_ms       REAL,
    data_fetch_latency_ms       REAL,
    cs_score                    REAL,
    cs_grade                    TEXT,
    body_spike                  REAL,
    volume_spike                REAL,
    volume_24h_usd              REAL,
    change_24h_pct              REAL,
    breakout_pct                REAL,
    flat_period_days            REAL,
    intensity_regime            TEXT,
    oi_state                    TEXT,
    flow_state                  TEXT,
    ob_state                    TEXT,
    liquidity_state             TEXT,
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
    atr_value                   REAL,
    atr_source                  TEXT,
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
    bybit_order_id                  TEXT,
    bybit_sl_order_id               TEXT,
    bybit_tp_order_id               TEXT,
    actual_entry_price              REAL,
    slippage_pct                    REAL,
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
    signal_to_decision_ms           REAL,
    signal_to_order_ms              REAL,
    signal_to_fill_ms               REAL,
    signal_to_protection_ms         REAL,
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

-- ── Demo Session Guard ───────────────────────────────────────────────────
-- Singleton row (id=1) counts successfully submitted Demo market orders.
-- A process restart does NOT reset this counter — that is intentional.
-- MAX_DEMO_TRADES_PER_SESSION is enforced in demo_guard.py.
CREATE TABLE IF NOT EXISTS demo_session_guard (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    trade_count  INTEGER NOT NULL DEFAULT 0
);
-- Ensure the singleton row always exists
INSERT OR IGNORE INTO demo_session_guard (id, trade_count) VALUES (1, 0);
"""


# ---------------------------------------------------------------------------
# Connection pool (simple module-level singleton for this service size)
# ---------------------------------------------------------------------------

_db_path: str = ""


def configure(db_path: str) -> None:
    """Called once at startup with the path from settings."""
    global _db_path
    _db_path = db_path


@asynccontextmanager
async def get_db() -> AsyncIterator[aiosqlite.Connection]:
    """Yield an open, WAL-configured aiosqlite connection.

    Callers should use this as::

        async with get_db() as db:
            await db.execute(...)
    """
    if not _db_path:
        raise RuntimeError("database.configure() has not been called")
    async with aiosqlite.connect(_db_path) as db:
        db.row_factory = aiosqlite.Row
        # Ensure WAL and FK are on for every connection (SQLite PRAGMAs are
        # per-connection; WAL persists in the file but FK does not).
        await db.execute("PRAGMA journal_mode = WAL")
        await db.execute("PRAGMA foreign_keys = ON")
        yield db


async def _run_migrations(db_path: str) -> None:
    """Idempotent schema migrations for columns added after initial release.

    Uses PRAGMA table_info to check for column existence before ALTER TABLE,
    so this is safe to call on both fresh and existing databases.
    """
    async with get_db() as db:
        # signals.binance_symbol — added in Microstructure Recorder v1
        async with db.execute("PRAGMA table_info(signals)") as cur:
            col_names = {row[1] async for row in cur}
        if "binance_symbol" not in col_names:
            await db.execute("ALTER TABLE signals ADD COLUMN binance_symbol TEXT")
            await db.commit()
            logger.info("Migration applied: signals.binance_symbol")

        # demo_session_guard — added for Demo Session Guard v1
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='demo_session_guard'"
        ) as cur:
            table_exists = await cur.fetchone()
        if not table_exists:
            await db.execute(
                "CREATE TABLE demo_session_guard ("
                "  id          INTEGER PRIMARY KEY CHECK (id = 1),"
                "  trade_count INTEGER NOT NULL DEFAULT 0"
                ")"
            )
            await db.execute(
                "INSERT OR IGNORE INTO demo_session_guard (id, trade_count) VALUES (1, 0)"
            )
            await db.commit()
            logger.info("Migration applied: demo_session_guard table")
        else:
            # Table exists — ensure the singleton row is present (idempotent)
            await db.execute(
                "INSERT OR IGNORE INTO demo_session_guard (id, trade_count) VALUES (1, 0)"
            )
            await db.commit()



async def init_db(db_path: str) -> None:
    """Create tables and indexes (idempotent).  Call once at startup."""
    configure(db_path)
    async with get_db() as db:
        # Execute the schema as individual statements to avoid issues with
        # aiosqlite not supporting executescript inside a transaction context.
        for stmt in _SCHEMA_SQL.split(";"):
            stmt = stmt.strip()
            if stmt:
                await db.execute(stmt)
        await db.commit()
    await _run_migrations(db_path)
    logger.info("Database initialised at %s", db_path)
