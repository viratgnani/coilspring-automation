"""CoilBrain V1 Position Manager.

Tracks active positions for each symbol and enforces the rule:
  ONE open position per symbol at any time.

DESIGN PRINCIPLES
-----------------
* State is persisted to SQLite — survives restarts.
* Synchronized with Bybit Demo on startup and after each order event.
* A new signal for a symbol with an OPEN position → IGNORE (EXISTING_POSITION).
* After a position closes, the next signal is evaluated normally.
* Symbols are NEVER permanently blacklisted.
* State machine: NO_POSITION → PENDING_ENTRY → OPEN → EXITING → CLOSED

State Transitions
-----------------
  NO_POSITION  ──(TRADE decision)──→  PENDING_ENTRY
  PENDING_ENTRY ──(order confirmed)──→ OPEN
  PENDING_ENTRY ──(order failed / timeout)──→ NO_POSITION
  OPEN          ──(begin exit)──→ EXITING
  EXITING       ──(exit confirmed)──→ CLOSED → NO_POSITION (archived)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position states
# ---------------------------------------------------------------------------

class PositionStatus(str, Enum):
    NO_POSITION = "NO_POSITION"
    PENDING_ENTRY = "PENDING_ENTRY"
    OPEN = "OPEN"
    EXITING = "EXITING"
    CLOSED = "CLOSED"


ACTIVE_STATES = {
    PositionStatus.PENDING_ENTRY,
    PositionStatus.OPEN,
    PositionStatus.EXITING,
}


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA = """
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
CREATE INDEX IF NOT EXISTS idx_pos_signal
    ON demo_positions(signal_id);
CREATE INDEX IF NOT EXISTS idx_pos_bybit_order
    ON demo_positions(bybit_order_id);
"""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """In-memory position record."""
    position_id: str
    symbol: str
    bybit_symbol: str
    signal_id: str
    status: PositionStatus

    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    position_size_usd: float | None = None
    quantity: float | None = None
    risk_usd: float | None = None
    rr_ratio: float | None = None
    leverage: float | None = None

    bybit_order_id: str | None = None
    bybit_sl_order_id: str | None = None
    bybit_tp_order_id: str | None = None

    actual_entry_price: float | None = None
    actual_exit_price: float | None = None
    pnl_usd: float | None = None
    mfe_pct: float | None = None
    mae_pct: float | None = None

    opened_at: float | None = None
    closed_at: float | None = None
    exit_reason: str | None = None

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Position Manager
# ---------------------------------------------------------------------------

class PositionManager:
    """Async position manager backed by aiosqlite.

    Usage:
        pm = PositionManager()
        await pm.init(db)

        state = await pm.get_state(db, "TAKE/USDT:USDT")
        # state.state == "NO_POSITION"

        pos = await pm.open_pending(db, signal_id=..., symbol=..., ...)
        await pm.confirm_open(db, pos.position_id, bybit_order_id=..., actual_entry=...)
        await pm.close(db, pos.position_id, exit_reason="TP_HIT", actual_exit=...)
    """

    async def ensure_schema(self, db: aiosqlite.Connection) -> None:
        """Create position tables if they don't exist."""
        for stmt in _SCHEMA.split(";"):
            stmt = stmt.strip()
            if stmt:
                await db.execute(stmt)
        await db.commit()

    async def get_active_position(
        self, db: aiosqlite.Connection, symbol: str
    ) -> Position | None:
        """Return the active position for a symbol, or None."""
        async with db.execute(
            """
            SELECT * FROM demo_positions
            WHERE symbol = ? AND status IN ('PENDING_ENTRY', 'OPEN', 'EXITING')
            ORDER BY created_at DESC LIMIT 1
            """,
            (symbol,),
        ) as cur:
            row = await cur.fetchone()
        return self._row_to_position(dict(row)) if row else None

    async def get_state_for_symbol(
        self, db: aiosqlite.Connection, symbol: str
    ) -> PositionStatus:
        """Return the current PositionStatus for a symbol (NO_POSITION if none active)."""
        pos = await self.get_active_position(db, symbol)
        if pos is None:
            return PositionStatus.NO_POSITION
        return pos.status

    async def open_pending(
        self,
        db: aiosqlite.Connection,
        *,
        position_id: str,
        signal_id: str,
        symbol: str,
        bybit_symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size_usd: float,
        quantity: float,
        risk_usd: float,
        rr_ratio: float,
        leverage: float,
    ) -> Position:
        """Record a new PENDING_ENTRY position before order submission."""
        now = time.time()
        pos = Position(
            position_id=position_id,
            symbol=symbol,
            bybit_symbol=bybit_symbol,
            signal_id=signal_id,
            status=PositionStatus.PENDING_ENTRY,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_usd=position_size_usd,
            quantity=quantity,
            risk_usd=risk_usd,
            rr_ratio=rr_ratio,
            leverage=leverage,
            created_at=now,
            updated_at=now,
        )
        await db.execute(
            """
            INSERT INTO demo_positions (
                position_id, symbol, bybit_symbol, signal_id, status,
                entry_price, stop_loss, take_profit, position_size_usd,
                quantity, risk_usd, rr_ratio, leverage,
                created_at, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?
            )
            """,
            (
                pos.position_id, pos.symbol, pos.bybit_symbol, pos.signal_id,
                pos.status.value,
                pos.entry_price, pos.stop_loss, pos.take_profit,
                pos.position_size_usd, pos.quantity, pos.risk_usd,
                pos.rr_ratio, pos.leverage,
                pos.created_at, pos.updated_at,
            ),
        )
        await db.commit()
        logger.info(
            "Position PENDING_ENTRY | id=%s symbol=%s entry=%.6f sl=%.6f tp=%.6f",
            position_id, symbol, entry_price, stop_loss, take_profit,
        )
        return pos

    async def confirm_open(
        self,
        db: aiosqlite.Connection,
        position_id: str,
        *,
        bybit_order_id: str,
        actual_entry_price: float,
        bybit_sl_order_id: str | None = None,
        bybit_tp_order_id: str | None = None,
    ) -> None:
        """Transition PENDING_ENTRY → OPEN after exchange confirms fill."""
        now = time.time()
        await db.execute(
            """
            UPDATE demo_positions
            SET status='OPEN',
                bybit_order_id=?,
                actual_entry_price=?,
                bybit_sl_order_id=?,
                bybit_tp_order_id=?,
                opened_at=?,
                updated_at=?
            WHERE position_id=?
            """,
            (
                bybit_order_id, actual_entry_price,
                bybit_sl_order_id, bybit_tp_order_id,
                now, now, position_id,
            ),
        )
        await db.commit()
        logger.info("Position OPEN | id=%s order=%s fill=%.6f", position_id, bybit_order_id, actual_entry_price)

    async def fail_pending(
        self, db: aiosqlite.Connection, position_id: str, reason: str
    ) -> None:
        """Transition PENDING_ENTRY → NO_POSITION (order failed)."""
        now = time.time()
        await db.execute(
            """
            UPDATE demo_positions
            SET status='CLOSED', exit_reason=?, closed_at=?, updated_at=?
            WHERE position_id=?
            """,
            (f"ENTRY_FAILED: {reason}", now, now, position_id),
        )
        await db.commit()
        logger.warning("Position entry FAILED | id=%s reason=%s", position_id, reason)

    async def begin_exit(
        self, db: aiosqlite.Connection, position_id: str
    ) -> None:
        """Transition OPEN → EXITING."""
        now = time.time()
        await db.execute(
            "UPDATE demo_positions SET status='EXITING', updated_at=? WHERE position_id=?",
            (now, position_id),
        )
        await db.commit()

    async def close(
        self,
        db: aiosqlite.Connection,
        position_id: str,
        *,
        exit_reason: str,
        actual_exit_price: float,
        pnl_usd: float | None = None,
        mfe_pct: float | None = None,
        mae_pct: float | None = None,
    ) -> None:
        """Transition EXITING/OPEN → CLOSED."""
        now = time.time()
        await db.execute(
            """
            UPDATE demo_positions
            SET status='CLOSED',
                actual_exit_price=?,
                exit_reason=?,
                pnl_usd=?,
                mfe_pct=?,
                mae_pct=?,
                closed_at=?,
                updated_at=?
            WHERE position_id=?
            """,
            (
                actual_exit_price, exit_reason,
                pnl_usd, mfe_pct, mae_pct,
                now, now, position_id,
            ),
        )
        await db.commit()
        logger.info(
            "Position CLOSED | id=%s reason=%s exit=%.6f pnl=%s",
            position_id, exit_reason, actual_exit_price,
            f"${pnl_usd:.2f}" if pnl_usd is not None else "N/A",
        )

    async def get_position_by_id(
        self, db: aiosqlite.Connection, position_id: str
    ) -> Position | None:
        async with db.execute(
            "SELECT * FROM demo_positions WHERE position_id=?", (position_id,)
        ) as cur:
            row = await cur.fetchone()
        return self._row_to_position(dict(row)) if row else None

    async def list_positions(
        self,
        db: aiosqlite.Connection,
        status: str | None = None,
    ) -> list[Position]:
        if status:
            async with db.execute(
                "SELECT * FROM demo_positions WHERE status=? ORDER BY created_at DESC",
                (status,),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with db.execute(
                "SELECT * FROM demo_positions ORDER BY created_at DESC"
            ) as cur:
                rows = await cur.fetchall()
        return [self._row_to_position(dict(r)) for r in rows]

    @staticmethod
    def _row_to_position(row: dict) -> Position:
        return Position(
            position_id=row["position_id"],
            symbol=row["symbol"],
            bybit_symbol=row["bybit_symbol"],
            signal_id=row["signal_id"],
            status=PositionStatus(row["status"]),
            entry_price=row.get("entry_price"),
            stop_loss=row.get("stop_loss"),
            take_profit=row.get("take_profit"),
            position_size_usd=row.get("position_size_usd"),
            quantity=row.get("quantity"),
            risk_usd=row.get("risk_usd"),
            rr_ratio=row.get("rr_ratio"),
            leverage=row.get("leverage"),
            bybit_order_id=row.get("bybit_order_id"),
            bybit_sl_order_id=row.get("bybit_sl_order_id"),
            bybit_tp_order_id=row.get("bybit_tp_order_id"),
            actual_entry_price=row.get("actual_entry_price"),
            actual_exit_price=row.get("actual_exit_price"),
            pnl_usd=row.get("pnl_usd"),
            mfe_pct=row.get("mfe_pct"),
            mae_pct=row.get("mae_pct"),
            opened_at=row.get("opened_at"),
            closed_at=row.get("closed_at"),
            exit_reason=row.get("exit_reason"),
            created_at=row.get("created_at", 0.0),
            updated_at=row.get("updated_at", 0.0),
        )
