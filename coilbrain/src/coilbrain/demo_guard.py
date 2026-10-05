"""Demo Session Guard module.

Provides a hard limit on the number of Demo market orders that can be placed
in a single session (or across restarts, since it persists).
"""

from __future__ import annotations

import logging
import sqlite3

import os
import aiosqlite

from coilbrain.decision_engine import Decision

logger = logging.getLogger(__name__)

# Configurable limit as per requirements (default 50 trades)
MAX_DEMO_TRADES_PER_SESSION = int(os.getenv("MAX_DEMO_TRADES", "50"))

async def check_demo_limit(db: aiosqlite.Connection) -> bool:
    """Check if the demo trade limit has been reached.
    
    Returns:
        True if limit is reached (guard should block).
        False if limit is not reached (guard allows).
    """
    async with db.execute("SELECT trade_count FROM demo_session_guard WHERE id = 1") as cursor:
        row = await cursor.fetchone()
        if not row:
            # Should not happen due to migration/schema, but fail-closed if it does
            logger.error("demo_session_guard singleton row missing!")
            return True
        count = row[0]
        if count >= MAX_DEMO_TRADES_PER_SESSION:
            return True
        return False

async def increment_demo_trade_count(db: aiosqlite.Connection) -> None:
    """Increment the demo trade count after a successful market order submission."""
    await db.execute(
        "UPDATE demo_session_guard SET trade_count = trade_count + 1 WHERE id = 1"
    )
    await db.commit()
    logger.info("Demo trade count incremented.")

async def get_current_count(db: aiosqlite.Connection) -> int:
    """Get the current number of demo trades executed."""
    async with db.execute("SELECT trade_count FROM demo_session_guard WHERE id = 1") as cursor:
        row = await cursor.fetchone()
        if not row:
            return 0
        return row[0]
