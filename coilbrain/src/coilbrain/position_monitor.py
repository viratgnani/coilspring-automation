"""CoilBrain Position Monitor.

Background loop that polls Bybit Demo API every 60 seconds to check
if any OPEN positions have been closed (TP_HIT, SL_HIT, or manually).

When a position is found to be closed on Bybit:
  1. Marks it CLOSED in demo_positions with exit_reason and actual_exit_price
  2. Logs the outcome to trade_journal

This is the missing piece that reconciles our local DB with Bybit's state.
Bybit executes TP/SL server-side; we just need to detect when they fired.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import httpx

from coilbrain import database
from coilbrain.bybit_demo import BybitDemoClient, BybitDemoAPIError
from coilbrain.position_manager import PositionManager, Position
from coilbrain.funded_challenge import challenge_manager

logger = logging.getLogger(__name__)

# How often to poll Bybit for closed positions (seconds)
POLL_INTERVAL_S = 60

# How far back to look in order history (seconds)
ORDER_LOOKBACK_S = 7 * 24 * 3600  # 7 days


class PositionMonitor:
    """Polls Bybit every POLL_INTERVAL_S seconds and reconciles OPEN positions.

    Usage (in main.py lifespan):
        monitor = PositionMonitor(bybit_client)
        asyncio.create_task(monitor.run())
    """

    def __init__(self, bybit_client: BybitDemoClient) -> None:
        self.bybit_client = bybit_client
        self.position_manager = PositionManager()
        self._running = True

    async def run(self) -> None:
        """Background task: poll forever until stopped."""
        logger.info("PositionMonitor: started (poll every %ds)", POLL_INTERVAL_S)
        # Give the app a moment to fully start before first poll
        await asyncio.sleep(30)

        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("PositionMonitor: unexpected error in poll loop")
            await asyncio.sleep(POLL_INTERVAL_S)

        logger.info("PositionMonitor: stopped")

    def stop(self) -> None:
        self._running = False

    async def _poll_once(self) -> None:
        """Check all OPEN positions in DB against Bybit and close any that fired."""
        async with database.get_db() as db:
            open_positions = await self.position_manager.list_positions(db, status="OPEN")

        if not open_positions:
            logger.debug("PositionMonitor: no open positions to check")
            return

        logger.info("PositionMonitor: checking %d open position(s)", len(open_positions))

        for pos in open_positions:
            try:
                await self._check_position(pos)
            except Exception:
                logger.exception("PositionMonitor: error checking position %s (%s)", pos.position_id, pos.bybit_symbol)

    async def _check_position(self, pos: Position) -> None:
        """Check a single position against Bybit and close it in DB if needed."""
        bybit_symbol = pos.bybit_symbol

        # Get current position size on Bybit
        try:
            bybit_pos = await self.bybit_client.get_position(bybit_symbol)
        except BybitDemoAPIError as e:
            logger.warning("PositionMonitor: Bybit API error for %s: %s", bybit_symbol, e)
            return

        if bybit_pos is not None and bybit_pos.size > 0:
            # Position still open on Bybit - log current state
            logger.debug(
                "PositionMonitor: %s still OPEN | size=%.4f unrealisedPnl=%.2f",
                bybit_symbol, bybit_pos.size, bybit_pos.unrealised_pnl
            )
            return

        # Position is CLOSED on Bybit (size=0) - find out how it closed
        logger.info("PositionMonitor: %s position CLOSED on Bybit - fetching close details", bybit_symbol)

        exit_price, exit_reason, pnl_usd = await self._get_close_details(pos, bybit_symbol)

        if exit_price is None:
            # Couldn't determine exit price - use entry price as fallback
            logger.warning(
                "PositionMonitor: Could not get exit details for %s, using entry price as fallback",
                bybit_symbol
            )
            exit_price = pos.actual_entry_price or pos.entry_price or 0.0
            exit_reason = exit_reason or "CLOSED_UNKNOWN"

        # Calculate PnL if not provided
        if pnl_usd is None and pos.actual_entry_price and exit_price and pos.quantity:
            # For long positions (we only go long currently)
            pnl_usd = (exit_price - pos.actual_entry_price) * pos.quantity

        logger.info(
            "PositionMonitor: Closing %s | reason=%s exit=%.6f pnl=%s",
            bybit_symbol, exit_reason, exit_price,
            f"${pnl_usd:.2f}" if pnl_usd is not None else "N/A"
        )

        async with database.get_db() as db:
            await self.position_manager.close(
                db,
                pos.position_id,
                exit_reason=exit_reason,
                actual_exit_price=exit_price,
                pnl_usd=pnl_usd,
            )
            try:
                bal = await self.bybit_client.get_balance("USDT")
                if bal and bal.total_equity > 0:
                    await challenge_manager.sync_balance(db, bal.total_equity)
            except Exception as e:
                logger.warning("PositionMonitor: failed to sync funded challenge balance: %s", e)

    async def _get_close_details(
        self, pos: Position, bybit_symbol: str
    ) -> tuple[Optional[float], str, Optional[float]]:
        """
        Try to determine exit price and reason from Bybit order/trade history.
        Returns (exit_price, exit_reason, pnl_usd).
        """
        # Strategy 1: Check closed PnL (most reliable)
        try:
            pnl_data = await self.bybit_client._get(
                "/v5/position/closed-pnl",
                params={
                    "category": "linear",
                    "symbol": bybit_symbol,
                    "limit": "10",
                },
            )
            pnl_list = pnl_data.get("result", {}).get("list", [])
            if pnl_list:
                # Most recent closed PnL entry
                entry = pnl_list[0]
                exit_price = float(entry.get("avgExitPrice", 0) or 0)
                closed_pnl = float(entry.get("closedPnl", 0) or 0)

                # Determine reason based on TP/SL prices
                tp = pos.take_profit or 0
                sl = pos.stop_loss or 0

                if exit_price > 0 and tp > 0 and abs(exit_price - tp) / tp < 0.005:
                    reason = "TP_HIT"
                elif exit_price > 0 and sl > 0 and abs(exit_price - sl) / sl < 0.01:
                    reason = "SL_HIT"
                elif closed_pnl > 0:
                    reason = "TP_HIT"
                elif closed_pnl < 0:
                    reason = "SL_HIT"
                else:
                    reason = "CLOSED_MANUAL"

                if exit_price > 0:
                    logger.info(
                        "PositionMonitor: %s closed-pnl: exit=%.6f pnl=%.2f reason=%s",
                        bybit_symbol, exit_price, closed_pnl, reason
                    )
                    return exit_price, reason, closed_pnl if closed_pnl != 0 else None

        except Exception as e:
            logger.warning("PositionMonitor: closed-pnl query failed for %s: %s", bybit_symbol, e)

        # Strategy 2: Check order history for filled TP/SL orders
        try:
            order_data = await self.bybit_client._get(
                "/v5/order/history",
                params={
                    "category": "linear",
                    "symbol": bybit_symbol,
                    "limit": "20",
                },
            )
            orders = order_data.get("result", {}).get("list", [])

            # Look for Filled reduce-only orders (these are TP/SL)
            for order in orders:
                if (order.get("orderStatus") == "Filled"
                        and order.get("reduceOnly") in (True, "true", "True")):
                    exit_price = float(order.get("avgPrice", 0) or 0)
                    stop_order_type = order.get("stopOrderType", "")
                    order_type = order.get("orderType", "")

                    if "TakeProfit" in stop_order_type or "tp" in order_type.lower():
                        reason = "TP_HIT"
                    elif "StopLoss" in stop_order_type or "sl" in order_type.lower():
                        reason = "SL_HIT"
                    else:
                        reason = "CLOSED_REDUCE_ONLY"

                    if exit_price > 0:
                        entry = pos.actual_entry_price or pos.entry_price or exit_price
                        pnl = (exit_price - entry) * (pos.quantity or 0)
                        return exit_price, reason, pnl if pnl != 0 else None

        except Exception as e:
            logger.warning("PositionMonitor: order history query failed for %s: %s", bybit_symbol, e)

        return None, "CLOSED_UNKNOWN", None


async def reconcile_stuck_positions(bybit_client: BybitDemoClient) -> None:
    """One-shot reconciliation: close any OPEN positions that are already closed on Bybit.

    Call this on startup to fix positions that were stuck OPEN across a restart.
    """
    logger.info("PositionMonitor: running startup reconciliation...")
    monitor = PositionMonitor(bybit_client)
    await monitor._poll_once()
    logger.info("PositionMonitor: startup reconciliation complete")
