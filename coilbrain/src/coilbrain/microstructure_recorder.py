"""Microstructure Recorder background task for CoilBrain.

Collects OI, funding, order book, and (optionally) aggr-trade data from
Binance USDM Futures for every symbol that has at least one active signal.

Architecture
------------
* ``start()``            -- called from lifespan after observatory.start().
* ``stop()``             -- called from lifespan before observatory.stop().
* Symbol picker loop     -- wakes every MS_PICKER_INTERVAL_S.
                            Builds active symbol set from OBSERVING signals.
                            Spawns / cleans up per-symbol tasks.
* ``_symbol_loop()``     -- one Task per distinct Binance symbol.
                            Fetches snapshot once per MS_POLL_INTERVAL_S.
                            Associates the snapshot with ALL active signals
                            for that symbol via the junction table.
                            Exits when no active signals remain.

Isolation
---------
The recorder never writes to price_observations or observation_errors.
A crash in any symbol loop is caught and logged; it cannot affect the
price Observatory.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from coilbrain import database
from coilbrain import microstructure_repository as micro_repo
from coilbrain import repository
from coilbrain.config import settings
from coilbrain.microstructure_fetcher import fetch_all
from coilbrain.observatory import get_known_symbols, to_binance_symbol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_ms_picker_task: asyncio.Task | None = None
_active_symbol_tasks: dict[str, asyncio.Task] = {}  # binance_symbol -> Task
_last_purge_ts: float = 0.0                          # Unix ts of last retention purge

_PURGE_INTERVAL_S: float = 86400.0  # Run purge once per day.


# ---------------------------------------------------------------------------
# Per-symbol observation loop
# ---------------------------------------------------------------------------

async def _symbol_loop(binance_symbol: str) -> None:
    """Collect microstructure snapshots for one symbol until it goes quiet.

    Runs as an asyncio.Task.  Exits (naturally, without cancellation) when
    no OBSERVING signals remain for this symbol.
    """
    logger.info("Microstructure: starting symbol loop | symbol=%s", binance_symbol)

    prev_oi_usdt: float | None = None
    last_snapshot_ts: float = time.time()
    missed_polls: int = 0
    stale_threshold = int(settings.ms_stale_threshold_multiplier)

    try:
        async with httpx.AsyncClient() as client:
            while True:
                # --- Check for active signals ------------------------------------
                try:
                    async with database.get_db() as db:
                        active = await micro_repo.fetch_active_signals_for_symbol(
                            db, binance_symbol
                        )
                except Exception as exc:
                    logger.error(
                        "Microstructure: DB error checking active signals for %s: %s",
                        binance_symbol, exc,
                    )
                    await asyncio.sleep(settings.ms_poll_interval_s)
                    continue

                if not active:
                    logger.info(
                        "Microstructure: no active signals for %s — exiting loop",
                        binance_symbol,
                    )
                    break

                # --- Fetch snapshot ----------------------------------------------
                since_ts = last_snapshot_ts
                try:
                    snap = await fetch_all(
                        client, binance_symbol, since_ts, prev_oi_usdt
                    )
                except Exception as exc:
                    logger.error(
                        "Microstructure: fetch_all failed for %s: %s",
                        binance_symbol, exc,
                    )
                    missed_polls += 1
                    if missed_polls >= stale_threshold:
                        logger.warning(
                            "Microstructure: stale data — %d consecutive missed polls for %s",
                            missed_polls, binance_symbol,
                        )
                        try:
                            async with database.get_db() as db:
                                await micro_repo.insert_micro_error(
                                    db, binance_symbol, "fetch_all",
                                    f"stale: {missed_polls} consecutive failures",
                                )
                        except Exception:
                            pass
                    await asyncio.sleep(settings.ms_poll_interval_s)
                    continue

                missed_polls = 0

                # --- Persist snapshot + junction rows ----------------------------
                try:
                    async with database.get_db() as db:
                        snapshot_id = await micro_repo.insert_snapshot(db, snap)
                        now = snap.snapshot_ts
                        for row in active:
                            elapsed_s = now - row["t0_timestamp"]
                            await micro_repo.insert_signal_snapshot(
                                db, row["signal_id"], snapshot_id, elapsed_s
                            )
                except Exception as exc:
                    logger.error(
                        "Microstructure: DB write failed for %s: %s",
                        binance_symbol, exc,
                    )

                # Update state for next tick
                if snap.oi_usdt is not None:
                    prev_oi_usdt = snap.oi_usdt
                last_snapshot_ts = snap.snapshot_ts

                await asyncio.sleep(settings.ms_poll_interval_s)

    except asyncio.CancelledError:
        logger.info("Microstructure: symbol loop cancelled | symbol=%s", binance_symbol)
        raise
    finally:
        _active_symbol_tasks.pop(binance_symbol, None)
        logger.info("Microstructure: symbol loop exited | symbol=%s", binance_symbol)


# ---------------------------------------------------------------------------
# Symbol picker loop
# ---------------------------------------------------------------------------

async def _ms_picker_loop() -> None:
    """Wake periodically; spawn symbol loops for newly active symbols."""
    global _last_purge_ts
    while True:
        try:
            # --- Daily retention purge ----------------------------------------
            now = time.time()
            retain_days = settings.ms_retain_days
            if retain_days > 0 and (now - _last_purge_ts) >= _PURGE_INTERVAL_S:
                older_than = now - retain_days * 86400.0
                try:
                    async with database.get_db() as db:
                        deleted = await micro_repo.purge_old_snapshots(db, older_than)
                    if deleted > 0:
                        logger.info(
                            "Microstructure: purged %d old snapshots (retain=%dd)",
                            deleted, retain_days,
                        )
                    _last_purge_ts = now
                except Exception as exc:
                    logger.error("Microstructure: daily purge failed: %s", exc)

            # 1. Collect known binance_symbol values for OBSERVING signals.
            async with database.get_db() as db:
                symbols = await micro_repo.fetch_active_binance_symbols(db)

            # 2. Back-fill binance_symbol for signals the observatory
            #    hasn't yet resolved (pre-Phase-6 compatibility path).
            async with database.get_db() as db:
                unresolved = await micro_repo.fetch_observing_without_binance_symbol(db)

            known = get_known_symbols()
            for row in unresolved:
                bs = to_binance_symbol(row["symbol"], known)
                if bs:
                    async with database.get_db() as db:
                        await repository.set_binance_symbol(db, row["signal_id"], bs)
                    if bs not in symbols:
                        symbols.append(bs)

            # 3. Respect symbol cap (take the most recent up to ms_max_symbols).
            if len(symbols) > settings.ms_max_symbols:
                logger.warning(
                    "Microstructure: %d active symbols exceeds cap %d — "
                    "recording only first %d",
                    len(symbols), settings.ms_max_symbols, settings.ms_max_symbols,
                )
                symbols = symbols[: settings.ms_max_symbols]

            # 4. Enforce priority selection and clean up tasks.
            tasks_to_await = []
            for sym in list(_active_symbol_tasks):
                task = _active_symbol_tasks[sym]
                if task.done():
                    _active_symbol_tasks.pop(sym, None)
                elif sym not in symbols:
                    logger.info("Microstructure: cancelling loop for dropped symbol %s", sym)
                    task.cancel()
                    tasks_to_await.append(task)

            if tasks_to_await:
                await asyncio.gather(*tasks_to_await, return_exceptions=True)

            # 5. Spawn new tasks for symbols not yet being tracked.
            for sym in symbols:
                if sym not in _active_symbol_tasks:
                    task = asyncio.create_task(
                        _symbol_loop(sym),
                        name=f"ms-{sym}",
                    )
                    _active_symbol_tasks[sym] = task
                    logger.info("Microstructure: spawned loop for %s", sym)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Microstructure picker error: %s", exc)

        await asyncio.sleep(settings.ms_picker_interval_s)


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

async def start() -> None:
    """Start the microstructure recorder.  Called from the FastAPI lifespan."""
    global _ms_picker_task
    _ms_picker_task = asyncio.create_task(
        _ms_picker_loop(), name="ms-picker"
    )
    logger.info("Microstructure Recorder started")


async def stop() -> None:
    """Cancel all recorder tasks.  Called from the FastAPI lifespan."""
    for sym, task in list(_active_symbol_tasks.items()):
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    _active_symbol_tasks.clear()

    if _ms_picker_task is not None:
        _ms_picker_task.cancel()
        try:
            await _ms_picker_task
        except (asyncio.CancelledError, Exception):
            pass
    logger.info("Microstructure Recorder stopped")
