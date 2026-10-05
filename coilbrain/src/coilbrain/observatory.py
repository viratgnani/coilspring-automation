"""Signal Observatory background task for CoilBrain.

This module runs entirely within the FastAPI asyncio event loop.
It never calls time.sleep() -- only asyncio.sleep().
It never blocks POST /signals.

Architecture
------------
* ``start()``          -- called from lifespan; starts the picker task.
* ``stop()``           -- called from lifespan; cancels all tasks cleanly.
* Picker task          -- wakes every OBS_PICKER_INTERVAL_S seconds,
                          picks up OPEN signals, spawns per-signal coroutines.
* ``observe_signal()`` -- per-signal coroutine; polls Binance, writes DB.
* Symbol registry      -- caches known Binance USDM symbols; refreshes every
                          OBS_REGISTRY_REFRESH_HOURS hours.
* ``to_binance_symbol()`` -- converts ccxt unified symbol to Binance REST symbol.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from coilbrain import database, repository
from coilbrain.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_picker_task: asyncio.Task | None = None
_active_observations: dict[str, asyncio.Task] = {}  # signal_id -> Task

# Known Binance USDM perpetual symbols (set of REST-format strings, e.g. "BTCUSDT")
_known_symbols: set[str] = set()
_registry_loaded_at: float = 0.0


def get_known_symbols() -> set[str]:
    """Return the current symbol registry (read-only for external callers)."""
    return _known_symbols


# ---------------------------------------------------------------------------
# Symbol mapping
# ---------------------------------------------------------------------------

def to_binance_symbol(ccxt_symbol: str, known: set[str] | None = None) -> str | None:
    """Convert a symbol to a Binance USDM Futures REST symbol.

    Accepts two input formats:

    1. CCXT unified format  (BASE/QUOTE:SETTLE) -- from Alpha Coiler / V4:
           "BTC/USDT:USDT"       -> "BTCUSDT"
           "1000PEPE/USDT:USDT"  -> "1000PEPEUSDT"
           "KOMA/USDT:USDT"      -> "KOMAUSDT"

    2. Raw Binance REST format  (already resolved) -- from CoilSpring V2:
           "BTCUSDT"             -> "BTCUSDT"    (validated against known set)
           "1000PEPEUSDT"        -> "1000PEPEUSDT"

    Returns None if the symbol cannot be mapped to a known USDM perpetual.
    CoilBrain will then log an observation error and skip polling.

    Parsing for CCXT format:
        BASE  = everything before "/"
        QUOTE = substring between "/" and ":"
        REST_SYMBOL = BASE + QUOTE
    """
    if not ccxt_symbol:
        logger.warning("Cannot parse ccxt symbol: %r", ccxt_symbol)
        return None

    # --- Fast path: raw Binance REST symbol (no "/" present) -----------------
    if "/" not in ccxt_symbol:
        if known is not None:
            if ccxt_symbol in known:
                return ccxt_symbol
            logger.warning(
                "Raw Binance symbol %r not found in known USDM set",
                ccxt_symbol,
            )
            return None
        return ccxt_symbol

    # --- CCXT unified format (BASE/QUOTE:SETTLE) -----------------------------
    try:
        slash = ccxt_symbol.index("/")
        colon = ccxt_symbol.index(":", slash)
        base = ccxt_symbol[:slash]
        quote = ccxt_symbol[slash + 1: colon]
        candidate = base + quote
    except (ValueError, IndexError):
        logger.warning("Cannot parse ccxt symbol: %r", ccxt_symbol)
        return None

    if known is not None:
        if candidate in known:
            return candidate
        logger.warning(
            "Derived Binance symbol %r (from %r) not found in known USDM set",
            candidate, ccxt_symbol,
        )
        return None

    return candidate



async def _load_symbol_registry(client: httpx.AsyncClient) -> set[str]:
    """Fetch active USDM PERPETUAL symbols from Binance exchangeInfo."""
    try:
        resp = await client.get(
            settings.obs_binance_exchange_info_url,
            timeout=settings.obs_http_timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            s["symbol"]
            for s in data.get("symbols", [])
            if (
                s.get("contractType") == "PERPETUAL"
                and s.get("status") == "TRADING"
                and s.get("quoteAsset") == "USDT"
            )
        }
    except Exception as exc:
        logger.error("Failed to load Binance symbol registry: %s", exc)
        return set()


async def _ensure_registry(client: httpx.AsyncClient) -> None:
    """Refresh the symbol registry if it is stale or empty."""
    global _known_symbols, _registry_loaded_at
    refresh_s = settings.obs_registry_refresh_hours * 3600
    if _known_symbols and (time.time() - _registry_loaded_at) < refresh_s:
        return
    fresh = await _load_symbol_registry(client)
    if fresh:
        _known_symbols = fresh
        _registry_loaded_at = time.time()
        logger.info("Symbol registry loaded: %d USDM perpetuals", len(_known_symbols))


# ---------------------------------------------------------------------------
# Price fetch
# ---------------------------------------------------------------------------

async def _fetch_price(client: httpx.AsyncClient, binance_symbol: str) -> float | None:
    """Return the current mark price for a Binance USDM symbol, or None on error."""
    try:
        resp = await client.get(
            settings.obs_binance_price_url,
            params={"symbol": binance_symbol},
            timeout=settings.obs_http_timeout_s,
        )
        resp.raise_for_status()
        return float(resp.json()["price"])
    except Exception as exc:
        logger.warning("Price fetch failed for %s: %s", binance_symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Polling schedule
# ---------------------------------------------------------------------------

def _next_poll_interval(elapsed_s: float) -> float:
    """Return the polling interval (seconds) appropriate for elapsed time."""
    if elapsed_s < settings.obs_phase1_duration_s:
        return settings.obs_phase1_interval_s
    return settings.obs_phase2_interval_s


def _compute_schedule(
    t0: float,
    last_obs_ts: float | None,
) -> tuple[float, float]:
    """Return (next_poll_at, interval_s) given t0 and the last observation time.

    On a fresh start, next_poll_at = t0 + first_interval.
    On restart recovery, next_poll_at = last_obs_ts + appropriate_interval.
    """
    if last_obs_ts is None:
        interval_s = _next_poll_interval(0.0)
        return t0 + interval_s, interval_s

    elapsed = last_obs_ts - t0
    interval_s = _next_poll_interval(elapsed)
    return last_obs_ts + interval_s, interval_s


# ---------------------------------------------------------------------------
# Per-signal observation coroutine
# ---------------------------------------------------------------------------

async def observe_signal(
    signal_id: str,
    symbol: str,
    binance_symbol: str,
    t0_price: float,
    t0_timestamp: float,
    last_obs_ts: float | None,
) -> None:
    """Observe a single signal for its full window.  Runs as an asyncio Task."""
    obs_window_s = settings.obs_window_hours * 3600
    deadline = t0_timestamp + obs_window_s

    # Running outcome state (held in memory; flushed to DB after every poll)
    max_price = t0_price
    min_price = t0_price
    time_to_max_s: float | None = None
    secs_to_1pct: float | None = None
    secs_to_5pct: float | None = None
    secs_to_10pct: float | None = None

    logger.info(
        "Observatory: starting observation | id=%s symbol=%s binance=%s t0=%.4f",
        signal_id, symbol, binance_symbol, t0_price,
    )

    async with httpx.AsyncClient() as client:
        next_poll_at, _ = _compute_schedule(t0_timestamp, last_obs_ts)

        while True:
            now = time.time()

            # If the window has expired, close out.
            if now >= deadline:
                break

            sleep_s = next_poll_at - now
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

            now = time.time()
            if now >= deadline:
                break

            price = await _fetch_price(client, binance_symbol)
            if price is None:
                async with database.get_db() as db:
                    await repository.insert_error(
                        db, signal_id,
                        f"price fetch failed for {binance_symbol}"
                    )
                # Advance schedule and retry next poll
                elapsed = now - t0_timestamp
                interval_s = _next_poll_interval(elapsed)
                next_poll_at = now + interval_s
                continue

            elapsed = now - t0_timestamp
            pct_from_t0 = (price - t0_price) / t0_price * 100.0

            # Update running extremes
            if price > max_price:
                max_price = price
                time_to_max_s = elapsed
            if price < min_price:
                min_price = price

            # Milestone: time to +1%, +5%, +10%
            if secs_to_1pct is None and pct_from_t0 >= 1.0:
                secs_to_1pct = elapsed
            if secs_to_5pct is None and pct_from_t0 >= 5.0:
                secs_to_5pct = elapsed
            if secs_to_10pct is None and pct_from_t0 >= 10.0:
                secs_to_10pct = elapsed

            mfe_pct = max(0.0, (max_price - t0_price) / t0_price * 100.0)
            mae_pct = max(0.0, (t0_price - min_price) / t0_price * 100.0)

            async with database.get_db() as db:
                await repository.insert_observation(
                    db, signal_id, now, elapsed, price, pct_from_t0
                )
                await repository.update_outcome(
                    db, signal_id,
                    mfe_pct, mae_pct,
                    max_price, min_price,
                    time_to_max_s,
                    secs_to_1pct, secs_to_5pct, secs_to_10pct,
                )

            interval_s = _next_poll_interval(elapsed)
            next_poll_at = now + interval_s

    # Mark complete
    async with database.get_db() as db:
        await repository.mark_complete(db, signal_id)

    _active_observations.pop(signal_id, None)
    logger.info("Observatory: observation complete | id=%s", signal_id)


# ---------------------------------------------------------------------------
# Picker loop
# ---------------------------------------------------------------------------

async def _picker_loop() -> None:
    """Wake periodically to pick up OPEN signals and spawn observation tasks."""
    async with httpx.AsyncClient() as client:
        while True:
            try:
                await _ensure_registry(client)

                async with database.get_db() as db:
                    open_signals = await repository.fetch_open_signals(db)

                for row in open_signals:
                    sid = row["signal_id"]
                    if sid in _active_observations:
                        continue  # already being observed

                    binance_sym = to_binance_symbol(row["symbol"], _known_symbols)
                    if binance_sym is None:
                        async with database.get_db() as db:
                            await repository.insert_error(
                                db, sid,
                                f"no valid Binance USDM market for symbol {row['symbol']!r}"
                            )
                        # Leave OPEN — will retry next picker cycle in case
                        # the registry is stale; if registry is fresh, the
                        # error persists harmlessly.
                        continue

                    async with database.get_db() as db:
                        await repository.mark_observing(db, sid)
                        await repository.set_binance_symbol(db, sid, binance_sym)

                    task = asyncio.create_task(
                        observe_signal(
                            signal_id=sid,
                            symbol=row["symbol"],
                            binance_symbol=binance_sym,
                            t0_price=row["t0_price"],
                            t0_timestamp=row["t0_timestamp"],
                            last_obs_ts=None,
                        ),
                        name=f"obs-{sid}",
                    )
                    _active_observations[sid] = task

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Picker loop error: %s", exc)

            await asyncio.sleep(settings.obs_picker_interval_s)


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

async def start() -> None:
    """Start the observatory.  Called from the FastAPI lifespan handler."""
    global _picker_task
    await _recover_observing_signals()
    _picker_task = asyncio.create_task(_picker_loop(), name="obs-picker")
    logger.info("Observatory started")


async def stop() -> None:
    """Cancel all observatory tasks.  Called from the FastAPI lifespan handler."""
    for sid, task in list(_active_observations.items()):
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    _active_observations.clear()

    if _picker_task is not None:
        _picker_task.cancel()
        try:
            await _picker_task
        except (asyncio.CancelledError, Exception):
            pass
    logger.info("Observatory stopped")


async def _recover_observing_signals() -> None:
    """On restart, resume any signals that were OBSERVING when we died."""
    async with httpx.AsyncClient() as client:
        await _ensure_registry(client)

    async with database.get_db() as db:
        recovering = await repository.fetch_recovering_signals(db)

    obs_window_s = settings.obs_window_hours * 3600

    for row in recovering:
        sid = row["signal_id"]
        t0 = row["t0_timestamp"]
        now = time.time()

        # If the window has already elapsed, close it out immediately.
        if now >= t0 + obs_window_s:
            async with database.get_db() as db:
                await repository.mark_complete(db, sid)
            logger.info("Observatory: recovered expired signal, marked COMPLETE | id=%s", sid)
            continue

        binance_sym = to_binance_symbol(row["symbol"], _known_symbols)
        if binance_sym is None:
            async with database.get_db() as db:
                await repository.insert_error(
                    db, sid,
                    f"recovery: no valid Binance USDM market for {row['symbol']!r}"
                )
                await repository.mark_error(db, sid)
            continue

        # Find where the schedule was
        async with database.get_db() as db:
            last_obs_ts = await repository.fetch_latest_observation_timestamp(db, sid)

        task = asyncio.create_task(
            observe_signal(
                signal_id=sid,
                symbol=row["symbol"],
                binance_symbol=binance_sym,
                t0_price=row["t0_price"],
                t0_timestamp=t0,
                last_obs_ts=last_obs_ts,
            ),
            name=f"obs-{sid}",
        )
        _active_observations[sid] = task
        logger.info("Observatory: resumed observation | id=%s", sid)
