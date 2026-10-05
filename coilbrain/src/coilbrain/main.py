"""CoilBrain FastAPI application entry-point."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncio
import aiosqlite
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from coilbrain import database, observatory, microstructure_recorder, repository
from coilbrain.bybit_demo import BybitDemoClient
from coilbrain.config import settings
from coilbrain.executor import TradeExecutor
from coilbrain.models import HealthResponse, SignalPayload, SignalRecord
from coilbrain.position_monitor import PositionMonitor, reconcile_stuck_positions
from coilbrain.funded_challenge import challenge_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging, initialise the database, and start the observatory."""
    logging.basicConfig(level=settings.log_level.upper())
    logger.info("CoilBrain starting up")
    await database.init_db(settings.database_url)
    await observatory.start()
    await microstructure_recorder.start()

    # Initialise position monitor
    position_monitor: PositionMonitor | None = None
    monitor_task: asyncio.Task | None = None
    try:
        bybit_client = BybitDemoClient.from_env()
        # Immediately reconcile any positions that were stuck OPEN across restarts
        await reconcile_stuck_positions(bybit_client)
        # Start background polling loop (every 60s)
        position_monitor = PositionMonitor(bybit_client)
        monitor_task = asyncio.create_task(position_monitor.run(), name="position_monitor")
        logger.info("PositionMonitor: background task started")

        # Initialise funded account evaluation tracking
        async with database.get_db() as db:
            await challenge_manager.ensure_schema(db)
            bal = await bybit_client.get_balance("USDT")
            if bal and bal.total_equity > 0:
                await challenge_manager.sync_balance(db, bal.total_equity)
                logger.info("FundedChallenge: Live balance synchronized: $%.2f", bal.total_equity)
    except Exception:
        logger.exception("PositionMonitor/FundedChallenge: failed to start")

    # Start periodic DB cleanup (every 6 hours) to prevent disk saturation
    db_cleanup_task = asyncio.create_task(_periodic_db_cleanup(), name="db_cleanup")

    yield

    # Shutdown
    db_cleanup_task.cancel()
    if position_monitor is not None:
        position_monitor.stop()
    if monitor_task is not None:
        monitor_task.cancel()
    await microstructure_recorder.stop()
    await observatory.stop()
    logger.info("CoilBrain shutting down")


async def _periodic_db_cleanup() -> None:
    """Delete signals older than 24h every 6 hours to keep DB size manageable."""
    import time
    CLEANUP_INTERVAL_S = 6 * 3600  # 6 hours
    KEEP_RECENT_S = 24 * 3600      # keep last 24h of signals
    await asyncio.sleep(300)       # wait 5 min after startup before first cleanup
    while True:
        try:
            cutoff = time.time() - KEEP_RECENT_S
            async with database.get_db() as db:
                async with db.execute(
                    "DELETE FROM signals WHERE received_timestamp < ?", (cutoff,)
                ) as cur:
                    deleted = cur.rowcount
                await db.commit()
            if deleted and deleted > 0:
                logger.info("DB cleanup: deleted %d old signals (older than 24h)", deleted)
        except Exception:
            logger.exception("DB cleanup: unexpected error")
        await asyncio.sleep(CLEANUP_INTERVAL_S)


app = FastAPI(
    title="CoilBrain",
    description="Trading signal analysis service for the CoilSpring ecosystem.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# -- Routes -------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["infra"],
)
async def health() -> HealthResponse:
    """Return a simple liveness response."""
    return HealthResponse(status="ok", service=settings.service_name)


@app.get(
    "/funded",
    summary="Funded Account Evaluation Dashboard",
    tags=["funded"],
)
async def funded_dashboard() -> dict:
    """Get live evaluation status, drawdown metrics, profit target, and trading days."""
    async with database.get_db() as db:
        try:
            bybit = BybitDemoClient.from_env()
            bal = await bybit.get_balance("USDT")
            if bal and bal.total_equity > 0:
                await challenge_manager.sync_balance(db, bal.total_equity)
        except Exception:
            pass
        return await challenge_manager.get_dashboard(db)


def _row_to_signal_record(row: dict) -> SignalRecord:
    """Reconstruct a SignalRecord from a signals DB row."""
    return SignalRecord(
        signal_id=row["signal_id"],
        symbol=row["symbol"],
        exchange=row["exchange"],
        trigger=row["trigger"],
        grade=row["grade"],
        score=row["score"],
        signal=row["signal"],
        oi_status=row["oi_status"],
        price=row["t0_price"],
        signal_timestamp=row["t0_timestamp"],
        change_24h_pct=row["change_24h_pct"],
        volume_24h_usd=row["volume_24h_usd"],
        breakout_pct=row["breakout_pct"],
        body_volume_spike=row["body_volume_spike"],
        volume_spike=row["volume_spike"],
        flat_period_days=row["flat_period_days"],
        received_timestamp=row["received_timestamp"],
        transport_latency_ms=row["transport_latency_ms"],
    )


@app.post(
    "/signals",
    response_model=SignalRecord,
    status_code=201,
    summary="Ingest a CoilSpring trading signal",
    tags=["signals"],
)
async def create_signal(
    request: Request,
    payload: SignalPayload,
    background_tasks: BackgroundTasks,
) -> SignalRecord:
    """Accept a parsed CoilSpring signal, persist it, and return the record.

    Idempotency
    -----------
    If the request carries an ``X-Idempotency-Key`` header the server uses
    that value as the canonical ``signal_id``.  On the first call the signal
    is inserted and the created record is returned (HTTP 201).  On any
    subsequent call with the same key the existing record is returned without
    creating a second row (HTTP 201 again, body identical to the first
    response).  This makes the endpoint safe to retry unconditionally.

    Without ``X-Idempotency-Key`` a new UUID4 is generated server-side on
    every call (backward-compatible with clients that do not supply the header).
    """
    # BLOCKER 3: Read idempotency key from header
    idempotency_key = request.headers.get("X-Idempotency-Key", "").strip()

    if idempotency_key:
        # Check for an existing signal with this key before inserting
        async with database.get_db() as db:
            existing = await repository.fetch_signal(db, idempotency_key)
        if existing:
            logger.info(
                "Idempotent replay | key=%s symbol=%s",
                idempotency_key, existing["symbol"],
            )
            return _row_to_signal_record(existing)
        signal_id = idempotency_key
    else:
        # No idempotency key supplied — generate one (backward compatible)
        signal_id = str(uuid.uuid4())

    received_ts = time.time()
    record = SignalRecord(
        **payload.model_dump(),
        signal_id=signal_id,
        received_timestamp=received_ts,
        transport_latency_ms=(received_ts - payload.signal_timestamp) * 1000.0,
    )

    async with database.get_db() as db:
        try:
            await repository.insert_signal(db, record)
        except aiosqlite.IntegrityError:
            # Race condition: another concurrent request inserted the same key
            # between our fetch and insert.  Fetch and return the winner.
            existing = await repository.fetch_signal(db, signal_id)
            if existing:
                logger.info(
                    "Idempotent race resolved | key=%s", signal_id,
                )
                return _row_to_signal_record(existing)
            raise  # unexpected — re-raise

    logger.info(
        "Signal stored | id=%s symbol=%s score=%.0f",
        record.signal_id,
        record.symbol,
        record.score,
    )

    # BLOCKER: V1 Decision/Execution wiring
    background_tasks.add_task(TradeExecutor().execute, record)

    return record


@app.get(
    "/signals/{signal_id}",
    response_model=SignalRecord,
    summary="Retrieve a stored signal by ID",
    tags=["signals"],
)
async def get_signal(signal_id: str) -> SignalRecord:
    """Return a previously ingested signal by its ``signal_id``."""
    async with database.get_db() as db:
        row = await repository.fetch_signal(db, signal_id)

    if row is None:
        raise HTTPException(status_code=404, detail="Signal not found")

    return _row_to_signal_record(row)


# -- Dev runner ---------------------------------------------------------------


if __name__ == "__main__":
    uvicorn.run(
        "coilbrain.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
