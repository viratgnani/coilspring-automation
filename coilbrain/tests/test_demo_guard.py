import time
from unittest.mock import AsyncMock, patch

import pytest
import httpx

from coilbrain import database
from coilbrain.models import SignalRecord
from coilbrain.microstructure_fetcher import MicrostructureSnapshot
from coilbrain.executor import TradeExecutor
from coilbrain.position_manager import PositionManager
from coilbrain import demo_guard
from coilbrain.bybit_demo import BybitDemoClient
from coilbrain.atr_fetcher import ATRResult

import tempfile
import os

@pytest.fixture(autouse=True)
async def setup_db():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        temp_db_path = f.name
    
    await database.init_db(temp_db_path)
    
    # Reset limit to 0 just in case
    async with database.get_db() as db:
        await db.execute("UPDATE demo_session_guard SET trade_count = 0 WHERE id = 1")
        await db.commit()
    
    yield
    
    # Cleanup
    if os.path.exists(temp_db_path):
        try:
            os.remove(temp_db_path)
        except OSError:
            pass

def make_payload(score=80.0, volume=5_000_000.0, symbol="BTCUSDT", signal_id="sig1"):
    return {
        "symbol": symbol,
        "exchange": "BINANCE",
        "trigger": "VOL_SPIKE",
        "grade": "A",
        "score": score,
        "signal": "LONG",
        "oi_status": "RISING",
        "price": 1000.0,
        "volume_24h_usd": volume,
        "signal_timestamp": time.time(),
        "change_24h_pct": 2.5,
        "breakout_pct": 1.0,
        "body_volume_spike": 6.0,
        "volume_spike": 12.0,
        "flat_period_days": 1.5,
    }

def make_snapshot(symbol="BTCUSDT", fresh=True, supportive=True):
    return MicrostructureSnapshot(
        binance_symbol=symbol,
        snapshot_ts=time.time() if fresh else time.time() - 30.0,
        oi_usdt=100_000_000.0,
        oi_delta_pct=2.0 if supportive else 0.0,
        funding_rate=0.01,
        mark_price=1000.0,
        next_funding_ts=time.time() + 3600.0,
        best_bid=999.9,
        best_ask=1000.1,
        spread_abs=0.2,
        spread_pct=0.02,
        bid_depth_usdt=500_000.0,
        ask_depth_usdt=200_000.0 if supportive else 500_000.0,
        ob_imbalance=0.2 if supportive else 0.0,
        trade_count=1000,
        traded_vol_usdt=1_000_000.0,
        buy_vol_usdt=600_000.0 if supportive else 500_000.0,
        sell_vol_usdt=400_000.0 if supportive else 500_000.0,
        trade_imbalance=0.2 if supportive else 0.0,
        used_weight_1m=100,
    )

@pytest.mark.asyncio
@patch("coilbrain.demo_guard.MAX_DEMO_TRADES_PER_SESSION", 1)
@patch("coilbrain.executor.fetch_all")
@patch("coilbrain.atr_fetcher.ATRFetcher.fetch_atr")
async def test_demo_guard_limits_to_one_trade(mock_fetch_atr, mock_fetch_all):
    mock_fetch_all.return_value = make_snapshot(supportive=True)
    mock_fetch_atr.return_value = ATRResult(atr=10.0, source="test", symbol="BTCUSDT", interval="1m", fetch_ts=time.time(), candles_used=14)

    client = AsyncMock(spec=BybitDemoClient)
    client.get_position.return_value = None
    
    order_mock = AsyncMock()
    order_mock.order_id = "mock_order_123"
    client.place_market_order.return_value = order_mock

    executor = TradeExecutor()
    executor.bybit_client = client

    # First trade: supportive signal
    rec1 = SignalRecord(signal_id="sig1", received_timestamp=time.time(), transport_latency_ms=0, **make_payload(symbol="BTCUSDT"))
    await executor.execute(rec1)

    # Should have placed order
    client.place_market_order.assert_called_once()
    
    # Check limit count in DB
    async with database.get_db() as db:
        count = await demo_guard.get_current_count(db)
        assert count == 1

    # Second trade: another supportive signal
    rec2 = SignalRecord(signal_id="sig2", received_timestamp=time.time(), transport_latency_ms=0, **make_payload(symbol="ETHUSDT"))
    await executor.execute(rec2)

    # Order count should still be 1 (ignored by guard)
    assert client.place_market_order.call_count == 1
    
    # Journal should reflect DEMO_SESSION_LIMIT
    async with database.get_db() as db:
        async with db.execute("SELECT decision, reasons_json FROM decision_journal WHERE signal_id='sig2'") as cur:
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == "IGNORE"
            assert "DEMO_SESSION_LIMIT" in row[1]

@pytest.mark.asyncio
@patch("coilbrain.executor.fetch_all")
async def test_demo_guard_ignore_signals_do_not_consume_slot(mock_fetch_all):
    # Unsupportive snapshot -> IGNORE decision
    mock_fetch_all.return_value = make_snapshot(supportive=False)
    
    client = AsyncMock(spec=BybitDemoClient)
    executor = TradeExecutor()
    executor.bybit_client = client

    rec = SignalRecord(signal_id="sig1", received_timestamp=time.time(), transport_latency_ms=0, **make_payload(symbol="BTCUSDT"))
    await executor.execute(rec)

    client.place_market_order.assert_not_called()

    async with database.get_db() as db:
        count = await demo_guard.get_current_count(db)
        assert count == 0

@pytest.mark.asyncio
async def test_demo_guard_persists_across_restart():
    # Force count to 1
    async with database.get_db() as db:
        await db.execute("UPDATE demo_session_guard SET trade_count = 1 WHERE id = 1")
        await db.commit()
    
    # Simulating restart by calling init_db again with the same temp db path
    async with database.get_db() as db:
        # Hack to get the current db path used by the fixture
        pass
    
    # Actually, we can just re-init with the global _db_path
    await database.init_db(database._db_path)
    
    async with database.get_db() as db:
        count = await demo_guard.get_current_count(db)
        # Should still be 1 (idempotent init)
        assert count == 1
