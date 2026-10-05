import asyncio
import time
import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from coilbrain import database, decision_repository as drepo
from coilbrain.config import settings
from coilbrain.decision_engine import Decision, DecisionConfig
from coilbrain.main import app
from coilbrain.microstructure_fetcher import MicrostructureSnapshot
from coilbrain.models import SignalPayload
from coilbrain.position_manager import PositionManager

# Since we use SQLite in-memory for testing, BackgroundTasks will run synchronously
# inside the test client when the request finishes, accessing the same DB connection!
client = TestClient(app)

@pytest.fixture(autouse=True)
async def db_setup():
    # Because we don't use TestClient as a context manager across all tests automatically,
    # let's just make sure the DB is initialized if the session fixture didn't.
    await database.init_db(settings.database_url)
    async with database.get_db() as db:
        await db.execute("DELETE FROM trade_journal")
        await db.execute("DELETE FROM decision_journal")
        await db.execute("DELETE FROM demo_positions")
        await db.execute("DELETE FROM signals")
        await db.commit()
    yield


def make_payload(score=80.0, volume=5_000_000.0, symbol="BTCUSDT"):
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

class MockATRResult:
    def __init__(self):
        self.atr = 15.0

@pytest.mark.asyncio
async def test_e2e_low_volume_ignored():
    # A) LOW VOLUME: signal -> IGNORE -> zero Bybit orders
    payload = make_payload(volume=100_000.0) # < 1M threshold
    snapshot = make_snapshot()
    
    with patch("coilbrain.executor.fetch_all", new=AsyncMock(return_value=snapshot)) as mock_fetch:
        with patch("coilbrain.bybit_demo.BybitDemoClient.place_market_order") as mock_order:
            with patch("coilbrain.bybit_demo.BybitDemoClient.from_env") as mock_client_factory:
                response = client.post("/signals", json=payload)
                assert response.status_code == 201

                mock_fetch.assert_called_once()
                mock_order.assert_not_called()
                
                async with database.get_db() as db:
                    async with db.execute("SELECT * FROM decision_journal") as cursor:
                        decisions = await cursor.fetchall()
                        # convert sqlite3.Row to dict
                        decisions = [dict(zip([column[0] for column in cursor.description], row)) for row in decisions]
                    assert len(decisions) >= 1
                    assert decisions[-1]["decision"] == "IGNORE", f"Decisions: {decisions[-1]}"
                    # The column might be reasons or reason
                    reason_str = decisions[-1].get("reasons_json", "")
                    assert "VOLUME" in reason_str, f"Decisions: {decisions[-1]}"


@pytest.mark.asyncio
async def test_e2e_stale_data_ignored():
    # B) STALE DATA: signal -> IGNORE -> zero Bybit orders
    payload = make_payload()
    snapshot = make_snapshot(fresh=False) # Age > 10s
    
    with patch("coilbrain.executor.fetch_all", new=AsyncMock(return_value=snapshot)) as mock_fetch:
        with patch("coilbrain.bybit_demo.BybitDemoClient.place_market_order") as mock_order:
            with patch("coilbrain.bybit_demo.BybitDemoClient.from_env"):
                response = client.post("/signals", json=payload)
                assert response.status_code == 201

                mock_order.assert_not_called()
                
                async with database.get_db() as db:
                    async with db.execute("SELECT * FROM decision_journal") as cursor:
                        decisions = await cursor.fetchall()
                        decisions = [dict(zip([column[0] for column in cursor.description], row)) for row in decisions]
                    assert len(decisions) >= 1
                    assert decisions[-1]["decision"] == "IGNORE", f"Decisions: {decisions[-1]}"
                    reason_str = decisions[-1].get("reasons_json", "")
                    assert "STALE" in reason_str, f"Decisions: {decisions[-1]}"


@pytest.mark.asyncio
async def test_e2e_existing_position_ignored():
    # C) EXISTING POSITION: signal -> IGNORE -> zero Bybit orders
    payload = make_payload()
    snapshot = make_snapshot(fresh=True, supportive=True)
    
    async with database.get_db() as db:
        pm = PositionManager()
        await pm.open_pending(
            db, 
            position_id="SIG123", 
            signal_id="SIG123", 
            symbol=payload["symbol"],
            bybit_symbol=payload["symbol"], 
            entry_price=1000.0,
            stop_loss=900.0,
            take_profit=1100.0,
            position_size_usd=10000.0,
            quantity=10.0,
            risk_usd=1000.0,
            rr_ratio=1.0,
            leverage=1.0
        )
        await pm.confirm_open(db, position_id="SIG123", bybit_order_id="ORD123", actual_entry_price=1000.0)
    
    with patch("coilbrain.executor.fetch_all", new=AsyncMock(return_value=snapshot)):
        with patch("coilbrain.bybit_demo.BybitDemoClient.place_market_order") as mock_order:
            with patch("coilbrain.bybit_demo.BybitDemoClient.from_env"):
                response = client.post("/signals", json=payload)
                assert response.status_code == 201

                mock_order.assert_not_called()
                
                async with database.get_db() as db:
                    async with db.execute("SELECT * FROM decision_journal") as cursor:
                        decisions = await cursor.fetchall()
                        decisions = [dict(zip([column[0] for column in cursor.description], row)) for row in decisions]
                    assert len(decisions) >= 1
                    assert decisions[-1]["decision"] == "IGNORE", f"Decisions: {decisions[-1]}"
                    reason_str = decisions[-1].get("reasons_json", "")
                    assert "POSITION" in reason_str, f"Decisions: {decisions[-1]}"


@pytest.mark.asyncio
async def test_e2e_valid_trade_flow():
    # D) VALID TRADE: signal -> fresh microstructure -> TRADE -> risk -> position recheck -> Demo market order -> SL -> TP -> journal
    payload = make_payload()
    snapshot = make_snapshot(fresh=True, supportive=True)
    
    mock_order_resp = AsyncMock()
    mock_order_resp.order_id = "DEMO123"
    
    mock_pos_resp = AsyncMock()
    mock_pos_resp.avg_price = 1005.0
    
    mock_client = AsyncMock()
    mock_client.place_market_order.return_value = mock_order_resp
    mock_client.get_position.return_value = mock_pos_resp
    
    with patch("coilbrain.executor.BybitDemoClient.from_env", return_value=mock_client):
        with patch("coilbrain.executor.fetch_all", new=AsyncMock(return_value=snapshot)):
            with patch("coilbrain.executor.ATRFetcher.fetch_atr", new=AsyncMock(return_value=MockATRResult())):
                response = client.post("/signals", json=payload)
                assert response.status_code == 201
                
                # Check that Bybit client was called properly
                
                async with database.get_db() as db:
                    async with db.execute("SELECT * FROM decision_journal") as cursor:
                        decisions = await cursor.fetchall()
                        decisions = [dict(zip([column[0] for column in cursor.description], row)) for row in decisions]
                    assert len(decisions) >= 1
                    assert decisions[-1]["decision"] == "TRADE", f"Decision was {decisions[-1]['decision']}, reason: {decisions[-1].get('reasons_json', '')}"
                    
                    async with db.execute("SELECT * FROM trade_journal") as cursor:
                        trades = await cursor.fetchall()
                        trades = [dict(zip([column[0] for column in cursor.description], row)) for row in trades]
                    assert len(trades) >= 1
                    trade = trades[-1]
                    mock_client.place_market_order.assert_called_once()
                    assert trade["bybit_order_id"] == "DEMO123"
                    assert trade["sl_confirmed_timestamp"] is not None
                    assert trade["tp_confirmed_timestamp"] is not None

                    # Verify PositionManager state is OPEN
                    pm = PositionManager()
                    state = await pm.get_state_for_symbol(db, payload["symbol"])
                    assert state == "OPEN"
