"""Tests for Execution Universe Safety Gate and ATR Fetcher Integration.

Verifies:
A. Valid crypto perpetual:
   ARKUSDT -> eligible -> valid ATR -> can proceed past RiskEngine
B. TradFi equity:
   ADBEUSDT -> UNSUPPORTED_INSTRUMENT -> zero market orders
C. Unknown instrument metadata:
   UNKNOWNUSDT -> UNSUPPORTED_INSTRUMENT -> zero market orders
D. Invalid ATR:
   ARKUSDT -> RISK_INVALID -> zero market orders
E. Existing stale-data gate still works:
   Stale snapshot -> STALE_MARKET_DATA -> zero market orders
"""

import json
import os
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from coilbrain import database, demo_guard
from coilbrain.atr_fetcher import ATRFetcher, ATRResult
from coilbrain.bybit_demo import BybitDemoClient, BybitDemoConfigError
from coilbrain.decision_engine import Decision, Reason
from coilbrain.executor import TradeExecutor
from coilbrain.instrument_validator import InstrumentValidator
from coilbrain.microstructure_fetcher import MicrostructureSnapshot
from coilbrain.models import SignalRecord
from coilbrain.risk_engine import RiskConfig, RiskEngine


@pytest.fixture(autouse=True)
async def setup_db():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        temp_db_path = f.name

    await database.init_db(temp_db_path)

    async with database.get_db() as db:
        await db.execute("UPDATE demo_session_guard SET trade_count = 0 WHERE id = 1")
        await db.commit()

    yield

    if os.path.exists(temp_db_path):
        try:
            os.remove(temp_db_path)
        except OSError:
            pass


def make_payload(score=80.0, volume=5_000_000.0, symbol="ARKUSDT", signal_id="sig_test_1"):
    return {
        "symbol": symbol,
        "exchange": "BINANCE",
        "trigger": "VOL_SPIKE",
        "grade": "A",
        "score": score,
        "signal": "LONG",
        "oi_status": "RISING",
        "price": 0.50,
        "volume_24h_usd": volume,
        "signal_timestamp": time.time(),
        "change_24h_pct": 5.0,
        "breakout_pct": 2.0,
        "body_volume_spike": 6.0,
        "volume_spike": 12.0,
        "flat_period_days": 2.0,
    }


def make_snapshot(symbol="ARKUSDT", fresh=True, supportive=True, mark_price=0.50):
    return MicrostructureSnapshot(
        binance_symbol=symbol,
        snapshot_ts=time.time() if fresh else time.time() - 30.0,
        oi_usdt=10_000_000.0,
        oi_delta_pct=3.0 if supportive else 0.0,
        funding_rate=0.005,
        mark_price=mark_price,
        next_funding_ts=time.time() + 3600.0,
        best_bid=mark_price - 0.0001,
        best_ask=mark_price + 0.0001,
        spread_abs=0.0002,
        spread_pct=0.0004,
        bid_depth_usdt=500_000.0,
        ask_depth_usdt=200_000.0 if supportive else 500_000.0,
        ob_imbalance=0.3 if supportive else 0.0,
        trade_count=500,
        traded_vol_usdt=2_000_000.0,
        buy_vol_usdt=1_200_000.0 if supportive else 500_000.0,
        sell_vol_usdt=800_000.0 if supportive else 500_000.0,
        trade_imbalance=0.25 if supportive else 0.0,
        used_weight_1m=50,
    )


# ── Metadata mocks ──────────────────────────────────────────────────────────

ARK_BYBIT_META = {
    "symbol": "ARKUSDT",
    "contractType": "LinearPerpetual",
    "status": "Trading",
    "quoteCoin": "USDT",
    "settleCoin": "USDT",
    "symbolType": "",
    "marketRegion": "",
    "underlyingTicker": "",
    "lotSizeFilter": {"qtyStep": "1", "minOrderQty": "1", "maxOrderQty": "100000"},
    "priceFilter": {"tickSize": "0.0001"},
}

ARK_BINANCE_META = {
    "symbol": "ARKUSDT",
    "contractType": "PERPETUAL",
    "underlyingType": "COIN",
    "quoteAsset": "USDT",
    "status": "TRADING",
}

ADBE_BYBIT_META = {
    "symbol": "ADBEUSDT",
    "contractType": "LinearPerpetual",
    "status": "Trading",
    "quoteCoin": "USDT",
    "settleCoin": "USDT",
    "symbolType": "stock",
    "marketRegion": "US",
    "underlyingTicker": "ADBE",
    "lotSizeFilter": {"qtyStep": "0.01", "minOrderQty": "0.01", "maxOrderQty": "1000"},
    "priceFilter": {"tickSize": "0.01"},
}

ADBE_BINANCE_META = {
    "symbol": "ADBEUSDT",
    "contractType": "TRADIFI_PERPETUAL",
    "underlyingType": "EQUITY",
    "quoteAsset": "USDT",
    "status": "TRADING",
    "subType": ["TradFi"],
}


def build_mock_bybit_client():
    client = AsyncMock(spec=BybitDemoClient)
    client.get_position.return_value = None

    order_mock = AsyncMock()
    order_mock.order_id = "demo_order_ark_123"
    client.place_market_order.return_value = order_mock

    async def get_instrument_info_side_effect(symbol):
        if symbol == "ARKUSDT":
            return ARK_BYBIT_META
        elif symbol == "ADBEUSDT":
            return ADBE_BYBIT_META
        raise BybitDemoConfigError(f"Instrument {symbol} not found or not trading.")

    client.get_instrument_info = AsyncMock(side_effect=get_instrument_info_side_effect)
    return client


# ── Test A: Valid Crypto Perpetual ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_valid_crypto_perpetual_reaches_risk_engine_and_executes():
    """ARKUSDT -> eligible -> valid ATR -> can proceed past RiskEngine."""
    client = build_mock_bybit_client()
    validator = InstrumentValidator(
        bybit_client=client,
        binance_cache={"ARKUSDT": ARK_BINANCE_META},
    )

    # 1. Direct validation check
    is_eligible, reason = await validator.validate_instrument("ARKUSDT")
    assert is_eligible is True
    assert reason == "OK"

    # 2. ATR fetcher instance invocation check
    fetcher = ATRFetcher()
    fake_atr = ATRResult(
        symbol="ARKUSDT",
        interval="1m",
        atr=0.015,
        fetch_ts=time.time(),
        candles_used=14,
    )
    with patch.object(fetcher, "fetch_atr", AsyncMock(return_value=fake_atr)) as mock_fetch:
        # 3. RiskEngine calculate check
        risk_engine = RiskEngine(RiskConfig())
        risk_res = risk_engine.calculate(
            entry_price=0.50,
            atr_result=fake_atr,
            spread_abs=0.0002,
        )
        assert risk_res.valid is True
        assert risk_res.stop_loss < 0.50
        assert risk_res.take_profit > 0.50
        assert risk_res.quantity > 0

        # 4. End-to-end execution through TradeExecutor
        executor = TradeExecutor(
            bybit_client=client,
            atr_fetcher=fetcher,
            instrument_validator=validator,
        )

        with patch("coilbrain.executor.fetch_all", AsyncMock(return_value=make_snapshot("ARKUSDT", supportive=True))):
            rec = SignalRecord(
                signal_id="sig_ark_1",
                received_timestamp=time.time(),
                transport_latency_ms=0,
                **make_payload(symbol="ARKUSDT", signal_id="sig_ark_1"),
            )
            await executor.execute(rec)

        # Verified: fetch_atr was called with client and 'ARKUSDT' (instance call, no argument shift)
        mock_fetch.assert_called_once()
        args, _ = mock_fetch.call_args
        assert isinstance(args[0], httpx.AsyncClient)
        assert args[1] == "ARKUSDT"

        # Verified: proceed past RiskEngine and market order placed
        client.place_market_order.assert_called_once()
        order_args = client.place_market_order.call_args[1]
        assert order_args["symbol"] == "ARKUSDT"
        assert order_args["side"] == "Buy"

        # Journal decision was TRADE
        async with database.get_db() as db:
            async with db.execute("SELECT decision, reasons_json FROM decision_journal WHERE signal_id='sig_ark_1'") as cur:
                row = await cur.fetchone()
                assert row is not None
                assert row[0] == "TRADE"


# ── Test B: TradFi Equity Synthetic ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_b_tradfi_equity_rejected_unsupported_instrument():
    """ADBEUSDT -> UNSUPPORTED_INSTRUMENT -> zero market orders."""
    client = build_mock_bybit_client()
    validator = InstrumentValidator(
        bybit_client=client,
        binance_cache={"ADBEUSDT": ADBE_BINANCE_META},
    )

    # 1. Direct validation check
    is_eligible, reason = await validator.validate_instrument("ADBEUSDT")
    assert is_eligible is False
    assert reason == Reason.UNSUPPORTED_INSTRUMENT

    # 2. TradeExecutor end-to-end execution
    fetcher = ATRFetcher()
    executor = TradeExecutor(
        bybit_client=client,
        atr_fetcher=fetcher,
        instrument_validator=validator,
    )

    with patch("coilbrain.executor.fetch_all", AsyncMock(return_value=make_snapshot("ADBEUSDT", supportive=True, mark_price=500.0))):
        rec = SignalRecord(
            signal_id="sig_adbe_1",
            received_timestamp=time.time(),
            transport_latency_ms=0,
            **make_payload(symbol="ADBEUSDT", signal_id="sig_adbe_1"),
        )
        await executor.execute(rec)

    # Verified: zero market orders placed
    client.place_market_order.assert_not_called()

    # Decision journal reflects IGNORE + UNSUPPORTED_INSTRUMENT
    async with database.get_db() as db:
        async with db.execute("SELECT decision, reasons_json FROM decision_journal WHERE signal_id='sig_adbe_1'") as cur:
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == "IGNORE"
            reasons = json.loads(row[1])
            assert Reason.UNSUPPORTED_INSTRUMENT in reasons


# ── Test C: Unknown Instrument Metadata ─────────────────────────────────────

@pytest.mark.asyncio
async def test_c_unknown_instrument_fails_closed():
    """Unknown instrument metadata -> UNSUPPORTED_INSTRUMENT -> zero market orders."""
    client = build_mock_bybit_client()
    validator = InstrumentValidator(
        bybit_client=client,
        binance_cache={},  # Empty cache, symbol not found
    )

    # 1. Direct validation fails closed
    is_eligible, reason = await validator.validate_instrument("UNKNOWNUSDT")
    assert is_eligible is False
    assert reason == Reason.UNSUPPORTED_INSTRUMENT

    # 2. TradeExecutor end-to-end execution
    fetcher = ATRFetcher()
    executor = TradeExecutor(
        bybit_client=client,
        atr_fetcher=fetcher,
        instrument_validator=validator,
    )

    with patch("coilbrain.executor.fetch_all", AsyncMock(return_value=make_snapshot("UNKNOWNUSDT", supportive=True))):
        rec = SignalRecord(
            signal_id="sig_unknown_1",
            received_timestamp=time.time(),
            transport_latency_ms=0,
            **make_payload(symbol="UNKNOWNUSDT", signal_id="sig_unknown_1"),
        )
        await executor.execute(rec)

    # Verified: zero market orders placed
    client.place_market_order.assert_not_called()

    # Decision journal reflects IGNORE + UNSUPPORTED_INSTRUMENT
    async with database.get_db() as db:
        async with db.execute("SELECT decision, reasons_json FROM decision_journal WHERE signal_id='sig_unknown_1'") as cur:
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == "IGNORE"
            reasons = json.loads(row[1])
            assert Reason.UNSUPPORTED_INSTRUMENT in reasons


# ── Test D: Invalid ATR ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_d_invalid_atr_results_in_risk_invalid_zero_orders():
    """Invalid ATR -> RISK_INVALID -> zero market orders."""
    client = build_mock_bybit_client()
    validator = InstrumentValidator(
        bybit_client=client,
        binance_cache={"ARKUSDT": ARK_BINANCE_META},
    )

    fetcher = ATRFetcher()
    # Mock ATRFetcher to return None (ATR genuinely unavailable or invalid)
    with patch.object(fetcher, "fetch_atr", AsyncMock(return_value=None)) as mock_fetch:
        executor = TradeExecutor(
            bybit_client=client,
            atr_fetcher=fetcher,
            instrument_validator=validator,
        )

        with patch("coilbrain.executor.fetch_all", AsyncMock(return_value=make_snapshot("ARKUSDT", supportive=True))):
            rec = SignalRecord(
                signal_id="sig_ark_invalid_atr",
                received_timestamp=time.time(),
                transport_latency_ms=0,
                **make_payload(symbol="ARKUSDT", signal_id="sig_ark_invalid_atr"),
            )
            await executor.execute(rec)

        mock_fetch.assert_called_once()

    # Verified: zero market orders placed
    client.place_market_order.assert_not_called()

    # Decision journal reflects IGNORE + RISK_INVALID
    async with database.get_db() as db:
        async with db.execute("SELECT decision, reasons_json FROM decision_journal WHERE signal_id='sig_ark_invalid_atr'") as cur:
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == "IGNORE"
            reasons = json.loads(row[1])
            assert "RISK_INVALID" in reasons


# ── Test E: Existing Stale-Data Gate ────────────────────────────────────────

@pytest.mark.asyncio
async def test_e_stale_data_gate_still_works():
    """Existing stale-data gate still works (snapshot age > 10s -> STALE_MARKET_DATA)."""
    client = build_mock_bybit_client()
    validator = InstrumentValidator(
        bybit_client=client,
        binance_cache={"ARKUSDT": ARK_BINANCE_META},
    )
    fetcher = ATRFetcher()
    executor = TradeExecutor(
        bybit_client=client,
        atr_fetcher=fetcher,
        instrument_validator=validator,
    )

    # Stale snapshot: 30 seconds old (> 10s max age threshold)
    stale_snapshot = make_snapshot("ARKUSDT", fresh=False, supportive=True)
    with patch("coilbrain.executor.fetch_all", AsyncMock(return_value=stale_snapshot)):
        rec = SignalRecord(
            signal_id="sig_stale_1",
            received_timestamp=time.time(),
            transport_latency_ms=0,
            **make_payload(symbol="ARKUSDT", signal_id="sig_stale_1"),
        )
        await executor.execute(rec)

    # Verified: zero market orders placed
    client.place_market_order.assert_not_called()

    # Decision journal reflects IGNORE + STALE_MARKET_DATA
    async with database.get_db() as db:
        async with db.execute("SELECT decision, reasons_json FROM decision_journal WHERE signal_id='sig_stale_1'") as cur:
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == "IGNORE"
            reasons = json.loads(row[1])
            assert Reason.STALE_MARKET_DATA in reasons


# ── Regression Test: ATRFetcher Call Signatures & Instance Method ───────────

@pytest.mark.asyncio
async def test_atr_fetcher_instance_method_signature_regression():
    """Regression test: ATRFetcher.fetch_atr is an instance method and does not swallow AttributeError."""
    fetcher = ATRFetcher()

    # Verify fetch_atr is an instance method
    import inspect
    assert inspect.iscoroutinefunction(fetcher.fetch_atr)

    # Verify calling it on instance with httpx client and symbol does not shift args
    fake_kline_row = [
        1600000000000, "0.50", "0.52", "0.49", "0.51", "1000",
        1600000059999, "500", 100, "50", "25", "0"
    ]
    # 17 rows -> 16 candles
    mock_klines = [fake_kline_row] * 17

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_klines
    mock_resp.raise_for_status = MagicMock()
    mock_client.get.return_value = mock_resp

    res = await fetcher.fetch_atr(mock_client, "ARKUSDT")
    assert res is not None
    assert res.symbol == "ARKUSDT"
    assert res.atr > 0
    assert res.candles_used == 14

    # Verify mock_client.get was called with symbol="ARKUSDT"
    mock_client.get.assert_called_once()
    params = mock_client.get.call_args[1]["params"]
    assert params["symbol"] == "ARKUSDT"
    assert params["interval"] == "1m"
