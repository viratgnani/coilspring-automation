"""Regression tests for CCXT symbol resolution and microstructure freshness."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from coilbrain.decision_engine import (
    Decision,
    DecisionConfig,
    DecisionEngine,
    PositionState,
    Reason,
    SignalFeatures,
)
from coilbrain.microstructure_fetcher import (
    MicrostructureSnapshot,
    fetch_all,
    fetch_funding,
    fetch_oi,
    fetch_book,
    normalize_binance_symbol,
)
from coilbrain.observatory import to_binance_symbol


class TestSymbolNormalizationRegression:
    """Regression suite covering the exact old-symbol failure modes."""

    def test_normalize_binance_symbol_ccxt_formats(self):
        """All symbols seen in live session normalize correctly to Binance REST format."""
        assert normalize_binance_symbol("MOVR/USDT:USDT") == "MOVRUSDT"
        assert normalize_binance_symbol("GTC/USDT:USDT") == "GTCUSDT"
        assert normalize_binance_symbol("SOMI/USDT:USDT") == "SOMIUSDT"
        assert normalize_binance_symbol("BTC/USDT:USDT") == "BTCUSDT"
        assert normalize_binance_symbol("1000PEPE/USDT:USDT") == "1000PEPEUSDT"
        assert normalize_binance_symbol("BTC/USDT") == "BTCUSDT"
        assert normalize_binance_symbol("BTCUSDT") == "BTCUSDT"

    def test_to_binance_symbol_single_argument_no_type_error(self):
        """to_binance_symbol can be called with 1 argument without raising TypeError."""
        result_movr = to_binance_symbol("MOVR/USDT:USDT")
        assert result_movr == "MOVRUSDT"

        result_gtc = to_binance_symbol("GTC/USDT:USDT")
        assert result_gtc == "GTCUSDT"

        result_somi = to_binance_symbol("SOMI/USDT:USDT")
        assert result_somi == "SOMIUSDT"

    def test_to_binance_symbol_backward_compatibility_with_known_set(self):
        """When known set is provided, full validation against the set is preserved."""
        known = {"MOVRUSDT", "BTCUSDT"}
        assert to_binance_symbol("MOVR/USDT:USDT", known) == "MOVRUSDT"
        assert to_binance_symbol("GTC/USDT:USDT", known) is None
        assert to_binance_symbol("MOVR/USDT:USDT", set()) is None

    @pytest.mark.asyncio
    async def test_fetch_all_requests_normalized_symbol(self):
        """fetch_all must request the normalized symbol (e.g. MOVRUSDT) even if CCXT symbol is passed."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)

        oi_resp = MagicMock()
        oi_resp.headers = {}
        oi_resp.json.return_value = {"openInterest": "10000.0"}
        oi_resp.raise_for_status = MagicMock()

        fund_resp = MagicMock()
        fund_resp.headers = {}
        fund_resp.json.return_value = {
            "symbol": "MOVRUSDT",
            "markPrice": "1.65",
            "lastFundingRate": "0.0001",
            "nextFundingTime": 1790000000000,
        }
        fund_resp.raise_for_status = MagicMock()

        book_resp = MagicMock()
        book_resp.headers = {}
        book_resp.json.return_value = {
            "bids": [["1.64", "100.0"]],
            "asks": [["1.66", "100.0"]],
        }
        book_resp.raise_for_status = MagicMock()

        # Route mocks based on URL
        async def mock_get(url, params=None, timeout=None):
            if "openInterest" in url:
                assert params["symbol"] == "MOVRUSDT", f"Requested bad symbol {params['symbol']}"
                return oi_resp
            elif "premiumIndex" in url:
                assert params["symbol"] == "MOVRUSDT", f"Requested bad symbol {params['symbol']}"
                return fund_resp
            elif "depth" in url:
                assert params["symbol"] == "MOVRUSDT", f"Requested bad symbol {params['symbol']}"
                return book_resp
            raise ValueError(f"Unexpected url {url}")

        mock_client.get = AsyncMock(side_effect=mock_get)

        with patch("coilbrain.microstructure_fetcher.settings") as s:
            s.ms_enable_oi = True
            s.ms_enable_funding = True
            s.ms_enable_book = True
            s.ms_enable_trades = False
            s.ms_book_depth_limit = 5
            s.ms_binance_oi_url = "https://fapi.binance.com/fapi/v1/openInterest"
            s.ms_binance_premium_url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            s.ms_binance_depth_url = "https://fapi.binance.com/fapi/v1/depth"
            s.ms_http_timeout_s = 5.0
            s.ms_rate_limit_warn_pct = 0.8

            snap = await fetch_all(mock_client, "MOVR/USDT:USDT", since_ts=time.time())

        assert snap.binance_symbol == "MOVRUSDT"
        assert snap.mark_price == 1.65
        assert snap.best_bid == 1.64
        assert snap.best_ask == 1.66

    def test_stale_snapshot_triggers_stale_market_data_gate(self):
        """A snapshot older than max_data_age_ms (10s) must trigger STALE_MARKET_DATA and reject."""
        engine = DecisionEngine(DecisionConfig(max_data_age_ms=10_000))
        now = time.time()

        snap_stale = MicrostructureSnapshot(
            binance_symbol="MOVRUSDT",
            snapshot_ts=now - 25.0,  # 25 seconds old -> stale
            oi_usdt=1_000_000.0,
            oi_delta_pct=1.0,
            funding_rate=0.0001,
            mark_price=1.65,
            next_funding_ts=now + 3600,
            best_bid=1.649,
            best_ask=1.651,
            spread_abs=0.002,
            spread_pct=0.12,
            bid_depth_usdt=50_000.0,
            ask_depth_usdt=40_000.0,
            ob_imbalance=0.11,
            trade_count=100,
            traded_vol_usdt=50_000.0,
            buy_vol_usdt=30_000.0,
            sell_vol_usdt=20_000.0,
            trade_imbalance=0.2,
        )

        signal = SignalFeatures(
            signal_id="sig-test",
            symbol="MOVR/USDT:USDT",
            score=95.0,
            grade="A",
            body_spike=5.0,
            volume_spike=5.0,
            volume_24h_usd=2_000_000.0,
            change_24h_pct=25.0,
            breakout_pct=5.0,
            flat_period_days=30.0,
            signal_timestamp=now,
        )
        pos = PositionState(symbol="MOVR/USDT:USDT", state="NO_POSITION")

        result = engine.decide(
            signal=signal,
            snapshot=snap_stale,
            position=pos,
            bybit_symbol="MOVRUSDT",
            fetch_start_ts=now - 0.2,
            fetch_end_ts=now,
        )

        assert result.decision == Decision.IGNORE
        assert Reason.STALE_MARKET_DATA in result.reasons

    def test_fresh_snapshot_passes_freshness_gate(self):
        """A snapshot younger than 10s passes freshness and is not rejected as stale."""
        engine = DecisionEngine(DecisionConfig(max_data_age_ms=10_000))
        now = time.time()

        snap_fresh = MicrostructureSnapshot(
            binance_symbol="MOVRUSDT",
            snapshot_ts=now - 0.5,  # 500ms old -> fresh
            oi_usdt=1_000_000.0,
            oi_delta_pct=1.0,
            funding_rate=0.0001,
            mark_price=1.65,
            next_funding_ts=now + 3600,
            best_bid=1.649,
            best_ask=1.651,
            spread_abs=0.002,
            spread_pct=0.12,
            bid_depth_usdt=50_000.0,
            ask_depth_usdt=40_000.0,
            ob_imbalance=0.11,
            trade_count=100,
            traded_vol_usdt=50_000.0,
            buy_vol_usdt=30_000.0,
            sell_vol_usdt=20_000.0,
            trade_imbalance=0.2,
        )

        signal = SignalFeatures(
            signal_id="sig-test",
            symbol="MOVR/USDT:USDT",
            score=95.0,
            grade="A",
            body_spike=5.0,
            volume_spike=5.0,
            volume_24h_usd=2_000_000.0,
            change_24h_pct=25.0,
            breakout_pct=5.0,
            flat_period_days=30.0,
            signal_timestamp=now,
        )
        pos = PositionState(symbol="MOVR/USDT:USDT", state="NO_POSITION")

        result = engine.decide(
            signal=signal,
            snapshot=snap_fresh,
            position=pos,
            bybit_symbol="MOVRUSDT",
            fetch_start_ts=now - 0.2,
            fetch_end_ts=now,
        )

        assert Reason.STALE_MARKET_DATA not in result.reasons
        assert Reason.MISSING_MARKET_DATA not in result.reasons
