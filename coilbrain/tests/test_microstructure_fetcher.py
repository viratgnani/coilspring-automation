"""Tests for microstructure_fetcher.py.

All HTTP calls are mocked — no real Binance requests are made.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from coilbrain.microstructure_fetcher import (
    BookData,
    FundingData,
    MicrostructureSnapshot,
    OIData,
    TradesData,
    check_rate_limit,
    fetch_aggr_trades,
    fetch_all,
    fetch_book,
    fetch_funding,
    fetch_oi,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(data: dict | list, status: int = 200, headers: dict | None = None) -> MagicMock:
    """Return a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.json.return_value = data
    resp.headers = headers or {}
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1. fetch_oi
# ---------------------------------------------------------------------------

class TestFetchOI:
    def test_success(self):
        resp = _mock_response({"openInterest": "28213.400", "symbol": "BTCUSDT"})
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)

        result = _run(fetch_oi(client, "BTCUSDT"))

        assert isinstance(result, OIData)
        assert result.oi_contracts == pytest.approx(28213.4)

    def test_http_error_returns_none(self):
        resp = _mock_response({}, status=500)
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)

        result = _run(fetch_oi(client, "BTCUSDT"))
        assert result is None

    def test_network_error_returns_none(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        result = _run(fetch_oi(client, "BTCUSDT"))
        assert result is None


# ---------------------------------------------------------------------------
# 2. fetch_funding
# ---------------------------------------------------------------------------

class TestFetchFunding:
    _PAYLOAD = {
        "symbol": "BTCUSDT",
        "markPrice": "65000.00",
        "lastFundingRate": "0.0001",
        "nextFundingTime": 1700000000000,  # ms
    }

    def test_success(self):
        resp = _mock_response(self._PAYLOAD)
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)

        result = _run(fetch_funding(client, "BTCUSDT"))

        assert isinstance(result, FundingData)
        assert result.mark_price == pytest.approx(65000.0)
        assert result.funding_rate == pytest.approx(0.0001)
        assert result.next_funding_ts == pytest.approx(1700000000.0)  # ms -> s

    def test_http_error_returns_none(self):
        resp = _mock_response({}, status=429)
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)

        assert _run(fetch_funding(client, "BTCUSDT")) is None

    def test_network_error_returns_none(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(side_effect=TimeoutError("timed out"))

        assert _run(fetch_funding(client, "BTCUSDT")) is None


# ---------------------------------------------------------------------------
# 3. fetch_book
# ---------------------------------------------------------------------------

class TestFetchBook:
    _PAYLOAD = {
        "bids": [["65000.00", "1.5"], ["64999.00", "2.0"]],
        "asks": [["65001.00", "1.0"], ["65002.00", "0.5"]],
    }

    def test_success_metrics(self):
        resp = _mock_response(self._PAYLOAD)
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)

        result = _run(fetch_book(client, "BTCUSDT"))

        assert isinstance(result, BookData)
        assert result.best_bid == pytest.approx(65000.0)
        assert result.best_ask == pytest.approx(65001.0)
        assert result.spread_abs == pytest.approx(1.0)
        # mid = 65000.5 ; spread_pct = 1/65000.5*100
        assert result.spread_pct == pytest.approx(1.0 / 65000.5 * 100.0, rel=1e-4)

    def test_bid_depth_calculation(self):
        resp = _mock_response(self._PAYLOAD)
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)

        result = _run(fetch_book(client, "BTCUSDT"))

        expected_bid_depth = 65000.0 * 1.5 + 64999.0 * 2.0
        assert result.bid_depth_usdt == pytest.approx(expected_bid_depth)

    def test_ask_depth_calculation(self):
        resp = _mock_response(self._PAYLOAD)
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)

        result = _run(fetch_book(client, "BTCUSDT"))

        expected_ask_depth = 65001.0 * 1.0 + 65002.0 * 0.5
        assert result.ask_depth_usdt == pytest.approx(expected_ask_depth)

    def test_ob_imbalance_range(self):
        resp = _mock_response(self._PAYLOAD)
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)

        result = _run(fetch_book(client, "BTCUSDT"))

        assert -1.0 <= result.ob_imbalance <= 1.0

    def test_empty_bids_returns_none(self):
        resp = _mock_response({"bids": [], "asks": [["65001.00", "1.0"]]})
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)

        assert _run(fetch_book(client, "BTCUSDT")) is None

    def test_http_error_returns_none(self):
        resp = _mock_response({}, status=400)
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)

        assert _run(fetch_book(client, "BTCUSDT")) is None


# ---------------------------------------------------------------------------
# 4. fetch_aggr_trades
# ---------------------------------------------------------------------------

class TestFetchAggrTrades:
    # m=True → maker is buyer → aggressive sell
    _TRADES = [
        {"p": "100.0", "q": "2.0", "m": False},  # aggressive buy: 200 USD
        {"p": "100.0", "q": "1.0", "m": True},   # aggressive sell: 100 USD
    ]

    def test_buy_sell_split(self):
        resp = _mock_response(self._TRADES)
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)

        result = _run(fetch_aggr_trades(client, "BTCUSDT", since_ts=1.0))

        assert isinstance(result, TradesData)
        assert result.trade_count == 2
        assert result.buy_vol_usdt == pytest.approx(200.0)
        assert result.sell_vol_usdt == pytest.approx(100.0)
        assert result.traded_vol_usdt == pytest.approx(300.0)

    def test_imbalance_positive_when_more_buys(self):
        resp = _mock_response(self._TRADES)
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)

        result = _run(fetch_aggr_trades(client, "BTCUSDT", since_ts=1.0))

        # (200 - 100) / 300 ≈ 0.333
        assert result.trade_imbalance == pytest.approx(100.0 / 300.0, rel=1e-4)

    def test_empty_trades_returns_zeros(self):
        resp = _mock_response([])
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)

        result = _run(fetch_aggr_trades(client, "BTCUSDT", since_ts=1.0))

        assert result.trade_count == 0
        assert result.trade_imbalance == 0.0

    def test_http_error_returns_none(self):
        resp = _mock_response({}, status=500)
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)

        assert _run(fetch_aggr_trades(client, "BTCUSDT", since_ts=1.0)) is None


# ---------------------------------------------------------------------------
# 5. fetch_all
# ---------------------------------------------------------------------------

OI_RESP = {"openInterest": "100.0", "symbol": "BTCUSDT"}
FUND_RESP = {
    "symbol": "BTCUSDT", "markPrice": "65000.0",
    "lastFundingRate": "0.0001", "nextFundingTime": 1700000000000,
}
BOOK_RESP = {
    "bids": [["65000.0", "1.0"]], "asks": [["65001.0", "1.0"]],
}


class TestFetchAll:
    def _client_returning(self, responses: list) -> AsyncMock:
        """Return a mock client whose .get() cycles through responses."""
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(side_effect=[_mock_response(r) for r in responses])
        return client

    def test_oi_usdt_computed_from_oi_and_mark_price(self):
        client = self._client_returning([OI_RESP, FUND_RESP, BOOK_RESP])
        with patch("coilbrain.microstructure_fetcher.settings") as ms:
            ms.ms_enable_oi = True
            ms.ms_enable_funding = True
            ms.ms_enable_book = True
            ms.ms_enable_trades = False
            ms.ms_book_depth_limit = 5
            ms.ms_binance_oi_url = "u"
            ms.ms_binance_premium_url = "u"
            ms.ms_binance_depth_url = "u"
            ms.ms_http_timeout_s = 5.0
            ms.ms_rate_limit_warn_pct = 0.8

            snap = _run(fetch_all(client, "BTCUSDT", since_ts=1.0))

        # 100.0 contracts × 65000.0 mark = 6_500_000 USD
        assert snap.oi_usdt == pytest.approx(6_500_000.0)

    def test_oi_delta_pct_computed_from_prev(self):
        client = self._client_returning([OI_RESP, FUND_RESP, BOOK_RESP])
        with patch("coilbrain.microstructure_fetcher.settings") as ms:
            ms.ms_enable_oi = True
            ms.ms_enable_funding = True
            ms.ms_enable_book = True
            ms.ms_enable_trades = False
            ms.ms_book_depth_limit = 5
            ms.ms_binance_oi_url = "u"
            ms.ms_binance_premium_url = "u"
            ms.ms_binance_depth_url = "u"
            ms.ms_http_timeout_s = 5.0
            ms.ms_rate_limit_warn_pct = 0.8

            snap = _run(fetch_all(client, "BTCUSDT", since_ts=1.0, prev_oi_usdt=5_000_000.0))

        # (6_500_000 - 5_000_000) / 5_000_000 * 100 = 30%
        assert snap.oi_delta_pct == pytest.approx(30.0)

    def test_oi_delta_none_when_no_prev(self):
        client = self._client_returning([OI_RESP, FUND_RESP, BOOK_RESP])
        with patch("coilbrain.microstructure_fetcher.settings") as ms:
            ms.ms_enable_oi = True
            ms.ms_enable_funding = True
            ms.ms_enable_book = True
            ms.ms_enable_trades = False
            ms.ms_book_depth_limit = 5
            ms.ms_binance_oi_url = "u"
            ms.ms_binance_premium_url = "u"
            ms.ms_binance_depth_url = "u"
            ms.ms_http_timeout_s = 5.0
            ms.ms_rate_limit_warn_pct = 0.8

            snap = _run(fetch_all(client, "BTCUSDT", since_ts=1.0, prev_oi_usdt=None))

        assert snap.oi_delta_pct is None

    def test_partial_failure_does_not_prevent_other_data(self):
        """If OI fetch fails, funding and book data should still be present."""
        oi_error_resp = _mock_response({}, status=500)
        fund_resp = _mock_response(FUND_RESP)
        book_resp = _mock_response(BOOK_RESP)

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(side_effect=[oi_error_resp, fund_resp, book_resp])

        with patch("coilbrain.microstructure_fetcher.settings") as ms:
            ms.ms_enable_oi = True
            ms.ms_enable_funding = True
            ms.ms_enable_book = True
            ms.ms_enable_trades = False
            ms.ms_book_depth_limit = 5
            ms.ms_binance_oi_url = "u"
            ms.ms_binance_premium_url = "u"
            ms.ms_binance_depth_url = "u"
            ms.ms_http_timeout_s = 5.0
            ms.ms_rate_limit_warn_pct = 0.8

            snap = _run(fetch_all(client, "BTCUSDT", since_ts=1.0))

        assert snap.oi_usdt is None
        assert snap.funding_rate is not None
        assert snap.best_bid is not None

    def test_trades_none_when_disabled(self):
        client = self._client_returning([OI_RESP, FUND_RESP, BOOK_RESP])
        with patch("coilbrain.microstructure_fetcher.settings") as ms:
            ms.ms_enable_oi = True
            ms.ms_enable_funding = True
            ms.ms_enable_book = True
            ms.ms_enable_trades = False
            ms.ms_book_depth_limit = 5
            ms.ms_binance_oi_url = "u"
            ms.ms_binance_premium_url = "u"
            ms.ms_binance_depth_url = "u"
            ms.ms_http_timeout_s = 5.0
            ms.ms_rate_limit_warn_pct = 0.8

            snap = _run(fetch_all(client, "BTCUSDT", since_ts=1.0))

        assert snap.trade_count is None
        assert snap.traded_vol_usdt is None

    def test_trades_present_when_enabled(self):
        trades_resp = [
            {"p": "100.0", "q": "2.0", "m": False},
            {"p": "100.0", "q": "1.0", "m": True},
        ]
        client = self._client_returning([OI_RESP, FUND_RESP, BOOK_RESP, trades_resp])
        with patch("coilbrain.microstructure_fetcher.settings") as ms:
            ms.ms_enable_oi = True
            ms.ms_enable_funding = True
            ms.ms_enable_book = True
            ms.ms_enable_trades = True
            ms.ms_book_depth_limit = 5
            ms.ms_binance_oi_url = "u"
            ms.ms_binance_premium_url = "u"
            ms.ms_binance_depth_url = "u"
            ms.ms_binance_aggtrades_url = "u"
            ms.ms_http_timeout_s = 5.0
            ms.ms_rate_limit_warn_pct = 0.8

            snap = _run(fetch_all(client, "BTCUSDT", since_ts=1.0))

        assert snap.trade_count == 2
        assert snap.traded_vol_usdt == pytest.approx(300.0)
        assert snap.buy_vol_usdt == pytest.approx(200.0)
        assert snap.sell_vol_usdt == pytest.approx(100.0)
        assert snap.trade_imbalance == pytest.approx(100.0 / 300.0, rel=1e-4)


    def test_returns_microstructure_snapshot(self):
        client = self._client_returning([OI_RESP, FUND_RESP, BOOK_RESP])
        with patch("coilbrain.microstructure_fetcher.settings") as ms:
            ms.ms_enable_oi = True
            ms.ms_enable_funding = True
            ms.ms_enable_book = True
            ms.ms_enable_trades = False
            ms.ms_book_depth_limit = 5
            ms.ms_binance_oi_url = "u"
            ms.ms_binance_premium_url = "u"
            ms.ms_binance_depth_url = "u"
            ms.ms_http_timeout_s = 5.0
            ms.ms_rate_limit_warn_pct = 0.8

            snap = _run(fetch_all(client, "BTCUSDT", since_ts=1.0))

        assert isinstance(snap, MicrostructureSnapshot)
        assert snap.binance_symbol == "BTCUSDT"


# ---------------------------------------------------------------------------
# 6. Rate-limit helper
# ---------------------------------------------------------------------------

class TestRateLimitCheck:
    def test_no_warning_below_threshold(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            check_rate_limit(100)   # 100/2400 = 4% — below 80%
        assert "rate-limit" not in caplog.text.lower()

    def test_warning_at_threshold(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            check_rate_limit(1920)  # 1920/2400 = 80%
        assert "rate-limit" in caplog.text.lower()

    def test_none_does_not_warn(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            check_rate_limit(None)
        assert "rate-limit" not in caplog.text.lower()


# ---------------------------------------------------------------------------
# 7. C4 — rate-limit warning wired into individual fetchers
# ---------------------------------------------------------------------------

class TestRateLimitFromFetcher:
    """Verify the C4 fix: each fetcher calls check_rate_limit when the
    Binance X-MBX-USED-WEIGHT-1M header is present at or above threshold."""

    def _resp_with_weight(self, data, weight: int) -> MagicMock:
        """Build a mock response that carries the weight header."""
        resp = _mock_response(data, headers={"X-MBX-USED-WEIGHT-1M": str(weight)})
        return resp

    def test_fetch_oi_warns_when_weight_at_threshold(self, caplog):
        """fetch_oi must trigger a rate-limit warning when header = 1920 (80%)."""
        import logging
        resp = self._resp_with_weight(
            {"openInterest": "100.0", "symbol": "BTCUSDT"}, weight=1920
        )
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)

        with caplog.at_level(logging.WARNING):
            result = _run(fetch_oi(client, "BTCUSDT"))

        assert result is not None
        assert "rate-limit" in caplog.text.lower()

    def test_fetch_oi_no_warning_when_weight_absent(self, caplog):
        """fetch_oi must not warn when the header is missing."""
        import logging
        resp = _mock_response({"openInterest": "100.0", "symbol": "BTCUSDT"})
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)

        with caplog.at_level(logging.WARNING):
            _run(fetch_oi(client, "BTCUSDT"))

        assert "rate-limit" not in caplog.text.lower()

    def test_fetch_book_warns_when_weight_at_threshold(self, caplog):
        """fetch_book must trigger a rate-limit warning when header >= threshold."""
        import logging
        book_data = {
            "bids": [["65000.0", "1.0"]],
            "asks": [["65001.0", "1.0"]],
        }
        resp = self._resp_with_weight(book_data, weight=1920)
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)

        with caplog.at_level(logging.WARNING):
            result = _run(fetch_book(client, "BTCUSDT"))

        assert result is not None
        assert "rate-limit" in caplog.text.lower()

    def test_fetch_funding_warns_when_weight_at_threshold(self, caplog):
        """fetch_funding must trigger a rate-limit warning when header >= threshold."""
        import logging
        fund_data = {
            "symbol": "BTCUSDT", "markPrice": "65000.0",
            "lastFundingRate": "0.0001", "nextFundingTime": 1700000000000,
        }
        resp = self._resp_with_weight(fund_data, weight=2000)
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)

        with caplog.at_level(logging.WARNING):
            result = _run(fetch_funding(client, "BTCUSDT"))

        assert result is not None
        assert "rate-limit" in caplog.text.lower()
