"""CoilBrain Bybit Demo API client.

SAFETY CONTRACT
---------------
* This module ONLY connects to Bybit DEMO (testnet).
* It NEVER silently falls back to production.
* API keys are read from environment variables ONLY.
* If BYBIT_DEMO_API_KEY or BYBIT_DEMO_API_SECRET are missing,
  all order methods raise BybitDemoConfigError immediately.
* The demo base URL is hardcoded — it cannot be overridden to production.

Environment Variables Required
-------------------------------
    BYBIT_DEMO_API_KEY      Bybit demo API key
    BYBIT_DEMO_API_SECRET   Bybit demo API secret

Optional:
    BYBIT_DEMO_BASE_URL     Override demo REST URL (for testing only).
                            Any non-demo URL will raise BybitDemoConfigError.

DEMO URLS
---------
    REST:       https://api-demo.bybit.com
    WebSocket:  wss://stream-demo.bybit.com  (not used in V1)

    These are the official Bybit demo endpoints.
    V1 uses REST only.

Usage
-----
    client = BybitDemoClient.from_env()
    symbol_ok = await client.validate_symbol("TAKEUSDT")
    balance = await client.get_balance()
    order = await client.place_market_order("TAKEUSDT", side="Buy", qty=100.0)
    await client.place_stop_loss(order_id, "TAKEUSDT", sl_price=0.085)
    await client.place_take_profit(order_id, "TAKEUSDT", tp_price=0.096)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

# Official Bybit Demo REST base URL
_BYBIT_DEMO_BASE_URL = "https://api-demo.bybit.com"

# Safety: reject any URL that looks like production
_FORBIDDEN_URL_FRAGMENTS = ["api.bybit.com", "mainnet", "prod"]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BybitDemoConfigError(Exception):
    """Raised when the client is misconfigured (missing keys, wrong URL)."""


class BybitDemoAPIError(Exception):
    """Raised when Bybit returns an API-level error."""
    def __init__(self, ret_code: int, ret_msg: str, extra: dict | None = None):
        self.ret_code = ret_code
        self.ret_msg = ret_msg
        self.extra = extra or {}
        super().__init__(f"Bybit API error {ret_code}: {ret_msg}")


# ---------------------------------------------------------------------------
# Response dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str
    qty: float
    order_type: str
    status: str
    avg_price: float | None
    raw: dict


@dataclass
class PositionInfo:
    symbol: str
    side: str
    size: float
    avg_price: float
    unrealised_pnl: float
    leverage: float
    raw: dict


@dataclass
class Balance:
    total_equity: float
    available_balance: float
    coin: str
    raw: dict


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class BybitDemoClient:
    """Async Bybit Demo REST client.

    Only connect via from_env() or explicit __init__ — never hardcode keys.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = _BYBIT_DEMO_BASE_URL,
        timeout_s: float = 10.0,
    ) -> None:
        # Safety: reject production-like URLs
        url_lower = base_url.lower()
        for fragment in _FORBIDDEN_URL_FRAGMENTS:
            if fragment in url_lower:
                raise BybitDemoConfigError(
                    f"Bybit Demo client refuses to connect to production-like URL: {base_url}. "
                    "Use BYBIT_DEMO_BASE_URL or default demo endpoint."
                )

        if not api_key or not api_secret:
            raise BybitDemoConfigError(
                "BYBIT_DEMO_API_KEY and BYBIT_DEMO_API_SECRET must be set. "
                "Do not use production credentials."
            )

        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_s
        self._instruments_cache: dict[str, dict] = {}

    @classmethod
    def from_env(cls) -> "BybitDemoClient":
        """Construct from environment variables.

        Raises BybitDemoConfigError if credentials are missing.
        Never reads production credentials.
        """
        key = os.environ.get("BYBIT_DEMO_API_KEY", "").strip()
        secret = os.environ.get("BYBIT_DEMO_API_SECRET", "").strip()
        base_url = os.environ.get("BYBIT_DEMO_BASE_URL", _BYBIT_DEMO_BASE_URL).strip()

        if not key or not secret:
            raise BybitDemoConfigError(
                "BYBIT_DEMO_API_KEY and BYBIT_DEMO_API_SECRET environment variables are required. "
                "Obtain demo credentials from Bybit testnet — do not use live account keys."
            )

        return cls(api_key=key, api_secret=secret, base_url=base_url)

    # ── Symbol validation ──────────────────────────────────────────────────


    async def get_instrument_info(self, symbol: str) -> dict:
        """Fetch and cache instrument metadata for step/tick rounding."""
        if symbol in self._instruments_cache:
            return self._instruments_cache[symbol]

        data = await self._get(
            "/v5/market/instruments-info",
            params={"category": "linear", "symbol": symbol},
            signed=False,
        )
        instruments = data.get("result", {}).get("list", [])
        for inst in instruments:
            if inst.get("symbol") == symbol and inst.get("status") == "Trading":
                self._instruments_cache[symbol] = inst
                return inst
        raise BybitDemoConfigError(f"Instrument {symbol} not found or not trading.")

    def round_qty(self, qty: float, info: dict) -> str:
        """Round quantity to the step size defined by lotSizeFilter."""
        lot_filter = info.get("lotSizeFilter", {})
        step = float(lot_filter.get("qtyStep", "0.001"))
        qty_rounded = int(qty / step) * step
        # Format without scientific notation, removing trailing zeros
        return f"{qty_rounded:.10f}".rstrip("0").rstrip(".")

    def round_price(self, price: float, info: dict) -> str:
        """Round price to the tick size defined by priceFilter."""
        tick = float(info.get("priceFilter", {}).get("tickSize", "0.1"))
        price_rounded = round(price / tick) * tick
        return f"{price_rounded:.10f}".rstrip("0").rstrip(".")

    async def validate_symbol(self, symbol: str) -> bool:

        """Return True if the symbol exists and is tradeable on Bybit Demo."""
        try:
            data = await self._get(
                "/v5/market/instruments-info",
                params={"category": "linear", "symbol": symbol},
                signed=False,
            )
            instruments = data.get("result", {}).get("list", [])
            for inst in instruments:
                if inst.get("symbol") == symbol and inst.get("status") == "Trading":
                    return True
            return False
        except Exception as exc:
            logger.warning("Symbol validation failed for %s: %s", symbol, exc)
            return False

    # ── Account / balance ──────────────────────────────────────────────────

    @staticmethod
    def _safe_float(val: object, default: float = 0.0) -> float:
        """Convert val to float; return default on empty string / None."""
        try:
            return float(val) if val not in (None, "", "None") else default
        except (ValueError, TypeError):
            return default

    async def get_balance(self, coin: str = "USDT") -> Balance:
        """Get unified trading account balance."""
        data = await self._get(
            "/v5/account/wallet-balance",
            params={"accountType": "UNIFIED", "coin": coin},
        )
        accounts = data.get("result", {}).get("list", [])
        for account in accounts:
            for coin_info in account.get("coin", []):
                if coin_info.get("coin") == coin:
                    return Balance(
                        total_equity=self._safe_float(coin_info.get("equity", 0)),
                        available_balance=self._safe_float(
                            coin_info.get("availableToWithdraw", 0)
                        ),
                        coin=coin,
                        raw=coin_info,
                    )
        # Fresh demo account may have no USDT row yet — return zero balance
        logger.warning(
            "Coin %s absent from balance response; returning 0 (fresh account?)", coin
        )
        return Balance(total_equity=0.0, available_balance=0.0, coin=coin, raw={})

    # ── Position retrieval ─────────────────────────────────────────────────

    async def get_position(self, symbol: str) -> PositionInfo | None:
        """Get current open position for a symbol. Returns None if flat."""
        data = await self._get(
            "/v5/position/list",
            params={"category": "linear", "symbol": symbol},
        )
        positions = data.get("result", {}).get("list", [])
        for pos in positions:
            size = float(pos.get("size", 0))
            if size > 0:
                return PositionInfo(
                    symbol=pos.get("symbol", symbol),
                    side=pos.get("side", ""),
                    size=size,
                    avg_price=float(pos.get("avgPrice", 0)),
                    unrealised_pnl=float(pos.get("unrealisedPnl", 0)),
                    leverage=float(pos.get("leverage", 1)),
                    raw=pos,
                )
        return None

    # ── Order submission ───────────────────────────────────────────────────

    async def place_market_order(
        self,
        symbol: str,
        side: str,            # "Buy" | "Sell"
        qty: float,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> OrderResult:
        """Place a market order. Returns the confirmed OrderResult or raises."""
        params: dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": self.round_qty(qty, await self.get_instrument_info(symbol)),
        }
        if reduce_only:
            params["reduceOnly"] = True
        if client_order_id:
            params["orderLinkId"] = client_order_id

        data = await self._post("/v5/order/create", params)
        result = data.get("result", {})
        return OrderResult(
            order_id=result.get("orderId", ""),
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="Market",
            status=result.get("orderStatus", ""),
            avg_price=None,   # filled via order query after fill
            raw=result,
        )

    async def get_order_status(self, symbol: str, order_id: str) -> dict:
        """Query the current status of an order."""
        data = await self._get(
            "/v5/order/realtime",
            params={"category": "linear", "symbol": symbol, "orderId": order_id},
        )
        orders = data.get("result", {}).get("list", [])
        return orders[0] if orders else {}

    async def place_stop_loss(
        self,
        symbol: str,
        sl_price: float,
        position_idx: int = 0,    # 0 = one-way, 1 = hedge long, 2 = hedge short
    ) -> dict:
        """Place/update a stop-loss on an open position."""
        params = {
            "category": "linear",
            "symbol": symbol,
            "stopLoss": self.round_price(sl_price, await self.get_instrument_info(symbol)),
            "slTriggerBy": "MarkPrice",
            "positionIdx": position_idx,
        }
        return await self._post("/v5/position/trading-stop", params)

    async def place_take_profit(
        self,
        symbol: str,
        tp_price: float,
        position_idx: int = 0,
    ) -> dict:
        """Place/update a take-profit on an open position."""
        params = {
            "category": "linear",
            "symbol": symbol,
            "takeProfit": self.round_price(tp_price, await self.get_instrument_info(symbol)),
            "tpTriggerBy": "MarkPrice",
            "positionIdx": position_idx,
        }
        return await self._post("/v5/position/trading-stop", params)

    async def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancel an open order."""
        params = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        }
        return await self._post("/v5/order/cancel", params)

    async def set_leverage(self, symbol: str, leverage: float) -> dict:
        """Set leverage for a symbol (both buy/sell side)."""
        params = {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        return await self._post("/v5/position/set-leverage", params)

    # ── Internal HTTP helpers ──────────────────────────────────────────────

    async def _get(
        self, path: str, params: dict | None = None, signed: bool = True
    ) -> dict:
        params = dict(params or {})
        if signed:
            params = self._sign_params(params)
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as c:
            resp = await c.get(path, params=params)
        resp.raise_for_status()
        return self._unwrap(resp.json())

    async def _post(self, path: str, body: dict) -> dict:
        body = self._sign_body(body)
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as c:
            resp = await c.post(path, json=body)
        resp.raise_for_status()
        return self._unwrap(resp.json())

    def _sign_params(self, params: dict) -> dict:
        """Add signature to GET query params."""
        ts = str(int(time.time() * 1000))
        params["api_key"] = self._api_key
        params["timestamp"] = ts
        params["recv_window"] = "20000"   # 20 s — covers typical clock skew
        query = urlencode(sorted(params.items()))
        sig = hmac.new(
            self._api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["sign"] = sig
        return params

    def _sign_body(self, body: dict) -> dict:
        """Add signature to POST body."""
        ts = str(int(time.time() * 1000))
        body = dict(body)
        body["api_key"] = self._api_key
        body["timestamp"] = ts
        body["recv_window"] = "20000"     # 20 s — covers typical clock skew
        # Sort for deterministic signature, formatting booleans as lowercase strings
        sorted_items = sorted([(k, str(v).lower() if isinstance(v, bool) else v) for k, v in body.items()])
        payload = urlencode(sorted_items)
        sig = hmac.new(
            self._api_secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()
        body["sign"] = sig
        return body

    @staticmethod
    def _unwrap(data: dict) -> dict:
        """Check Bybit ret_code and raise BybitDemoAPIError on failure."""
        ret_code = data.get("retCode", -1)
        if ret_code != 0:
            raise BybitDemoAPIError(
                ret_code=ret_code,
                ret_msg=data.get("retMsg", "unknown"),
                extra=data,
            )
        return data
