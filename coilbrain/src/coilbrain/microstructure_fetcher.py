"""Binance USDM Futures microstructure data fetchers.

Each function is a thin async wrapper around one Binance REST endpoint.
All functions return a typed dataclass or None on failure; errors are
logged but never raised so that a single bad endpoint cannot kill the
entire snapshot cycle.

Enabled endpoints and their weight (Binance weight-per-minute budget: 2400):
    openInterest   weight 1
    premiumIndex   weight 1   (also provides mark price for OI-USD calculation)
    depth          weight 2   (limit <= 100)
    aggTrades      weight 20  (disabled by default: MS_ENABLE_TRADES=false)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from coilbrain.config import settings

logger = logging.getLogger(__name__)

# Binance rate-limit header (weight consumed in the last rolling minute).
_WEIGHT_HEADER = "X-MBX-USED-WEIGHT-1M"
_BINANCE_WEIGHT_LIMIT = 2400


# ---------------------------------------------------------------------------
# Typed result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OIData:
    oi_contracts: float          # raw open interest in base-asset units


@dataclass
class FundingData:
    mark_price: float
    funding_rate: float
    next_funding_ts: float       # Unix seconds (Binance gives milliseconds)


@dataclass
class BookData:
    best_bid: float
    best_ask: float
    spread_abs: float
    spread_pct: float
    bid_depth_usdt: float        # USD notional in top N bid levels
    ask_depth_usdt: float        # USD notional in top N ask levels
    ob_imbalance: float          # (bid_depth - ask_depth) / (bid_depth + ask_depth)


@dataclass
class TradesData:
    trade_count: int
    traded_vol_usdt: float
    buy_vol_usdt: float          # aggressive buyer side
    sell_vol_usdt: float         # aggressive seller side
    trade_imbalance: float       # (buy - sell) / total; 0 when total == 0


@dataclass
class MicrostructureSnapshot:
    """Combined snapshot for a single symbol at a single point in time."""

    binance_symbol: str
    snapshot_ts: float           # time.time() at fetch moment

    # OI (NULL when ms_enable_oi=false or fetch failed)
    oi_usdt: float | None        # oi_contracts × mark_price
    oi_delta_pct: float | None   # % change vs previous snapshot; filled by recorder

    # Funding / mark price (NULL when ms_enable_funding=false or fetch failed)
    funding_rate: float | None
    mark_price: float | None
    next_funding_ts: float | None

    # Order book (NULL when ms_enable_book=false or fetch failed)
    best_bid: float | None
    best_ask: float | None
    spread_abs: float | None
    spread_pct: float | None
    bid_depth_usdt: float | None
    ask_depth_usdt: float | None
    ob_imbalance: float | None

    # Aggr trades (NULL when ms_enable_trades=false or fetch failed)
    trade_count: int | None
    traded_vol_usdt: float | None
    buy_vol_usdt: float | None
    sell_vol_usdt: float | None
    trade_imbalance: float | None

    # Rate-limit diagnostic (set by fetch_all from response headers)
    used_weight_1m: int | None = field(default=None)


# ---------------------------------------------------------------------------
# Rate-limit header helper
# ---------------------------------------------------------------------------

def _extract_weight(response: httpx.Response) -> int | None:
    """Return the Binance X-MBX-USED-WEIGHT-1M header value, or None."""
    val = response.headers.get(_WEIGHT_HEADER)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return None


def check_rate_limit(used: int | None) -> None:
    """Log a warning if the used weight approaches the Binance limit."""
    if used is None:
        return
    pct = used / _BINANCE_WEIGHT_LIMIT
    if pct >= settings.ms_rate_limit_warn_pct:
        logger.warning(
            "Binance rate-limit warning: used %d / %d (%.0f%%)",
            used, _BINANCE_WEIGHT_LIMIT, pct * 100,
        )


def normalize_binance_symbol(symbol: str) -> str:
    """Normalize a symbol to Binance USDM perpetual REST format (e.g. BTCUSDT).

    Accepts:
    - CCXT unified format: "BTC/USDT:USDT" -> "BTCUSDT"
    - Raw REST format:     "BTCUSDT"       -> "BTCUSDT"
    - Spot/margin slash:   "BTC/USDT"      -> "BTCUSDT"
    """
    if not symbol:
        return ""
    if "/" in symbol:
        try:
            slash = symbol.index("/")
            colon = symbol.index(":", slash) if ":" in symbol[slash:] else len(symbol)
            base = symbol[:slash]
            quote = symbol[slash + 1:colon]
            return (base + quote).strip().upper()
        except Exception:
            return symbol.replace("/", "").replace(":", "").strip().upper()
    return symbol.strip().upper()


# ---------------------------------------------------------------------------
# Individual endpoint fetchers
# ---------------------------------------------------------------------------

async def fetch_oi(
    client: httpx.AsyncClient, binance_symbol: str
) -> OIData | None:
    """Fetch current open interest (contracts) for a USDM symbol."""
    binance_symbol = normalize_binance_symbol(binance_symbol)
    try:
        resp = await client.get(
            settings.ms_binance_oi_url,
            params={"symbol": binance_symbol},
            timeout=settings.ms_http_timeout_s,
        )
        resp.raise_for_status()
        check_rate_limit(_extract_weight(resp))
        data = resp.json()
        return OIData(oi_contracts=float(data["openInterest"]))
    except Exception as exc:
        logger.warning("OI fetch failed for %s: %s", binance_symbol, exc)
        return None


async def fetch_funding(
    client: httpx.AsyncClient, binance_symbol: str
) -> FundingData | None:
    """Fetch premiumIndex (mark price, funding rate, next funding time)."""
    binance_symbol = normalize_binance_symbol(binance_symbol)
    try:
        resp = await client.get(
            settings.ms_binance_premium_url,
            params={"symbol": binance_symbol},
            timeout=settings.ms_http_timeout_s,
        )
        resp.raise_for_status()
        check_rate_limit(_extract_weight(resp))
        data = resp.json()
        return FundingData(
            mark_price=float(data["markPrice"]),
            funding_rate=float(data["lastFundingRate"]),
            next_funding_ts=float(data["nextFundingTime"]) / 1000.0,  # ms -> s
        )
    except Exception as exc:
        logger.warning("Funding fetch failed for %s: %s", binance_symbol, exc)
        return None


async def fetch_book(
    client: httpx.AsyncClient,
    binance_symbol: str,
    depth_limit: int | None = None,
) -> BookData | None:
    """Fetch order book depth and compute spread/imbalance metrics."""
    binance_symbol = normalize_binance_symbol(binance_symbol)
    limit = depth_limit if depth_limit is not None else settings.ms_book_depth_limit
    try:
        resp = await client.get(
            settings.ms_binance_depth_url,
            params={"symbol": binance_symbol, "limit": limit},
            timeout=settings.ms_http_timeout_s,
        )
        resp.raise_for_status()
        check_rate_limit(_extract_weight(resp))
        data = resp.json()

        bids: list[list[str]] = data.get("bids", [])
        asks: list[list[str]] = data.get("asks", [])

        if not bids or not asks:
            logger.warning("Empty order book for %s", binance_symbol)
            return None

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2.0
        spread_abs = best_ask - best_bid
        spread_pct = (spread_abs / mid * 100.0) if mid > 0 else 0.0

        bid_depth = sum(float(p) * float(q) for p, q in bids)
        ask_depth = sum(float(p) * float(q) for p, q in asks)
        total_depth = bid_depth + ask_depth
        ob_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0

        return BookData(
            best_bid=best_bid,
            best_ask=best_ask,
            spread_abs=spread_abs,
            spread_pct=spread_pct,
            bid_depth_usdt=bid_depth,
            ask_depth_usdt=ask_depth,
            ob_imbalance=ob_imbalance,
        )
    except Exception as exc:
        logger.warning("Book fetch failed for %s: %s", binance_symbol, exc)
        return None


async def fetch_aggr_trades(
    client: httpx.AsyncClient,
    binance_symbol: str,
    since_ts: float,          # Unix seconds — fetch trades since this timestamp
    limit: int = 500,
) -> TradesData | None:
    """Fetch aggregated trades since ``since_ts`` and compute buy/sell volumes."""
    binance_symbol = normalize_binance_symbol(binance_symbol)
    try:
        start_ms = int(since_ts * 1000)
        resp = await client.get(
            settings.ms_binance_aggtrades_url,
            params={"symbol": binance_symbol, "startTime": start_ms, "limit": limit},
            timeout=settings.ms_http_timeout_s,
        )
        resp.raise_for_status()
        check_rate_limit(_extract_weight(resp))
        trades: list[dict[str, Any]] = resp.json()

        if not trades:
            return TradesData(
                trade_count=0, traded_vol_usdt=0.0,
                buy_vol_usdt=0.0, sell_vol_usdt=0.0, trade_imbalance=0.0,
            )

        buy_vol = 0.0
        sell_vol = 0.0
        for t in trades:
            notional = float(t["p"]) * float(t["q"])
            # m=True → maker is buyer → aggressive seller (market sell)
            if t["m"]:
                sell_vol += notional
            else:
                buy_vol += notional

        total = buy_vol + sell_vol
        imbalance = (buy_vol - sell_vol) / total if total > 0 else 0.0

        return TradesData(
            trade_count=len(trades),
            traded_vol_usdt=total,
            buy_vol_usdt=buy_vol,
            sell_vol_usdt=sell_vol,
            trade_imbalance=imbalance,
        )
    except Exception as exc:
        logger.warning("AggTrades fetch failed for %s: %s", binance_symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Combined fetcher (runs all enabled endpoints concurrently)
# ---------------------------------------------------------------------------

async def _none() -> None:
    """Placeholder coroutine for disabled endpoints."""
    return None


async def fetch_all(
    client: httpx.AsyncClient,
    binance_symbol: str,
    since_ts: float,
    prev_oi_usdt: float | None = None,
) -> MicrostructureSnapshot:
    """Fetch all enabled microstructure data concurrently.

    Uses asyncio.gather so OI, funding, and book all fire in parallel.
    Any individual failure returns None for that data type without
    affecting the others.

    ``prev_oi_usdt`` is used to compute oi_delta_pct; pass None on the
    first tick for a symbol.
    """
    binance_symbol = normalize_binance_symbol(binance_symbol)
    snapshot_ts = time.time()

    oi_coro = fetch_oi(client, binance_symbol) if settings.ms_enable_oi else _none()
    fund_coro = fetch_funding(client, binance_symbol) if settings.ms_enable_funding else _none()
    book_coro = fetch_book(client, binance_symbol) if settings.ms_enable_book else _none()
    trades_coro = fetch_aggr_trades(client, binance_symbol, since_ts) if settings.ms_enable_trades else _none()

    oi, funding, book, trades = await asyncio.gather(
        oi_coro, fund_coro, book_coro, trades_coro,
        return_exceptions=True,
    )

    # Treat exceptions from gather as None (already logged inside each fetcher,
    # but gather(return_exceptions=True) wraps unexpected errors as well).
    if isinstance(oi, Exception):
        logger.warning("Unexpected OI error for %s: %s", binance_symbol, oi)
        oi = None
    if isinstance(funding, Exception):
        logger.warning("Unexpected funding error for %s: %s", binance_symbol, funding)
        funding = None
    if isinstance(book, Exception):
        logger.warning("Unexpected book error for %s: %s", binance_symbol, book)
        book = None
    if isinstance(trades, Exception):
        logger.warning("Unexpected trades error for %s: %s", binance_symbol, trades)
        trades = None

    # Compute OI in USD using mark price from the funding response.
    oi_usdt: float | None = None
    if isinstance(oi, OIData) and isinstance(funding, FundingData):
        oi_usdt = oi.oi_contracts * funding.mark_price
    elif isinstance(oi, OIData) and funding is None:
        # Funding failed but OI succeeded; USD notional unavailable.
        oi_usdt = None

    # OI delta
    oi_delta_pct: float | None = None
    if oi_usdt is not None and prev_oi_usdt is not None and prev_oi_usdt > 0:
        oi_delta_pct = (oi_usdt - prev_oi_usdt) / prev_oi_usdt * 100.0

    return MicrostructureSnapshot(
        binance_symbol=binance_symbol,
        snapshot_ts=snapshot_ts,
        # OI
        oi_usdt=oi_usdt,
        oi_delta_pct=oi_delta_pct,
        # Funding
        funding_rate=funding.funding_rate if isinstance(funding, FundingData) else None,
        mark_price=funding.mark_price if isinstance(funding, FundingData) else None,
        next_funding_ts=funding.next_funding_ts if isinstance(funding, FundingData) else None,
        # Book
        best_bid=book.best_bid if isinstance(book, BookData) else None,
        best_ask=book.best_ask if isinstance(book, BookData) else None,
        spread_abs=book.spread_abs if isinstance(book, BookData) else None,
        spread_pct=book.spread_pct if isinstance(book, BookData) else None,
        bid_depth_usdt=book.bid_depth_usdt if isinstance(book, BookData) else None,
        ask_depth_usdt=book.ask_depth_usdt if isinstance(book, BookData) else None,
        ob_imbalance=book.ob_imbalance if isinstance(book, BookData) else None,
        # Trades
        trade_count=trades.trade_count if isinstance(trades, TradesData) else None,
        traded_vol_usdt=trades.traded_vol_usdt if isinstance(trades, TradesData) else None,
        buy_vol_usdt=trades.buy_vol_usdt if isinstance(trades, TradesData) else None,
        sell_vol_usdt=trades.sell_vol_usdt if isinstance(trades, TradesData) else None,
        trade_imbalance=trades.trade_imbalance if isinstance(trades, TradesData) else None,
    )
