"""CoilBrain ATR Calculator — Real Volatility for Risk Engine.

Fetches recent 1-minute Binance USDM Futures klines and computes ATR(14)
(Average True Range over 14 periods on the 1m timeframe).

Design
------
* Completely separate from the decision engine and the per-signal
  microstructure recorder.
* Async; uses httpx.  Can be called just before risk calculation.
* Fail-closed: if ATR is unavailable or stale, returns None.
  The caller (RiskEngine) must treat None as a hard gate.
* The fetch is lightweight: 15 kline rows = negligible payload.
* Binance endpoint: GET /fapi/v1/klines (USDM Futures, public, weight 1).

ATR(14) Definition
------------------
  True Range_i = max(
      high_i - low_i,
      |high_i - close_{i-1}|,
      |low_i  - close_{i-1}|,
  )

  ATR(14) = SMA(True Range, 14)     ← simple average for V1 transparency
            (Wilder's EMA is a minor refinement; SMA is correct for V1)

Usage
-----
    fetcher = ATRFetcher()
    atr = await fetcher.fetch_atr(client, "TAKEUSDT")
    if atr is None:
        # fail-closed — cannot size position
        ...
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

from coilbrain.config import settings

logger = logging.getLogger(__name__)

# Binance USDM klines endpoint
_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"

# We need 15 candles to produce 14 True Ranges.
# (TR_1 requires close_{0}, so we need candle[0] through candle[14].)
_KLINE_LIMIT = 16     # a small safety buffer
_ATR_PERIOD = 14      # canonical ATR period


@dataclass
class ATRResult:
    """Result of an ATR fetch."""
    symbol: str
    interval: str           # "1m"
    atr: float              # ATR(14) in price units
    fetch_ts: float         # Unix time of fetch
    candles_used: int       # number of candles used in calculation
    source: str = "binance_1m"


class ATRFetcher:
    """Fetches 1m klines and computes ATR(14).

    Designed to be called once per signal on the critical path.
    Total round-trip is one lightweight REST call (~100ms typical).
    """

    def __init__(
        self,
        klines_url: str = _KLINES_URL,
        http_timeout_s: float | None = None,
        max_atr_age_s: float = 30.0,    # reject ATR older than this
    ) -> None:
        self._url = klines_url
        self._timeout = http_timeout_s or settings.ms_http_timeout_s
        self._max_age_s = max_atr_age_s

    async def fetch_atr(
        self,
        client: httpx.AsyncClient,
        binance_symbol: str,
        period: int = _ATR_PERIOD,
    ) -> ATRResult | None:
        """Fetch recent 1m klines and return ATR(period).

        Returns None on any failure — caller must fail-closed.
        """
        try:
            from coilbrain.microstructure_fetcher import normalize_binance_symbol
            norm_symbol = normalize_binance_symbol(binance_symbol)
            if not norm_symbol:
                logger.warning("ATR fetch: unable to normalize symbol %s", binance_symbol)
                return None

            fetch_start = time.time()
            resp = await client.get(
                self._url,
                params={
                    "symbol": norm_symbol,
                    "interval": "1m",
                    "limit": period + 2,      # +2: need one extra prev-close + buffer
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            klines = resp.json()

            if not klines or len(klines) < period + 1:
                logger.warning(
                    "ATR fetch: insufficient klines for %s (got %d, need %d)",
                    norm_symbol, len(klines), period + 1,
                )
                return None

            # Kline columns: [open_ts, open, high, low, close, vol, ...]
            atr = self._calculate_atr(klines, period)
            if atr is None or atr <= 0:
                logger.warning("ATR calculation returned invalid value for %s: %s", norm_symbol, atr)
                return None

            fetch_ts = time.time()
            logger.debug(
                "ATR(%d) for %s = %.8f (fetch %.0fms)",
                period, norm_symbol, atr,
                (fetch_ts - fetch_start) * 1000,
            )
            return ATRResult(
                symbol=norm_symbol,
                interval="1m",
                atr=atr,
                fetch_ts=fetch_ts,
                candles_used=period,
            )

        except Exception as exc:
            logger.warning("ATR fetch failed for %s: %s", binance_symbol, exc)
            return None

    def is_fresh(self, result: ATRResult | None, now: float | None = None) -> bool:
        """Return True if the ATR result is fresh enough to use."""
        if result is None:
            return False
        age = (now or time.time()) - result.fetch_ts
        return age <= self._max_age_s

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _calculate_atr(klines: list, period: int) -> float | None:
        """Compute ATR(period) using simple average of True Range.

        True Range = max(H-L, |H - prev_C|, |L - prev_C|)
        ATR(N) = mean(TR_{1..N})

        V1 uses simple average (SMA) for transparency.
        Wilder's EMA smoothing is a minor refinement left for V2.
        """
        if len(klines) < period + 1:
            return None

        true_ranges: list[float] = []
        # Use the last (period+1) candles to produce period TR values.
        # klines[-1] is the most recent (possibly incomplete) candle — skip it.
        # Use klines[-(period+2) : -1] to get period+1 completed candles.
        recent = klines[-(period + 2) : -1]      # period+1 completed candles

        if len(recent) < period + 1:
            # Fallback: use whatever we have from start
            recent = klines[: period + 1]
        if len(recent) < 2:
            return None

        for i in range(1, len(recent)):
            try:
                high = float(recent[i][2])
                low = float(recent[i][3])
                prev_close = float(recent[i - 1][4])
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close),
                )
                true_ranges.append(tr)
            except (IndexError, ValueError, TypeError) as exc:
                logger.warning("ATR kline parse error at index %d: %s", i, exc)
                return None

        if not true_ranges:
            return None

        # Use the last `period` TR values to match ATR(14) semantics
        tr_window = true_ranges[-period:]
        return sum(tr_window) / len(tr_window)
