"""Shared Pydantic request/response models used across the API."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Infrastructure ─────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    service: str


# ── Signal contract ────────────────────────────────────────────────────────────


class SignalPayload(BaseModel):
    """Validated body for POST /signals.

    Field availability notes (Alpha Coiler source)
    -----------------------------------------------
    symbol            result.symbol
    exchange          cfg.exchange.exchange_id
    trigger           constant "ALPHA_COILER" (no equivalent field in ScoreResult)
    grade             result.grade
    score             result.alpha_score
    signal            constant "PUMP" (detector only fires on bullish setups)
    oi_status         derived: OI_PRESENT / OI_NEUTRAL / OI_UNKNOWN
    price             result.last_price
    change_24h_pct    UNAVAILABLE — no 24h change is computed in CoilSpring
    volume_24h_usd    bundle.market_data.quote_volume_24h (real USDT ticker vol)
    breakout_pct      UNAVAILABLE — body_mult is a relative multiplier, not a
                      price-move percentage; no % is computed in _factor_breakout
    body_volume_spike factor_metrics["breakout_spike"]["body_mult"] or null
    volume_spike      factor_metrics["breakout_spike"]["vol_mult"] or null
    flat_period_days  factor_metrics["low_volatility_coil"]["coil_window"] / 24
    signal_timestamp  time.time() at the moment of signal emission
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    # ── Identity ───────────────────────────────────────────────────────────────
    symbol: str = Field(description="Trading pair, e.g. KOMA/USDT:USDT")
    exchange: str = Field(description="Exchange identifier, e.g. bybit")

    # ── Alert metadata ─────────────────────────────────────────────────────────
    trigger: str = Field(description="Alert type, e.g. SNIPER TRIGGER or ALPHA_COILER")
    grade: str = Field(description="Signal grade, e.g. A")
    score: float = Field(description="Composite signal score out of 100")
    signal: str = Field(description="Direction: PUMP or DUMP")
    oi_status: str = Field(description="Open-interest label, e.g. OI_PRESENT")

    # ── Price / market data ────────────────────────────────────────────────────
    price: float = Field(gt=0, description="Last price in USDT (must be > 0)")

    change_24h_pct: Optional[float] = Field(
        default=None,
        description=(
            "24 h price change in percent. "
            "Null when the signal source does not compute a 24 h change "
            "(Alpha Coiler measures drift over the OI lookback window, not 24 h)."
        ),
    )

    volume_24h_usd: float = Field(
        ge=0,
        description="24 h quote volume in raw USD, from exchange ticker.",
    )

    breakout_pct: Optional[float] = Field(
        default=None,
        description=(
            "Breakout price-move magnitude in percent. "
            "Null when the signal source does not compute a price-move percentage "
            "(Alpha Coiler stores body_mult/vol_mult multipliers, not a price %)."
        ),
    )

    body_volume_spike: Optional[float] = Field(
        default=None,
        ge=0,
        description="Body candle volume spike multiplier vs flat-period average (e.g. 3.3).",
    )
    volume_spike: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total volume spike multiplier vs flat-period average (e.g. 8.6).",
    )
    flat_period_days: Optional[float] = Field(
        default=None,
        ge=0,
        description="Duration of the detected coil/flat period in days (coil_window candles / 24).",
    )

    # ── Timing ─────────────────────────────────────────────────────────────────
    signal_timestamp: float = Field(
        description="Unix timestamp (seconds, UTC) when the signal source emitted the signal."
    )


class SignalRecord(SignalPayload):
    """A received signal stored in memory, augmented with server-generated fields."""

    signal_id: str = Field(description="Server-assigned UUID4 identifier")
    received_timestamp: float = Field(
        description="Unix timestamp (seconds, UTC) when CoilBrain received the POST"
    )
    transport_latency_ms: float = Field(
        description="Latency from signal emission to receipt, in milliseconds"
    )
