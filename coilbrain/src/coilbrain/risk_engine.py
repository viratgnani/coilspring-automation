"""CoilBrain V1 Risk Engine — Real ATR-Based SL/TP.

Calculates position size, stop-loss, take-profit, and risk/reward ratio
for a TRADE decision.

DESIGN PRINCIPLES
-----------------
* Completely separate from decision logic.
* No trade execution here — pure calculation.
* All parameters configurable and documented.
* Hard maximum on risk per trade (never exceed configured limit).
* No martingale, no averaging down, no revenge sizing.
* SL/TP derived from REAL ATR(14) on 1-minute candles.
  ATR is fetched by atr_fetcher.ATRFetcher before calling this engine.
  If ATR is None or stale → fail-closed (valid=False, error set).
* Every result is fully documented.

SL/TP METHOD (V1)
-----------------
V1 uses ATR(14) on the 1m timeframe for volatility:

  stop_distance = ATR(14) × sl_atr_multiple
  stop_distance = clamp(
      stop_distance,
      max(spread_abs × spread_sl_buffer, entry_price × min_stop_pct),
      entry_price × max_stop_pct
  )

  stop_loss   = entry_price − stop_distance   (LONG only in V1)
  take_profit = entry_price + stop_distance × rr_ratio

Spread is used as a safety floor (you must always stop beyond the spread),
not as the primary volatility measure.

SCORE WEIGHTS (INITIAL HYPOTHESIS — NOT STATISTICALLY VALIDATED)
----------------------------------------------------------------
The calling decision engine uses:
    CoilSpring score     weight = 25
    Intensity regime     weight = 15
    OI state             weight = 20
    Trade flow           weight = 20
    Order book           weight = 15
    Spread/liquidity     weight =  5
    min_trade_score      = 55
    min_supportive_components = 2
    max_adverse_components    < 2

These are V1 initial hypothesis parameters.
They have NOT been back-tested or statistically validated.
Do not claim they are optimal or proven.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from coilbrain.atr_fetcher import ATRResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    """All V1 risk parameters — inject and document; never hardcode.

    INITIAL HYPOTHESIS PARAMETERS — not statistically validated.
    """

    # ── Account / sizing ──────────────────────────────────────────────────
    account_balance_usd: float = 5000.0   # $5,000 Funded Account evaluation balance
    max_risk_pct_per_trade: float = 2.2   # Target % of balance to risk per trade (aggressive eval sizing)
    hard_max_single_risk_pct: float = 3.0 # Hard ceiling: Prop firm 3% max single position drawdown
    max_position_usd: float = 7500.0      # Cap on notional position size (USD, up to 1.5x leverage)

    # ── SL/TP methodology ─────────────────────────────────────────────────
    # Multiplier applied to ATR(14) to set stop distance.
    # A value of 2.0 means: stop = 2 × ATR below entry.
    sl_atr_multiple: float = 2.0

    # Risk:Reward ratio used to set TP
    rr_ratio: float = 2.0

    # Spread safety buffer: stop must be at least this many × spread away.
    # Prevents stops from being inside the bid-ask spread.
    spread_sl_buffer: float = 1.5        # stop >= 1.5 × spread_abs

    # Minimum stop distance as % of price (absolute floor if ATR is tiny)
    min_stop_pct: float = 0.005          # 0.5 %

    # Maximum stop distance as % of price (prevents absurdly wide stops)
    max_stop_pct: float = 0.05           # 5.0 %

    # ── Leverage ─────────────────────────────────────────────────────────
    max_leverage: float = 3.0            # Hard cap for demo safety


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RiskResult:
    """Complete risk calculation output.

    All intermediate values are exposed so the caller can inspect / log them.
    atr_value: the raw ATR(14) value in price units (None if ATR was unavailable)
    atr_source: how the stop was ultimately sized ("atr14_1m" | "min_stop_floor" | "spread_floor" | "error")
    """
    # Inputs
    entry_price: float
    account_balance_usd: float

    # ATR data used
    atr_value: float | None       # raw ATR(14) in price units
    atr_source: str               # "atr14_1m" | "min_stop_floor" | "spread_floor" | "error"

    # Position parameters
    stop_distance: float          # price distance from entry to SL
    stop_loss: float              # absolute SL price (for LONG: entry - stop_distance)
    take_profit: float            # absolute TP price (for LONG: entry + stop_distance * rr)
    stop_pct: float               # stop_distance / entry * 100
    rr_ratio: float

    # Sizing
    risk_usd: float               # $ amount risked on this trade
    position_size_usd: float      # total notional position
    quantity: float               # base-asset quantity
    leverage: float

    # Flags
    capped_by_max_position: bool
    capped_by_max_risk: bool
    valid: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Risk Engine
# ---------------------------------------------------------------------------

class RiskEngine:
    """V1 risk calculator — uses real ATR(14) from 1m candles.

    The ATR must be fetched separately by ATRFetcher before calling calculate().
    If atr_result is None → fail-closed (returns RiskResult with valid=False).

    Usage:
        fetcher = ATRFetcher()
        atr = await fetcher.fetch_atr(http_client, "TAKEUSDT")

        risk = RiskEngine(RiskConfig(...))
        result = risk.calculate(entry_price=0.089, atr_result=atr, spread_abs=0.00004)
        if not result.valid:
            # fail-closed — do not trade
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.cfg = config or RiskConfig()

    def calculate(
        self,
        entry_price: float,
        atr_result: ATRResult | None,
        spread_abs: float | None = None,
    ) -> RiskResult:
        """Calculate position parameters for a LONG entry at entry_price.

        V1 only supports LONG (CoilSpring fires on PUMP setups).
        Returns a RiskResult; never raises.

        Parameters
        ----------
        entry_price:  mark price at decision time
        atr_result:   ATRResult from ATRFetcher.fetch_atr(); None → fail-closed
        spread_abs:   current best-ask minus best-bid; used as spread safety floor
        """
        cfg = self.cfg

        try:
            if entry_price <= 0:
                return self._error(entry_price, None, "entry_price must be > 0")

            # ── ATR gate — fail-closed ────────────────────────────────────
            if atr_result is None:
                return self._error(entry_price, None, "ATR unavailable — fail-closed")

            if atr_result.atr <= 0:
                return self._error(entry_price, atr_result.atr,
                                   f"ATR({atr_result.atr}) is non-positive — fail-closed")

            atr = atr_result.atr
            atr_source = "atr14_1m"

            # ── Stop distance from ATR ────────────────────────────────────
            stop_distance = atr * cfg.sl_atr_multiple

            # Floor 1: spread safety — stop must clear the bid-ask spread
            if spread_abs is not None and spread_abs > 0:
                spread_floor = spread_abs * cfg.spread_sl_buffer
                if stop_distance < spread_floor:
                    stop_distance = spread_floor
                    atr_source = "spread_floor"

            # Floor 2: minimum stop % of price
            min_stop = entry_price * cfg.min_stop_pct
            if stop_distance < min_stop:
                stop_distance = min_stop
                atr_source = "min_stop_floor"

            # Ceiling: maximum stop % of price
            max_stop = entry_price * cfg.max_stop_pct
            if stop_distance > max_stop:
                stop_distance = max_stop

            stop_pct = stop_distance / entry_price * 100.0
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + stop_distance * cfg.rr_ratio

            # ── Sizing ────────────────────────────────────────────────────
            # Never exceed max_risk_pct_per_trade or hard 3% single position drawdown limit
            hard_limit_pct = getattr(cfg, "hard_max_single_risk_pct", 3.0)
            effective_risk_pct = min(cfg.max_risk_pct_per_trade, hard_limit_pct)
            max_risk_usd = cfg.account_balance_usd * effective_risk_pct / 100.0
            risk_usd = max_risk_usd
            position_size_usd = risk_usd / (stop_distance / entry_price)

            capped_by_max_position = False
            capped_by_max_risk = False

            if position_size_usd > cfg.max_position_usd:
                position_size_usd = cfg.max_position_usd
                risk_usd = position_size_usd * (stop_distance / entry_price)
                capped_by_max_position = True

            if risk_usd > max_risk_usd:
                risk_usd = max_risk_usd
                position_size_usd = risk_usd / (stop_distance / entry_price)
                capped_by_max_risk = True

            # Inviolable hard ceiling: max single position drawdown (3% = $150 on $5000)
            hard_ceiling_usd = cfg.account_balance_usd * (hard_limit_pct / 100.0)
            if risk_usd > hard_ceiling_usd:
                risk_usd = hard_ceiling_usd
                position_size_usd = risk_usd / (stop_distance / entry_price)
                capped_by_max_risk = True

            # ── Quantity ──────────────────────────────────────────────────
            quantity = position_size_usd / entry_price

            # ── Leverage ──────────────────────────────────────────────────
            leverage = min(
                position_size_usd / (risk_usd if risk_usd > 0 else 1),
                cfg.max_leverage,
            )

            return RiskResult(
                entry_price=entry_price,
                account_balance_usd=cfg.account_balance_usd,
                atr_value=atr,
                atr_source=atr_source,
                stop_distance=stop_distance,
                stop_loss=stop_loss,
                take_profit=take_profit,
                stop_pct=stop_pct,
                rr_ratio=cfg.rr_ratio,
                risk_usd=risk_usd,
                position_size_usd=position_size_usd,
                quantity=quantity,
                leverage=leverage,
                capped_by_max_position=capped_by_max_position,
                capped_by_max_risk=capped_by_max_risk,
                valid=True,
            )

        except Exception as exc:
            logger.exception("RiskEngine.calculate raised: %s", exc)
            return self._error(entry_price, None, str(exc))

    def _error(
        self,
        entry_price: float,
        atr_value: float | None,
        msg: str,
    ) -> RiskResult:
        return RiskResult(
            entry_price=entry_price,
            account_balance_usd=self.cfg.account_balance_usd,
            atr_value=atr_value,
            atr_source="error",
            stop_distance=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            stop_pct=0.0,
            rr_ratio=self.cfg.rr_ratio,
            risk_usd=0.0,
            position_size_usd=0.0,
            quantity=0.0,
            leverage=0.0,
            capped_by_max_position=False,
            capped_by_max_risk=False,
            valid=False,
            error=msg,
        )
