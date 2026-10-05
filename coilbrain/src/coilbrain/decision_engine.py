"""CoilBrain V1 Decision Engine.

Deterministic TRADE / IGNORE classification based on:
  - CoilSpring signal features
  - Fresh microstructure snapshot
  - Current position state
  - Configurable hard gates and scoring weights

DESIGN PRINCIPLES
-----------------
* No LLM on the critical path.
* All thresholds documented and configurable.
* Every decision accompanied by machine-readable reasons.
* Fail-closed: ambiguity → IGNORE.
* Decision score is NOT a calibrated probability — do not present it as one.

DECISION FLOW
-------------
1. Hard safety gates (any failure → immediate IGNORE)
   - existing open position
   - 24h volume < threshold
   - market data missing or stale
   - symbol not available on Bybit Demo
   - spread beyond safety threshold
   - invalid price / depth / OI

2. Feature classification
   - intensity regime (NORMAL / ELEVATED / EXTREME)
   - OI state (SUPPORTIVE / NEUTRAL / ADVERSE)
   - order-book state (SUPPORTIVE / NEUTRAL / ADVERSE)
   - flow state (SUPPORTIVE / NEUTRAL / ADVERSE)
   - liquidity state (ACCEPTABLE / TIGHT / WIDE)

3. Scoring (configurable weights, max 100)

4. Threshold gate → TRADE or IGNORE
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from coilbrain.microstructure_fetcher import MicrostructureSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Decision(str, Enum):
    TRADE = "TRADE"
    IGNORE = "IGNORE"


class IntensityRegime(str, Enum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    EXTREME = "EXTREME"


class MarketState(str, Enum):
    SUPPORTIVE = "SUPPORTIVE"
    NEUTRAL = "NEUTRAL"
    ADVERSE = "ADVERSE"


class LiquidityState(str, Enum):
    ACCEPTABLE = "ACCEPTABLE"
    TIGHT = "TIGHT"      # spread very low — unusual, note it
    WIDE = "WIDE"        # spread too high — gate


# ---------------------------------------------------------------------------
# Decision reasons (machine-readable)
# ---------------------------------------------------------------------------

class Reason:
    # Hard gates — IGNORE
    EXISTING_POSITION = "EXISTING_POSITION"
    LOW_24H_VOLUME = "LOW_24H_VOLUME"
    MISSING_MARKET_DATA = "INSUFFICIENT_MARKET_DATA"
    STALE_MARKET_DATA = "STALE_MARKET_DATA"
    BYBIT_SYMBOL_UNAVAILABLE = "BYBIT_SYMBOL_UNAVAILABLE"
    UNSUPPORTED_INSTRUMENT = "UNSUPPORTED_INSTRUMENT"
    WIDE_SPREAD = "WIDE_SPREAD"
    INVALID_PRICE = "INVALID_PRICE"
    INVALID_DEPTH = "INVALID_DEPTH"
    CONFLICTING_MICROSTRUCTURE = "CONFLICTING_MICROSTRUCTURE"
    INSUFFICIENT_CONFIRMATION = "INSUFFICIENT_CONFIRMATION"

    # Positive reasons — TRADE components
    VOLUME_OK = "VOLUME_OK"
    INTENSITY_ELEVATED = "INTENSITY_ELEVATED"
    INTENSITY_EXTREME = "INTENSITY_EXTREME"
    OI_SUPPORTIVE = "OI_SUPPORTIVE"
    FLOW_SUPPORTIVE = "FLOW_SUPPORTIVE"
    ORDERBOOK_SUPPORTIVE = "ORDERBOOK_SUPPORTIVE"
    SPREAD_ACCEPTABLE = "SPREAD_ACCEPTABLE"
    SCORE_ABOVE_THRESHOLD = "SCORE_ABOVE_THRESHOLD"
    ASYMMETRIC_SQUEEZE = "ASYMMETRIC_SQUEEZE"

    # Neutral / informational
    INTENSITY_NORMAL = "INTENSITY_NORMAL"
    OI_NEUTRAL = "OI_NEUTRAL"
    FLOW_NEUTRAL = "FLOW_NEUTRAL"
    ORDERBOOK_NEUTRAL = "ORDERBOOK_NEUTRAL"


# ---------------------------------------------------------------------------
# V1 Thresholds (all configurable — do not hardcode logic around these)
# ---------------------------------------------------------------------------

@dataclass
class DecisionConfig:
    """All V1 decision thresholds in one place.

    Intended to be injected by the caller or read from config/env.
    Every threshold is documented.
    """

    # ── Hard gate: volume ──────────────────────────────────────────────────
    min_24h_volume_usd: float = 1_000_000.0

    # ── Hard gate: data freshness ──────────────────────────────────────────
    # Maximum age of a microstructure snapshot before it is considered stale.
    max_data_age_ms: float = 10_000.0     # 10 seconds

    # ── Hard gate: spread ──────────────────────────────────────────────────
    max_spread_pct: float = 0.5           # 0.5 % spread maximum

    # ── Intensity regime thresholds ────────────────────────────────────────
    elevated_body_spike: float = 5.0
    elevated_volume_spike: float = 10.0
    extreme_body_spike: float = 10.0
    extreme_volume_spike: float = 20.0

    # ── OI classification ─────────────────────────────────────────────────
    oi_supportive_delta_pct: float = 0.5  # OI must rise >= 0.5% to be supportive
    oi_adverse_delta_pct: float = -2.0    # OI falling >= 2% is adverse

    # ── Order-book classification ──────────────────────────────────────────
    ob_supportive_imbalance: float = 0.1  # bid_depth >= ask_depth + 10% → supportive
    ob_adverse_imbalance: float = -0.1    # ask_depth >> bid_depth → adverse

    # ── Trade flow classification ──────────────────────────────────────────
    flow_supportive_imbalance: float = 0.1   # >10% net buy flow → supportive
    flow_adverse_imbalance: float = -0.1

    # ── Decision score weights (must sum to 100) ───────────────────────────
    # Each component is rated 0-1 and multiplied by its weight.
    weight_signal_score: float = 25.0     # CoilSpring composite score (0-100 → 0-1)
    weight_intensity: float = 15.0        # EXTREME=1.0, ELEVATED=0.7, NORMAL=0.4
    weight_oi: float = 20.0              # SUPPORTIVE=1.0, NEUTRAL=0.5, ADVERSE=0.0
    weight_flow: float = 20.0            # same scale
    weight_orderbook: float = 15.0       # same scale
    weight_spread: float = 5.0           # ACCEPTABLE=1.0, TIGHT=0.7, WIDE=0.0

    # ── Decision gate ─────────────────────────────────────────────────────
    # Minimum score to emit TRADE.  Also all components must be non-adversarial.
    min_trade_score: float = 55.0

    # ── Confirmation: number of supportive components required ─────────────
    # Even if score is high, we require at least N components to agree.
    # Prevents a single runaway metric from causing a trade.
    min_supportive_components: int = 2

    # ── Asymmetric squeeze path thresholds ─────────────────────────────────
    squeeze_min_score: float = 85.0
    squeeze_min_flow_imbalance: float = 0.20
    squeeze_max_ob_imbalance: float = -0.10


# ---------------------------------------------------------------------------
# Input / Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SignalFeatures:
    """Normalized CoilSpring signal features passed into the decision engine."""
    signal_id: str
    symbol: str
    score: float            # CoilSpring score 0–100
    grade: str
    body_spike: float       # body_volume_spike
    volume_spike: float     # volume_spike
    volume_24h_usd: float
    change_24h_pct: float | None
    breakout_pct: float | None
    flat_period_days: float | None
    signal_timestamp: float


@dataclass
class PositionState:
    """Current position state for the symbol being evaluated."""
    symbol: str
    state: str              # NO_POSITION | PENDING_ENTRY | OPEN | EXITING | CLOSED


@dataclass
class DataFreshness:
    """Freshness assessment for each microstructure component."""
    snapshot_age_ms: float
    is_fresh: bool
    missing_critical: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)


@dataclass
class FeatureClassification:
    """Classified market state derived from microstructure."""
    intensity: IntensityRegime
    oi_state: MarketState
    flow_state: MarketState
    ob_state: MarketState
    liquidity_state: LiquidityState


@dataclass
class DecisionResult:
    """Complete output of the decision engine."""
    decision: Decision
    decision_score: float                  # 0–100, V1 rule score (not a probability)
    reasons: list[str]
    signal_features: SignalFeatures
    feature_classification: FeatureClassification | None
    microstructure: MicrostructureSnapshot | None
    data_freshness: DataFreshness | None
    bybit_symbol: str | None               # resolved Bybit Demo symbol

    # Latency timestamps (all Unix seconds)
    decision_start_ts: float = 0.0
    market_data_start_ts: float = 0.0
    market_data_complete_ts: float = 0.0
    decision_ts: float = 0.0

    # Derived latency (ms)
    signal_to_decision_ms: float = 0.0
    data_fetch_latency_ms: float = 0.0

    # Extra context
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Decision Engine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """V1 deterministic signal decision engine.

    Inject a DecisionConfig to control all thresholds.
    Call decide() with a fresh microstructure snapshot.
    """

    def __init__(self, config: DecisionConfig | None = None) -> None:
        self.cfg = config or DecisionConfig()

    # ── Main entry point ──────────────────────────────────────────────────

    def decide(
        self,
        signal: SignalFeatures,
        snapshot: MicrostructureSnapshot | None,
        position: PositionState,
        bybit_symbol: str | None = None,
        fetch_start_ts: float | None = None,
        fetch_end_ts: float | None = None,
    ) -> DecisionResult:
        """Run the full V1 decision pipeline.

        Returns a DecisionResult with TRADE or IGNORE + all evidence.
        This method never raises; all errors → IGNORE.
        """
        decision_start = time.time()
        reasons: list[str] = []

        try:
            result = self._run_pipeline(
                signal=signal,
                snapshot=snapshot,
                position=position,
                bybit_symbol=bybit_symbol,
                reasons=reasons,
                decision_start=decision_start,
                fetch_start_ts=fetch_start_ts,
                fetch_end_ts=fetch_end_ts,
            )
        except Exception as exc:
            logger.exception("DecisionEngine.decide raised unexpectedly: %s", exc)
            result = DecisionResult(
                decision=Decision.IGNORE,
                decision_score=0.0,
                reasons=[f"INTERNAL_ERROR: {exc}"],
                signal_features=signal,
                feature_classification=None,
                microstructure=snapshot,
                data_freshness=None,
                bybit_symbol=bybit_symbol,
            )

        result.decision_ts = time.time()
        result.signal_to_decision_ms = (result.decision_ts - signal.signal_timestamp) * 1000.0
        if fetch_start_ts and fetch_end_ts:
            result.data_fetch_latency_ms = (fetch_end_ts - fetch_start_ts) * 1000.0
        result.decision_start_ts = decision_start

        path = result.extra.get("decision_path")
        if result.decision == Decision.TRADE and path:
            logger.info("Trade authorized via path: %s", path)

        logger.info(
            "Decision: %s%s | symbol=%s score=%.1f reasons=%s",
            result.decision.value,
            f" (path: {path})" if path else "",
            signal.symbol,
            result.decision_score,
            result.reasons,
        )
        return result

    # ── Pipeline ──────────────────────────────────────────────────────────

    def _run_pipeline(
        self,
        signal: SignalFeatures,
        snapshot: MicrostructureSnapshot | None,
        position: PositionState,
        bybit_symbol: str | None,
        reasons: list[str],
        decision_start: float,
        fetch_start_ts: float | None,
        fetch_end_ts: float | None,
    ) -> DecisionResult:

        def _ignore(reason: str) -> DecisionResult:
            reasons.append(reason)
            return DecisionResult(
                decision=Decision.IGNORE,
                decision_score=0.0,
                reasons=reasons,
                signal_features=signal,
                feature_classification=None,
                microstructure=snapshot,
                data_freshness=freshness,
                bybit_symbol=bybit_symbol,
                market_data_start_ts=fetch_start_ts or 0.0,
                market_data_complete_ts=fetch_end_ts or 0.0,
            )

        # ── Gate 0: position state ─────────────────────────────────────────
        freshness: DataFreshness | None = None

        if position.state in ("OPEN", "PENDING_ENTRY", "EXITING"):
            return _ignore(Reason.EXISTING_POSITION)

        # ── Gate 1: 24h volume ─────────────────────────────────────────────
        if signal.volume_24h_usd < self.cfg.min_24h_volume_usd:
            return _ignore(Reason.LOW_24H_VOLUME)
        reasons.append(Reason.VOLUME_OK)

        # ── Gate 2: Bybit symbol availability ─────────────────────────────
        if bybit_symbol is None:
            return _ignore(Reason.BYBIT_SYMBOL_UNAVAILABLE)

        # ── Gate 3: data presence and freshness ───────────────────────────
        freshness = self._assess_freshness(snapshot, decision_start)
        if not freshness.is_fresh:
            reason = (
                Reason.MISSING_MARKET_DATA
                if snapshot is None
                else Reason.STALE_MARKET_DATA
            )
            return _ignore(reason)
        if freshness.missing_critical:
            reasons.append(Reason.MISSING_MARKET_DATA)
            return _ignore(Reason.MISSING_MARKET_DATA)

        assert snapshot is not None  # guaranteed by freshness check above

        # ── Gate 4: price sanity ───────────────────────────────────────────
        price = snapshot.mark_price or snapshot.best_bid
        if not price or price <= 0:
            return _ignore(Reason.INVALID_PRICE)

        # ── Gate 5: depth sanity ───────────────────────────────────────────
        if snapshot.bid_depth_usdt is None or snapshot.ask_depth_usdt is None:
            return _ignore(Reason.INVALID_DEPTH)

        # ── Gate 6: spread ────────────────────────────────────────────────
        if snapshot.spread_pct is not None and snapshot.spread_pct > self.cfg.max_spread_pct:
            return _ignore(Reason.WIDE_SPREAD)
        reasons.append(Reason.SPREAD_ACCEPTABLE)

        # ── Feature classification ─────────────────────────────────────────
        classification = self._classify_features(signal, snapshot)

        # Intensity reason
        if classification.intensity == IntensityRegime.EXTREME:
            reasons.append(Reason.INTENSITY_EXTREME)
        elif classification.intensity == IntensityRegime.ELEVATED:
            reasons.append(Reason.INTENSITY_ELEVATED)
        else:
            reasons.append(Reason.INTENSITY_NORMAL)

        # OI reason
        if classification.oi_state == MarketState.SUPPORTIVE:
            reasons.append(Reason.OI_SUPPORTIVE)
        elif classification.oi_state == MarketState.ADVERSE:
            pass  # adverse is recorded in final logic
        else:
            reasons.append(Reason.OI_NEUTRAL)

        # Flow reason
        if classification.flow_state == MarketState.SUPPORTIVE:
            reasons.append(Reason.FLOW_SUPPORTIVE)
        elif classification.flow_state == MarketState.ADVERSE:
            pass
        else:
            reasons.append(Reason.FLOW_NEUTRAL)

        # OB reason
        if classification.ob_state == MarketState.SUPPORTIVE:
            reasons.append(Reason.ORDERBOOK_SUPPORTIVE)
        elif classification.ob_state == MarketState.ADVERSE:
            pass
        else:
            reasons.append(Reason.ORDERBOOK_NEUTRAL)

        # ── Scoring ────────────────────────────────────────────────────────
        score = self._calculate_score(signal, classification)

        # ── Confirmation check ─────────────────────────────────────────────
        supportive = sum([
            classification.oi_state == MarketState.SUPPORTIVE,
            classification.flow_state == MarketState.SUPPORTIVE,
            classification.ob_state == MarketState.SUPPORTIVE,
            classification.intensity in (IntensityRegime.ELEVATED, IntensityRegime.EXTREME),
        ])

        adverse = sum([
            classification.oi_state == MarketState.ADVERSE,
            classification.flow_state == MarketState.ADVERSE,
            classification.ob_state == MarketState.ADVERSE,
        ])

        if adverse >= 2:
            return DecisionResult(
                decision=Decision.IGNORE,
                decision_score=score,
                reasons=reasons + [Reason.CONFLICTING_MICROSTRUCTURE],
                signal_features=signal,
                feature_classification=classification,
                microstructure=snapshot,
                data_freshness=freshness,
                bybit_symbol=bybit_symbol,
                market_data_start_ts=fetch_start_ts or 0.0,
                market_data_complete_ts=fetch_end_ts or 0.0,
            )

        # ── Decision Qualification Paths ──────────────────────────────────
        # Path 1: NORMAL_V1
        #   - score >= min_trade_score (55)
        #   - supportive components >= min_supportive_components (2)
        #   - adverse components <= 1 (verified by adverse < 2 check above)
        normal_qualifies = (
            score >= self.cfg.min_trade_score
            and supportive >= self.cfg.min_supportive_components
        )

        # Path 2: ASYMMETRIC_SQUEEZE
        # ALL must be true:
        #   - CoilSpring score >= squeeze_min_score (85)
        #   - trade_imbalance >= squeeze_min_flow_imbalance (+0.20)
        #   - ob_imbalance <= squeeze_max_ob_imbalance (-0.10)
        #   - score >= min_trade_score (55)
        #   - adverse components <= 1 (verified by adverse < 2 check above)
        # This path bypasses ONLY the normal min_supportive_components >= 2 requirement.
        is_asymmetric_squeeze = (
            signal.score >= self.cfg.squeeze_min_score
            and snapshot.trade_imbalance is not None
            and snapshot.trade_imbalance >= self.cfg.squeeze_min_flow_imbalance
            and snapshot.ob_imbalance is not None
            and snapshot.ob_imbalance <= self.cfg.squeeze_max_ob_imbalance
        )
        squeeze_qualifies = is_asymmetric_squeeze and (score >= self.cfg.min_trade_score)

        if normal_qualifies:
            authorized_path = "NORMAL_V1"
            reasons.append(Reason.SCORE_ABOVE_THRESHOLD)
            qualifying_paths = ["NORMAL_V1"]
            if squeeze_qualifies:
                qualifying_paths.append("ASYMMETRIC_SQUEEZE")
        elif squeeze_qualifies:
            authorized_path = "ASYMMETRIC_SQUEEZE"
            reasons.append(Reason.ASYMMETRIC_SQUEEZE)
            qualifying_paths = ["ASYMMETRIC_SQUEEZE"]
        else:
            reasons.append(Reason.INSUFFICIENT_CONFIRMATION)
            return DecisionResult(
                decision=Decision.IGNORE,
                decision_score=score,
                reasons=reasons,
                signal_features=signal,
                feature_classification=classification,
                microstructure=snapshot,
                data_freshness=freshness,
                bybit_symbol=bybit_symbol,
                market_data_start_ts=fetch_start_ts or 0.0,
                market_data_complete_ts=fetch_end_ts or 0.0,
            )

        return DecisionResult(
            decision=Decision.TRADE,
            decision_score=score,
            reasons=reasons,
            signal_features=signal,
            feature_classification=classification,
            microstructure=snapshot,
            data_freshness=freshness,
            bybit_symbol=bybit_symbol,
            market_data_start_ts=fetch_start_ts or 0.0,
            market_data_complete_ts=fetch_end_ts or 0.0,
            extra={
                "decision_path": authorized_path,
                "qualifying_paths": qualifying_paths,
            },
        )

    # ── Sub-routines ──────────────────────────────────────────────────────

    def _assess_freshness(
        self,
        snapshot: MicrostructureSnapshot | None,
        now: float,
    ) -> DataFreshness:
        if snapshot is None:
            return DataFreshness(
                snapshot_age_ms=float("inf"),
                is_fresh=False,
                missing_critical=["snapshot"],
            )
        age_ms = (now - snapshot.snapshot_ts) * 1000.0
        missing_critical = []
        missing_optional = []

        for attr in ("mark_price", "best_bid", "best_ask", "bid_depth_usdt", "ask_depth_usdt"):
            if getattr(snapshot, attr) is None:
                missing_critical.append(attr)
        for attr in ("oi_usdt", "trade_imbalance", "ob_imbalance"):
            if getattr(snapshot, attr) is None:
                missing_optional.append(attr)

        return DataFreshness(
            snapshot_age_ms=age_ms,
            is_fresh=(age_ms <= self.cfg.max_data_age_ms) and not missing_critical,
            missing_critical=missing_critical,
            missing_optional=missing_optional,
        )

    def _classify_intensity(self, signal: SignalFeatures) -> IntensityRegime:
        bs, vs = signal.body_spike, signal.volume_spike
        if bs >= self.cfg.extreme_body_spike and vs >= self.cfg.extreme_volume_spike:
            return IntensityRegime.EXTREME
        if bs >= self.cfg.elevated_body_spike and vs >= self.cfg.elevated_volume_spike:
            return IntensityRegime.ELEVATED
        return IntensityRegime.NORMAL

    def _classify_oi(self, snapshot: MicrostructureSnapshot) -> MarketState:
        delta = snapshot.oi_delta_pct
        if delta is None:
            return MarketState.NEUTRAL
        if delta >= self.cfg.oi_supportive_delta_pct:
            return MarketState.SUPPORTIVE
        if delta <= self.cfg.oi_adverse_delta_pct:
            return MarketState.ADVERSE
        return MarketState.NEUTRAL

    def _classify_flow(self, snapshot: MicrostructureSnapshot) -> MarketState:
        imb = snapshot.trade_imbalance
        if imb is None:
            return MarketState.NEUTRAL
        if imb >= self.cfg.flow_supportive_imbalance:
            return MarketState.SUPPORTIVE
        if imb <= self.cfg.flow_adverse_imbalance:
            return MarketState.ADVERSE
        return MarketState.NEUTRAL

    def _classify_ob(self, snapshot: MicrostructureSnapshot) -> MarketState:
        imb = snapshot.ob_imbalance
        if imb is None:
            return MarketState.NEUTRAL
        if imb >= self.cfg.ob_supportive_imbalance:
            return MarketState.SUPPORTIVE
        if imb <= self.cfg.ob_adverse_imbalance:
            return MarketState.ADVERSE
        return MarketState.NEUTRAL

    def _classify_liquidity(self, snapshot: MicrostructureSnapshot) -> LiquidityState:
        sp = snapshot.spread_pct
        if sp is None:
            return LiquidityState.ACCEPTABLE
        if sp > self.cfg.max_spread_pct:
            return LiquidityState.WIDE
        if sp < 0.01:
            return LiquidityState.TIGHT
        return LiquidityState.ACCEPTABLE

    def _classify_features(
        self,
        signal: SignalFeatures,
        snapshot: MicrostructureSnapshot,
    ) -> FeatureClassification:
        return FeatureClassification(
            intensity=self._classify_intensity(signal),
            oi_state=self._classify_oi(snapshot),
            flow_state=self._classify_flow(snapshot),
            ob_state=self._classify_ob(snapshot),
            liquidity_state=self._classify_liquidity(snapshot),
        )

    def _calculate_score(
        self,
        signal: SignalFeatures,
        fc: FeatureClassification,
    ) -> float:
        """Weighted rule score (0–100).  Weights are documented and configurable.

        NOT a calibrated probability.
        """
        cfg = self.cfg

        # Signal component: raw CoilSpring score normalised to [0,1]
        signal_norm = min(signal.score / 100.0, 1.0)

        # Intensity component
        intensity_map = {
            IntensityRegime.EXTREME: 1.0,
            IntensityRegime.ELEVATED: 0.7,
            IntensityRegime.NORMAL: 0.4,
        }
        intensity_norm = intensity_map[fc.intensity]

        # Market-state components
        state_map = {
            MarketState.SUPPORTIVE: 1.0,
            MarketState.NEUTRAL: 0.5,
            MarketState.ADVERSE: 0.0,
        }
        oi_norm = state_map[fc.oi_state]
        flow_norm = state_map[fc.flow_state]
        ob_norm = state_map[fc.ob_state]

        # Spread component
        liq_map = {
            LiquidityState.ACCEPTABLE: 1.0,
            LiquidityState.TIGHT: 0.7,
            LiquidityState.WIDE: 0.0,
        }
        spread_norm = liq_map[fc.liquidity_state]

        score = (
            signal_norm * cfg.weight_signal_score
            + intensity_norm * cfg.weight_intensity
            + oi_norm * cfg.weight_oi
            + flow_norm * cfg.weight_flow
            + ob_norm * cfg.weight_orderbook
            + spread_norm * cfg.weight_spread
        )
        return round(score, 2)
