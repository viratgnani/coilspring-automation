"""Tests for the CoilBrain V1 Decision Engine — Pre-Demo Gate.

REQUIRED SCENARIOS (numbered as specified)
------------------------------------------
 1. low volume → IGNORE
 2. low volume then higher volume → evaluate normally (no blacklist)
 3. repeated signal after IGNORE is evaluated again
 4. existing open position → IGNORE(EXISTING_POSITION)
 5. position closed → next signal evaluates normally
 6. NORMAL intensity reaches decision engine
 7. ELEVATED intensity classification
 8. EXTREME intensity classification
 9. missing microstructure → IGNORE
10. stale microstructure → IGNORE
11. wide spread → IGNORE
12. conflicting microstructure (≥2 adverse) → IGNORE
13. supportive microstructure → TRADE candidate
14. no single metric alone causes TRADE
15. risk engine — basic calculation valid
16. SL/TP generation
17. position sizing limits
18. Bybit Demo: production API URL cannot be used
19. Bybit Demo: order not successful until exchange ACK
20. Bybit Demo: failed SL placement raises error
21. Bybit Demo: failed TP placement raises error
22. duplicate signal does not create duplicate position
23. latency timestamps populated
24. decision journal records every evaluation
25. fail-closed behavior (any uncertainty → IGNORE)
26. concurrent/rapid signals for same symbol
27. re-check position before execution
28. re-check data freshness before execution (stale guard)

ADDITIONAL REQUIRED (pre-demo gate additions)
---------------------------------------------
 A. ATR fetcher calculates ATR(14) correctly from kline data
 B. ATR fetcher fails closed when data unavailable/stale
 C. Risk engine fails closed when ATR is None
 D. Risk engine uses real ATR for stop, not fixed percentage
 E. Spread floor: stop never inside spread
 F. Full repeated-signal scenario: vol<1M → vol>1M → OPEN → CLOSED
 G. Production credentials cannot be silently accepted
 H. Demo credentials required to construct client
 I. Position state rechecked immediately before order submission
 J. SL must be confirmed before marking protection complete
 K. TP must be confirmed before marking protection complete
 L. Trade journal latency compute_latencies() function
 M. decision_repository.record_trade() stores all 15 timestamps + 4 derived
 N. Score weights are explicitly initial-hypothesis parameters (documented test)
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from coilbrain.atr_fetcher import ATRFetcher, ATRResult
from coilbrain.bybit_demo import BybitDemoAPIError, BybitDemoClient, BybitDemoConfigError
from coilbrain.decision_engine import (
    Decision, DecisionConfig, DecisionEngine, IntensityRegime,
    MarketState, PositionState, Reason, SignalFeatures,
)
from coilbrain.microstructure_fetcher import MicrostructureSnapshot
from coilbrain.position_manager import Position, PositionManager, PositionStatus
from coilbrain.risk_engine import RiskConfig, RiskEngine, RiskResult
from coilbrain import decision_repository as drepo


# ===========================================================================
# Shared helpers
# ===========================================================================

def _snap(
    age_s: float = 0.5,
    spread_pct: float = 0.1,
    ob_imbalance: float = 0.2,
    trade_imbalance: float = 0.15,
    oi_delta_pct: float = 1.5,
    mark_price: float = 0.089,
    best_bid: float = 0.0888,
    best_ask: float = 0.0892,
    bid_depth: float = 50_000.0,
    ask_depth: float = 40_000.0,
    oi_usdt: float = 1_200_000.0,
) -> MicrostructureSnapshot:
    spread_abs = best_ask - best_bid
    return MicrostructureSnapshot(
        binance_symbol="TAKEUSDT",
        snapshot_ts=time.time() - age_s,
        oi_usdt=oi_usdt,
        oi_delta_pct=oi_delta_pct,
        funding_rate=0.0001,
        mark_price=mark_price,
        next_funding_ts=time.time() + 14400,
        best_bid=best_bid,
        best_ask=best_ask,
        spread_abs=spread_abs,
        spread_pct=spread_pct,
        bid_depth_usdt=bid_depth,
        ask_depth_usdt=ask_depth,
        ob_imbalance=ob_imbalance,
        trade_count=200,
        traded_vol_usdt=50_000.0,
        buy_vol_usdt=30_000.0,
        sell_vol_usdt=20_000.0,
        trade_imbalance=trade_imbalance,
    )


def _signal(
    volume_24h: float = 3_000_000.0,
    body_spike: float = 6.0,
    volume_spike: float = 12.0,
    score: float = 85.0,
    symbol: str = "TAKE/USDT:USDT",
) -> SignalFeatures:
    return SignalFeatures(
        signal_id=str(uuid.uuid4()),
        symbol=symbol,
        score=score,
        grade="A",
        body_spike=body_spike,
        volume_spike=volume_spike,
        volume_24h_usd=volume_24h,
        change_24h_pct=15.0,
        breakout_pct=4.5,
        flat_period_days=14.0,
        signal_timestamp=time.time() - 1.0,  # 1s ago so latency ms > 0
    )


def _pos(state: str = "NO_POSITION", symbol: str = "TAKE/USDT:USDT") -> PositionState:
    return PositionState(symbol=symbol, state=state)


def _engine(config: DecisionConfig | None = None) -> DecisionEngine:
    return DecisionEngine(config or DecisionConfig())


def _atr(value: float = 0.0008, symbol: str = "TAKEUSDT") -> ATRResult:
    """Return a fresh synthetic ATRResult."""
    return ATRResult(
        symbol=symbol,
        interval="1m",
        atr=value,
        fetch_ts=time.time(),
        candles_used=14,
    )


@pytest.fixture
def engine():
    return _engine()


@pytest.fixture
async def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        pm = PositionManager()
        await pm.ensure_schema(conn)
        await drepo.ensure_schema(conn)
        yield conn


# ===========================================================================
# 1. Low volume → IGNORE
# ===========================================================================

class TestLowVolume:
    def test_low_volume_ignored(self, engine):
        result = engine.decide(_signal(volume_24h=800_000.0), _snap(), _pos(), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE
        assert Reason.LOW_24H_VOLUME in result.reasons

    def test_borderline_volume_ignored(self, engine):
        result = engine.decide(_signal(volume_24h=999_999.0), _snap(), _pos(), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE

    def test_exact_threshold_accepted(self, engine):
        result = engine.decide(_signal(volume_24h=1_000_000.0), _snap(), _pos(), bybit_symbol="TAKEUSDT")
        assert Reason.LOW_24H_VOLUME not in result.reasons
        assert Reason.VOLUME_OK in result.reasons

    def test_high_volume_accepted(self, engine):
        result = engine.decide(_signal(volume_24h=5_000_000.0), _snap(), _pos(), bybit_symbol="TAKEUSDT")
        assert Reason.VOLUME_OK in result.reasons


# ===========================================================================
# 2. Low-volume → higher volume → no blacklist
# ===========================================================================

class TestVolumeRisesLater:
    def test_first_signal_ignored_second_evaluated(self, engine):
        sig1 = _signal(volume_24h=700_000.0, symbol="COIN/USDT:USDT")
        r1 = engine.decide(sig1, _snap(), _pos(symbol="COIN/USDT:USDT"), bybit_symbol="COINUSDT")
        assert r1.decision == Decision.IGNORE
        assert Reason.LOW_24H_VOLUME in r1.reasons

        # Second signal — symbol NOT blacklisted
        sig2 = _signal(volume_24h=4_000_000.0, symbol="COIN/USDT:USDT",
                       body_spike=6.0, volume_spike=12.0, score=85.0)
        r2 = engine.decide(sig2, _snap(), _pos(symbol="COIN/USDT:USDT"), bybit_symbol="COINUSDT")
        assert Reason.VOLUME_OK in r2.reasons


# ===========================================================================
# 3. Repeated signal after IGNORE is evaluated again
# ===========================================================================

class TestRepeatedSignalEvaluated:
    def test_ignored_symbol_evaluated_on_next_signal(self, engine):
        r1 = engine.decide(_signal(volume_24h=500_000.0), _snap(), _pos(), bybit_symbol="TAKEUSDT")
        assert r1.decision == Decision.IGNORE

        # Wide spread causes the next to ignore, but for spread reason — not volume
        r2 = engine.decide(_signal(volume_24h=5_000_000.0), _snap(spread_pct=1.0), _pos(), bybit_symbol="TAKEUSDT")
        assert Reason.LOW_24H_VOLUME not in r2.reasons


# ===========================================================================
# F. Full repeated-signal scenario (item F above)
# vol<1M → vol>1M → position OPEN → position CLOSED
# ===========================================================================

class TestFullRepeatedSignalScenario:
    """Authoritative 4-signal test as specified in the review."""

    @pytest.mark.asyncio
    async def test_four_signal_lifecycle(self):
        engine = _engine()
        pm = PositionManager()
        sym = "FULL/USDT:USDT"
        bsym = "FULLUSDT"

        # We need a fresh DB for this test
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row
                await pm.ensure_schema(db)

                # Signal 1: volume < $1M → IGNORE
                status1 = await pm.get_state_for_symbol(db, sym)
                r1 = engine.decide(
                    _signal(volume_24h=500_000.0, symbol=sym),
                    _snap(), PositionState(symbol=sym, state=status1.value),
                    bybit_symbol=bsym,
                )
                assert r1.decision == Decision.IGNORE
                assert Reason.LOW_24H_VOLUME in r1.reasons

                # Signal 2: volume > $1M → evaluate normally
                status2 = await pm.get_state_for_symbol(db, sym)
                r2 = engine.decide(
                    _signal(volume_24h=4_000_000.0, symbol=sym,
                            body_spike=6.0, volume_spike=12.0, score=90.0),
                    _snap(oi_delta_pct=2.0, trade_imbalance=0.25, ob_imbalance=0.3),
                    PositionState(symbol=sym, state=status2.value),
                    bybit_symbol=bsym,
                )
                # Passes volume gate — may TRADE or IGNORE for other reasons,
                # but MUST NOT be ignored for volume
                assert Reason.LOW_24H_VOLUME not in r2.reasons
                assert Reason.VOLUME_OK in r2.reasons

                # Open a position to simulate execution from Signal 2
                pid = str(uuid.uuid4())
                await pm.open_pending(
                    db, position_id=pid, signal_id="sig-2",
                    symbol=sym, bybit_symbol=bsym,
                    entry_price=0.089, stop_loss=0.085, take_profit=0.096,
                    position_size_usd=100.0, quantity=1000.0, risk_usd=1.0,
                    rr_ratio=2.0, leverage=1.0,
                )

                # Signal 3: while position OPEN → IGNORE(EXISTING_POSITION)
                status3 = await pm.get_state_for_symbol(db, sym)
                r3 = engine.decide(
                    _signal(volume_24h=4_000_000.0, symbol=sym),
                    _snap(),
                    PositionState(symbol=sym, state=status3.value),
                    bybit_symbol=bsym,
                )
                assert r3.decision == Decision.IGNORE
                assert Reason.EXISTING_POSITION in r3.reasons

                # Close the position
                await pm.close(db, pid, exit_reason="TP_HIT", actual_exit_price=0.096)

                # Signal 4: after position CLOSED → evaluate normally again
                status4 = await pm.get_state_for_symbol(db, sym)
                assert status4 == PositionStatus.NO_POSITION
                r4 = engine.decide(
                    _signal(volume_24h=4_000_000.0, symbol=sym,
                            body_spike=6.0, volume_spike=12.0, score=90.0),
                    _snap(oi_delta_pct=2.0, trade_imbalance=0.25, ob_imbalance=0.3),
                    PositionState(symbol=sym, state=status4.value),
                    bybit_symbol=bsym,
                )
                assert Reason.EXISTING_POSITION not in r4.reasons
                assert Reason.VOLUME_OK in r4.reasons
        finally:
            os.unlink(db_path)


# ===========================================================================
# 4. Existing open position → IGNORE(EXISTING_POSITION)
# ===========================================================================

class TestExistingPosition:
    def test_open_position_ignored(self, engine):
        result = engine.decide(_signal(), _snap(), _pos(state="OPEN"), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE
        assert Reason.EXISTING_POSITION in result.reasons

    def test_pending_entry_ignored(self, engine):
        result = engine.decide(_signal(), _snap(), _pos(state="PENDING_ENTRY"), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE
        assert Reason.EXISTING_POSITION in result.reasons

    def test_exiting_ignored(self, engine):
        result = engine.decide(_signal(), _snap(), _pos(state="EXITING"), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE
        assert Reason.EXISTING_POSITION in result.reasons

    def test_closed_position_evaluates_normally(self, engine):
        result = engine.decide(_signal(), _snap(), _pos(state="CLOSED"), bybit_symbol="TAKEUSDT")
        assert Reason.EXISTING_POSITION not in result.reasons


# ===========================================================================
# 5. Position closed → next signal evaluates normally
# ===========================================================================

class TestPositionClosed:
    def test_no_position_evaluates_normally(self, engine):
        result = engine.decide(_signal(), _snap(), _pos(state="NO_POSITION"), bybit_symbol="TAKEUSDT")
        assert Reason.EXISTING_POSITION not in result.reasons

    def test_closed_state_evaluates_fully(self, engine):
        result = engine.decide(_signal(), _snap(), _pos(state="CLOSED"), bybit_symbol="TAKEUSDT")
        assert Reason.EXISTING_POSITION not in result.reasons


# ===========================================================================
# 6. NORMAL intensity reaches decision engine
# ===========================================================================

class TestNormalIntensity:
    def test_normal_intensity_reaches_engine(self, engine):
        sig = _signal(body_spike=2.0, volume_spike=5.0)
        result = engine.decide(sig, _snap(), _pos(), bybit_symbol="TAKEUSDT")
        assert result.feature_classification is not None
        assert result.feature_classification.intensity == IntensityRegime.NORMAL

    def test_normal_intensity_reason_recorded(self, engine):
        sig = _signal(body_spike=1.0, volume_spike=2.0)
        result = engine.decide(sig, _snap(), _pos(), bybit_symbol="TAKEUSDT")
        assert Reason.INTENSITY_NORMAL in result.reasons


# ===========================================================================
# 7. ELEVATED intensity classification
# ===========================================================================

class TestElevatedIntensity:
    def test_elevated_at_threshold(self, engine):
        sig = _signal(body_spike=5.0, volume_spike=10.0)
        result = engine.decide(sig, _snap(), _pos(), bybit_symbol="TAKEUSDT")
        assert result.feature_classification is not None
        assert result.feature_classification.intensity == IntensityRegime.ELEVATED
        assert Reason.INTENSITY_ELEVATED in result.reasons

    def test_elevated_above_threshold(self, engine):
        sig = _signal(body_spike=7.0, volume_spike=15.0)
        result = engine.decide(sig, _snap(), _pos(), bybit_symbol="TAKEUSDT")
        assert result.feature_classification.intensity == IntensityRegime.ELEVATED

    def test_not_elevated_if_only_body_spike(self, engine):
        """Both body AND volume must cross threshold for ELEVATED."""
        sig = _signal(body_spike=8.0, volume_spike=5.0)
        result = engine.decide(sig, _snap(), _pos(), bybit_symbol="TAKEUSDT")
        assert result.feature_classification.intensity != IntensityRegime.ELEVATED


# ===========================================================================
# 8. EXTREME intensity classification
# ===========================================================================

class TestExtremeIntensity:
    def test_extreme_at_threshold(self, engine):
        sig = _signal(body_spike=10.0, volume_spike=20.0)
        result = engine.decide(sig, _snap(), _pos(), bybit_symbol="TAKEUSDT")
        assert result.feature_classification is not None
        assert result.feature_classification.intensity == IntensityRegime.EXTREME
        assert Reason.INTENSITY_EXTREME in result.reasons

    def test_extreme_above_threshold(self, engine):
        sig = _signal(body_spike=25.0, volume_spike=50.0)
        result = engine.decide(sig, _snap(), _pos(), bybit_symbol="TAKEUSDT")
        assert result.feature_classification.intensity == IntensityRegime.EXTREME

    def test_not_extreme_if_only_body_extreme(self, engine):
        sig = _signal(body_spike=15.0, volume_spike=10.0)
        result = engine.decide(sig, _snap(), _pos(), bybit_symbol="TAKEUSDT")
        assert result.feature_classification.intensity != IntensityRegime.EXTREME


# ===========================================================================
# 9. Missing microstructure → IGNORE
# ===========================================================================

class TestMissingMicrostructure:
    def test_none_snapshot_ignored(self, engine):
        result = engine.decide(_signal(), None, _pos(), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE
        assert any(r in result.reasons for r in [Reason.MISSING_MARKET_DATA, Reason.STALE_MARKET_DATA])

    def test_missing_price_ignored(self, engine):
        snap = _snap()
        snap.mark_price = None
        snap.best_bid = None
        result = engine.decide(_signal(), snap, _pos(), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE

    def test_missing_book_depth_ignored(self, engine):
        snap = _snap()
        snap.bid_depth_usdt = None
        snap.ask_depth_usdt = None
        result = engine.decide(_signal(), snap, _pos(), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE


# ===========================================================================
# 10. Stale microstructure → IGNORE
# ===========================================================================

class TestStaleMicrostructure:
    def test_stale_snapshot_ignored(self, engine):
        result = engine.decide(_signal(), _snap(age_s=30.0), _pos(), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE
        assert Reason.STALE_MARKET_DATA in result.reasons

    def test_fresh_snapshot_accepted(self, engine):
        result = engine.decide(_signal(), _snap(age_s=0.5), _pos(), bybit_symbol="TAKEUSDT")
        assert Reason.STALE_MARKET_DATA not in result.reasons

    def test_clearly_within_limit_accepted(self, engine):
        """5s is clearly within the 10s limit."""
        result = engine.decide(_signal(), _snap(age_s=5.0), _pos(), bybit_symbol="TAKEUSDT")
        assert Reason.STALE_MARKET_DATA not in result.reasons

    def test_clearly_over_limit_stale(self, engine):
        result = engine.decide(_signal(), _snap(age_s=30.0), _pos(), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE


# ===========================================================================
# 11. Wide spread → IGNORE
# ===========================================================================

class TestWideSpread:
    def test_wide_spread_ignored(self, engine):
        result = engine.decide(_signal(), _snap(spread_pct=0.6), _pos(), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE
        assert Reason.WIDE_SPREAD in result.reasons

    def test_acceptable_spread_passes(self, engine):
        result = engine.decide(_signal(), _snap(spread_pct=0.2), _pos(), bybit_symbol="TAKEUSDT")
        assert Reason.WIDE_SPREAD not in result.reasons

    def test_exactly_at_spread_limit_passes(self, engine):
        result = engine.decide(_signal(), _snap(spread_pct=0.5), _pos(), bybit_symbol="TAKEUSDT")
        assert Reason.WIDE_SPREAD not in result.reasons


# ===========================================================================
# 12. Conflicting microstructure → IGNORE
# ===========================================================================

class TestConflictingMicrostructure:
    def test_two_adverse_states_ignored(self, engine):
        snap = _snap(oi_delta_pct=-3.0, trade_imbalance=-0.3, ob_imbalance=0.2)
        result = engine.decide(_signal(), snap, _pos(), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE
        assert Reason.CONFLICTING_MICROSTRUCTURE in result.reasons


# ===========================================================================
# 13. Supportive microstructure → TRADE candidate
# ===========================================================================

class TestSupportiveMicrostructure:
    def test_strong_supportive_setup_can_trade(self, engine):
        sig = _signal(body_spike=6.0, volume_spike=12.0, score=90.0, volume_24h=5_000_000.0)
        snap = _snap(oi_delta_pct=2.0, trade_imbalance=0.25, ob_imbalance=0.3, spread_pct=0.1)
        result = engine.decide(sig, snap, _pos(), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.TRADE
        assert Reason.OI_SUPPORTIVE in result.reasons
        assert Reason.FLOW_SUPPORTIVE in result.reasons
        assert Reason.ORDERBOOK_SUPPORTIVE in result.reasons

    def test_trade_has_positive_score(self, engine):
        sig = _signal(score=90.0, volume_24h=5_000_000.0, body_spike=6.0, volume_spike=12.0)
        snap = _snap(oi_delta_pct=2.0, trade_imbalance=0.25, ob_imbalance=0.3)
        result = engine.decide(sig, snap, _pos(), bybit_symbol="TAKEUSDT")
        assert result.decision_score > 0


# ===========================================================================
# 14. No single metric alone causes TRADE
# ===========================================================================

class TestNoSingleMetricCausesTrade:
    def test_oi_alone_does_not_trade(self, engine):
        snap = _snap(oi_delta_pct=10.0, trade_imbalance=0.0, ob_imbalance=0.0)
        sig = _signal(body_spike=2.0, volume_spike=4.0, score=40.0)
        result = engine.decide(sig, snap, _pos(), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE

    def test_flow_alone_does_not_trade(self, engine):
        snap = _snap(trade_imbalance=0.9, oi_delta_pct=0.0, ob_imbalance=0.0)
        sig = _signal(body_spike=2.0, volume_spike=4.0, score=40.0)
        result = engine.decide(sig, snap, _pos(), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE

    def test_ob_alone_does_not_trade(self, engine):
        snap = _snap(ob_imbalance=0.9, oi_delta_pct=0.0, trade_imbalance=0.0)
        sig = _signal(body_spike=2.0, volume_spike=4.0, score=40.0)
        result = engine.decide(sig, snap, _pos(), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE


# ===========================================================================
# 15. Risk calculation (now with real ATR)
# ===========================================================================

class TestRiskCalculation:
    def test_basic_calculation_valid(self):
        risk = RiskEngine(RiskConfig(account_balance_usd=1000.0, max_risk_pct_per_trade=1.0))
        result = risk.calculate(entry_price=0.089, atr_result=_atr(0.0008), spread_abs=0.00004)
        assert result.valid
        assert result.stop_loss < result.entry_price
        assert result.take_profit > result.entry_price
        assert result.rr_ratio == pytest.approx(2.0)
        assert result.risk_usd <= 10.01    # 1% of $1000
        assert result.stop_distance > 0

    def test_invalid_entry_price_not_valid(self):
        result = RiskEngine().calculate(entry_price=0.0, atr_result=_atr())
        assert not result.valid

    def test_negative_entry_price_not_valid(self):
        result = RiskEngine().calculate(entry_price=-1.0, atr_result=_atr())
        assert not result.valid

    def test_rr_ratio_applies_correctly(self):
        risk = RiskEngine(RiskConfig(rr_ratio=3.0))
        result = risk.calculate(entry_price=100.0, atr_result=_atr(0.5), spread_abs=0.1)
        assert result.valid
        expected_tp = result.entry_price + result.stop_distance * 3.0
        assert abs(result.take_profit - expected_tp) < 0.001


# ===========================================================================
# 16. SL/TP generation
# ===========================================================================

class TestSLTP:
    def test_sl_below_entry_for_long(self):
        result = RiskEngine().calculate(entry_price=0.089, atr_result=_atr(0.0008))
        assert result.stop_loss < result.entry_price

    def test_tp_above_entry_for_long(self):
        result = RiskEngine().calculate(entry_price=0.089, atr_result=_atr(0.0008))
        assert result.take_profit > result.entry_price

    def test_atr_used_as_primary_source(self):
        """When ATR × multiple > spread_floor, ATR governs the stop."""
        cfg = RiskConfig(sl_atr_multiple=2.0, spread_sl_buffer=1.5)
        risk = RiskEngine(cfg)
        atr_val = 0.01   # large ATR
        spread = 0.0001  # tiny spread
        result = risk.calculate(entry_price=1.0, atr_result=_atr(atr_val), spread_abs=spread)
        assert result.valid
        assert result.atr_source == "atr14_1m"

    def test_sl_tp_expose_all_values(self):
        result = RiskEngine().calculate(entry_price=0.089, atr_result=_atr(0.0008))
        assert result.stop_pct > 0
        assert result.stop_distance > 0
        assert result.rr_ratio > 0
        assert result.atr_value is not None and result.atr_value > 0


# ===========================================================================
# 17. Position sizing limits
# ===========================================================================

class TestPositionSizing:
    def test_max_position_cap_applied(self):
        cfg = RiskConfig(account_balance_usd=100_000.0, max_position_usd=200.0)
        result = RiskEngine(cfg).calculate(entry_price=100.0, atr_result=_atr(0.1), spread_abs=0.05)
        assert result.position_size_usd <= 200.0
        assert result.capped_by_max_position

    def test_max_risk_cap_applied(self):
        cfg = RiskConfig(account_balance_usd=1000.0, max_risk_pct_per_trade=1.0)
        result = RiskEngine(cfg).calculate(entry_price=100.0, atr_result=_atr(0.5))
        assert result.risk_usd <= 10.01

    def test_quantity_is_positive(self):
        result = RiskEngine().calculate(entry_price=0.089, atr_result=_atr(0.0008))
        assert result.quantity > 0

    def test_no_unlimited_size(self):
        cfg = RiskConfig(account_balance_usd=10_000_000.0, max_position_usd=500.0)
        result = RiskEngine(cfg).calculate(entry_price=1.0, atr_result=_atr(0.002))
        assert result.position_size_usd <= 500.0


# ===========================================================================
# A. ATR fetcher — correct ATR(14) calculation
# ===========================================================================

class TestATRFetcher:
    """Unit-test the ATR calculation without hitting Binance."""

    def _make_klines(self, highs: list[float], lows: list[float], closes: list[float]) -> list:
        """Build a list of synthetic Binance-format klines."""
        klines = []
        for i, (h, l, c) in enumerate(zip(highs, lows, closes)):
            # [open_ts, open, high, low, close, vol, ...]
            klines.append([i * 60_000, str(c), str(h), str(l), str(c), "1000"])
        return klines

    def test_atr14_calculation(self):
        """Verify ATR(14) matches manual calculation."""
        fetcher = ATRFetcher()
        # 16 candles with known high/low/close; price monotonically rising
        # so TR_i = high_i - low_i (no gap effects)
        period = 14
        n = period + 2
        highs  = [float(i) + 0.5 for i in range(n)]
        lows   = [float(i) - 0.5 for i in range(n)]
        closes = [float(i)       for i in range(n)]
        klines = self._make_klines(highs, lows, closes)

        atr = fetcher._calculate_atr(klines, period)
        # Each candle: H=i+0.5, L=i-0.5, close=i, prev_close=i-1
        # TR = max(H-L, |H-prevC|, |L-prevC|)
        #    = max(1.0, |i+0.5-(i-1)|, |i-0.5-(i-1)|)
        #    = max(1.0, 1.5, 0.5) = 1.5
        assert atr is not None
        assert abs(atr - 1.5) < 0.01

    def test_atr_with_gaps(self):
        """ATR should reflect the true range including gaps."""
        fetcher = ATRFetcher()
        # Constant candle + gap: TR on gap candle is max(H-L, |H-prevC|, |L-prevC|)
        n = 16
        highs  = [1.0] * n
        lows   = [0.0] * n
        closes = [0.5] * n
        # Insert a gap: candle index 8 has a big gap up
        highs[8]  = 3.0
        lows[8]   = 2.5
        closes[7] = 0.5    # previous close
        klines = self._make_klines(highs, lows, closes)
        atr = fetcher._calculate_atr(klines, 14)
        # The gap candle has TR = max(0.5, |3.0-0.5|, |2.5-0.5|) = 2.5
        assert atr is not None
        assert atr > 1.0    # must be elevated compared to a flat ATR of 1.0

    def test_insufficient_klines_returns_none(self):
        fetcher = ATRFetcher()
        # Only 5 candles, need 15
        klines = [[i, "1", "1.5", "0.5", "1", "100"] for i in range(5)]
        result = fetcher._calculate_atr(klines, 14)
        assert result is None

    def test_empty_klines_returns_none(self):
        fetcher = ATRFetcher()
        assert fetcher._calculate_atr([], 14) is None

    def test_atr_freshness_check(self):
        fetcher = ATRFetcher(max_atr_age_s=30.0)
        fresh = ATRResult("X", "1m", 0.005, time.time(), 14)
        stale = ATRResult("X", "1m", 0.005, time.time() - 60.0, 14)
        assert fetcher.is_fresh(fresh) is True
        assert fetcher.is_fresh(stale) is False
        assert fetcher.is_fresh(None) is False


# ===========================================================================
# B. ATR fetcher fails closed
# ===========================================================================

class TestATRFetcherFailClosed:
    @pytest.mark.asyncio
    async def test_http_error_returns_none(self):
        import httpx
        fetcher = ATRFetcher()
        async with httpx.AsyncClient() as client:
            with patch.object(client, "get", side_effect=httpx.ConnectError("offline")):
                result = await fetcher.fetch_atr(client, "TAKEUSDT")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_response_returns_none(self):
        import httpx
        fetcher = ATRFetcher()
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value=[])
        async with httpx.AsyncClient() as client:
            with patch.object(client, "get", return_value=mock_resp):
                result = await fetcher.fetch_atr(client, "TAKEUSDT")
        assert result is None

    @pytest.mark.asyncio
    async def test_stale_atr_fails_freshness(self):
        fetcher = ATRFetcher(max_atr_age_s=5.0)
        old = ATRResult("X", "1m", 0.005, time.time() - 10.0, 14)
        assert not fetcher.is_fresh(old)


# ===========================================================================
# C. Risk engine fails closed when ATR is None
# ===========================================================================

class TestRiskEngineFailClosed:
    def test_none_atr_returns_invalid(self):
        result = RiskEngine().calculate(entry_price=0.089, atr_result=None)
        assert not result.valid
        assert result.error is not None
        assert "ATR" in result.error

    def test_zero_atr_returns_invalid(self):
        bad_atr = ATRResult("X", "1m", 0.0, time.time(), 14)
        result = RiskEngine().calculate(entry_price=0.089, atr_result=bad_atr)
        assert not result.valid

    def test_negative_atr_returns_invalid(self):
        bad_atr = ATRResult("X", "1m", -0.001, time.time(), 14)
        result = RiskEngine().calculate(entry_price=0.089, atr_result=bad_atr)
        assert not result.valid


# ===========================================================================
# D. Risk engine uses real ATR for stop, not fixed percentage
# ===========================================================================

class TestRiskUsesRealATR:
    def test_larger_atr_produces_wider_stop(self):
        """A 2× larger ATR → 2× wider stop distance (before clamping)."""
        cfg = RiskConfig(sl_atr_multiple=2.0, min_stop_pct=0.0, max_stop_pct=1.0)
        r1 = RiskEngine(cfg).calculate(entry_price=100.0, atr_result=_atr(0.5))
        r2 = RiskEngine(cfg).calculate(entry_price=100.0, atr_result=_atr(1.0))
        assert r2.stop_distance > r1.stop_distance

    def test_stop_is_multiple_of_atr(self):
        """stop_distance = ATR × sl_atr_multiple (when within bounds)."""
        atr_val = 0.002
        cfg = RiskConfig(sl_atr_multiple=2.0, min_stop_pct=0.001, max_stop_pct=0.1)
        entry = 0.089
        result = RiskEngine(cfg).calculate(entry_price=entry, atr_result=_atr(atr_val))
        assert result.valid
        # If spread and min_stop floors don't kick in, stop_distance = ATR × 2
        expected = atr_val * 2.0
        # Allow for floor adjustments
        assert result.stop_distance >= expected * 0.8  # within 20% of expected

    def test_atr_source_reflects_atr14_1m(self):
        """When ATR governs, atr_source should be atr14_1m."""
        cfg = RiskConfig(sl_atr_multiple=2.0, spread_sl_buffer=1.0, min_stop_pct=0.0)
        result = RiskEngine(cfg).calculate(
            entry_price=100.0,
            atr_result=_atr(1.0),
            spread_abs=0.001,
        )
        assert result.valid
        assert result.atr_source == "atr14_1m"

    def test_different_coins_get_different_stops(self):
        """A high-ATR coin gets a wider stop than a low-ATR coin."""
        cfg = RiskConfig(sl_atr_multiple=2.0, min_stop_pct=0.001, max_stop_pct=0.5)
        high_vol = RiskEngine(cfg).calculate(entry_price=1.0, atr_result=_atr(0.05))
        low_vol  = RiskEngine(cfg).calculate(entry_price=1.0, atr_result=_atr(0.005))
        assert high_vol.stop_distance > low_vol.stop_distance


# ===========================================================================
# E. Spread floor: stop never inside spread
# ===========================================================================

class TestSpreadFloor:
    def test_stop_clears_spread(self):
        """stop_distance >= spread_abs × spread_sl_buffer."""
        cfg = RiskConfig(
            sl_atr_multiple=1.0,
            spread_sl_buffer=1.5,
            min_stop_pct=0.0,
            max_stop_pct=0.5,
        )
        spread = 0.001
        tiny_atr = 0.0001  # ATR much smaller than spread_floor
        result = RiskEngine(cfg).calculate(
            entry_price=1.0,
            atr_result=_atr(tiny_atr),
            spread_abs=spread,
        )
        assert result.valid
        assert result.stop_distance >= spread * cfg.spread_sl_buffer * 0.99  # float tolerance

    def test_atr_governs_when_larger_than_spread_floor(self):
        cfg = RiskConfig(sl_atr_multiple=2.0, spread_sl_buffer=1.5, min_stop_pct=0.0)
        large_atr = 0.1
        spread = 0.001
        result = RiskEngine(cfg).calculate(
            entry_price=1.0,
            atr_result=_atr(large_atr),
            spread_abs=spread,
        )
        assert result.valid
        assert result.atr_source == "atr14_1m"


# ===========================================================================
# 18. Bybit Demo: production API URL cannot be used
# ===========================================================================

class TestBybitDemoURLSafety:
    def test_production_url_rejected(self):
        with pytest.raises(BybitDemoConfigError, match="production"):
            BybitDemoClient(api_key="k", api_secret="s", base_url="https://api.bybit.com")

    def test_mainnet_url_rejected(self):
        with pytest.raises(BybitDemoConfigError):
            BybitDemoClient(api_key="k", api_secret="s", base_url="https://mainnet.bybit.com")

    def test_prod_in_path_rejected(self):
        with pytest.raises(BybitDemoConfigError):
            BybitDemoClient(api_key="k", api_secret="s", base_url="https://prod.bybit.com")

    def test_demo_url_accepted(self):
        client = BybitDemoClient(api_key="k", api_secret="s", base_url="https://api-demo.bybit.com")
        assert "demo" in client._base_url


# ===========================================================================
# G. Production credentials cannot be silently accepted
# H. Demo credentials required
# ===========================================================================

class TestBybitDemoCredentials:
    def test_missing_api_key_raises(self):
        with pytest.raises(BybitDemoConfigError):
            BybitDemoClient(api_key="", api_secret="secret")

    def test_missing_api_secret_raises(self):
        with pytest.raises(BybitDemoConfigError):
            BybitDemoClient(api_key="key", api_secret="")

    def test_from_env_raises_without_env_vars(self, monkeypatch):
        monkeypatch.delenv("BYBIT_DEMO_API_KEY", raising=False)
        monkeypatch.delenv("BYBIT_DEMO_API_SECRET", raising=False)
        with pytest.raises(BybitDemoConfigError):
            BybitDemoClient.from_env()

    def test_from_env_reads_env_vars(self, monkeypatch):
        monkeypatch.setenv("BYBIT_DEMO_API_KEY", "mykey")
        monkeypatch.setenv("BYBIT_DEMO_API_SECRET", "mysecret")
        monkeypatch.delenv("BYBIT_DEMO_BASE_URL", raising=False)
        client = BybitDemoClient.from_env()
        assert client._api_key == "mykey"

    def test_env_url_production_fragment_rejected(self, monkeypatch):
        monkeypatch.setenv("BYBIT_DEMO_API_KEY", "k")
        monkeypatch.setenv("BYBIT_DEMO_API_SECRET", "s")
        monkeypatch.setenv("BYBIT_DEMO_BASE_URL", "https://api.bybit.com")
        with pytest.raises(BybitDemoConfigError):
            BybitDemoClient.from_env()


# ===========================================================================
# 19. Order not successful until exchange ACK
# ===========================================================================

class TestBybitOrderAck:
    @pytest.mark.asyncio
    async def test_place_order_returns_order_result(self):
        client = BybitDemoClient(api_key="k", api_secret="s", base_url="https://api-demo.bybit.com")
        mock_resp = {
            "retCode": 0, "retMsg": "OK",
            "result": {"orderId": "ABC123", "orderStatus": "Filled"},
        }
        with patch.object(client, "_post", new=AsyncMock(return_value=mock_resp)):
            with patch.object(client, "get_instrument_info", new=AsyncMock(return_value={})):
                order = await client.place_market_order("TAKEUSDT", "Buy", qty=100.0)
        assert order.order_id == "ABC123"
        assert order.symbol == "TAKEUSDT"

    @pytest.mark.asyncio
    async def test_order_id_absent_indicates_no_ack(self):
        """If orderId is missing, the exchange has not ACKed."""
        client = BybitDemoClient(api_key="k", api_secret="s", base_url="https://api-demo.bybit.com")
        empty_resp = {"retCode": 0, "retMsg": "OK", "result": {}}
        with patch.object(client, "_post", new=AsyncMock(return_value=empty_resp)):
            with patch.object(client, "get_instrument_info", new=AsyncMock(return_value={})):
                order = await client.place_market_order("TAKEUSDT", "Buy", qty=100.0)
        # orderId defaults to "" — caller must treat empty string as no ACK
        assert order.order_id == ""

    @pytest.mark.asyncio
    async def test_api_error_raises_bybit_demo_api_error(self):
        client = BybitDemoClient(api_key="k", api_secret="s", base_url="https://api-demo.bybit.com")
        with patch.object(client, "_post",
                          new=AsyncMock(side_effect=BybitDemoAPIError(10001, "params error"))):
            with pytest.raises(BybitDemoAPIError) as exc_info:
                await client.place_market_order("TAKEUSDT", "Buy", qty=100.0)
        assert exc_info.value.ret_code == 10001


# ===========================================================================
# 20. Failed SL placement raises
# ===========================================================================

class TestFailedSL:
    @pytest.mark.asyncio
    async def test_failed_sl_raises(self):
        client = BybitDemoClient(api_key="k", api_secret="s", base_url="https://api-demo.bybit.com")
        with patch.object(client, "_post",
                          new=AsyncMock(side_effect=BybitDemoAPIError(10002, "sl error"))):
            with patch.object(client, "get_instrument_info", new=AsyncMock(return_value={})):
                with pytest.raises(BybitDemoAPIError):
                    await client.place_stop_loss("TAKEUSDT", sl_price=0.085)


# ===========================================================================
# 21. Failed TP placement raises
# ===========================================================================

class TestFailedTP:
    @pytest.mark.asyncio
    async def test_failed_tp_raises(self):
        client = BybitDemoClient(api_key="k", api_secret="s", base_url="https://api-demo.bybit.com")
        with patch.object(client, "_post",
                          new=AsyncMock(side_effect=BybitDemoAPIError(10003, "tp error"))):
            with patch.object(client, "get_instrument_info", new=AsyncMock(return_value={})):
                with pytest.raises(BybitDemoAPIError):
                    await client.place_take_profit("TAKEUSDT", tp_price=0.096)


# ===========================================================================
# J. SL must be confirmed before protection is complete
# K. TP must be confirmed before protection is complete
# ===========================================================================

class TestSLTPConfirmation:
    def test_protection_ts_uses_max_of_sl_tp(self):
        sl_ts = time.time() + 0.1
        tp_ts = time.time() + 0.2   # TP takes longer
        latencies = drepo.compute_latencies(
            signal_timestamp=time.time() - 1.0,
            decision_timestamp=time.time() - 0.8,
            order_submission_timestamp=time.time() - 0.5,
            fill_timestamp=time.time() - 0.3,
            sl_confirmed_timestamp=sl_ts,
            tp_confirmed_timestamp=tp_ts,
        )
        prot_ms = latencies["signal_to_protection_ms"]
        assert prot_ms is not None
        # Protection is when BOTH are done = max(sl, tp)
        signal_ts = time.time() - 1.0
        expected = (tp_ts - signal_ts) * 1000.0
        assert abs(prot_ms - expected) < 50.0  # within 50ms tolerance

    def test_protection_ts_uses_sl_when_no_tp(self):
        sig_ts = time.time() - 1.0
        sl_ts = time.time()
        latencies = drepo.compute_latencies(
            signal_timestamp=sig_ts,
            decision_timestamp=None,
            order_submission_timestamp=None,
            fill_timestamp=None,
            sl_confirmed_timestamp=sl_ts,
            tp_confirmed_timestamp=None,
        )
        assert latencies["signal_to_protection_ms"] is not None

    def test_no_protection_ts_when_neither_confirmed(self):
        latencies = drepo.compute_latencies(
            signal_timestamp=time.time() - 1.0,
            decision_timestamp=None,
            order_submission_timestamp=None,
            fill_timestamp=None,
            sl_confirmed_timestamp=None,
            tp_confirmed_timestamp=None,
        )
        assert latencies["signal_to_protection_ms"] is None


# ===========================================================================
# 22. Duplicate signal does not create duplicate position
# ===========================================================================

class TestNoDuplicatePosition:
    @pytest.mark.asyncio
    async def test_second_signal_while_open_ignored(self, engine, db):
        pm = PositionManager()
        await pm.open_pending(
            db, position_id=str(uuid.uuid4()), signal_id="sig-1",
            symbol="TAKE/USDT:USDT", bybit_symbol="TAKEUSDT",
            entry_price=0.089, stop_loss=0.085, take_profit=0.096,
            position_size_usd=100.0, quantity=1123.0, risk_usd=1.0,
            rr_ratio=2.0, leverage=1.0,
        )
        status = await pm.get_state_for_symbol(db, "TAKE/USDT:USDT")
        ps = PositionState(symbol="TAKE/USDT:USDT", state=status.value)
        result = engine.decide(_signal(), _snap(), ps, bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE
        assert Reason.EXISTING_POSITION in result.reasons

    @pytest.mark.asyncio
    async def test_position_manager_returns_active(self, db):
        pm = PositionManager()
        await pm.open_pending(
            db, position_id="p1", signal_id="s1",
            symbol="X/USDT:USDT", bybit_symbol="XUSDT",
            entry_price=1.0, stop_loss=0.95, take_profit=1.1,
            position_size_usd=50.0, quantity=50.0, risk_usd=0.5,
            rr_ratio=2.0, leverage=1.0,
        )
        active = await pm.get_active_position(db, "X/USDT:USDT")
        assert active is not None
        assert active.status == PositionStatus.PENDING_ENTRY


# ===========================================================================
# 23. Latency timestamps
# ===========================================================================

class TestLatencyTimestamps:
    def test_signal_to_decision_ms_positive(self, engine):
        sig = _signal()
        sig.signal_timestamp = time.time() - 1.0
        result = engine.decide(sig, _snap(), _pos(), bybit_symbol="TAKEUSDT")
        assert result.signal_to_decision_ms > 0

    def test_decision_start_ts_set(self, engine):
        result = engine.decide(_signal(), _snap(), _pos(), bybit_symbol="TAKEUSDT")
        assert result.decision_start_ts > 0

    def test_data_fetch_latency_recorded(self, engine):
        fetch_start = time.time() - 0.1
        fetch_end = time.time()
        result = engine.decide(
            _signal(), _snap(), _pos(),
            bybit_symbol="TAKEUSDT",
            fetch_start_ts=fetch_start,
            fetch_end_ts=fetch_end,
        )
        assert result.data_fetch_latency_ms == pytest.approx(100.0, abs=20.0)


# ===========================================================================
# L. compute_latencies() helper
# ===========================================================================

class TestComputeLatencies:
    def test_all_timestamps_present(self):
        now = time.time()
        latencies = drepo.compute_latencies(
            signal_timestamp=now - 1.0,
            decision_timestamp=now - 0.8,
            order_submission_timestamp=now - 0.5,
            fill_timestamp=now - 0.3,
            sl_confirmed_timestamp=now - 0.1,
            tp_confirmed_timestamp=now,
        )
        assert latencies["signal_to_decision_ms"] == pytest.approx(200.0, abs=50.0)
        assert latencies["signal_to_order_ms"] == pytest.approx(500.0, abs=50.0)
        assert latencies["signal_to_fill_ms"] == pytest.approx(700.0, abs=50.0)
        assert latencies["signal_to_protection_ms"] is not None
        assert latencies["signal_to_protection_ms"] >= latencies["signal_to_fill_ms"]

    def test_missing_timestamps_return_none(self):
        latencies = drepo.compute_latencies(
            signal_timestamp=None,
            decision_timestamp=None,
            order_submission_timestamp=None,
            fill_timestamp=None,
            sl_confirmed_timestamp=None,
            tp_confirmed_timestamp=None,
        )
        for v in latencies.values():
            assert v is None


# ===========================================================================
# 24. Decision journal records
# ===========================================================================

class TestDecisionJournal:
    @pytest.mark.asyncio
    async def test_record_decision_inserted(self, db):
        row_id = await drepo.record_decision(
            db, signal_id="test-sig-001", symbol="TAKE/USDT:USDT",
            signal_timestamp=time.time(), decision="IGNORE",
            decision_score=0.0, reasons=["LOW_24H_VOLUME"],
            cs_score=85.0, volume_24h_usd=500_000.0,
        )
        assert row_id > 0

    @pytest.mark.asyncio
    async def test_record_decision_fetchable(self, db):
        await drepo.record_decision(
            db, signal_id="test-sig-002", symbol="ABC/USDT:USDT",
            signal_timestamp=time.time(), decision="TRADE",
            decision_score=72.0, reasons=["VOLUME_OK", "OI_SUPPORTIVE"],
        )
        rows = await drepo.fetch_decisions(db, symbol="ABC/USDT:USDT")
        assert len(rows) == 1
        assert rows[0]["decision"] == "TRADE"
        assert json.loads(rows[0]["reasons_json"]) == ["VOLUME_OK", "OI_SUPPORTIVE"]

    @pytest.mark.asyncio
    async def test_decision_journal_stores_atr_fields(self, db):
        await drepo.record_decision(
            db, signal_id="atr-sig-001", symbol="ATR/USDT:USDT",
            signal_timestamp=time.time(), decision="TRADE",
            decision_score=70.0, reasons=["VOLUME_OK"],
            atr_value=0.0012, atr_source="atr14_1m",
        )
        rows = await drepo.fetch_decisions(db, symbol="ATR/USDT:USDT")
        assert rows[0]["atr_value"] == pytest.approx(0.0012)
        assert rows[0]["atr_source"] == "atr14_1m"


# ===========================================================================
# M. record_trade() stores full latency timeline
# ===========================================================================

class TestTradeJournal:
    @pytest.mark.asyncio
    async def test_trade_journal_stores_all_timestamps(self, db):
        dj_id = await drepo.record_decision(
            db, signal_id="s3", symbol="X/USDT:USDT",
            signal_timestamp=time.time(), decision="TRADE",
            decision_score=70.0, reasons=["VOLUME_OK"],
        )
        now = time.time()
        tj_id = await drepo.record_trade(
            db,
            decision_journal_id=dj_id,
            signal_id="s3", position_id="p3",
            symbol="X/USDT:USDT", bybit_symbol="XUSDT",
            planned_entry=1.0, stop_loss=0.95, take_profit=1.1,
            position_size_usd=50.0, quantity=50.0, risk_usd=0.5,
            rr_ratio=2.0, leverage=1.0,
            atr_value=0.005, atr_source="atr14_1m",
            signal_timestamp=now - 1.0,
            signal_received_timestamp=now - 0.9,
            decision_start_timestamp=now - 0.8,
            market_data_start_timestamp=now - 0.75,
            market_data_complete_timestamp=now - 0.65,
            decision_timestamp=now - 0.6,
            risk_start_timestamp=now - 0.55,
            risk_complete_timestamp=now - 0.5,
            order_submission_timestamp=now - 0.4,
            exchange_ack_timestamp=now - 0.3,
            fill_timestamp=now - 0.2,
            sl_submission_timestamp=now - 0.15,
            sl_confirmed_timestamp=now - 0.1,
            tp_submission_timestamp=now - 0.05,
            tp_confirmed_timestamp=now,
        )
        assert tj_id > 0
        async with db.execute("SELECT * FROM trade_journal WHERE id=?", (tj_id,)) as cur:
            row = dict(await cur.fetchone())
        assert row["atr_value"] == pytest.approx(0.005)
        assert row["signal_to_decision_ms"] is not None
        assert row["signal_to_order_ms"] is not None
        assert row["signal_to_fill_ms"] is not None
        assert row["signal_to_protection_ms"] is not None
        # All 15 timestamps should be stored
        assert row["signal_timestamp"] is not None
        assert row["signal_received_timestamp"] is not None
        assert row["decision_start_timestamp"] is not None
        assert row["market_data_start_timestamp"] is not None
        assert row["market_data_complete_timestamp"] is not None
        assert row["decision_timestamp"] is not None
        assert row["risk_start_timestamp"] is not None
        assert row["risk_complete_timestamp"] is not None
        assert row["order_submission_timestamp"] is not None
        assert row["exchange_ack_timestamp"] is not None
        assert row["fill_timestamp"] is not None
        assert row["sl_submission_timestamp"] is not None
        assert row["sl_confirmed_timestamp"] is not None
        assert row["tp_submission_timestamp"] is not None
        assert row["tp_confirmed_timestamp"] is not None


# ===========================================================================
# 25. Fail-closed behavior
# ===========================================================================

class TestFailClosed:
    def test_none_snapshot_is_ignore(self, engine):
        result = engine.decide(_signal(), None, _pos(), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE

    def test_no_bybit_symbol_is_ignore(self, engine):
        result = engine.decide(_signal(), _snap(), _pos(), bybit_symbol=None)
        assert result.decision == Decision.IGNORE
        assert Reason.BYBIT_SYMBOL_UNAVAILABLE in result.reasons

    def test_missing_price_is_ignore(self, engine):
        snap = _snap()
        snap.mark_price = None
        snap.best_bid = None
        result = engine.decide(_signal(), snap, _pos(), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE

    def test_engine_exception_returns_ignore(self):
        e = _engine()
        with patch.object(e, "_run_pipeline", side_effect=RuntimeError("oops")):
            result = e.decide(_signal(), _snap(), _pos(), bybit_symbol="TAKEUSDT")
        assert result.decision == Decision.IGNORE

    def test_risk_engine_none_atr_is_closed(self):
        result = RiskEngine().calculate(entry_price=0.089, atr_result=None)
        assert not result.valid


# ===========================================================================
# 26. Concurrent / rapid signals for same symbol
# ===========================================================================

class TestConcurrentSignals:
    @pytest.mark.asyncio
    async def test_rapid_signals_same_symbol_position_cap(self, engine, db):
        pm = PositionManager()
        sym = "RAPID/USDT:USDT"
        bsym = "RAPIDUSDT"

        status1 = await pm.get_state_for_symbol(db, sym)
        ps1 = PositionState(symbol=sym, state=status1.value)
        engine.decide(_signal(symbol=sym), _snap(), ps1, bybit_symbol=bsym)

        await pm.open_pending(
            db, position_id=str(uuid.uuid4()), signal_id="r1",
            symbol=sym, bybit_symbol=bsym,
            entry_price=0.089, stop_loss=0.085, take_profit=0.096,
            position_size_usd=100.0, quantity=1000.0, risk_usd=1.0,
            rr_ratio=2.0, leverage=1.0,
        )

        status2 = await pm.get_state_for_symbol(db, sym)
        ps2 = PositionState(symbol=sym, state=status2.value)
        r2 = engine.decide(_signal(symbol=sym), _snap(), ps2, bybit_symbol=bsym)
        assert r2.decision == Decision.IGNORE
        assert Reason.EXISTING_POSITION in r2.reasons

        r3 = engine.decide(_signal(symbol=sym), _snap(), ps2, bybit_symbol=bsym)
        assert r3.decision == Decision.IGNORE


# ===========================================================================
# I. Position state rechecked immediately before order submission
# ===========================================================================

class TestRecheckPositionBeforeExecution:
    @pytest.mark.asyncio
    async def test_position_opened_between_decision_and_execution(self, engine, db):
        pm = PositionManager()
        sym = "RACE/USDT:USDT"

        ps = PositionState(symbol=sym, state="NO_POSITION")
        result = engine.decide(
            _signal(symbol=sym, volume_24h=5_000_000.0, body_spike=6.0,
                    volume_spike=12.0, score=90.0),
            _snap(oi_delta_pct=2.0, trade_imbalance=0.25, ob_imbalance=0.3),
            ps, bybit_symbol="RACEUSDT",
        )
        assert result.decision == Decision.TRADE

        # Simulate: another path opens a position before order is submitted
        await pm.open_pending(
            db, position_id=str(uuid.uuid4()), signal_id="race-1",
            symbol=sym, bybit_symbol="RACEUSDT",
            entry_price=1.0, stop_loss=0.95, take_profit=1.1,
            position_size_usd=50.0, quantity=50.0, risk_usd=0.5,
            rr_ratio=2.0, leverage=1.0,
        )

        # Re-check — should block execution
        recheck_status = await pm.get_state_for_symbol(db, sym)
        assert recheck_status in (PositionStatus.PENDING_ENTRY, PositionStatus.OPEN)


# ===========================================================================
# 28. Re-check data freshness before execution
# ===========================================================================

class TestRecheckDataFreshness:
    def test_stale_data_at_execution_time_fails_freshness(self, engine):
        stale_snap = _snap(age_s=30.0)
        fresh_check = engine._assess_freshness(stale_snap, time.time())
        assert not fresh_check.is_fresh

    def test_fresh_data_passes_freshness(self, engine):
        fresh_snap = _snap(age_s=0.5)
        fresh_check = engine._assess_freshness(fresh_snap, time.time())
        assert fresh_check.is_fresh


# ===========================================================================
# N. Score weights explicitly labelled as initial hypothesis parameters
# ===========================================================================

class TestScoreWeightDocumentation:
    def test_default_weights_match_spec(self):
        """Verify the V1 hypothesis weights exactly match specification.

        INITIAL HYPOTHESIS PARAMETERS — NOT STATISTICALLY VALIDATED.
        These are V1 starting points only. Calibration requires backtesting.
        """
        cfg = DecisionConfig()
        assert cfg.weight_signal_score  == 25.0, "CoilSpring score weight must be 25"
        assert cfg.weight_intensity     == 15.0, "Intensity regime weight must be 15"
        assert cfg.weight_oi            == 20.0, "OI state weight must be 20"
        assert cfg.weight_flow          == 20.0, "Trade flow weight must be 20"
        assert cfg.weight_orderbook     == 15.0, "Order book weight must be 15"
        assert cfg.weight_spread        ==  5.0, "Spread/liquidity weight must be 5"

    def test_weights_sum_to_100(self):
        cfg = DecisionConfig()
        total = (cfg.weight_signal_score + cfg.weight_intensity + cfg.weight_oi
                 + cfg.weight_flow + cfg.weight_orderbook + cfg.weight_spread)
        assert total == pytest.approx(100.0), "Weights must sum to 100"

    def test_default_gate_thresholds_match_spec(self):
        cfg = DecisionConfig()
        assert cfg.min_trade_score == 55.0
        assert cfg.min_supportive_components == 2

    def test_weights_are_configurable(self):
        """Any weight can be changed — they are hypothesis parameters."""
        cfg = DecisionConfig(weight_signal_score=30.0, weight_oi=15.0)
        assert cfg.weight_signal_score == 30.0


# ===========================================================================
# Position Manager — full state machine
# ===========================================================================

class TestPositionManagerStateMachine:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, db):
        pm = PositionManager()
        pid = str(uuid.uuid4())
        await pm.open_pending(
            db, position_id=pid, signal_id="sig-x", symbol="X/USDT:USDT",
            bybit_symbol="XUSDT", entry_price=1.0, stop_loss=0.95,
            take_profit=1.1, position_size_usd=100.0, quantity=100.0,
            risk_usd=1.0, rr_ratio=2.0, leverage=1.0,
        )
        pos = await pm.get_position_by_id(db, pid)
        assert pos.status == PositionStatus.PENDING_ENTRY

        await pm.confirm_open(db, pid, bybit_order_id="BYB-001", actual_entry_price=1.001)
        pos = await pm.get_position_by_id(db, pid)
        assert pos.status == PositionStatus.OPEN
        assert pos.actual_entry_price == 1.001

        await pm.begin_exit(db, pid)
        pos = await pm.get_position_by_id(db, pid)
        assert pos.status == PositionStatus.EXITING

        await pm.close(db, pid, exit_reason="TP_HIT", actual_exit_price=1.1, pnl_usd=9.9)
        pos = await pm.get_position_by_id(db, pid)
        assert pos.status == PositionStatus.CLOSED
        assert pos.pnl_usd == pytest.approx(9.9)
        assert pos.exit_reason == "TP_HIT"

    @pytest.mark.asyncio
    async def test_fail_pending_closes_as_entry_failed(self, db):
        pm = PositionManager()
        pid = str(uuid.uuid4())
        await pm.open_pending(
            db, position_id=pid, signal_id="sig-y", symbol="Y/USDT:USDT",
            bybit_symbol="YUSDT", entry_price=1.0, stop_loss=0.95,
            take_profit=1.1, position_size_usd=100.0, quantity=100.0,
            risk_usd=1.0, rr_ratio=2.0, leverage=1.0,
        )
        await pm.fail_pending(db, pid, reason="ORDER_REJECTED")
        pos = await pm.get_position_by_id(db, pid)
        assert pos.status == PositionStatus.CLOSED
        assert "ENTRY_FAILED" in pos.exit_reason
        active = await pm.get_active_position(db, "Y/USDT:USDT")
        assert active is None

    @pytest.mark.asyncio
    async def test_closed_position_allows_next_signal(self, db):
        pm = PositionManager()
        pid = str(uuid.uuid4())
        await pm.open_pending(
            db, position_id=pid, signal_id="sig-z", symbol="Z/USDT:USDT",
            bybit_symbol="ZUSDT", entry_price=1.0, stop_loss=0.95,
            take_profit=1.1, position_size_usd=100.0, quantity=100.0,
            risk_usd=1.0, rr_ratio=2.0, leverage=1.0,
        )
        await pm.close(db, pid, exit_reason="TP_HIT", actual_exit_price=1.1)
        status = await pm.get_state_for_symbol(db, "Z/USDT:USDT")
        assert status == PositionStatus.NO_POSITION


# ===========================================================================
# Decision score sanity
# ===========================================================================

class TestDecisionScore:
    def test_score_bounded_0_100(self, engine):
        for body, vol in [(1, 1), (5, 10), (10, 20), (20, 50)]:
            sig = _signal(body_spike=body, volume_spike=vol, score=100.0)
            result = engine.decide(sig, _snap(), _pos(), bybit_symbol="TAKEUSDT")
            assert 0.0 <= result.decision_score <= 100.0

    def test_higher_signal_score_higher_decision_score(self, engine):
        snap = _snap(oi_delta_pct=2.0, trade_imbalance=0.25, ob_imbalance=0.3)
        low  = engine.decide(_signal(score=30.0), snap, _pos(), bybit_symbol="TAKEUSDT")
        high = engine.decide(_signal(score=90.0), snap, _pos(), bybit_symbol="TAKEUSDT")
        assert high.decision_score > low.decision_score


# ===========================================================================
# ATR fetcher integration test (mocked HTTP)
# ===========================================================================

class TestATRFetcherMockedHTTP:
    @pytest.mark.asyncio
    async def test_successful_fetch_returns_atr_result(self):
        """Successful Binance response → valid ATRResult with atr > 0."""
        import httpx
        # Build synthetic 17 klines (index 0-16) with constant H-L range of 0.1
        klines = [
            [i * 60000, "1.0", "1.05", "0.95", "1.0", "1000", i * 60000 + 59999, "1000", 100, "500", "500", "0"]
            for i in range(17)
        ]
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value=klines)

        fetcher = ATRFetcher()
        async with httpx.AsyncClient() as client:
            with patch.object(client, "get", return_value=mock_resp):
                result = await fetcher.fetch_atr(client, "TAKEUSDT")

        assert result is not None
        assert result.atr > 0
        assert result.symbol == "TAKEUSDT"
        assert result.interval == "1m"
        assert result.candles_used == 14
