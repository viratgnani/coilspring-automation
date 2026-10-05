"""Unit tests for the ASYMMETRIC_SQUEEZE decision path.

Covers all 12 required scenarios:
 1. Valid asymmetric squeeze → TRADE
 2. Score 84.99 → not squeeze
 3. Flow 0.199 → not squeeze
 4. OB -0.099 → not squeeze
 5. Strong score + supportive flow + neutral OB → normal rules still apply
 6. Strong score + supportive flow + supportive OB → normal rules still apply
 7. Asymmetric squeeze + stale data → IGNORE
 8. Asymmetric squeeze + invalid ATR → IGNORE
 9. Asymmetric squeeze + existing position → IGNORE
10. Asymmetric squeeze + invalid instrument → IGNORE
11. Both paths qualifying → deterministic behavior, no duplicate trade
12. SL/TP/risk behavior unchanged.
"""

from __future__ import annotations

import time
import uuid
import pytest

from coilbrain.atr_fetcher import ATRResult
from coilbrain.decision_engine import (
    Decision,
    DecisionConfig,
    DecisionEngine,
    IntensityRegime,
    MarketState,
    PositionState,
    Reason,
    SignalFeatures,
)
from coilbrain.instrument_validator import InstrumentValidator
from coilbrain.microstructure_fetcher import MicrostructureSnapshot
from coilbrain.risk_engine import RiskConfig, RiskEngine


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _signal(
    score: float = 88.0,
    volume_24h: float = 5_000_000.0,
    body_spike: float = 2.0,
    volume_spike: float = 2.0,
    symbol: str = "MOVR/USDT:USDT",
) -> SignalFeatures:
    """Helper returning a SignalFeatures instance.

    Default body_spike and volume_spike are low (NORMAL intensity) so intensity
    does NOT act as a supportive component unless explicitly tested.
    """
    return SignalFeatures(
        signal_id=str(uuid.uuid4()),
        symbol=symbol,
        score=score,
        grade="A" if score >= 85 else "B",
        body_spike=body_spike,
        volume_spike=volume_spike,
        volume_24h_usd=volume_24h,
        change_24h_pct=8.5,
        breakout_pct=3.2,
        flat_period_days=7.0,
        signal_timestamp=time.time() - 1.0,
    )


def _snap(
    trade_imbalance: float = 0.25,
    ob_imbalance: float = -0.20,
    oi_delta_pct: float = 0.0,
    age_s: float = 0.5,
    spread_pct: float = 0.1,
    mark_price: float = 10.0,
    bid_depth: float = 50_000.0,
    ask_depth: float = 50_000.0,
) -> MicrostructureSnapshot:
    """Helper returning a MicrostructureSnapshot instance.

    Default values:
    - trade_imbalance = +0.25 (FLOW SUPPORTIVE, meets squeeze >= +0.20)
    - ob_imbalance = -0.20 (OB ADVERSE, meets squeeze <= -0.10)
    - oi_delta_pct = 0.0 (OI NEUTRAL)
    - supportive components = 1 (Flow)
    - adverse components = 1 (OB)
    """
    best_bid = mark_price * (1.0 - spread_pct / 200.0)
    best_ask = mark_price * (1.0 + spread_pct / 200.0)
    return MicrostructureSnapshot(
        binance_symbol="MOVRUSDT",
        snapshot_ts=time.time() - age_s,
        oi_usdt=1_500_000.0,
        oi_delta_pct=oi_delta_pct,
        funding_rate=0.0001,
        mark_price=mark_price,
        next_funding_ts=time.time() + 14400,
        best_bid=best_bid,
        best_ask=best_ask,
        spread_abs=best_ask - best_bid,
        spread_pct=spread_pct,
        bid_depth_usdt=bid_depth,
        ask_depth_usdt=ask_depth,
        ob_imbalance=ob_imbalance,
        trade_count=350,
        traded_vol_usdt=80_000.0,
        buy_vol_usdt=50_000.0,
        sell_vol_usdt=30_000.0,
        trade_imbalance=trade_imbalance,
    )


def _pos(state: str = "NO_POSITION", symbol: str = "MOVR/USDT:USDT") -> PositionState:
    return PositionState(symbol=symbol, state=state)


@pytest.fixture
def engine() -> DecisionEngine:
    return DecisionEngine(DecisionConfig())


# ===========================================================================
# 1. Valid asymmetric squeeze → TRADE
# ===========================================================================

def test_1_valid_asymmetric_squeeze_trades(engine: DecisionEngine):
    """Score >= 85, flow >= +0.20, OB <= -0.10 qualifies via ASYMMETRIC_SQUEEZE.

    With only 1 supportive component (Flow) and 1 adverse (OB), normal V1 would
    reject (supportive < 2). The ASYMMETRIC_SQUEEZE path authorizes the trade.
    """
    sig = _signal(score=88.0)
    snap = _snap(trade_imbalance=0.25, ob_imbalance=-0.20, oi_delta_pct=0.0)
    result = engine.decide(sig, snap, _pos(), bybit_symbol="MOVRUSDT")

    assert result.decision == Decision.TRADE
    assert Reason.ASYMMETRIC_SQUEEZE in result.reasons
    assert Reason.SCORE_ABOVE_THRESHOLD not in result.reasons
    assert Reason.INSUFFICIENT_CONFIRMATION not in result.reasons
    assert result.extra.get("decision_path") == "ASYMMETRIC_SQUEEZE"
    assert result.extra.get("qualifying_paths") == ["ASYMMETRIC_SQUEEZE"]


# ===========================================================================
# 2. Score 84.99 → not squeeze
# ===========================================================================

def test_2_score_84_99_not_squeeze(engine: DecisionEngine):
    """CoilSpring score of 84.99 is below 85.0 threshold, rejecting squeeze."""
    sig = _signal(score=84.99)
    snap = _snap(trade_imbalance=0.25, ob_imbalance=-0.20, oi_delta_pct=0.0)
    result = engine.decide(sig, snap, _pos(), bybit_symbol="MOVRUSDT")

    assert result.decision == Decision.IGNORE
    assert Reason.INSUFFICIENT_CONFIRMATION in result.reasons
    assert Reason.ASYMMETRIC_SQUEEZE not in result.reasons


# ===========================================================================
# 3. Flow 0.199 → not squeeze
# ===========================================================================

def test_3_flow_0_199_not_squeeze(engine: DecisionEngine):
    """Flow imbalance of 0.199 is below +0.20 squeeze threshold, rejecting squeeze."""
    sig = _signal(score=88.0)
    snap = _snap(trade_imbalance=0.199, ob_imbalance=-0.20, oi_delta_pct=0.0)
    result = engine.decide(sig, snap, _pos(), bybit_symbol="MOVRUSDT")

    assert result.decision == Decision.IGNORE
    assert Reason.INSUFFICIENT_CONFIRMATION in result.reasons
    assert Reason.ASYMMETRIC_SQUEEZE not in result.reasons


# ===========================================================================
# 4. OB -0.099 → not squeeze
# ===========================================================================

def test_4_ob_neg_0_099_not_squeeze(engine: DecisionEngine):
    """OB imbalance of -0.099 is above -0.10 threshold, rejecting squeeze."""
    sig = _signal(score=88.0)
    snap = _snap(trade_imbalance=0.25, ob_imbalance=-0.099, oi_delta_pct=0.0)
    result = engine.decide(sig, snap, _pos(), bybit_symbol="MOVRUSDT")

    assert result.decision == Decision.IGNORE
    assert Reason.INSUFFICIENT_CONFIRMATION in result.reasons
    assert Reason.ASYMMETRIC_SQUEEZE not in result.reasons


# ===========================================================================
# 5. Strong score + supportive flow + neutral OB → normal rules still apply
# ===========================================================================

def test_5_strong_score_supportive_flow_neutral_ob(engine: DecisionEngine):
    """OB is neutral (0.0), so squeeze does not apply. Normal rules govern.

    Case A: Only 1 supportive component (Flow) → normal path fails (IGNORE).
    Case B: 2 supportive components (Flow + OI) → normal path succeeds (TRADE via NORMAL_V1).
    """
    # Case A: supportive < 2
    sig_a = _signal(score=90.0)
    snap_a = _snap(trade_imbalance=0.25, ob_imbalance=0.0, oi_delta_pct=0.0)
    res_a = engine.decide(sig_a, snap_a, _pos(), bybit_symbol="MOVRUSDT")
    assert res_a.decision == Decision.IGNORE
    assert Reason.INSUFFICIENT_CONFIRMATION in res_a.reasons
    assert Reason.ASYMMETRIC_SQUEEZE not in res_a.reasons

    # Case B: supportive >= 2 (OI is supportive)
    sig_b = _signal(score=90.0)
    snap_b = _snap(trade_imbalance=0.25, ob_imbalance=0.0, oi_delta_pct=2.0)
    res_b = engine.decide(sig_b, snap_b, _pos(), bybit_symbol="MOVRUSDT")
    assert res_b.decision == Decision.TRADE
    assert res_b.extra.get("decision_path") == "NORMAL_V1"
    assert Reason.SCORE_ABOVE_THRESHOLD in res_b.reasons
    assert Reason.ASYMMETRIC_SQUEEZE not in res_b.reasons


# ===========================================================================
# 6. Strong score + supportive flow + supportive OB → normal rules still apply
# ===========================================================================

def test_6_strong_score_supportive_flow_supportive_ob(engine: DecisionEngine):
    """OB is supportive (+0.25), not <= -0.10. Squeeze does not apply.

    With flow and OB both supportive, normal V1 rules qualify the trade.
    """
    sig = _signal(score=90.0)
    snap = _snap(trade_imbalance=0.25, ob_imbalance=0.25, oi_delta_pct=0.0)
    result = engine.decide(sig, snap, _pos(), bybit_symbol="MOVRUSDT")

    assert result.decision == Decision.TRADE
    assert result.extra.get("decision_path") == "NORMAL_V1"
    assert Reason.SCORE_ABOVE_THRESHOLD in result.reasons
    assert Reason.ORDERBOOK_SUPPORTIVE in result.reasons
    assert Reason.ASYMMETRIC_SQUEEZE not in result.reasons


# ===========================================================================
# 7. Asymmetric squeeze + stale data → IGNORE
# ===========================================================================

def test_7_asymmetric_squeeze_stale_data_ignored(engine: DecisionEngine):
    """Squeeze criteria met, but snapshot age > 10s. Stale data gate blocks."""
    sig = _signal(score=88.0)
    snap = _snap(trade_imbalance=0.25, ob_imbalance=-0.20, age_s=15.0)
    result = engine.decide(sig, snap, _pos(), bybit_symbol="MOVRUSDT")

    assert result.decision == Decision.IGNORE
    assert Reason.STALE_MARKET_DATA in result.reasons
    assert Reason.ASYMMETRIC_SQUEEZE not in result.reasons


# ===========================================================================
# 8. Asymmetric squeeze + invalid ATR → IGNORE
# ===========================================================================

def test_8_asymmetric_squeeze_invalid_atr_ignored(engine: DecisionEngine):
    """Squeeze qualifies in DecisionEngine, but invalid ATR fails RiskEngine."""
    sig = _signal(score=88.0)
    snap = _snap(trade_imbalance=0.25, ob_imbalance=-0.20)
    decision_res = engine.decide(sig, snap, _pos(), bybit_symbol="MOVRUSDT")
    assert decision_res.decision == Decision.TRADE
    assert decision_res.extra.get("decision_path") == "ASYMMETRIC_SQUEEZE"

    # Downstream RiskEngine evaluation with missing / invalid ATR
    risk_engine = RiskEngine()
    risk_none = risk_engine.calculate(
        entry_price=snap.mark_price,
        atr_result=None,
        spread_abs=snap.spread_abs,
    )
    assert not risk_none.valid
    assert risk_none.error is not None
    assert "ATR" in risk_none.error

    # Invalid non-positive ATR
    invalid_atr = ATRResult(
        symbol="MOVRUSDT", interval="1m", atr=0.0, fetch_ts=time.time(), candles_used=14
    )
    risk_zero = risk_engine.calculate(
        entry_price=snap.mark_price,
        atr_result=invalid_atr,
        spread_abs=snap.spread_abs,
    )
    assert not risk_zero.valid

    # Executor logic overrides to IGNORE when risk is invalid
    final_decision = Decision.TRADE if risk_none.valid else Decision.IGNORE
    assert final_decision == Decision.IGNORE


# ===========================================================================
# 9. Asymmetric squeeze + existing position → IGNORE
# ===========================================================================

def test_9_asymmetric_squeeze_existing_position_ignored(engine: DecisionEngine):
    """Squeeze criteria met, but symbol already has an OPEN position."""
    sig = _signal(score=88.0)
    snap = _snap(trade_imbalance=0.25, ob_imbalance=-0.20)
    pos = _pos(state="OPEN")
    result = engine.decide(sig, snap, pos, bybit_symbol="MOVRUSDT")

    assert result.decision == Decision.IGNORE
    assert Reason.EXISTING_POSITION in result.reasons
    assert Reason.ASYMMETRIC_SQUEEZE not in result.reasons


# ===========================================================================
# 10. Asymmetric squeeze + invalid instrument → IGNORE
# ===========================================================================

@pytest.mark.asyncio
async def test_10_asymmetric_squeeze_invalid_instrument_ignored(engine: DecisionEngine):
    """Squeeze qualifies in DecisionEngine, but instrument validation rejects."""
    sig = _signal(score=88.0, symbol="AAPL/USD:USD")
    snap = _snap(trade_imbalance=0.25, ob_imbalance=-0.20)
    
    # If bybit_symbol is None at decision time, gate blocks immediately
    res_none = engine.decide(sig, snap, _pos(), bybit_symbol=None)
    assert res_none.decision == Decision.IGNORE
    assert Reason.BYBIT_SYMBOL_UNAVAILABLE in res_none.reasons

    # If evaluated through InstrumentValidator: ineligible symbol is rejected
    validator = InstrumentValidator(
        binance_cache={
            "AAPLUSDT": {
                "contractType": "TRADIFI_PERPETUAL",
                "underlyingType": "EQUITY",
                "quoteAsset": "USDT",
                "status": "TRADING",
            }
        }
    )
    is_eligible, reason = await validator.validate_instrument("AAPLUSDT")
    assert is_eligible is False
    assert reason == Reason.UNSUPPORTED_INSTRUMENT


# ===========================================================================
# 11. Both paths qualifying → deterministic behavior, no duplicate trade
# ===========================================================================

def test_11_both_paths_qualifying_deterministic_no_duplicate(engine: DecisionEngine):
    """When both NORMAL_V1 and ASYMMETRIC_SQUEEZE conditions are satisfied:

    - Score = 88.0 (>= 85 and >= 55)
    - Flow = +0.25 (>= +0.20, supportive)
    - OB = -0.15 (<= -0.10, adverse)
    - OI = +2.0% (>= +0.5%, supportive)
    Here supportive = 2 (Flow + OI), adverse = 1 (OB).
    Both NORMAL_V1 and ASYMMETRIC_SQUEEZE qualify.

    Deterministic behavior:
    - Exactly 1 DecisionResult returned
    - Decision is TRADE
    - Authorized path is NORMAL_V1 (since normal requirements met without bypass)
    - Qualifying paths recorded as ['NORMAL_V1', 'ASYMMETRIC_SQUEEZE']
    - No duplicate decision or ambiguity
    """
    sig = _signal(score=88.0)
    snap = _snap(trade_imbalance=0.25, ob_imbalance=-0.15, oi_delta_pct=2.0)

    res1 = engine.decide(sig, snap, _pos(), bybit_symbol="MOVRUSDT")
    res2 = engine.decide(sig, snap, _pos(), bybit_symbol="MOVRUSDT")

    assert res1.decision == Decision.TRADE
    assert res2.decision == Decision.TRADE
    assert res1.extra.get("decision_path") == "NORMAL_V1"
    assert res2.extra.get("decision_path") == "NORMAL_V1"
    assert res1.extra.get("qualifying_paths") == ["NORMAL_V1", "ASYMMETRIC_SQUEEZE"]
    assert res2.extra.get("qualifying_paths") == ["NORMAL_V1", "ASYMMETRIC_SQUEEZE"]
    assert Reason.SCORE_ABOVE_THRESHOLD in res1.reasons


# ===========================================================================
# 12. SL/TP/risk behavior unchanged
# ===========================================================================

def test_12_sl_tp_risk_behavior_unchanged(engine: DecisionEngine):
    """RiskEngine calculation, ATR stop (2x ATR, max 5%), 2R TP, and $200 position

    cap remain completely unchanged for ASYMMETRIC_SQUEEZE authorized trades.
    """
    sig = _signal(score=88.0)
    snap = _snap(trade_imbalance=0.25, ob_imbalance=-0.20, mark_price=10.0)
    decision = engine.decide(sig, snap, _pos(), bybit_symbol="MOVRUSDT")
    assert decision.decision == Decision.TRADE
    assert decision.extra.get("decision_path") == "ASYMMETRIC_SQUEEZE"

    # RiskEngine evaluates sizing and bracket
    risk_engine = RiskEngine(RiskConfig(
        account_balance_usd=1000.0,
        max_risk_pct_per_trade=1.0,  # $10 risk budget
        max_position_usd=200.0,      # $200 notional ceiling
        sl_atr_multiple=2.0,         # 2x ATR stop
        rr_ratio=2.0,                # 2R TP
        max_stop_pct=0.05,           # 5% max SL
    ))

    atr_result = ATRResult(
        symbol="MOVRUSDT", interval="1m", atr=0.20, fetch_ts=time.time(), candles_used=14
    )

    risk = risk_engine.calculate(
        entry_price=10.0,
        atr_result=atr_result,
        spread_abs=0.01,
    )

    assert risk.valid
    # Stop loss = 10.0 - (2.0 * 0.20) = 9.60 (distance = 0.40, 4.0% <= 5.0%)
    assert pytest.approx(risk.stop_loss, abs=1e-4) == 9.60
    # Take profit = 10.0 + 2 * 0.40 = 10.80 (2R)
    assert pytest.approx(risk.take_profit, abs=1e-4) == 10.80
    assert pytest.approx(risk.rr_ratio, abs=1e-4) == 2.0
    # Sizing: $10 risk / $0.40 distance = 25 units ($250 notional), clamped by $200 max = 20.0 units
    assert pytest.approx(risk.quantity, abs=1e-4) == 20.0
    assert pytest.approx(risk.position_size_usd, abs=1e-4) == 200.0
