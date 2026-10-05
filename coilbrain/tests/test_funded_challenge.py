import os
import tempfile
import pytest
import aiosqlite

from coilbrain import database
from coilbrain.funded_challenge import (
    FundedChallengeManager,
    EvaluationStep,
    INITIAL_BALANCE_USD,
    STEP_1_PROFIT_TARGET_PCT,
    STEP_2_PROFIT_TARGET_PCT,
    MIN_TRADING_DAYS,
    MAX_SINGLE_POS_RISK_PCT,
)
from coilbrain.risk_engine import RiskConfig, RiskEngine
from coilbrain.atr_fetcher import ATRResult


@pytest.fixture
async def temp_db():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        temp_path = f.name
    await database.init_db(temp_path)
    yield temp_path
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_funded_challenge_initial_state(temp_db):
    mgr = FundedChallengeManager()
    async with database.get_db() as db:
        await mgr.ensure_schema(db)
        state = await mgr.get_state(db)
        assert state.account_id == "bybit_demo_5k"
        assert state.step == EvaluationStep.STEP_1
        assert state.initial_balance == 5000.0
        assert state.step_starting_balance == 5000.0
        assert state.current_balance == 5000.0
        assert state.trading_days == []
        assert not state.is_funded


@pytest.mark.asyncio
async def test_drawdown_guards(temp_db):
    mgr = FundedChallengeManager()
    async with database.get_db() as db:
        await mgr.ensure_schema(db)

        # Normal equity: $5000 -> Allowed
        can_trade, reason = await mgr.check_can_trade(db, 5000.0)
        assert can_trade is True
        assert reason == "PERMITTED"

        # Small loss: $4900 (-2.0%) -> Allowed
        can_trade, reason = await mgr.check_can_trade(db, 4900.0)
        assert can_trade is True

        # Daily loss reaches 4.0% ($4800 on $5000 starting day) -> Blocked by 3.8% Daily Guard
        can_trade, reason = await mgr.check_can_trade(db, 4800.0)
        assert can_trade is False
        assert "DAILY_DRAWDOWN_GUARD_HALT" in reason

        # Reset daily starting balance to simulate next day, but severe overall loss ($4550 = 9% drawdown)
        state = await mgr.get_state(db)
        state.daily_start_balance = 4600.0
        await mgr._save_state(db, state)

        # $4550 equity against $5000 initial is 9.0% drawdown -> Blocked by 8.0% Max DD Guard
        can_trade, reason = await mgr.check_can_trade(db, 4550.0)
        assert can_trade is False
        assert "MAX_DRAWDOWN" in reason


@pytest.mark.asyncio
async def test_aggressive_sizing_and_pass_preservation(temp_db):
    mgr = FundedChallengeManager()
    async with database.get_db() as db:
        await mgr.ensure_schema(db)
        state = await mgr.get_state(db)

        # Step 1, balance $5000, 0 trading days -> Aggressive evaluation sizing
        sizing = mgr.get_sizing_parameters(state)
        assert sizing["risk_pct"] == 2.2
        assert sizing["max_position_usd"] == 7500.0
        assert sizing["is_preservation_mode"] is False

        # Simulate profit target reached ($5550 = +11%), but only 2 trading days
        state.current_balance = 5550.0
        state.trading_days = ["2026-10-01", "2026-10-02"]
        sizing = mgr.get_sizing_parameters(state)

        # Should activate preservation mode (0.2% micro-risk) to protect profits!
        assert sizing["is_preservation_mode"] is True
        assert sizing["risk_pct"] == 0.2
        assert "PASS_PRESERVATION_MODE" in sizing["reason"]


@pytest.mark.asyncio
async def test_step_progression_to_funded(temp_db):
    mgr = FundedChallengeManager()
    async with database.get_db() as db:
        await mgr.ensure_schema(db)

        # 1. Step 1 Pass: +10% ($5500) and 5 distinct trading days
        state = await mgr.get_state(db)
        state.current_balance = 5520.0
        state.trading_days = [f"2026-10-0{i}" for i in range(1, 6)]
        await mgr._check_step_milestones(state)
        await mgr._save_state(db, state)

        # Should transition to Step 2
        reloaded = await mgr.get_state(db)
        assert reloaded.step == EvaluationStep.STEP_2
        assert reloaded.trading_days == []  # Reset for Step 2
        assert reloaded.step_starting_balance == 5520.0

        # 2. Step 2 Pass: +5% on $5520 (target: +$276 -> ~$5800) and 5 trading days
        reloaded.current_balance = 5810.0
        reloaded.trading_days = [f"2026-10-1{i}" for i in range(1, 6)]
        await mgr._check_step_milestones(reloaded)
        await mgr._save_state(db, reloaded)

        # Should transition to FUNDED!
        final_state = await mgr.get_state(db)
        assert final_state.step == EvaluationStep.FUNDED
        assert final_state.is_funded is True

        # Funded sizing should be sustainable payout mode
        sizing = mgr.get_sizing_parameters(final_state)
        assert sizing["risk_pct"] == 1.5
        assert sizing["reason"] == "FUNDED_PAYOUT_MODE"


@pytest.mark.asyncio
async def test_hard_max_single_position_risk_enforced():
    # Verify RiskEngine hard ceiling of 3%
    cfg = RiskConfig(
        account_balance_usd=5000.0,
        max_risk_pct_per_trade=5.0,  # Intentional override attempt to exceed 3%
        hard_max_single_risk_pct=3.0,
        max_position_usd=20000.0,
    )
    engine = RiskEngine(cfg)
    atr = ATRResult(atr=0.02, source="test", symbol="TESTUSDT", interval="1m", fetch_ts=0, candles_used=14)
    res = engine.calculate(entry_price=1.0, atr_result=atr)

    assert res.valid
    # 3% of $5000 is $150.0 — risk_usd must NEVER exceed $150.0!
    assert res.risk_usd <= 150.0 + 1e-4
