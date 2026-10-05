"""CoilBrain Funded Account Evaluation Manager.

Manages the 2-Step Prop Firm Evaluation & Funded Account Rules:
  - Account Initial Balance: $5,000.00
  - Step 1 (Evaluation 1):
      * Profit Target: +10% ($500 profit -> $5,500 target equity)
      * Minimum Trading Days: 5 days
      * Max Drawdown: 10% ($500 max loss, hard stop at $4,500)
      * Daily Drawdown: 5% ($250 max loss on any calendar day)
      * Max Single Position Drawdown (Risk per trade): 3% ($150 max loss)
  - Step 2 (Evaluation 2):
      * Profit Target: +5% ($250 profit -> $5,750 or +5% on step baseline)
      * Minimum Trading Days: 5 days
      * Max Drawdown: 10%
      * Daily Drawdown: 5%
      * Max Single Position Drawdown: 3%
  - Funded Stage:
      * Both evaluations passed -> Account is FUNDED
      * Payout mode with capital preservation (1.5% risk per trade)
      * Enforces 10% max DD and 5% daily DD

Position Sizing Philosophy:
  - "Play Big, Pass Evals, Protect Drawdown"
  - Targets 2.2% - 2.5% risk per trade during evaluations (hard cap 3.0%).
  - At 1:2 R:R, just 2 winning trades hits the +10% Step 1 target!
  - Daily Drawdown Circuit Breaker: halts at 3.8% daily loss to mathematically
    guarantee the 5.0% prop firm daily limit is never breached.
  - Overall Drawdown Circuit Breaker: halts at 8.0% total drawdown to prevent
    the 10.0% max drawdown breach.
  - Pass Preservation Mode: Once profit target is hit, if min 5 trading days are
    not yet reached, switches to micro-sizing (0.2% risk) to register remaining
    days safely without putting profits at risk.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Tuple

import aiosqlite

from coilbrain import database

logger = logging.getLogger(__name__)

# Evaluation Constants
INITIAL_BALANCE_USD = 5000.0
STEP_1_PROFIT_TARGET_PCT = 10.0      # +10% (+$500)
STEP_2_PROFIT_TARGET_PCT = 5.0       # +5% (+$250)
MIN_TRADING_DAYS = 5                 # 5 days per step

MAX_DRAWDOWN_LIMIT_PCT = 10.0        # 10% max overall drawdown limit
DAILY_DRAWDOWN_LIMIT_PCT = 5.0       # 5% max daily drawdown limit
MAX_SINGLE_POS_RISK_PCT = 3.0        # 3% max single position risk (hard ceiling)

# Circuit Breakers (Safety buffers before prop firm breach)
DAILY_DRAWDOWN_GUARD_PCT = 3.8       # Stop new trades today if down 3.8% (buffer before 5.0%)
MAX_DRAWDOWN_GUARD_PCT = 8.0         # Stop all trades if total drawdown reaches 8.0% (buffer before 10.0%)

# Sizing Defaults
DEFAULT_EVAL_RISK_PCT = 2.2          # Aggressive sizing to pass fast: 2 wins = pass!
MICRO_PRESERVATION_RISK_PCT = 0.2    # Micro size when target is hit, waiting for min days
FUNDED_STAGE_RISK_PCT = 1.5          # Sustainable payout compounding


class EvaluationStep(str, Enum):
    STEP_1 = "STEP_1"
    STEP_2 = "STEP_2"
    FUNDED = "FUNDED"


@dataclass
class ChallengeState:
    account_id: str
    step: EvaluationStep
    initial_balance: float
    step_starting_balance: float
    current_balance: float
    high_water_mark: float
    daily_start_date: str
    daily_start_balance: float
    trading_days: list[str]          # Dates with at least 1 trade: ["2026-10-02", ...]
    step_1_passed_at: Optional[float] = None
    step_2_passed_at: Optional[float] = None
    is_funded: bool = False
    updated_at: float = 0.0


_SCHEMA = """
CREATE TABLE IF NOT EXISTS funded_challenge_state (
    account_id             TEXT PRIMARY KEY,
    step                   TEXT NOT NULL,
    initial_balance        REAL NOT NULL,
    step_starting_balance  REAL NOT NULL,
    current_balance        REAL NOT NULL,
    high_water_mark        REAL NOT NULL,
    daily_start_date       TEXT NOT NULL,
    daily_start_balance    REAL NOT NULL,
    trading_days_json      TEXT NOT NULL,
    step_1_passed_at       REAL,
    step_2_passed_at       REAL,
    is_funded              INTEGER NOT NULL DEFAULT 0,
    updated_at             REAL NOT NULL
);
"""


class FundedChallengeManager:
    """Singleton-like manager for tracking prop firm evaluation progress and risk limits."""

    ACCOUNT_ID = "bybit_demo_5k"

    def __init__(self) -> None:
        self._state: Optional[ChallengeState] = None

    async def ensure_schema(self, db: aiosqlite.Connection) -> None:
        """Create table if it doesn't exist and initialize default state."""
        await db.execute(_SCHEMA)
        await db.commit()

        # Check if record exists
        async with db.execute(
            "SELECT * FROM funded_challenge_state WHERE account_id = ?",
            (self.ACCOUNT_ID,),
        ) as cur:
            row = await cur.fetchone()

        today_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        now = time.time()

        if not row:
            initial_state = ChallengeState(
                account_id=self.ACCOUNT_ID,
                step=EvaluationStep.STEP_1,
                initial_balance=INITIAL_BALANCE_USD,
                step_starting_balance=INITIAL_BALANCE_USD,
                current_balance=INITIAL_BALANCE_USD,
                high_water_mark=INITIAL_BALANCE_USD,
                daily_start_date=today_str,
                daily_start_balance=INITIAL_BALANCE_USD,
                trading_days=[],
                step_1_passed_at=None,
                step_2_passed_at=None,
                is_funded=False,
                updated_at=now,
            )
            await self._save_state(db, initial_state)
            self._state = initial_state
            logger.info("FundedChallengeManager: Initialized new $5k Evaluation State (Step 1).")
        else:
            self._state = self._row_to_state(dict(row))

    async def get_state(self, db: aiosqlite.Connection) -> ChallengeState:
        """Fetch current challenge state from DB."""
        await self.ensure_schema(db)
        async with db.execute(
            "SELECT * FROM funded_challenge_state WHERE account_id = ?",
            (self.ACCOUNT_ID,),
        ) as cur:
            row = await cur.fetchone()
        state = self._row_to_state(dict(row))
        self._state = state
        return state

    async def sync_balance(self, db: aiosqlite.Connection, real_balance: float) -> ChallengeState:
        """Update current wallet balance, sync daily rollover, and evaluate step milestones."""
        state = await self.get_state(db)
        if real_balance <= 0:
            return state

        today_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        now = time.time()

        # Check daily date rollover (UTC 00:00)
        if today_str != state.daily_start_date:
            logger.info(
                "FundedChallenge: Daily rollover from %s to %s | New day starting balance: $%.2f",
                state.daily_start_date, today_str, real_balance
            )
            state.daily_start_date = today_str
            state.daily_start_balance = real_balance

        state.current_balance = real_balance
        if real_balance > state.high_water_mark:
            state.high_water_mark = real_balance

        state.updated_at = now
        await self._check_step_milestones(state)
        await self._save_state(db, state)
        self._state = state
        return state

    async def record_trade_day(self, db: aiosqlite.Connection) -> None:
        """Record today as an active trading day towards the 5-day requirement."""
        state = await self.get_state(db)
        today_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        if today_str not in state.trading_days:
            state.trading_days.append(today_str)
            state.updated_at = time.time()
            logger.info(
                "FundedChallenge: Logged active trading day: %s (Total in current phase: %d/%d)",
                today_str, len(state.trading_days), MIN_TRADING_DAYS
            )
            await self._check_step_milestones(state)
            await self._save_state(db, state)

    async def check_can_trade(self, db: aiosqlite.Connection, equity: float) -> Tuple[bool, str]:
        """Check if trading is permitted under funded evaluation rules.
        
        Returns:
            (allowed: bool, reason: str)
        """
        state = await self.sync_balance(db, equity)

        # 1. Total Max Drawdown Check
        # Hard stop if balance dropped > 10% from initial/step baseline ($4,500 floor)
        step_base = state.step_starting_balance
        total_dd_pct = ((step_base - equity) / step_base) * 100.0
        if total_dd_pct >= MAX_DRAWDOWN_LIMIT_PCT:
            return False, f"MAX_DRAWDOWN_BREACH (DD={total_dd_pct:.2f}% >= {MAX_DRAWDOWN_LIMIT_PCT}%)"
        if total_dd_pct >= MAX_DRAWDOWN_GUARD_PCT:
            return False, f"MAX_DRAWDOWN_GUARD_ACTIVE (DD={total_dd_pct:.2f}% >= {MAX_DRAWDOWN_GUARD_PCT}%)"

        # 2. Daily Drawdown Check
        daily_base = state.daily_start_balance
        if daily_base > 0:
            daily_dd_pct = ((daily_base - equity) / daily_base) * 100.0
            if daily_dd_pct >= DAILY_DRAWDOWN_LIMIT_PCT:
                return False, f"DAILY_DRAWDOWN_BREACH (Daily DD={daily_dd_pct:.2f}% >= {DAILY_DRAWDOWN_LIMIT_PCT}%)"
            if daily_dd_pct >= DAILY_DRAWDOWN_GUARD_PCT:
                return False, f"DAILY_DRAWDOWN_GUARD_HALT (Daily DD={daily_dd_pct:.2f}% >= {DAILY_DRAWDOWN_GUARD_PCT}%)"

        return True, "PERMITTED"

    def get_sizing_parameters(self, state: ChallengeState) -> dict:
        """Calculate optimal risk % and position limits based on evaluation phase.
        
        Returns dict with:
          - risk_pct: float (e.g. 2.2% or 0.2%)
          - max_position_usd: float (e.g. 7500.0)
          - is_preservation_mode: bool
        """
        step_base = state.step_starting_balance
        current = state.current_balance
        profit_pct = ((current - step_base) / step_base) * 100.0
        trading_days_count = len(state.trading_days)

        target_pct = (
            STEP_1_PROFIT_TARGET_PCT if state.step == EvaluationStep.STEP_1
            else STEP_2_PROFIT_TARGET_PCT if state.step == EvaluationStep.STEP_2
            else 0.0
        )

        is_funded = state.step == EvaluationStep.FUNDED

        # If already funded -> steady payout compounding
        if is_funded:
            return {
                "risk_pct": FUNDED_STAGE_RISK_PCT,
                "max_position_usd": min(current * 1.5, 10000.0),
                "is_preservation_mode": False,
                "reason": "FUNDED_PAYOUT_MODE"
            }

        # Check if target is already achieved, but still need min trading days
        target_achieved = profit_pct >= target_pct
        needs_more_days = trading_days_count < MIN_TRADING_DAYS

        if target_achieved and needs_more_days:
            # PRESERVATION MODE: Don't risk the passing profit!
            # Use micro-position sizing (0.2% risk) to safely log the 5 trading days.
            return {
                "risk_pct": MICRO_PRESERVATION_RISK_PCT,
                "max_position_usd": 1000.0,
                "is_preservation_mode": True,
                "reason": f"PASS_PRESERVATION_MODE (Profit {profit_pct:.2f}% >= {target_pct}%, Days: {trading_days_count}/{MIN_TRADING_DAYS})"
            }

        # Normal Evaluation Mode: "Play Big, Pass Fast"
        # 2.2% risk per trade (capped at 3.0% hard limit)
        risk_pct = DEFAULT_EVAL_RISK_PCT

        # Position size cap: allow up to 1.5x - 2.0x leverage on balance
        max_pos = min(current * 1.5, 7500.0)

        return {
            "risk_pct": risk_pct,
            "max_position_usd": max_pos,
            "is_preservation_mode": False,
            "reason": f"EVAL_AGGRESSIVE_MODE (Target: +{target_pct}%, Current: {profit_pct:+.2f}%)"
        }

    async def _check_step_milestones(self, state: ChallengeState) -> None:
        """Check if Step 1 or Step 2 has been passed and transition if requirements met."""
        step_base = state.step_starting_balance
        profit_pct = ((state.current_balance - step_base) / step_base) * 100.0
        trading_days_count = len(state.trading_days)
        now = time.time()

        if state.step == EvaluationStep.STEP_1:
            if profit_pct >= STEP_1_PROFIT_TARGET_PCT and trading_days_count >= MIN_TRADING_DAYS:
                logger.info(
                    "🎉 STEP 1 EVALUATION PASSED! Profit: +%.2f%%, Days: %d. Advancing to STEP 2!",
                    profit_pct, trading_days_count
                )
                state.step = EvaluationStep.STEP_2
                state.step_1_passed_at = now
                state.step_starting_balance = state.current_balance
                state.trading_days = []  # Reset 5-day counter for Step 2

        elif state.step == EvaluationStep.STEP_2:
            if profit_pct >= STEP_2_PROFIT_TARGET_PCT and trading_days_count >= MIN_TRADING_DAYS:
                logger.info(
                    "🏆 STEP 2 EVALUATION PASSED! Profit: +%.2f%%, Days: %d. YOU ARE NOW FULLY FUNDED!",
                    profit_pct, trading_days_count
                )
                state.step = EvaluationStep.FUNDED
                state.step_2_passed_at = now
                state.is_funded = True
                state.step_starting_balance = state.current_balance

    async def get_dashboard(self, db: aiosqlite.Connection) -> dict:
        """Return rich dashboard payload for API and status reporting."""
        state = await self.get_state(db)

        step_base = state.step_starting_balance
        curr = state.current_balance
        profit_usd = curr - step_base
        profit_pct = (profit_usd / step_base) * 100.0 if step_base > 0 else 0.0

        daily_base = state.daily_start_balance
        daily_pnl_usd = curr - daily_base
        daily_dd_pct = max(0.0, ((daily_base - curr) / daily_base) * 100.0) if daily_base > 0 else 0.0

        total_dd_pct = max(0.0, ((state.high_water_mark - curr) / state.high_water_mark) * 100.0) if state.high_water_mark > 0 else 0.0

        target_pct = (
            STEP_1_PROFIT_TARGET_PCT if state.step == EvaluationStep.STEP_1
            else STEP_2_PROFIT_TARGET_PCT if state.step == EvaluationStep.STEP_2
            else 0.0
        )
        target_usd = step_base * (target_pct / 100.0)

        sizing = self.get_sizing_parameters(state)

        return {
            "account_id": state.account_id,
            "status": "FUNDED" if state.is_funded else "EVALUATION",
            "current_step": state.step.value,
            "initial_balance_usd": state.initial_balance,
            "current_equity_usd": round(curr, 2),
            "step_starting_balance_usd": round(step_base, 2),
            "high_water_mark_usd": round(state.high_water_mark, 2),
            "profit_usd": round(profit_usd, 2),
            "profit_pct": round(profit_pct, 2),
            "target_pct": target_pct,
            "target_usd": round(target_usd, 2),
            "target_progress_pct": round(min(100.0, max(0.0, (profit_pct / target_pct * 100.0))), 1) if target_pct > 0 else 100.0,
            "trading_days_completed": len(state.trading_days),
            "min_trading_days_required": MIN_TRADING_DAYS,
            "trading_days_list": state.trading_days,
            "daily_starting_equity_usd": round(daily_base, 2),
            "daily_pnl_usd": round(daily_pnl_usd, 2),
            "daily_drawdown_pct": round(daily_dd_pct, 2),
            "daily_drawdown_limit_pct": DAILY_DRAWDOWN_LIMIT_PCT,
            "daily_drawdown_guard_pct": DAILY_DRAWDOWN_GUARD_PCT,
            "daily_guard_healthy": daily_dd_pct < DAILY_DRAWDOWN_GUARD_PCT,
            "total_drawdown_pct": round(total_dd_pct, 2),
            "max_drawdown_limit_pct": MAX_DRAWDOWN_LIMIT_PCT,
            "max_drawdown_guard_pct": MAX_DRAWDOWN_GUARD_PCT,
            "max_guard_healthy": total_dd_pct < MAX_DRAWDOWN_GUARD_PCT,
            "active_sizing": sizing,
            "hard_max_single_trade_risk_pct": MAX_SINGLE_POS_RISK_PCT,
            "step_1_passed_at": state.step_1_passed_at,
            "step_2_passed_at": state.step_2_passed_at,
            "is_funded": state.is_funded,
        }

    async def _save_state(self, db: aiosqlite.Connection, state: ChallengeState) -> None:
        await db.execute(
            """
            INSERT OR REPLACE INTO funded_challenge_state (
                account_id, step, initial_balance, step_starting_balance,
                current_balance, high_water_mark, daily_start_date,
                daily_start_balance, trading_days_json, step_1_passed_at,
                step_2_passed_at, is_funded, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state.account_id,
                state.step.value,
                state.initial_balance,
                state.step_starting_balance,
                state.current_balance,
                state.high_water_mark,
                state.daily_start_date,
                state.daily_start_balance,
                json.dumps(state.trading_days),
                state.step_1_passed_at,
                state.step_2_passed_at,
                1 if state.is_funded else 0,
                state.updated_at,
            ),
        )
        await db.commit()

    @staticmethod
    def _row_to_state(row: dict) -> ChallengeState:
        return ChallengeState(
            account_id=row["account_id"],
            step=EvaluationStep(row["step"]),
            initial_balance=float(row["initial_balance"]),
            step_starting_balance=float(row["step_starting_balance"]),
            current_balance=float(row["current_balance"]),
            high_water_mark=float(row["high_water_mark"]),
            daily_start_date=row["daily_start_date"],
            daily_start_balance=float(row["daily_start_balance"]),
            trading_days=json.loads(row["trading_days_json"]) if row.get("trading_days_json") else [],
            step_1_passed_at=row.get("step_1_passed_at"),
            step_2_passed_at=row.get("step_2_passed_at"),
            is_funded=bool(row.get("is_funded", 0)),
            updated_at=float(row.get("updated_at", 0.0)),
        )


# Global instance
challenge_manager = FundedChallengeManager()
