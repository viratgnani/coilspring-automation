import asyncio
import logging
import time
from typing import Any

import httpx

from coilbrain import database, decision_repository as drepo
from coilbrain import demo_guard
from coilbrain.atr_fetcher import ATRFetcher
from coilbrain.bybit_demo import BybitDemoAPIError, BybitDemoClient
from coilbrain.config import settings
from coilbrain.decision_engine import (
    DataFreshness,
    Decision,
    DecisionConfig,
    DecisionEngine,
    SignalFeatures,
)
from coilbrain.instrument_validator import InstrumentValidator
from coilbrain.microstructure_fetcher import fetch_all
from coilbrain.models import SignalRecord
from coilbrain.position_manager import PositionManager
from coilbrain.risk_engine import RiskConfig, RiskEngine
from coilbrain.funded_challenge import challenge_manager

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Orchestrates the decision and execution flow for a single signal."""

    def __init__(
        self,
        bybit_client: BybitDemoClient | None = None,
        atr_fetcher: ATRFetcher | None = None,
        instrument_validator: InstrumentValidator | None = None,
    ) -> None:
        self.decision_engine = DecisionEngine(DecisionConfig())
        self.risk_engine = RiskEngine(RiskConfig())
        self.position_manager = PositionManager()
        self.atr_fetcher = atr_fetcher or ATRFetcher()
        if bybit_client is not None:
            self.bybit_client = bybit_client
        else:
            try:
                self.bybit_client = BybitDemoClient.from_env()
            except Exception as e:
                logger.error("Failed to initialize Bybit Demo client: %s", e)
                self.bybit_client = None
        self.instrument_validator = instrument_validator or InstrumentValidator(self.bybit_client)

    async def execute(self, record: SignalRecord) -> None:
        """Process a signal end-to-end."""
        # 1. Fetch immediate microstructure
        fetch_start = time.time()
        from coilbrain.observatory import to_binance_symbol
        binance_symbol = to_binance_symbol(record.symbol) or record.symbol
        
        async with httpx.AsyncClient() as http_client:
            snapshot = await fetch_all(http_client, binance_symbol, since_ts=record.received_timestamp)
        fetch_end = time.time()

        # 2. Extract features
        features = SignalFeatures(
            signal_id=record.signal_id,
            symbol=record.symbol,
            score=record.score,
            grade=record.grade,
            body_spike=record.body_volume_spike or 0.0,
            volume_spike=record.volume_spike or 0.0,
            volume_24h_usd=record.volume_24h_usd,
            change_24h_pct=record.change_24h_pct,
            breakout_pct=record.breakout_pct,
            flat_period_days=record.flat_period_days,
            signal_timestamp=record.signal_timestamp,
        )

        from coilbrain.decision_engine import PositionState
        # 3. Position check (DecisionEngine needs to know if we are flat)
        async with database.get_db() as db:
            pos_state_str = await self.position_manager.get_state_for_symbol(db, record.symbol)
            pos_state = PositionState(symbol=record.symbol, state=pos_state_str)

        # 4. Decide
        decision_start = time.time()
        decision_result = self.decision_engine.decide(
            signal=features,
            snapshot=snapshot,
            position=pos_state,
            bybit_symbol=record.symbol,
            fetch_start_ts=fetch_start,
            fetch_end_ts=fetch_end,
        )
        decision_end = time.time()

        if decision_result.decision == Decision.TRADE:
            # Execution universe safety gate
            async with httpx.AsyncClient() as http_client:
                is_eligible, reason = await self.instrument_validator.validate_instrument(
                    symbol=record.symbol,
                    binance_symbol=binance_symbol,
                    http_client=http_client,
                    bybit_client=self.bybit_client,
                )
            if not is_eligible:
                decision_result.decision = Decision.IGNORE
                decision_result.reasons = [reason]
                logger.warning("Execution universe safety gate rejected %s: %s", record.symbol, reason)
            else:
                async with database.get_db() as db:
                    limit_reached = await demo_guard.check_demo_limit(db)
                if limit_reached:
                    decision_result.decision = Decision.IGNORE
                    decision_result.reasons = ["DEMO_SESSION_LIMIT"]
                    logger.warning("Demo session trade limit reached. Overriding %s to IGNORE.", record.symbol)

        # 5. Journal decision kwargs
        decision_kwargs = dict(
            signal_id=record.signal_id,
            symbol=record.symbol,
            signal_timestamp=record.signal_timestamp,
            decision=decision_result.decision.value,
            decision_score=decision_result.decision_score,
            reasons=decision_result.reasons,
            decision_start_ts=decision_start,
            market_data_start_ts=fetch_start,
            market_data_complete_ts=fetch_end,
            decision_ts=decision_end,
            snapshot_age_ms=decision_result.data_freshness.snapshot_age_ms if decision_result.data_freshness else None,
            intensity_regime=decision_result.feature_classification.intensity.name if decision_result.feature_classification else None,
            oi_state=decision_result.feature_classification.oi_state.name if decision_result.feature_classification else None,
            flow_state=decision_result.feature_classification.flow_state.name if decision_result.feature_classification else None,
            ob_state=decision_result.feature_classification.ob_state.name if decision_result.feature_classification else None,
            liquidity_state=decision_result.feature_classification.liquidity_state.name if decision_result.feature_classification else None,
            atr_source=None,
            atr_value=None,
        )

        if decision_result.decision != Decision.TRADE:
            async with database.get_db() as db:
                await drepo.record_decision(db, **decision_kwargs)
            logger.info("Signal %s -> %s (%s)", record.symbol, decision_result.decision, decision_result.reasons)
            return

        # 6. Risk (ATR)
        # We need ATR for Risk Engine
        if not self.bybit_client:
            decision_kwargs["decision"] = Decision.IGNORE.value
            decision_kwargs["reasons"] = ["BYBIT_CLIENT_MISSING"]
            async with database.get_db() as db:
                await drepo.record_decision(db, **decision_kwargs)
            return

        # Fetch live Bybit balance for accurate evaluation sizing
        account_balance = 5000.0
        try:
            if hasattr(self.bybit_client, "get_balance"):
                bal_obj = await self.bybit_client.get_balance("USDT")
                if bal_obj and hasattr(bal_obj, "total_equity") and isinstance(bal_obj.total_equity, (int, float)) and bal_obj.total_equity > 0:
                    account_balance = float(bal_obj.total_equity)
        except Exception as e:
            logger.warning("Could not fetch live Bybit balance: %s; using fallback $5000", e)

        # Enforce Funded Account Evaluation Drawdown Guards (5% daily DD, 10% max DD)
        async with database.get_db() as db:
            can_trade, guard_reason = await challenge_manager.check_can_trade(db, account_balance)
            if not can_trade:
                decision_kwargs["decision"] = Decision.IGNORE.value
                decision_kwargs["reasons"] = [guard_reason]
                await drepo.record_decision(db, **decision_kwargs)
                logger.warning("Funded Evaluation Guard blocked trade on %s: %s", record.symbol, guard_reason)
                return

            state = await challenge_manager.get_state(db)
            sizing_params = challenge_manager.get_sizing_parameters(state)

        # Dynamic risk sizing tailored to funded evaluation status
        risk_cfg = RiskConfig(
            account_balance_usd=account_balance,
            max_risk_pct_per_trade=sizing_params["risk_pct"],
            max_position_usd=sizing_params["max_position_usd"],
            hard_max_single_risk_pct=3.0,
        )
        engine_to_use = RiskEngine(risk_cfg)

        async with httpx.AsyncClient() as http_client:
            atr_result = await self.atr_fetcher.fetch_atr(http_client, binance_symbol)

        risk_start = time.time()
        risk = engine_to_use.calculate(
            entry_price=snapshot.mark_price,
            atr_result=atr_result,
            spread_abs=snapshot.spread_abs,
        )
        risk_end = time.time()

        if atr_result:
            decision_kwargs["atr_source"] = "atr14_1m"
            decision_kwargs["atr_value"] = atr_result.atr

        if not risk.valid:
            decision_kwargs["decision"] = Decision.IGNORE.value
            decision_kwargs["reasons"] = ["RISK_INVALID"]
            async with database.get_db() as db:
                await drepo.record_decision(db, **decision_kwargs)
            logger.warning("Signal %s Risk Invalid", record.symbol)
            return

        # Update decision record with Risk details since it passed
        decision_kwargs["decision_ts"] = risk_end
        async with database.get_db() as db:
            decision_id = await drepo.record_decision(db, **decision_kwargs)

        bybit_symbol = to_binance_symbol(record.symbol) or record.symbol.replace("/", "").split(":")[0]

        # 7. Final execution gate - Check Position State again
        async with database.get_db() as db:
            final_pos_state = await self.position_manager.get_state_for_symbol(db, record.symbol)
            if final_pos_state not in ("NO_POSITION", "CLOSED"):
                logger.warning("Race condition: Position for %s became %s", record.symbol, final_pos_state)
                return
            await self.position_manager.open_pending(
                db, 
                position_id=record.signal_id, 
                signal_id=record.signal_id, 
                symbol=record.symbol,
                bybit_symbol=bybit_symbol, 
                entry_price=risk.entry_price,
                stop_loss=risk.stop_loss,
                take_profit=risk.take_profit,
                position_size_usd=risk.quantity * risk.entry_price,
                quantity=risk.quantity,
                risk_usd=risk.quantity * abs(risk.entry_price - risk.stop_loss),
                rr_ratio=risk.rr_ratio,
                leverage=risk.leverage
            )

        # 8. Execution
        trade_kwargs = dict(
            decision_journal_id=decision_id,
            signal_id=record.signal_id,
            position_id=record.signal_id,
            symbol=record.symbol,
            bybit_symbol=bybit_symbol,
            planned_entry=snapshot.mark_price,
            stop_loss=risk.stop_loss,
            take_profit=risk.take_profit,
            position_size_usd=risk.quantity * snapshot.mark_price,
            quantity=risk.quantity,
            risk_usd=risk.risk_usd,
            rr_ratio=risk.rr_ratio if hasattr(risk, 'rr_ratio') else 0.0,
            leverage=risk.leverage if hasattr(risk, 'leverage') else 1.0,
            bybit_order_id=None,
            actual_entry_price=None,
            fill_timestamp=None,
            sl_submission_timestamp=None,
            sl_confirmed_timestamp=None,
            tp_submission_timestamp=None,
            tp_confirmed_timestamp=None,
            order_submission_timestamp=None,
        )

        try:
            trade_kwargs["order_submission_timestamp"] = time.time()
            order = await self.bybit_client.place_market_order(
                symbol=bybit_symbol,
                side="Buy",
                qty=risk.quantity,
            )
            if not order.order_id:
                raise BybitDemoAPIError(0, "No orderId returned")
            trade_kwargs["bybit_order_id"] = order.order_id
            trade_kwargs["fill_timestamp"] = time.time()

            # Increment the demo session guard limit and record trading day
            async with database.get_db() as db:
                await demo_guard.increment_demo_trade_count(db)
                await challenge_manager.record_trade_day(db)

            # Confirm pos via API
            pos_info = await self.bybit_client.get_position(bybit_symbol)
            if pos_info:
                trade_kwargs["actual_entry_price"] = pos_info.avg_price

            async with database.get_db() as db:
                await self.position_manager.confirm_open(db, position_id=record.signal_id, bybit_order_id=trade_kwargs["bybit_order_id"], actual_entry_price=trade_kwargs.get("actual_entry_price", snapshot.mark_price))

            # Place SL
            trade_kwargs["sl_submission_timestamp"] = time.time()
            await self.bybit_client.place_stop_loss(bybit_symbol, risk.stop_loss)
            trade_kwargs["sl_confirmed_timestamp"] = time.time()

            # Place TP
            trade_kwargs["tp_submission_timestamp"] = time.time()
            await self.bybit_client.place_take_profit(bybit_symbol, risk.take_profit)
            trade_kwargs["tp_confirmed_timestamp"] = time.time()

        except BybitDemoAPIError as e:
            logger.error("Bybit execution failed for %s (%s): %s", record.symbol, bybit_symbol, e)
            status = "PROTECTION_FAILED" if trade_kwargs.get("fill_timestamp") else "ENTRY_FAILED"
            async with database.get_db() as db:
                if status == "ENTRY_FAILED":
                    await self.position_manager.fail_pending(db, record.signal_id, str(e))

        except Exception as e:
            logger.exception("Unexpected execution error for %s (%s)", record.symbol, bybit_symbol)
            status = "PROTECTION_FAILED" if trade_kwargs.get("fill_timestamp") else "ENTRY_FAILED"
            async with database.get_db() as db:
                if status == "ENTRY_FAILED":
                    await self.position_manager.fail_pending(db, record.signal_id, str(e))

        finally:
            async with database.get_db() as db:
                await drepo.record_trade(db, **trade_kwargs)
                logger.info("Trade %s finalized", record.symbol)

