#!/usr/bin/env python3
"""
CoilBrain V1 -- One-Shot Bybit Demo Smoke Test
==============================================

PURPOSE
-------
End-to-end connectivity and execution smoke test using real Bybit Demo
credentials and the existing production CoilBrain V1 components.

SAFETY CONTRACT
---------------
* ONLY connects to https://api-demo.bybit.com -- never production.
* Credentials must exist in environment (BYBIT_DEMO_API_KEY / BYBIT_DEMO_API_SECRET).
* Uses production DecisionEngine / RiskEngine / PositionManager / BybitDemoClient.
* Uses a single controlled synthetic CoilSpring-compatible signal.
* Does NOT connect to CoilSpring V2 or modify it in any way.
* No automatic retry of a market order.
* If any pre-order safety check fails -> ZERO orders placed.
* If order fills but SL/TP fails -> reports UNPROTECTED, does NOT retry.
* Never hides, fabricates, or suppresses any result.

USAGE
-----
    # From the coilbrain/ project root:
    python scripts/smoke_test_bybit_demo.py --dry-run
    python scripts/smoke_test_bybit_demo.py --symbol BTCUSDT

DO NOT modify CoilSpring V2 or restart PM2 after running this script.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap: allow running from any working directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)   # coilbrain/
_SRC = os.path.join(_PROJECT_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ---------------------------------------------------------------------------
# Load .env from project root (so credentials set there are visible via
# os.environ without needing a manual `export` in the shell).
# Uses python-dotenv if available; falls back to a minimal manual parser.
# ---------------------------------------------------------------------------
def _load_dotenv(env_path: str) -> None:
    """Load key=value lines from env_path into os.environ (no-op if missing)."""
    if not os.path.isfile(env_path):
        return
    try:
        from dotenv import load_dotenv as _ld
        _ld(dotenv_path=env_path, override=False)
        return
    except ImportError:
        pass
    # Minimal fallback parser
    with open(env_path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

_load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

import httpx

from coilbrain import database
from coilbrain import decision_repository as drepo
from coilbrain.atr_fetcher import ATRFetcher
from coilbrain.bybit_demo import (
    BybitDemoAPIError,
    BybitDemoClient,
    BybitDemoConfigError,
)
from coilbrain.decision_engine import (
    Decision,
    DecisionConfig,
    DecisionEngine,
    PositionState,
    SignalFeatures,
)
from coilbrain.decision_repository import ensure_schema
from coilbrain.microstructure_fetcher import fetch_all
from coilbrain.models import SignalRecord
from coilbrain.position_manager import PositionManager
from coilbrain.risk_engine import RiskConfig, RiskEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_test")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BYBIT_DEMO_BASE_URL = "https://api-demo.bybit.com"

# Synthetic signal: high-confidence A-grade parameters that are designed to
# pass the volume gate and score well, so the decision depends entirely on
# live microstructure. Adjust if you want a guaranteed IGNORE.
SYNTHETIC_DEFAULTS = dict(
    exchange="bybit",
    trigger="SMOKE_TEST",
    grade="A",
    score=90.0,
    signal="PUMP",
    oi_status="OI_PRESENT",
    volume_24h_usd=500_000_000.0,   # well above volume gate
    body_volume_spike=4.0,
    volume_spike=8.0,
    flat_period_days=7.0,
    change_24h_pct=None,
    breakout_pct=None,
)

# Separate DB so production data is never touched
SMOKE_DB = "coilbrain_smoke_test.db"


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------
@dataclass
class SmokeReport:
    base_url: str = BYBIT_DEMO_BASE_URL
    credentials_present: bool = False
    connectivity_ok: bool = False
    account_balance_usdt: float | None = None

    symbol: str = ""
    symbol_valid: bool = False
    qty_step: str = ""
    min_qty: str = ""
    tick_size: str = ""

    signal_id: str = ""
    signal_timestamp: float = 0.0
    signal_price: float = 0.0

    fetch_start_ts: float = 0.0
    fetch_end_ts: float = 0.0
    fetch_latency_ms: float = 0.0
    mark_price: float | None = None
    spread_abs: float | None = None
    snapshot_age_ms: float | None = None

    atr: float | None = None
    atr_source: str = ""

    decision: str = ""
    decision_score: float = 0.0
    decision_reasons: list[str] = field(default_factory=list)
    decision_latency_ms: float = 0.0

    risk_valid: bool = False
    risk_error: str | None = None
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    stop_pct: float | None = None
    rr_ratio: float | None = None
    risk_usd: float | None = None
    quantity: float | None = None
    leverage: float | None = None
    position_size_usd: float | None = None

    dry_run: bool = False
    order_id: str | None = None
    order_submission_ts: float | None = None
    fill_ts: float | None = None
    actual_entry: float | None = None
    sl_submission_ts: float | None = None
    sl_confirmed_ts: float | None = None
    tp_submission_ts: float | None = None
    tp_confirmed_ts: float | None = None

    final_position_state: str = "UNKNOWN"
    position_side: str | None = None
    position_size_live: float | None = None
    position_pnl: float | None = None
    execution_status: str = "NOT_ATTEMPTED"

    decision_journal_id: int | None = None
    trade_journal_id: int | None = None
    errors: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        logger.error("SMOKE ERROR: %s", msg)
        self.errors.append(msg)

    # ── Formatting helpers ────────────────────────────────────────────────

    def print_report(self) -> None:
        sep = "=" * 72

        def section(title: str) -> None:
            print(f"\n  --- {title}")

        def row(label: str, value: str) -> None:
            print(f"    {label:<32} {value}")

        def yn(v: bool | None) -> str:
            return "YES" if v else "NO"

        def ff(v: float | None, p: int = 6) -> str:
            return f"{v:.{p}f}" if v is not None else "--"

        def ts(v: float | None) -> str:
            if v is None:
                return "--"
            import datetime
            return datetime.datetime.fromtimestamp(v).strftime("%H:%M:%S.%f")[:-3]

        print()
        print(sep)
        print("  CoilBrain V1 -- Bybit Demo Smoke Test Report")
        print(sep)

        section("PRE-FLIGHT")
        row("Base URL", self.base_url)
        row("Credentials present", yn(self.credentials_present))
        row("Demo connectivity", yn(self.connectivity_ok))
        row("Account equity (USDT)", ff(self.account_balance_usdt, 2))

        section("INSTRUMENT")
        row("Symbol", self.symbol)
        row("Symbol valid on Demo", yn(self.symbol_valid))
        row("Qty step", self.qty_step or "--")
        row("Min qty", self.min_qty or "--")
        row("Price tick size", self.tick_size or "--")

        section("SYNTHETIC SIGNAL")
        row("Signal ID", self.signal_id or "--")
        row("Signal timestamp", ts(self.signal_timestamp))
        row("Price (live mark)", ff(self.signal_price))

        section("MICROSTRUCTURE FETCH (live)")
        row("Fetch latency", f"{self.fetch_latency_ms:.1f} ms")
        row("Mark price", ff(self.mark_price))
        row("Spread (abs)", ff(self.spread_abs, 8))
        row("Snapshot age", f"{self.snapshot_age_ms:.0f} ms" if self.snapshot_age_ms is not None else "--")

        section("ATR(14) -- 1m candles, Binance USDM")
        row("ATR value", ff(self.atr, 8))
        row("ATR source", self.atr_source or "--")

        section("DECISION ENGINE")
        row("Decision", self.decision or "--")
        row("Score", f"{self.decision_score:.2f}")
        row("Reasons", ", ".join(self.decision_reasons) if self.decision_reasons else "--")
        row("Decision latency", f"{self.decision_latency_ms:.1f} ms")

        section("RISK ENGINE")
        row("Risk valid", yn(self.risk_valid))
        if self.risk_error:
            row("Risk error", self.risk_error)
        row("Entry price", ff(self.entry_price))
        row("Stop loss", ff(self.stop_loss))
        row("Take profit", ff(self.take_profit))
        row("Stop %", f"{self.stop_pct:.4f}%" if self.stop_pct is not None else "--")
        row("R:R", f"{self.rr_ratio:.2f}" if self.rr_ratio is not None else "--")
        row("Risk USD", ff(self.risk_usd, 4))
        row("Quantity", ff(self.quantity, 8))
        row("Position size USD", ff(self.position_size_usd, 4))
        row("Leverage", ff(self.leverage, 2))

        section("EXECUTION")
        row("Mode", "DRY RUN (no orders)" if self.dry_run else "LIVE DEMO")
        row("Execution status", self.execution_status)
        row("Order ID", self.order_id or "--")
        row("Order submitted at", ts(self.order_submission_ts))
        row("Fill confirmed at", ts(self.fill_ts))
        row("Actual entry price", ff(self.actual_entry))
        row("SL submitted at", ts(self.sl_submission_ts))
        row("SL confirmed at", ts(self.sl_confirmed_ts))
        row("TP submitted at", ts(self.tp_submission_ts))
        row("TP confirmed at", ts(self.tp_confirmed_ts))

        section("EXECUTION LATENCY BREAKDOWN")
        row("Market data fetch", f"{self.fetch_latency_ms:.1f} ms")
        row("Decision engine", f"{self.decision_latency_ms:.1f} ms")
        if self.order_submission_ts and self.fetch_start_ts:
            row("Signal -> order submit",
                f"{(self.order_submission_ts - self.fetch_start_ts) * 1000:.1f} ms")
        if self.fill_ts and self.fetch_start_ts:
            row("Signal -> fill confirmed",
                f"{(self.fill_ts - self.fetch_start_ts) * 1000:.1f} ms")
        if self.sl_confirmed_ts and self.fetch_start_ts:
            row("Signal -> SL confirmed",
                f"{(self.sl_confirmed_ts - self.fetch_start_ts) * 1000:.1f} ms")
        if self.tp_confirmed_ts and self.fetch_start_ts:
            row("Signal -> TP confirmed",
                f"{(self.tp_confirmed_ts - self.fetch_start_ts) * 1000:.1f} ms")

        section("FINAL POSITION STATE (from Bybit Demo)")
        row("State", self.final_position_state)
        row("Side", self.position_side or "--")
        row("Size", ff(self.position_size_live, 8))
        row("Unrealised PnL", ff(self.position_pnl, 4))
        row("Decision journal row", str(self.decision_journal_id) if self.decision_journal_id else "--")
        row("Trade journal row", str(self.trade_journal_id) if self.trade_journal_id else "--")

        if self.errors:
            print("\n  ERRORS:")
            for e in self.errors:
                print(f"    x {e}")

        # Final banner
        status = self.execution_status
        print()
        if status == "TRADE_COMPLETE":
            print(f"  >>> PASS: Trade fully protected (SL + TP confirmed) <<<")
        elif status == "IGNORED":
            print(f"  >>> PASS: Decision=IGNORE -- ZERO orders placed <<<")
        elif status == "DRY_RUN":
            print(f"  >>> DRY RUN COMPLETE -- no orders placed <<<")
        elif status == "UNPROTECTED":
            print(f"  >>> WARNING: Position OPEN but UNPROTECTED (SL/TP failed) <<<")
        else:
            print(f"  >>> FAIL: {status} <<<")
        print(sep)
        print()


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
async def preflight(report: SmokeReport) -> BybitDemoClient | None:
    print("\n[PRE-FLIGHT] Verifying credentials and Demo connectivity...")

    api_key = os.environ.get("BYBIT_DEMO_API_KEY", "").strip()
    api_secret = os.environ.get("BYBIT_DEMO_API_SECRET", "").strip()

    if not api_key or not api_secret:
        report.add_error(
            "BYBIT_DEMO_API_KEY and/or BYBIT_DEMO_API_SECRET are not set. "
            "Export them before running this script."
        )
        return None
    report.credentials_present = True
    logger.info("Credentials present: YES (values hidden)")

    # Confirm URL is demo-only
    base_url = os.environ.get("BYBIT_DEMO_BASE_URL", BYBIT_DEMO_BASE_URL).strip()
    if "api-demo.bybit.com" not in base_url:
        report.add_error(
            f"BYBIT_DEMO_BASE_URL points to a non-demo URL: {base_url!r}. "
            "Only https://api-demo.bybit.com is permitted."
        )
        return None
    report.base_url = base_url
    logger.info("Base URL: %s  <-- DEMO only, confirmed", base_url)

    try:
        client = BybitDemoClient.from_env()
    except BybitDemoConfigError as exc:
        report.add_error(f"BybitDemoClient.from_env() failed: {exc}")
        return None

    # Connectivity: wallet balance
    try:
        balance = await client.get_balance("USDT")
        report.connectivity_ok = True
        report.account_balance_usdt = balance.total_equity
        logger.info(
            "Demo connectivity OK -- USDT equity=%.2f  available=%.2f",
            balance.total_equity, balance.available_balance,
        )
    except Exception as exc:
        report.add_error(f"Connectivity check failed (get_balance): {exc}")
        return None

    return client


# ---------------------------------------------------------------------------
# Instrument validation
# ---------------------------------------------------------------------------
async def validate_instrument(
    client: BybitDemoClient, symbol: str, report: SmokeReport
) -> dict | None:
    print(f"[INSTRUMENT] Querying {symbol!r} instrument metadata...")
    try:
        info = await client.get_instrument_info(symbol)
    except BybitDemoConfigError as exc:
        report.add_error(f"Instrument not found or not trading: {exc}")
        return None
    except Exception as exc:
        report.add_error(f"get_instrument_info failed: {exc}")
        return None

    lot = info.get("lotSizeFilter", {})
    pf = info.get("priceFilter", {})
    report.symbol_valid = True
    report.qty_step = lot.get("qtyStep", "?")
    report.min_qty = lot.get("minOrderQty", "?")
    report.tick_size = pf.get("tickSize", "?")

    logger.info(
        "Instrument %s: status=%s  qtyStep=%s  minQty=%s  tickSize=%s",
        symbol, info.get("status"), report.qty_step, report.min_qty, report.tick_size,
    )
    return info


# ---------------------------------------------------------------------------
# Main smoke-test flow
# ---------------------------------------------------------------------------
async def run_smoke_test(symbol: str, dry_run: bool) -> SmokeReport:
    report = SmokeReport(symbol=symbol, dry_run=dry_run)

    # Init smoke-test DB (separate from production)
    print(f"\n[DB] Initialising smoke-test DB: {SMOKE_DB}")
    await database.init_db(SMOKE_DB)
    async with database.get_db() as db:
        await ensure_schema(db)

    # Pre-flight
    client = await preflight(report)
    if client is None:
        report.execution_status = "PREFLIGHT_FAILED"
        return report

    # Instrument metadata
    inst_info = await validate_instrument(client, symbol, report)
    if inst_info is None:
        report.execution_status = "INSTRUMENT_INVALID"
        return report

    # Live microstructure fetch
    print(f"[MICROSTRUCTURE] Fetching live snapshot for {symbol!r}...")
    fetch_start = time.time()
    async with httpx.AsyncClient() as http:
        snapshot = await fetch_all(http, symbol, since_ts=fetch_start)
    fetch_end = time.time()

    report.fetch_start_ts = fetch_start
    report.fetch_end_ts = fetch_end
    report.fetch_latency_ms = (fetch_end - fetch_start) * 1000
    report.mark_price = snapshot.mark_price
    report.spread_abs = snapshot.spread_abs
    report.snapshot_age_ms = (time.time() - snapshot.snapshot_ts) * 1000

    logger.info(
        "Snapshot: mark=%.8f  bid=%.8f  ask=%.8f  spread=%.8f  "
        "oi_usdt=%.0f  funding=%.8f  age_ms=%.0f",
        snapshot.mark_price or 0, snapshot.best_bid or 0, snapshot.best_ask or 0,
        snapshot.spread_abs or 0, snapshot.oi_usdt or 0, snapshot.funding_rate or 0,
        report.snapshot_age_ms,
    )

    if not snapshot.mark_price or snapshot.mark_price <= 0:
        report.add_error(f"No valid mark price for {symbol} -- cannot proceed")
        report.execution_status = "NO_PRICE"
        return report

    # Synthetic signal at live price
    print("[SIGNAL] Building synthetic CoilSpring-compatible signal...")
    now = time.time()
    sig_id = str(uuid.uuid4())
    record = SignalRecord(
        signal_id=sig_id,
        received_timestamp=now,
        transport_latency_ms=0.0,
        symbol=symbol,
        price=snapshot.mark_price,
        signal_timestamp=now,
        **SYNTHETIC_DEFAULTS,
    )
    report.signal_id = sig_id
    report.signal_timestamp = now
    report.signal_price = snapshot.mark_price
    logger.info("Signal: id=%s  price=%.8f  score=%.1f  grade=%s",
                sig_id, snapshot.mark_price, record.score, record.grade)

    # ATR(14) on 1m candles
    print("[ATR] Fetching ATR(14) from Binance 1m klines...")
    async with httpx.AsyncClient() as http:
        atr_fetcher = ATRFetcher()
        atr_result = await atr_fetcher.fetch_atr(http, symbol)

    if atr_result:
        report.atr = atr_result.atr
        report.atr_source = atr_result.source
        logger.info("ATR(14) = %.8f  source=%s", atr_result.atr, atr_result.source)
    else:
        report.atr_source = "unavailable"
        logger.warning("ATR fetch returned None -- risk engine will fail-closed")

    # DecisionEngine
    print("[DECISION] Running V1 decision engine...")
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

    pos_mgr = PositionManager()
    async with database.get_db() as db:
        pos_state_str = await pos_mgr.get_state_for_symbol(db, symbol)
    pos_state = PositionState(symbol=symbol, state=pos_state_str)

    engine = DecisionEngine(DecisionConfig())
    d_start = time.time()
    result = engine.decide(
        signal=features,
        snapshot=snapshot,
        position=pos_state,
        bybit_symbol=symbol,
        fetch_start_ts=fetch_start,
        fetch_end_ts=fetch_end,
    )
    d_end = time.time()

    report.decision = result.decision.value
    report.decision_score = result.decision_score
    report.decision_reasons = list(result.reasons)
    report.decision_latency_ms = (d_end - d_start) * 1000
    logger.info("Decision: %s  score=%.2f  reasons=%s",
                result.decision.value, result.decision_score, result.reasons)

    # Common decision journal kwargs
    dkw: dict[str, Any] = dict(
        signal_id=record.signal_id,
        symbol=record.symbol,
        signal_timestamp=record.signal_timestamp,
        decision=result.decision.value,
        decision_score=result.decision_score,
        reasons=result.reasons,
        decision_start_ts=d_start,
        market_data_start_ts=fetch_start,
        market_data_complete_ts=fetch_end,
        decision_ts=d_end,
        snapshot_age_ms=result.data_freshness.snapshot_age_ms if result.data_freshness else None,
        intensity_regime=result.feature_classification.intensity.name if result.feature_classification else None,
        oi_state=result.feature_classification.oi_state.name if result.feature_classification else None,
        flow_state=result.feature_classification.flow_state.name if result.feature_classification else None,
        ob_state=result.feature_classification.ob_state.name if result.feature_classification else None,
        liquidity_state=result.feature_classification.liquidity_state.name if result.feature_classification else None,
        atr_source=report.atr_source if atr_result else None,
        atr_value=report.atr,
    )

    # IGNORE branch
    if result.decision != Decision.TRADE:
        async with database.get_db() as db:
            report.decision_journal_id = await drepo.record_decision(db, **dkw)
        report.execution_status = "IGNORED"
        logger.info("IGNORE -- ZERO orders placed. Reasons: %s", result.reasons)
        try:
            fp = await client.get_position(symbol)
            report.final_position_state = "OPEN" if fp else "FLAT"
        except Exception:
            report.final_position_state = "QUERY_FAILED"
        return report

    # RiskEngine
    print("[RISK] Calculating position sizing...")
    risk_eng = RiskEngine(RiskConfig())
    risk = risk_eng.calculate(
        entry_price=snapshot.mark_price,
        atr_result=atr_result,
        spread_abs=snapshot.spread_abs,
    )

    report.risk_valid = risk.valid
    report.risk_error = risk.error
    report.entry_price = risk.entry_price
    report.stop_loss = risk.stop_loss
    report.take_profit = risk.take_profit
    report.stop_pct = risk.stop_pct
    report.rr_ratio = risk.rr_ratio
    report.risk_usd = risk.risk_usd
    report.quantity = risk.quantity
    report.position_size_usd = risk.position_size_usd
    report.leverage = risk.leverage

    if not risk.valid:
        dkw["decision"] = Decision.IGNORE.value
        dkw["reasons"] = ["RISK_INVALID"]
        async with database.get_db() as db:
            report.decision_journal_id = await drepo.record_decision(db, **dkw)
        report.add_error(f"Risk engine fail-closed: {risk.error}")
        report.execution_status = "RISK_FAILED"
        return report

    logger.info(
        "Risk: entry=%.8f  SL=%.8f  TP=%.8f  stop_pct=%.4f%%  "
        "qty=%.8f  risk_usd=%.4f  RR=%.2f  leverage=%.2f",
        risk.entry_price, risk.stop_loss, risk.take_profit, risk.stop_pct,
        risk.quantity, risk.risk_usd, risk.rr_ratio, risk.leverage,
    )

    # Journal TRADE decision
    async with database.get_db() as db:
        report.decision_journal_id = await drepo.record_decision(db, **dkw)

    # Dry-run exit
    if dry_run:
        print("[DRY RUN] Halting before order placement (--dry-run mode).")
        report.execution_status = "DRY_RUN"
        return report

    # Final position recheck (race-condition guard)
    print("[RECHECK] Re-verifying position state before order...")
    async with database.get_db() as db:
        recheck_state = await pos_mgr.get_state_for_symbol(db, symbol)
    if recheck_state not in ("NO_POSITION", "CLOSED", None):
        report.add_error(
            f"Position recheck FAILED: state={recheck_state!r}. ZERO orders placed."
        )
        report.execution_status = "POSITION_CONFLICT"
        return report

    # Open pending in position manager
    async with database.get_db() as db:
        await pos_mgr.open_pending(
            db,
            position_id=sig_id,
            signal_id=sig_id,
            symbol=symbol,
            bybit_symbol=symbol,
            entry_price=risk.entry_price,
            stop_loss=risk.stop_loss,
            take_profit=risk.take_profit,
            position_size_usd=risk.position_size_usd,
            quantity=risk.quantity,
            risk_usd=risk.risk_usd,
            rr_ratio=risk.rr_ratio,
            leverage=risk.leverage,
        )

    # Trade journal kwargs
    tkw: dict[str, Any] = dict(
        decision_journal_id=report.decision_journal_id,
        signal_id=sig_id,
        position_id=sig_id,
        symbol=symbol,
        bybit_symbol=symbol,
        planned_entry=snapshot.mark_price,
        stop_loss=risk.stop_loss,
        take_profit=risk.take_profit,
        position_size_usd=risk.position_size_usd,
        quantity=risk.quantity,
        risk_usd=risk.risk_usd,
        rr_ratio=risk.rr_ratio,
        leverage=risk.leverage,
        atr_value=risk.atr_value,
        atr_source=risk.atr_source,
        bybit_order_id=None,
        actual_entry_price=None,
        fill_timestamp=None,
        sl_submission_timestamp=None,
        sl_confirmed_timestamp=None,
        tp_submission_timestamp=None,
        tp_confirmed_timestamp=None,
        order_submission_timestamp=None,
    )

    exec_status = "ENTRY_FAILED"

    try:
        # ONE market order -- NEVER retry
        print(f"[ORDER] Submitting ONE market Buy: {symbol}  qty={risk.quantity:.8f}...")
        tkw["order_submission_timestamp"] = time.time()
        report.order_submission_ts = tkw["order_submission_timestamp"]

        order = await client.place_market_order(
            symbol=symbol,
            side="Buy",
            qty=risk.quantity,
            client_order_id=sig_id[:36],
        )

        if not order.order_id:
            raise BybitDemoAPIError(0, "Bybit returned empty orderId")

        report.order_id = order.order_id
        tkw["bybit_order_id"] = order.order_id
        tkw["fill_timestamp"] = time.time()
        report.fill_ts = tkw["fill_timestamp"]
        logger.info("Order ACK: orderId=%s", order.order_id)

        # Confirm position via exchange query
        print("[FILL] Querying position confirmation...")
        await asyncio.sleep(0.6)     # brief wait for fill propagation
        pos_info = await client.get_position(symbol)
        if pos_info:
            tkw["actual_entry_price"] = pos_info.avg_price
            report.actual_entry = pos_info.avg_price
            report.position_side = pos_info.side
            report.position_size_live = pos_info.size
            report.position_pnl = pos_info.unrealised_pnl
            logger.info(
                "Position confirmed: side=%s  size=%.8f  avgPrice=%.8f  pnl=%.4f",
                pos_info.side, pos_info.size, pos_info.avg_price, pos_info.unrealised_pnl,
            )
        else:
            logger.warning("Position query returned None after fill; using mark price")
            tkw["actual_entry_price"] = snapshot.mark_price

        async with database.get_db() as db:
            await pos_mgr.confirm_open(
                db,
                position_id=sig_id,
                bybit_order_id=order.order_id,
                actual_entry_price=tkw["actual_entry_price"] or snapshot.mark_price,
            )

        # Stop Loss
        print(f"[SL] Submitting stop loss: {risk.stop_loss:.8f}...")
        tkw["sl_submission_timestamp"] = time.time()
        report.sl_submission_ts = tkw["sl_submission_timestamp"]
        await client.place_stop_loss(symbol, risk.stop_loss)
        tkw["sl_confirmed_timestamp"] = time.time()
        report.sl_confirmed_ts = tkw["sl_confirmed_timestamp"]
        logger.info("SL confirmed at %.8f", risk.stop_loss)

        # Take Profit
        print(f"[TP] Submitting take profit: {risk.take_profit:.8f}...")
        tkw["tp_submission_timestamp"] = time.time()
        report.tp_submission_ts = tkw["tp_submission_timestamp"]
        await client.place_take_profit(symbol, risk.take_profit)
        tkw["tp_confirmed_timestamp"] = time.time()
        report.tp_confirmed_ts = tkw["tp_confirmed_timestamp"]
        logger.info("TP confirmed at %.8f", risk.take_profit)

        exec_status = "TRADE_COMPLETE"

    except BybitDemoAPIError as exc:
        report.add_error(f"Bybit API error: {exc}")
        if tkw.get("fill_timestamp"):
            exec_status = "UNPROTECTED"
        else:
            exec_status = "ENTRY_FAILED"
            async with database.get_db() as db:
                await pos_mgr.fail_pending(db, symbol, str(exc))

    except Exception as exc:
        report.add_error(f"Unexpected execution error: {exc}")
        if tkw.get("fill_timestamp"):
            exec_status = "UNPROTECTED"
        else:
            exec_status = "ENTRY_FAILED"
            async with database.get_db() as db:
                await pos_mgr.fail_pending(db, symbol, str(exc))

    finally:
        report.execution_status = exec_status
        async with database.get_db() as db:
            try:
                report.trade_journal_id = await drepo.record_trade(db, **tkw)
                logger.info(
                    "Trade journal written: id=%s  status=%s",
                    report.trade_journal_id, exec_status,
                )
            except Exception as je:
                report.add_error(f"Trade journal write failed: {je}")

    # Final position query
    print("[POSITION] Final position state from Bybit Demo...")
    try:
        fp = await client.get_position(symbol)
        if fp:
            report.final_position_state = "OPEN"
            report.position_side = fp.side
            report.position_size_live = fp.size
            report.position_pnl = fp.unrealised_pnl
        else:
            report.final_position_state = "FLAT"
    except Exception as exc:
        report.add_error(f"Final position query failed: {exc}")
        report.final_position_state = "QUERY_FAILED"

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CoilBrain V1 -- One-shot Bybit Demo smoke test"
    )
    p.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Bybit linear perpetual symbol (default: BTCUSDT)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Decision + risk only; place ZERO orders",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbol = args.symbol.strip().upper()

    print()
    print("=" * 72)
    print("  CoilBrain V1 -- Bybit Demo Smoke Test")
    print(f"  Symbol : {symbol}")
    print(f"  Mode   : {'DRY RUN (no orders)' if args.dry_run else 'LIVE DEMO EXECUTION'}")
    print("=" * 72)

    if not args.dry_run:
        print(
            "\n  NOTICE: This places a real order on Bybit DEMO (not production).\n"
            "  The position will remain open until manually closed or hit SL/TP.\n"
        )

    report = asyncio.run(run_smoke_test(symbol=symbol, dry_run=args.dry_run))
    report.print_report()

    fail_statuses = {"PREFLIGHT_FAILED", "INSTRUMENT_INVALID", "ENTRY_FAILED",
                     "RISK_FAILED", "NO_PRICE", "POSITION_CONFLICT"}
    if report.errors or report.execution_status in fail_statuses:
        sys.exit(1)


if __name__ == "__main__":
    main()
