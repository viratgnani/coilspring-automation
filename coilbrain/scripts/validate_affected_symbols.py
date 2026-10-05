"""Validation script for symbol resolution, live snapshot freshness, and stale-data safety.

SAFETY:
- Read-only live market data fetch from Binance public REST API.
- NO order placement or exchange credentials used.
- Validates MAX_DEMO_TRADES_PER_SESSION = 1.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add coilbrain src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx
from coilbrain.decision_engine import (
    Decision,
    DecisionConfig,
    DecisionEngine,
    PositionState,
    Reason,
    SignalFeatures,
)
from coilbrain.demo_guard import MAX_DEMO_TRADES_PER_SESSION
from coilbrain.microstructure_fetcher import (
    MicrostructureSnapshot,
    fetch_all,
    normalize_binance_symbol,
)
from coilbrain.observatory import to_binance_symbol


async def main() -> None:
    print("=" * 70)
    print("COILBRAIN V1 -- AFFECTED SYMBOLS LIVE VALIDATION")
    print("=" * 70)

    # 1. Check Demo Session Guard constant
    print(f"\n[DEMO GUARD] MAX_DEMO_TRADES_PER_SESSION = {MAX_DEMO_TRADES_PER_SESSION}")
    assert MAX_DEMO_TRADES_PER_SESSION == 1, "MAX_DEMO_TRADES_PER_SESSION must be 1"

    # 2. Test the 3 affected symbols from the live session
    affected_symbols = ["MOVR/USDT:USDT", "GTC/USDT:USDT", "SOMI/USDT:USDT"]
    engine = DecisionEngine(DecisionConfig(max_data_age_ms=10_000))

    print("\n[LIVE MICROSTRUCTURE FETCH & FRESHNESS CHECK]")
    async with httpx.AsyncClient() as client:
        for sym in affected_symbols:
            norm = normalize_binance_symbol(sym)
            obs_norm = to_binance_symbol(sym)
            print(f"\nSymbol: {sym}")
            print(f"  -> normalize_binance_symbol: {norm}")
            print(f"  -> to_binance_symbol:        {obs_norm}")
            assert norm == obs_norm, f"Mismatch: {norm} vs {obs_norm}"

            fetch_start = time.time()
            snap = await fetch_all(client, sym, since_ts=fetch_start)
            fetch_end = time.time()
            snapshot_age_s = time.time() - snap.snapshot_ts
            fetch_latency_ms = (fetch_end - fetch_start) * 1000

            print(f"  Binance Symbol requested: {snap.binance_symbol}")
            print(f"  Mark price:               {snap.mark_price}")
            print(f"  Best Bid / Ask:           {snap.best_bid} / {snap.best_ask}")
            print(f"  Bid Depth USDT:           {snap.bid_depth_usdt:,.2f}")
            print(f"  Ask Depth USDT:           {snap.ask_depth_usdt:,.2f}")
            print(f"  Spread:                   {snap.spread_pct:.4f}%")
            print(f"  Open Interest USDT:       {snap.oi_usdt:,.2f}" if snap.oi_usdt else "  OI: None")
            print(f"  Funding rate:             {snap.funding_rate}")
            print(f"  Fetch latency:            {fetch_latency_ms:.1f} ms")
            print(f"  Actual snapshot age:      {snapshot_age_s:.3f} s")

            # Assertions for fresh snapshot
            assert snap.mark_price is not None and snap.mark_price > 0, "Mark price missing"
            assert snap.best_bid is not None and snap.best_bid > 0, "Best bid missing"
            assert snap.best_ask is not None and snap.best_ask > 0, "Best ask missing"
            assert snap.bid_depth_usdt is not None, "Bid depth missing"
            assert snap.ask_depth_usdt is not None, "Ask depth missing"
            assert snapshot_age_s < 10.0, f"Snapshot age {snapshot_age_s}s exceeded 10.0s threshold!"
            print(f"  [PASS] Fresh snapshot confirmed (< 10s)")

            # Test through DecisionEngine freshness gate
            now = time.time()
            features = SignalFeatures(
                signal_id=f"val-{norm}",
                symbol=sym,
                score=95.0,
                grade="A",
                body_spike=5.0,
                volume_spike=5.0,
                volume_24h_usd=5_000_000.0,
                change_24h_pct=25.0,
                breakout_pct=5.0,
                flat_period_days=30.0,
                signal_timestamp=now,
            )
            pos = PositionState(symbol=sym, state="NO_POSITION")
            decision_res = engine.decide(
                signal=features,
                snapshot=snap,
                position=pos,
                bybit_symbol=norm,
                fetch_start_ts=fetch_start,
                fetch_end_ts=fetch_end,
            )
            assert Reason.STALE_MARKET_DATA not in decision_res.reasons, "Fresh data incorrectly marked STALE!"
            assert Reason.MISSING_MARKET_DATA not in decision_res.reasons, "Fresh data incorrectly marked MISSING!"
            print(f"  [PASS] DecisionEngine accepts data as fresh (reasons: {decision_res.reasons[:3]}...)")

            # 3. Test stale data zero-order behavior
            stale_snap = MicrostructureSnapshot(
                binance_symbol=norm,
                snapshot_ts=now - 25.0,  # 25 seconds old
                oi_usdt=snap.oi_usdt,
                oi_delta_pct=snap.oi_delta_pct,
                funding_rate=snap.funding_rate,
                mark_price=snap.mark_price,
                next_funding_ts=snap.next_funding_ts,
                best_bid=snap.best_bid,
                best_ask=snap.best_ask,
                spread_abs=snap.spread_abs,
                spread_pct=snap.spread_pct,
                bid_depth_usdt=snap.bid_depth_usdt,
                ask_depth_usdt=snap.ask_depth_usdt,
                ob_imbalance=snap.ob_imbalance,
                trade_count=snap.trade_count,
                traded_vol_usdt=snap.traded_vol_usdt,
                buy_vol_usdt=snap.buy_vol_usdt,
                sell_vol_usdt=snap.sell_vol_usdt,
                trade_imbalance=snap.trade_imbalance,
            )
            stale_res = engine.decide(
                signal=features,
                snapshot=stale_snap,
                position=pos,
                bybit_symbol=norm,
                fetch_start_ts=now - 0.2,
                fetch_end_ts=now,
            )
            assert stale_res.decision == Decision.IGNORE, f"Expected IGNORE for stale data, got {stale_res.decision}"
            assert Reason.STALE_MARKET_DATA in stale_res.reasons, "STALE_MARKET_DATA reason missing"
            print(f"  [PASS] Stale snapshot correctly failed closed with IGNORE(STALE_MARKET_DATA) -> Zero orders")

    print("\n" + "=" * 70)
    print("ALL VALIDATION CHECKS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
