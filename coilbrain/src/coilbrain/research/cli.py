"""CLI interface for the Research Engine."""

import argparse
import asyncio
import sys

from coilbrain import database
from coilbrain.config import settings
from coilbrain.research import engine
from coilbrain.research.analysis import regime_core, exporter
from coilbrain.research.analysis.cli import _print_regime_case_study

def main():
    parser = argparse.ArgumentParser(description="CoilBrain Research Engine V1")
    parser.add_argument("--generate", action="store_true", help="Generate research outcomes for new COMPLETE signals")
    parser.add_argument("--symbol", type=str, help="Output MYX-style investigation timeline for a symbol")
    parser.add_argument("--signal-id", type=str, help="Output timeline for a specific signal")
    parser.add_argument("--summary", action="store_true", help="Output bucketed descriptive statistics")
    parser.add_argument("--runner-threshold", type=float, default=2.0, help="MFE >= this value is considered a runner (default: 2.0)")

    # Regime Analysis V2 arguments
    parser.add_argument("--regime", action="store_true", help="Run Regime Research Engine V2")
    parser.add_argument("--regime-symbol", type=str, help="Filter regime output to a specific symbol case study")
    parser.add_argument("--regime-signal-id", type=str, help="Filter regime output to a specific signal case study")
    parser.add_argument("--min-sample", type=int, default=30, help="Minimum sample size to mark as OK")

    args = parser.parse_args()

    # The SQLite driver needs an event loop
    asyncio.run(_run(args))


async def _run(args):
    database.configure(settings.database_url)
    # Ensure tables exist
    await database.init_db(settings.database_url)

    async with database.get_db() as db:
        if args.generate:
            processed = await engine.run_generation(db)
            print(f"Generated outcomes for {processed} signals.")
            return

        if args.symbol:
            await _print_symbol_timeline(db, args.symbol)
            return

        if args.signal_id:
            await _print_signal_timeline(db, args.signal_id)
            return

        if args.summary:
            await _print_summary(db, args.runner_threshold)
            return

        if args.regime:
            print(f"\nRunning Regime Research Engine V2... (min_sample={args.min_sample})")
            res = await regime_core.build_regime_dataset(db, min_sample=args.min_sample)
            
            if args.regime_signal_id or args.regime_symbol:
                _print_regime_case_study(res, args.regime_signal_id, args.regime_symbol)
            else:
                exporter.export_regime_results(res)
                print("\nValidation Checks:")
                for k, v in res["dataset_summary"].items():
                    print(f"  {k}: {v}")
                print("\nExported 8 research regime CSVs.")
            return

    # If no arguments provided, print help
    print("Please provide an action: --generate, --symbol <sym>, --signal-id <id>, --summary, or --regime.")


async def _print_symbol_timeline(db, symbol: str):
    print(f"--- Research Timeline for {symbol} ---")
    async with db.execute(
        "SELECT signal_id FROM research_signal_features WHERE symbol = ? ORDER BY t0_timestamp ASC",
        (symbol,)
    ) as cur:
        rows = await cur.fetchall()
    
    if not rows:
        print("No research data found for symbol.")
        return
    
    for row in rows:
        await _print_signal_timeline(db, row[0])
        print("\n" + "="*80 + "\n")


async def _print_signal_timeline(db, signal_id: str):
    async with db.execute(
        "SELECT * FROM research_signal_features WHERE signal_id = ?", (signal_id,)
    ) as cur:
        sig = await cur.fetchone()

    if not sig:
        print(f"Signal {signal_id} not found in research tables.")
        return

    print(f"Signal ID: {signal_id}  |  Symbol: {sig['symbol']}")
    print(f"Grade: {sig['grade']}  |  Score: {sig['score']:.1f}")
    print(f"T0 Price: {sig['t0_price']:.4f}")
    print(f"Overall MFE: {sig['mfe_pct']:.2f}%  |  MAE: {sig['mae_pct']:.2f}%")
    
    print("\n--- Microstructure Trajectory ---")
    
    # Print T0 baseline
    _print_row(
        "T0",
        price=sig['t0_price'],
        return_pct=0.0,
        oi=sig['t0_oi_usdt'],
        oi_delta=sig['t0_oi_delta_pct'],
        funding=sig['t0_funding_rate'],
        spread=sig['t0_spread_pct'],
        ob_imb=sig['t0_ob_imbalance'],
        buy_vol=sig['t0_buy_vol_usdt'],
        sell_vol=sig['t0_sell_vol_usdt'],
        trd_imb=sig['t0_trade_imbalance']
    )

    # Print horizons
    async with db.execute(
        "SELECT * FROM research_horizon_outcomes WHERE signal_id = ? ORDER BY horizon_s ASC",
        (signal_id,)
    ) as cur:
        horizons = await cur.fetchall()

    for h in horizons:
        _print_row(
            f"T+{h['horizon_s']}s",
            price=h['price'],
            return_pct=h['return_pct'],
            oi=h['oi_usdt'],
            oi_delta=h['oi_delta_from_t0_pct'],
            funding=h['funding_rate'],
            spread=h['spread_pct'],
            ob_imb=h['ob_imbalance'],
            buy_vol=h['buy_vol_usdt'],
            sell_vol=h['sell_vol_usdt'],
            trd_imb=h['trade_imbalance']
        )


def _print_row(label, price, return_pct, oi, oi_delta, funding, spread, ob_imb, buy_vol, sell_vol, trd_imb):
    def fmt(v, precision=2, is_pct=False):
        if v is None: return "NULL"
        suffix = "%" if is_pct else ""
        return f"{v:.{precision}f}{suffix}"

    print(f"{label:>8} | "
          f"Prc: {fmt(price, 4)} ({fmt(return_pct, 2, True):>7}) | "
          f"OI: {fmt(oi, 0)} ({fmt(oi_delta, 2, True):>7}) | "
          f"Fund: {fmt(funding, 4)} | "
          f"Sprd: {fmt(spread, 3, True)} | "
          f"OB Imb: {fmt(ob_imb, 2)} | "
          f"Buy: {fmt(buy_vol, 0)} | "
          f"Sell: {fmt(sell_vol, 0)} | "
          f"Trd Imb: {fmt(trd_imb, 2)}")


async def _print_summary(db, runner_threshold: float):
    print("--- Research Summary ---")
    print(f"Runner threshold: MFE >= {runner_threshold}%\n")
    
    async with db.execute("SELECT COUNT(*) FROM research_signal_features") as cur:
        total_signals = (await cur.fetchone())[0]
    
    if total_signals == 0:
        print("No data available.")
        return

    print(f"Total Signals: {total_signals}")
    
    # 1. Overall signal outcomes per horizon
    print("\n--- Overall Forward Returns ---")
    for h in engine.HORIZONS_S:
        async with db.execute(f"""
            SELECT COUNT(return_pct), AVG(return_pct)
            FROM research_horizon_outcomes
            WHERE horizon_s = {h} AND return_pct IS NOT NULL
        """) as cur:
            cnt, mean_ret = await cur.fetchone()
        
        async with db.execute(f"""
            SELECT COUNT(return_pct)
            FROM research_horizon_outcomes
            WHERE horizon_s = {h} AND return_pct > 0
        """) as cur:
            pos_cnt = (await cur.fetchone())[0]
            
        pos_pct = (pos_cnt / cnt * 100) if cnt else 0
        m = f"{mean_ret:.2f}%" if mean_ret is not None else "N/A"
        print(f"T+{h:<4}s | N={cnt:<4} | Mean Ret: {m:>7} | Positive: {pos_pct:.1f}%")

    # 2. Runner vs Non-Runner
    async with db.execute(f"""
        SELECT 
            SUM(CASE WHEN mfe_pct >= {runner_threshold} THEN 1 ELSE 0 END) as runners,
            SUM(CASE WHEN mfe_pct < {runner_threshold} THEN 1 ELSE 0 END) as non_runners
        FROM research_signal_features
    """) as cur:
        runners, non_runners = await cur.fetchone()
        
    print(f"\n--- Runner Comparison (N_Runners={runners or 0}, N_NonRunners={non_runners or 0}) ---")
    
    for h in [60, 300, 3600]:
        print(f"\nHorizon T+{h}s Microstructure Deltas from T0:")
        
        # We look at average OI delta, OB imbalance, Trade Imbalance
        async with db.execute(f"""
            SELECT 
                AVG(o.oi_delta_from_t0_pct),
                AVG(o.ob_imbalance_delta_from_t0),
                AVG(o.trade_imbalance_delta)
            FROM research_horizon_outcomes o
            JOIN research_signal_features f ON o.signal_id = f.signal_id
            WHERE o.horizon_s = {h} AND f.mfe_pct >= {runner_threshold}
        """) as cur:
            r_oi, r_ob, r_trd = await cur.fetchone()
            
        async with db.execute(f"""
            SELECT 
                AVG(o.oi_delta_from_t0_pct),
                AVG(o.ob_imbalance_delta_from_t0),
                AVG(o.trade_imbalance_delta)
            FROM research_horizon_outcomes o
            JOIN research_signal_features f ON o.signal_id = f.signal_id
            WHERE o.horizon_s = {h} AND f.mfe_pct < {runner_threshold}
        """) as cur:
            nr_oi, nr_ob, nr_trd = await cur.fetchone()
            
        def f(val): return f"{val:.3f}" if val is not None else "NULL"
        
        print(f"  Runners     -> OI Delta: {f(r_oi):>7}% | OB Imb Delta: {f(r_ob):>6} | Trd Imb Delta: {f(r_trd):>6}")
        print(f"  Non-Runners -> OI Delta: {f(nr_oi):>7}% | OB Imb Delta: {f(nr_ob):>6} | Trd Imb Delta: {f(nr_trd):>6}")


if __name__ == "__main__":
    main()
