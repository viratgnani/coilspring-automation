"""CLI for Research Analysis V1."""

import argparse
import asyncio
from typing import Any

from coilbrain import database
from coilbrain.config import settings
from coilbrain.research.analysis import core, exporter, regime_core


def main():
    parser = argparse.ArgumentParser(description="Research Analysis V1")
    parser.add_argument("--forward", action="store_true", help="Run forward returns analysis")
    parser.add_argument("--oi", action="store_true", help="Run OI trajectory analysis")
    parser.add_argument("--ob", action="store_true", help="Run OB imbalance trajectory analysis")
    parser.add_argument("--trade", action="store_true", help="Run Trade imbalance trajectory analysis")
    parser.add_argument("--combined", action="store_true", help="Run combined states analysis")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    parser.add_argument("--min-sample", type=int, default=30, help="Minimum sample size to mark as OK")
    
    parser.add_argument("--regime", action="store_true", help="Run Regime Research Engine V2")
    parser.add_argument("--regime-symbol", type=str, help="Filter regime output to a specific symbol case study")
    parser.add_argument("--regime-signal-id", type=str, help="Filter regime output to a specific signal case study")
    
    args = parser.parse_args()
    asyncio.run(_run(args))


async def _run(args):
    # Determine which database URL to use, default to settings, but user expects
    # it to use research_snapshot.db if DATABASE_URL points to it.
    database.configure(settings.database_url)
    
    do_forward = args.forward or args.all
    do_oi = args.oi or args.all
    do_ob = args.ob or args.all
    do_trade = args.trade or args.all
    do_combined = args.combined or args.all
    do_regime = args.regime

    if not any([do_forward, do_oi, do_ob, do_trade, do_combined, do_regime]):
        print("Please specify at least one analysis to run: --forward, --oi, --ob, --trade, --combined, --regime, or --all")
        return

    async with database.get_db() as db:
        if do_regime:
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
                
        if do_forward:
            print("\nRunning Forward Returns Analysis...")
            res = await core.get_forward_returns(db, min_sample=args.min_sample)
            _print_results(res)
            exporter.export_forward_returns(res)
            print("Saved to research_analysis_forward_returns.csv")

        if do_oi:
            print("\nRunning OI Trajectory Analysis...")
            res = await core.get_microstructure_buckets(
                db, 
                feature_column="oi_delta_from_t0_pct",
                bucket_boundaries=[
                    ("< -1%", None, -1.0),
                    ("-1% to 0%", -1.0, 0.0),
                    ("0% to +1%", 0.0, 1.0),
                    ("+1% to +3%", 1.0, 3.0),
                    ("> +3%", 3.0, None)
                ],
                min_sample=args.min_sample
            )
            _print_feature_results(res)
            exporter.export_microstructure_buckets(res, "research_analysis_oi_buckets.csv")
            print("Saved to research_analysis_oi_buckets.csv")

        if do_ob:
            print("\nRunning OB Imbalance Trajectory Analysis...")
            # For imablances we'll just bucket loosely by quantiles or logical values, e.g. < -0.2, -0.2 to 0, 0 to 0.2, > 0.2
            res = await core.get_microstructure_buckets(
                db, 
                feature_column="ob_imbalance_delta_from_t0",
                bucket_boundaries=[
                    ("< -0.2", None, -0.2),
                    ("-0.2 to 0", -0.2, 0.0),
                    ("0 to +0.2", 0.0, 0.2),
                    ("> +0.2", 0.2, None)
                ],
                min_sample=args.min_sample
            )
            _print_feature_results(res)
            exporter.export_microstructure_buckets(res, "research_analysis_ob_buckets.csv")
            print("Saved to research_analysis_ob_buckets.csv")

        if do_trade:
            print("\nRunning Trade Imbalance Trajectory Analysis...")
            res = await core.get_microstructure_buckets(
                db, 
                feature_column="trade_imbalance_delta",
                bucket_boundaries=[
                    ("< -0.5", None, -0.5),
                    ("-0.5 to 0", -0.5, 0.0),
                    ("0 to +0.5", 0.0, 0.5),
                    ("> +0.5", 0.5, None)
                ],
                min_sample=args.min_sample
            )
            _print_feature_results(res)
            exporter.export_microstructure_buckets(res, "research_analysis_trade_buckets.csv")
            print("Saved to research_analysis_trade_buckets.csv")

        if do_combined:
            print("\nRunning Combined States Analysis...")
            res = await core.get_combined_buckets(db, min_sample=args.min_sample)
            _print_results(res)
            exporter.export_combined_buckets(res)
            print("Saved to research_analysis_combined_buckets.csv")


def _print_results(results: list[dict[str, Any]]):
    for r in results:
        b = r["bucket"]
        print(f"Info: T+{r['information_time_s']}s | Future: T+{r['future_horizon_s']}s "
              f"(Duration: {r['forward_duration_s']}s) | Bucket: {b.bucket_name}")
        def fmt(v): return f"{v:.2f}%" if v is not None else "NULL"
        print(f"  N: {b.n} (Symbols: {b.unique_symbols}) | Quality: {b.sample_quality}")
        if b.n > 0:
            print(f"  Mean: {fmt(b.mean_forward_return)} | Median: {fmt(b.median_forward_return)} | Pos: {fmt(b.positive_return_pct)}")
            
def _print_feature_results(data: dict[str, Any]):
    print(f"Feature: {data['feature']}")
    _print_results(data["results"])


def _print_regime_case_study(results: dict[str, Any], signal_id: str | None, symbol: str | None) -> None:
    print(f"\n--- REGIME CASE STUDY ---")
    states = results.get("states", [])
    
    # Filter by signal_id or symbol
    if signal_id:
        target_states = [s for s in states if s["signal_id"] == signal_id]
        if not target_states:
            print(f"No data found for signal_id: {signal_id}")
            return
    elif symbol:
        # Group by signal_id for the symbol
        target_signals = set(s["signal_id"] for s in states if s["symbol"] == symbol)
        if not target_signals:
            print(f"No data found for symbol: {symbol}")
            return
        # Just pick the first signal for the case study view
        chosen_id = list(target_signals)[0]
        print(f"Found {len(target_signals)} signals for {symbol}. Displaying the first one: {chosen_id}")
        target_states = [s for s in states if s["signal_id"] == chosen_id]
    else:
        return
        
    target_states.sort(key=lambda x: x["information_horizon"])
    
    print(f"{'Horizon':<10} {'State':<25} {'Price':<12} {'OI USDT':<15} {'OB Imbalance':<15} {'Trade Imbalance':<15} {'Price Ret':<10}")
    print("-" * 110)
    
    for st in target_states:
        h = st["information_horizon"]
        state = st["state"]
        price = f"{st['price']:.5f}" if st["price"] is not None else "NULL"
        oi = f"{st['oi_usdt']:.0f}" if st["oi_usdt"] is not None else "NULL"
        ob = f"{st['ob_imbalance']:.3f}" if st["ob_imbalance"] is not None else "NULL"
        tr = f"{st['trade_imbalance']:.3f}" if st["trade_imbalance"] is not None else "NULL"
        ret = f"{st['price_return_from_t0']:.3f}%" if st["price_return_from_t0"] is not None else "NULL"
        
        print(f"{h:<10} {state:<25} {price:<12} {oi:<15} {ob:<15} {tr:<15} {ret:<10}")
        
    # Also grab forward returns for this signal
    fwd_rets = results.get("state_fwd_returns", [])
    # Wait, state_fwd_returns is aggregated. We don't have individual forward returns in the output dict easily accessible for printing, 
    # but the prompt mainly asked for the time-series state sequence for the case study.
    print("\n-------------------------")


if __name__ == "__main__":
    main()
