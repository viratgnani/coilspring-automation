"""Core analytical logic for Research Analysis V1."""

import statistics
from dataclasses import dataclass
from typing import Any, Sequence

import aiosqlite

MIN_SAMPLE = 30


@dataclass
class BucketResult:
    bucket_name: str
    n: int
    unique_symbols: int
    mean_forward_return: float | None
    median_forward_return: float | None
    positive_return_pct: float | None
    stddev: float | None
    sample_quality: str
    remaining_mfe: str = "NOT_AVAILABLE"
    reason: str = "insufficient point-in-time data"


def _aggregate(bucket_name: str, symbol_returns: list[tuple[str, float]], min_sample: int = MIN_SAMPLE) -> BucketResult:
    n = len(symbol_returns)
    unique_symbols = len(set(sym for sym, _ in symbol_returns))
    
    if n == 0:
        return BucketResult(bucket_name, 0, 0, None, None, None, None, "LOW_SAMPLE")

    returns = [r for _, r in symbol_returns]
    mean_ret = statistics.mean(returns)
    median_ret = statistics.median(returns)
    pos_pct = sum(1 for r in returns if r > 0) / n * 100.0
    stddev = statistics.stdev(returns) if n > 1 else None
    
    sample_quality = "OK" if n >= min_sample else "LOW_SAMPLE"

    return BucketResult(
        bucket_name=bucket_name,
        n=n,
        unique_symbols=unique_symbols,
        mean_forward_return=mean_ret,
        median_forward_return=median_ret,
        positive_return_pct=pos_pct,
        stddev=stddev,
        sample_quality=sample_quality
    )


async def get_forward_returns(
    db: aiosqlite.Connection,
    information_horizons: Sequence[int] = (30, 60, 120),
    min_sample: int = MIN_SAMPLE,
) -> list[dict[str, Any]]:
    """Calculate forward returns from information horizons to future horizons."""
    # We join research_horizon_outcomes (info) with research_horizon_outcomes (future)
    query = """
        SELECT
            i.signal_id,
            f_feat.symbol,
            i.horizon_s AS info_horizon_s,
            i.actual_elapsed_s AS info_elapsed_s,
            f.horizon_s AS future_horizon_s,
            f.actual_elapsed_s AS future_elapsed_s,
            i.price AS info_price,
            f.price AS future_price
        FROM research_horizon_outcomes i
        JOIN research_signal_features f_feat ON i.signal_id = f_feat.signal_id
        JOIN research_horizon_outcomes f ON i.signal_id = f.signal_id
        WHERE i.horizon_s IN ({})
          AND f.horizon_s > i.horizon_s
          AND i.price IS NOT NULL
          AND f.price IS NOT NULL
    """.format(",".join("?" for _ in information_horizons))
    
    async with db.execute(query, information_horizons) as cur:
        rows = await cur.fetchall()

    # Group by (info_horizon, future_horizon)
    grouped: dict[tuple[int, int], list[tuple[str, float]]] = {}
    for row in rows:
        sym = row["symbol"]
        info_h = row["info_horizon_s"]
        fut_h = row["future_horizon_s"]
        
        info_price = row["info_price"]
        fut_price = row["future_price"]
        
        fwd_ret = (fut_price / info_price - 1.0) * 100.0
        
        grouped.setdefault((info_h, fut_h), []).append((sym, fwd_ret))

    results = []
    # Sort groups logically
    for (info_h, fut_h) in sorted(grouped.keys()):
        bucket = _aggregate(f"T+{info_h}s -> T+{fut_h}s", grouped[(info_h, fut_h)], min_sample)
        results.append({
            "information_time_s": info_h,
            "future_horizon_s": fut_h,
            "forward_duration_s": fut_h - info_h,  # Nominal forward duration
            "bucket": bucket,
        })
        
    return results


async def get_microstructure_buckets(
    db: aiosqlite.Connection,
    feature_column: str,
    bucket_boundaries: list[tuple[str, float | None, float | None]],
    information_horizons: Sequence[int] = (30, 60, 120),
    min_sample: int = MIN_SAMPLE,
) -> dict[str, Any]:
    """Bucket signals by a microstructure feature and analyze future returns."""
    # future_horizon_s is hardcoded to specific ones we want to inspect for simplicity, 
    # but we can group them by the nominal forward duration.
    # To keep it manageable, we'll bucket by (info_horizon, future_horizon, bucket_name)
    
    query = f"""
        SELECT
            i.signal_id,
            f_feat.symbol,
            i.horizon_s AS info_horizon_s,
            f.horizon_s AS future_horizon_s,
            i.{feature_column} AS feature_val,
            i.price AS info_price,
            f.price AS future_price
        FROM research_horizon_outcomes i
        JOIN research_signal_features f_feat ON i.signal_id = f_feat.signal_id
        JOIN research_horizon_outcomes f ON i.signal_id = f.signal_id
        WHERE i.horizon_s IN ({",".join("?" for _ in information_horizons)})
          AND f.horizon_s > i.horizon_s
          AND i.price IS NOT NULL
          AND f.price IS NOT NULL
          AND i.{feature_column} IS NOT NULL
    """
    
    async with db.execute(query, information_horizons) as cur:
        rows = await cur.fetchall()
        
    # grouped[info_horizon][future_horizon][bucket_name] = [(sym, fwd_ret)]
    grouped: dict[int, dict[int, dict[str, list[tuple[str, float]]]]] = {}
    
    for row in rows:
        sym = row["symbol"]
        info_h = row["info_horizon_s"]
        fut_h = row["future_horizon_s"]
        val = row["feature_val"]
        fwd_ret = (row["future_price"] / row["info_price"] - 1.0) * 100.0
        
        bucket_name = None
        for b_name, b_min, b_max in bucket_boundaries:
            if b_min is not None and val < b_min:
                continue
            if b_max is not None and val >= b_max:
                continue
            bucket_name = b_name
            break
            
        if bucket_name:
            if info_h not in grouped:
                grouped[info_h] = {}
            if fut_h not in grouped[info_h]:
                grouped[info_h][fut_h] = {}
            if bucket_name not in grouped[info_h][fut_h]:
                grouped[info_h][fut_h][bucket_name] = []
                
            grouped[info_h][fut_h][bucket_name].append((sym, fwd_ret))

    results = []
    for info_h in sorted(grouped.keys()):
        for fut_h in sorted(grouped[info_h].keys()):
            for b_name, _, _ in bucket_boundaries:
                vals = grouped[info_h][fut_h].get(b_name, [])
                bucket_res = _aggregate(b_name, vals, min_sample)
                results.append({
                    "information_time_s": info_h,
                    "future_horizon_s": fut_h,
                    "forward_duration_s": fut_h - info_h,
                    "bucket": bucket_res
                })
                
    return {"feature": feature_column, "results": results}


async def get_combined_buckets(
    db: aiosqlite.Connection,
    information_horizons: Sequence[int] = (30, 60, 120),
    min_sample: int = MIN_SAMPLE,
) -> list[dict[str, Any]]:
    """Test combined microstructure states."""
    
    # Conditions:
    # OI increasing = oi_delta_from_t0_pct > 0
    # Trade imbalance increasing = trade_imbalance_delta > 0
    # OB imbalance increasing = ob_imbalance_delta_from_t0 > 0
    
    query = f"""
        SELECT
            i.signal_id,
            f_feat.symbol,
            i.horizon_s AS info_horizon_s,
            f.horizon_s AS future_horizon_s,
            i.oi_delta_from_t0_pct,
            i.trade_imbalance_delta,
            i.ob_imbalance_delta_from_t0,
            i.price AS info_price,
            f.price AS future_price
        FROM research_horizon_outcomes i
        JOIN research_signal_features f_feat ON i.signal_id = f_feat.signal_id
        JOIN research_horizon_outcomes f ON i.signal_id = f.signal_id
        WHERE i.horizon_s IN ({",".join("?" for _ in information_horizons)})
          AND f.horizon_s > i.horizon_s
          AND i.price IS NOT NULL
          AND f.price IS NOT NULL
    """
    
    async with db.execute(query, information_horizons) as cur:
        rows = await cur.fetchall()

    grouped: dict[int, dict[int, dict[str, list[tuple[str, float]]]]] = {}
    
    for row in rows:
        sym = row["symbol"]
        info_h = row["info_horizon_s"]
        fut_h = row["future_horizon_s"]
        
        oi_delta = row["oi_delta_from_t0_pct"]
        trd_delta = row["trade_imbalance_delta"]
        ob_delta = row["ob_imbalance_delta_from_t0"]
        
        if oi_delta is None: continue
        
        fwd_ret = (row["future_price"] / row["info_price"] - 1.0) * 100.0
        
        buckets = []
        if trd_delta is not None:
            if oi_delta > 0 and trd_delta > 0:
                buckets.append("OI UP + Trade Imb UP")
            if oi_delta > 0 and trd_delta < 0:
                buckets.append("OI UP + Trade Imb DOWN")
            if oi_delta < 0 and trd_delta > 0:
                buckets.append("OI DOWN + Trade Imb UP")
                
        if ob_delta is not None:
            if oi_delta > 0 and ob_delta > 0:
                buckets.append("OI UP + OB Imb UP")
            if oi_delta > 0 and ob_delta < 0:
                buckets.append("OI UP + OB Imb DOWN")
                
        for b_name in buckets:
            if info_h not in grouped: grouped[info_h] = {}
            if fut_h not in grouped[info_h]: grouped[info_h][fut_h] = {}
            if b_name not in grouped[info_h][fut_h]: grouped[info_h][fut_h][b_name] = []
            grouped[info_h][fut_h][b_name].append((sym, fwd_ret))

    results = []
    for info_h in sorted(grouped.keys()):
        for fut_h in sorted(grouped[info_h].keys()):
            for b_name in sorted(grouped[info_h][fut_h].keys()):
                vals = grouped[info_h][fut_h][b_name]
                bucket_res = _aggregate(b_name, vals, min_sample)
                results.append({
                    "information_time_s": info_h,
                    "future_horizon_s": fut_h,
                    "forward_duration_s": fut_h - info_h,
                    "bucket": bucket_res
                })
                
    return results
