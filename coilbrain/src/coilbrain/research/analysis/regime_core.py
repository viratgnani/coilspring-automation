"""Core logic for Regime Research Engine V2."""

import collections
import statistics
import logging
from typing import Any, Sequence, Dict, List, Tuple
import aiosqlite

from coilbrain.research.analysis import regime_v1
from coilbrain.research.analysis.core import MIN_SAMPLE

logger = logging.getLogger(__name__)

HORIZONS = [0, 30, 60, 120, 300, 600, 1800, 3600]
EVENT_CLUSTER_SECONDS = 3600

async def build_regime_dataset(db: aiosqlite.Connection, min_sample: int = MIN_SAMPLE) -> Dict[str, Any]:
    """
    Builds the regime dataset by querying ALL historical signals and horizons,
    applying the classification logic, and computing transitions and forward returns.
    """
    # 1. Fetch T0 Features for all signals
    # 2. Fetch Horizon Outcomes for all signals
    # 3. Perform the analysis
    
    # Let's extract T0 base features first
    query_t0 = """
        SELECT
            f.signal_id,
            f.symbol,
            f.t0_timestamp,
            f.t0_price,
            f.t0_snapshot_ts,
            f.t0_oi_usdt,
            f.t0_oi_delta_pct,
            f.t0_funding_rate,
            f.t0_spread_abs,
            f.t0_spread_pct,
            f.t0_ob_imbalance,
            f.t0_trade_count,
            f.t0_traded_vol_usdt,
            f.t0_buy_vol_usdt,
            f.t0_sell_vol_usdt,
            f.t0_trade_imbalance,
            s.bid_depth_usdt,
            s.ask_depth_usdt
        FROM research_signal_features f
        LEFT JOIN microstructure_snapshots s
               ON s.binance_symbol = f.symbol AND s.snapshot_ts = f.t0_snapshot_ts
    """
    
    async with db.execute(query_t0) as cur:
        t0_rows = await cur.fetchall()
        t0_keys = [description[0] for description in cur.description]
        t0_data = [dict(zip(t0_keys, row)) for row in t0_rows]

    # Map T0 data by signal_id
    signals_t0 = {row["signal_id"]: row for row in t0_data}

    # Fetch all horizon outcomes
    query_horizons = """
        SELECT
            h.signal_id,
            h.horizon_s,
            h.target_elapsed_s,
            h.actual_elapsed_s,
            h.alignment_error_s,
            h.observation_ts,
            h.price,
            h.return_pct,
            h.snapshot_ts,
            h.oi_usdt,
            h.oi_delta_pct,
            h.funding_rate,
            h.spread_abs,
            h.spread_pct,
            h.ob_imbalance,
            h.trade_count,
            h.traded_vol_usdt,
            h.buy_vol_usdt,
            h.sell_vol_usdt,
            h.trade_imbalance,
            h.oi_delta_from_t0_pct,
            h.spread_delta_from_t0_pct,
            h.ob_imbalance_delta_from_t0,
            h.buy_volume_delta,
            h.sell_volume_delta,
            h.trade_imbalance_delta,
            s.bid_depth_usdt,
            s.ask_depth_usdt
        FROM research_horizon_outcomes h
        JOIN research_signal_features f ON h.signal_id = f.signal_id
        LEFT JOIN microstructure_snapshots s
               ON s.binance_symbol = f.symbol AND s.snapshot_ts = h.snapshot_ts
        ORDER BY h.signal_id, h.horizon_s
    """

    async with db.execute(query_horizons) as cur:
        h_rows = await cur.fetchall()
        h_keys = [description[0] for description in cur.description]
        h_data = [dict(zip(h_keys, row)) for row in h_rows]

    # Group horizons by signal_id
    horizons_by_signal = collections.defaultdict(dict)
    for row in h_data:
        horizons_by_signal[row["signal_id"]][row["horizon_s"]] = row
        
    return process_raw_dataset(signals_t0, horizons_by_signal, min_sample)


def safe_div(num, den):
    if num is None or den is None or den == 0:
        return None
    return num / den

def safe_sub(a, b):
    if a is None or b is None:
        return None
    return a - b


def process_raw_dataset(signals_t0: Dict[str, Dict], horizons_by_signal: Dict[str, Dict], min_sample: int):
    # We will build:
    # 1. State at each horizon
    # 2. Transitions
    # 3. Forward returns for each state
    # 4. Forward returns for each transition
    # 5. Buckets for continuous features
    # 6. Combined states
    # 7. Symbol summaries
    # 8. Dataset summaries

    def _bucket_feature(val, feature_type):
        if feature_type == "oi_delta":
            if val < -1.0: return "< -1%"
            if val <= 0.0: return "-1% to 0%"
            if val <= 1.0: return "0% to +1%"
            if val <= 3.0: return "+1% to +3%"
            return "> +3%"
        if feature_type == "ob_delta" or feature_type == "trade_imbalance_delta":
            if val < -0.2: return "< -0.2"
            if val <= 0.0: return "-0.2 to 0"
            if val <= 0.2: return "0 to +0.2"
            return "> +0.2"
        return "ALL"
        
    states_list = []
    transitions_list = []
    
    # Format: (info_horizon, state, future_horizon) -> [(forward_return_pct, symbol, event_id, split), ...]
    state_fwd_returns = collections.defaultdict(list)
    
    # Format: (from_h, from_s, to_h, to_s, future_h) -> [(forward_return_pct, symbol, event_id, split), ...]
    transition_fwd_returns = collections.defaultdict(list)
    
    feature_fwd_returns = collections.defaultdict(list) # For buckets
    combined_counts = collections.defaultdict(list) # For combined states
    
    total_signals = len(signals_t0)
    total_symbols = len(set(row["symbol"] for row in signals_t0.values()))
    
    # -----------------------------------------------------------------
    # Event Clustering & Chronological Splitting
    # -----------------------------------------------------------------
    sorted_signals = sorted(signals_t0.values(), key=lambda x: (x["symbol"], x["t0_timestamp"]))
    events_by_signal = {}
    event_id_counter = 0
    last_sym = None
    last_ts = None
    event_start_times = {}
    
    for row in sorted_signals:
        sig_id = row["signal_id"]
        sym = row["symbol"]
        ts = row["t0_timestamp"]
        
        if sym != last_sym or last_ts is None or (ts - last_ts) > EVENT_CLUSTER_SECONDS:
            event_id_counter += 1
            event_start_times[event_id_counter] = ts
            
        events_by_signal[sig_id] = event_id_counter
        last_sym = sym
        last_ts = ts
        
    sorted_events = sorted(event_start_times.items(), key=lambda x: x[1])
    n_events = len(sorted_events)
    train_idx = int(n_events * 0.6)
    val_idx = int(n_events * 0.8)
    
    event_to_split = {}
    for i, (eid, _) in enumerate(sorted_events):
        if i < train_idx:
            event_to_split[eid] = "TRAIN"
        elif i < val_idx:
            event_to_split[eid] = "VAL"
        else:
            event_to_split[eid] = "TEST"
            
    splits_by_signal = {sig_id: event_to_split[eid] for sig_id, eid in events_by_signal.items()}
    # -----------------------------------------------------------------
    
    signals_with_price = 0
    signals_with_micro = 0
    
    symbol_stats = collections.defaultdict(lambda: {
        "signal_count": 0,
        "micro_count": 0,
        "state_counts": collections.Counter(),
        "transition_counts": collections.Counter(),
        "fwd_returns": []
    })

    for signal_id, t0_row in signals_t0.items():
        symbol = t0_row["symbol"]
        sym_stats = symbol_stats[symbol]
        sym_stats["signal_count"] += 1
        
        has_price = t0_row["t0_price"] is not None
        has_micro = t0_row["t0_oi_usdt"] is not None or t0_row["t0_ob_imbalance"] is not None
        
        if has_price:
            signals_with_price += 1
        if has_micro:
            signals_with_micro += 1
            sym_stats["micro_count"] += 1
            
        h_dict = horizons_by_signal.get(signal_id, {})
        
        # Build horizon 0 data
        h0 = {
            "signal_id": signal_id,
            "horizon_s": 0,
            "target_elapsed_s": 0,
            "actual_elapsed_s": 0,
            "alignment_error_s": 0,
            "observation_ts": t0_row["t0_timestamp"],
            "price": t0_row["t0_price"],
            "return_pct": 0.0,
            "snapshot_ts": t0_row["t0_snapshot_ts"],
            "oi_usdt": t0_row["t0_oi_usdt"],
            "oi_delta_pct": t0_row["t0_oi_delta_pct"],
            "funding_rate": t0_row["t0_funding_rate"],
            "spread_abs": t0_row["t0_spread_abs"],
            "spread_pct": t0_row["t0_spread_pct"],
            "ob_imbalance": t0_row["t0_ob_imbalance"],
            "trade_count": t0_row["t0_trade_count"],
            "traded_vol_usdt": t0_row["t0_traded_vol_usdt"],
            "buy_vol_usdt": t0_row["t0_buy_vol_usdt"],
            "sell_vol_usdt": t0_row["t0_sell_vol_usdt"],
            "trade_imbalance": t0_row["t0_trade_imbalance"],
            "oi_delta_from_t0_pct": 0.0,
            "spread_delta_from_t0_pct": 0.0,
            "ob_imbalance_delta_from_t0": 0.0,
            "buy_volume_delta": 0.0,
            "sell_volume_delta": 0.0,
            "trade_imbalance_delta": 0.0,
            "bid_depth_usdt": t0_row["bid_depth_usdt"],
            "ask_depth_usdt": t0_row["ask_depth_usdt"],
        }
        
        all_h = {0: h0}
        for h in HORIZONS:
            if h != 0 and h in h_dict:
                all_h[h] = h_dict[h]
                
        # Calculate states at each horizon
        states_by_h = {}
        for i, h in enumerate(HORIZONS):
            if h not in all_h:
                continue
            curr = all_h[h]
            prev = all_h[HORIZONS[i-1]] if i > 0 and HORIZONS[i-1] in all_h else None
            
            price_return_from_t0 = safe_div(safe_sub(curr["price"], t0_row["t0_price"]), t0_row["t0_price"])
            if price_return_from_t0 is not None:
                price_return_from_t0 *= 100.0
                
            price_change_prev = safe_div(safe_sub(curr["price"], prev["price"]), prev["price"]) if prev else None
            if price_change_prev is not None:
                price_change_prev *= 100.0
                
            oi_change_prev = safe_sub(curr["oi_usdt"], prev["oi_usdt"]) if prev else None
            if oi_change_prev is not None and prev and prev["oi_usdt"]:
                oi_change_prev = (oi_change_prev / prev["oi_usdt"]) * 100.0
                
            res = regime_v1.classify_state(
                price_return_from_t0=price_return_from_t0,
                oi_delta_from_t0_pct=curr["oi_delta_from_t0_pct"],
                trade_imbalance=curr["trade_imbalance"],
                price_change_prev=price_change_prev,
                oi_change_prev=oi_change_prev,
                trade_imbalance_delta=curr["trade_imbalance_delta"]
            )
            state = res["state"]
            states_by_h[h] = state
            sym_stats["state_counts"][state] += 1
            
            # Record state
            buy_sell_ratio = safe_div(curr["buy_vol_usdt"], curr["sell_vol_usdt"])
            
            states_list.append({
                "signal_id": signal_id,
                "symbol": symbol,
                "information_horizon": h,
                "state": state,
                "regime_information_basis": res["regime_information_basis"],
                "price_basis": res["price_basis"],
                "oi_basis": res["oi_basis"],
                "trade_basis": res["trade_basis"],
                "effective_price_change": res["effective_price_change"],
                "effective_oi_change": res["effective_oi_change"],
                "effective_trade_measure": res["effective_trade_measure"],
                "regime_definition_version": regime_v1.REGIME_DEFINITION_VERSION,
                "price": curr["price"],
                "price_return_from_t0": price_return_from_t0,
                "price_change_from_previous_horizon": price_change_prev,
                "oi_usdt": curr["oi_usdt"],
                "oi_delta_from_t0_pct": curr["oi_delta_from_t0_pct"],
                "oi_delta_from_previous_horizon_pct": oi_change_prev,
                "ob_imbalance": curr["ob_imbalance"],
                "ob_delta_from_t0": curr["ob_imbalance_delta_from_t0"],
                "ob_delta_from_previous_horizon": safe_sub(curr["ob_imbalance"], prev["ob_imbalance"]) if prev else None,
                "bid_depth_usdt": curr["bid_depth_usdt"],
                "ask_depth_usdt": curr["ask_depth_usdt"],
                "spread_pct": curr["spread_pct"],
                "trade_imbalance": curr["trade_imbalance"],
                "trade_delta_from_t0": curr["trade_imbalance_delta"],
                "trade_delta_from_previous_horizon": safe_sub(curr["trade_imbalance"], prev["trade_imbalance"]) if prev else None,
                "trade_count": curr["trade_count"],
                "traded_vol_usdt": curr["traded_vol_usdt"],
                "buy_vol_usdt": curr["buy_vol_usdt"],
                "sell_vol_usdt": curr["sell_vol_usdt"],
                "buy_minus_sell_usdt": safe_sub(curr["buy_vol_usdt"], curr["sell_vol_usdt"]),
                "buy_sell_ratio": buy_sell_ratio,
                "funding_rate": curr["funding_rate"],
                "funding_change_from_t0": safe_sub(curr["funding_rate"], t0_row["t0_funding_rate"])
            })

            # For combined states and continuous buckets (using T+30 and above, usually)
            if h > 0:
                # Combined States
                oi_delta = curr["oi_delta_from_t0_pct"]
                ob_delta = curr["ob_imbalance_delta_from_t0"]
                tr_imb = curr["trade_imbalance"]
                
                if oi_delta is not None and ob_delta is not None:
                    if oi_delta > 0 and ob_delta > 0:
                        combined_counts[("OI_UP_OB_UP", h)].append(signal_id)
                    if oi_delta > 0 and ob_delta < 0:
                        combined_counts[("OI_UP_OB_DOWN", h)].append(signal_id)
                if oi_delta is not None and tr_imb is not None:
                    if oi_delta > 0 and tr_imb > 0:
                        combined_counts[("OI_UP_TRADE_UP", h)].append(signal_id)
                    if oi_delta > 0 and tr_imb < 0:
                        combined_counts[("OI_UP_TRADE_DOWN", h)].append(signal_id)
                        
        # State Transitions
        for i in range(len(HORIZONS) - 1):
            h_from = HORIZONS[i]
            h_to = HORIZONS[i+1]
            if h_from in states_by_h and h_to in states_by_h:
                s_from = states_by_h[h_from]
                s_to = states_by_h[h_to]
                transitions_list.append({
                    "signal_id": signal_id,
                    "symbol": symbol,
                    "from_horizon": h_from,
                    "from_state": s_from,
                    "to_horizon": h_to,
                    "to_state": s_to
                })
                sym_stats["transition_counts"][f"{s_from} -> {s_to}"] += 1
                
        # Forward Returns (State)
        for h_info in HORIZONS:
            if h_info not in all_h:
                continue
            state = states_by_h[h_info]
            info_price = all_h[h_info]["price"]
            if not info_price:
                continue
                
            for h_fut in HORIZONS:
                if h_fut <= h_info:
                    continue
                if h_fut not in all_h:
                    continue
                    
                fut_price = all_h[h_fut]["price"]
                if not fut_price:
                    continue
                    
                # STRICT NO-LOOKAHEAD: compute from info_price to fut_price
                fwd_ret = (fut_price / info_price - 1.0) * 100.0
                split = splits_by_signal[signal_id]
                event_id = events_by_signal[signal_id]
                state_fwd_returns[(h_info, state, h_fut)].append((fwd_ret, symbol, event_id, split))
                sym_stats["fwd_returns"].append((fwd_ret, symbol, event_id, split))
                
        # Forward Returns (Transitions)
        for i in range(len(HORIZONS) - 1):
            h_from = HORIZONS[i]
            h_to = HORIZONS[i+1]
            if h_from not in all_h or h_to not in all_h:
                continue
                
            s_from = states_by_h[h_from]
            s_to = states_by_h[h_to]
            
            # The transition completes at h_to. We measure returns from h_to to future horizons.
            info_price = all_h[h_to]["price"]
            if not info_price:
                continue
                
            for h_fut in HORIZONS:
                if h_fut <= h_to:
                    continue
                if h_fut not in all_h:
                    continue
                    
                fut_price = all_h[h_fut]["price"]
                if not fut_price:
                    continue
                    
                fwd_ret = (fut_price / info_price - 1.0) * 100.0
                split = splits_by_signal[signal_id]
                event_id = events_by_signal[signal_id]
                transition_fwd_returns[(h_from, s_from, h_to, s_to, h_fut)].append((fwd_ret, symbol, event_id, split))

        # Forward Returns (Features)
        for h_info in HORIZONS:
            if h_info <= 0 or h_info not in all_h:
                continue
            info_price = all_h[h_info]["price"]
            if not info_price:
                continue
                
            curr = all_h[h_info]
            bucket_vals = []
            if curr["oi_delta_from_t0_pct"] is not None:
                bucket_vals.append(("oi_delta", _bucket_feature(curr["oi_delta_from_t0_pct"], "oi_delta")))
            if curr["ob_imbalance_delta_from_t0"] is not None:
                bucket_vals.append(("ob_delta", _bucket_feature(curr["ob_imbalance_delta_from_t0"], "ob_delta")))
            if curr["trade_imbalance_delta"] is not None:
                bucket_vals.append(("trade_imbalance_delta", _bucket_feature(curr["trade_imbalance_delta"], "trade_imbalance_delta")))
                
            for h_fut in HORIZONS:
                if h_fut <= h_info or h_fut not in all_h:
                    continue
                fut_price = all_h[h_fut]["price"]
                if not fut_price:
                    continue
                    
                fwd_ret = (fut_price / info_price - 1.0) * 100.0
                split = splits_by_signal[signal_id]
                event_id = events_by_signal[signal_id]
                
                for f_type, b_name in bucket_vals:
                    feature_fwd_returns[(f_type, b_name, h_info, h_fut)].append((fwd_ret, symbol, event_id, split))

    # -------------------------------------------------------------
    # Aggregate Forward Returns (Robust)
    # -------------------------------------------------------------
    def _agg_robust(returns_data: List[Tuple[float, str, int, str]]):
        def _compute(data):
            n = len(data)
            if n == 0:
                return 0, 0, 0, None, None, None, None, None, None, None, None, "LOW_SAMPLE"
            rets = [d[0] for d in data]
            unique_syms = len(set(d[1] for d in data))
            unique_evts = len(set(d[2] for d in data))
            mean_r = statistics.mean(rets)
            med_r = statistics.median(rets)
            pos_pct = sum(1 for r in rets if r > 0) / n * 100.0
            std_r = statistics.stdev(rets) if n > 1 else None
            min_r = min(rets)
            max_r = max(rets)
            if n >= 2:
                try:
                    q = statistics.quantiles(rets, n=4)
                    p25, p75 = q[0], q[2]
                except Exception:
                    p25, p75 = None, None
            else:
                p25, p75 = None, None
            quality = "OK" if n >= min_sample else "LOW_SAMPLE"
            return n, unique_syms, unique_evts, mean_r, med_r, pos_pct, std_r, min_r, max_r, p25, p75, quality

        results = {}
        for sp in ["ALL", "TRAIN", "VAL", "TEST"]:
            sp_data = returns_data if sp == "ALL" else [d for d in returns_data if d[3] == sp]
            results[sp] = _compute(sp_data)
        return results

    state_fwd_results = []
    for (info_h, state, fut_h), rets_data in state_fwd_returns.items():
        agg_res = _agg_robust(rets_data)
        for sp, (n, uniq_sym, uniq_evt, mean_r, med_r, pos_pct, std_r, min_r, max_r, p25, p75, quality) in agg_res.items():
            state_fwd_results.append({
                "information_horizon": info_h,
                "state": state,
                "future_horizon": fut_h,
                "split": sp,
                "N": n,
                "unique_symbols": uniq_sym,
                "unique_events": uniq_evt,
                "mean_return": mean_r,
                "median_return": med_r,
                "positive_return_pct": pos_pct,
                "stddev": std_r,
                "min_return": min_r,
                "max_return": max_r,
                "25th_pct": p25,
                "75th_pct": p75,
                "sample_quality": quality
            })
            
    transition_fwd_results = []
    for (h_from, s_from, h_to, s_to, h_fut), rets_data in transition_fwd_returns.items():
        agg_res = _agg_robust(rets_data)
        for sp, (n, uniq_sym, uniq_evt, mean_r, med_r, pos_pct, std_r, min_r, max_r, p25, p75, quality) in agg_res.items():
            transition_fwd_results.append({
                "from_horizon": h_from,
                "from_state": s_from,
                "to_horizon": h_to,
                "to_state": s_to,
                "future_horizon": h_fut,
                "split": sp,
                "N": n,
                "unique_symbols": uniq_sym,
                "unique_events": uniq_evt,
                "mean_return": mean_r,
                "median_return": med_r,
                "positive_return_pct": pos_pct,
                "stddev": std_r,
                "min_return": min_r,
                "max_return": max_r,
                "25th_pct": p25,
                "75th_pct": p75,
                "sample_quality": quality
            })
            
    feature_bucket_results = []
    for (f_type, b_name, h_info, h_fut), rets_data in feature_fwd_returns.items():
        agg_res = _agg_robust(rets_data)
        for sp, (n, uniq_sym, uniq_evt, mean_r, med_r, pos_pct, std_r, min_r, max_r, p25, p75, quality) in agg_res.items():
            feature_bucket_results.append({
                "feature": f_type,
                "bucket": b_name,
                "information_horizon": h_info,
                "future_horizon": h_fut,
                "split": sp,
                "N": n,
                "unique_symbols": uniq_sym,
                "unique_events": uniq_evt,
                "mean_return": mean_r,
                "median_return": med_r,
                "positive_return_pct": pos_pct,
                "stddev": std_r,
                "min_return": min_r,
                "max_return": max_r,
                "25th_pct": p25,
                "75th_pct": p75,
                "sample_quality": quality
            })

    # Combined states
    combined_state_results = []
    for (comb_name, h), sigs in combined_counts.items():
        combined_state_results.append({
            "combined_state": comb_name,
            "horizon": h,
            "N": len(sigs)
        })

    # Symbol Summary
    symbol_summary_results = []
    for sym, stats in symbol_stats.items():
        agg_res = _agg_robust(stats["fwd_returns"])
        all_res = agg_res["ALL"]
        n, _, _, mean_r, med_r, pos_pct, std_r, min_r, max_r, _, _, quality = all_res
        symbol_summary_results.append({
            "symbol": sym,
            "signal_count": stats["signal_count"],
            "usable_microstructure_signal_count": stats["micro_count"],
            "state_counts": dict(stats["state_counts"]),
            "transition_counts": dict(stats["transition_counts"]),
            "mean_forward_return": mean_r,
            "N_forward_returns": n
        })
        
    dataset_summary = {
        "TOTAL_SIGNALS": total_signals,
        "TOTAL_SYMBOLS": total_symbols,
        "SIGNALS_WITH_PRICE_DATA": signals_with_price,
        "SIGNALS_WITH_MICROSTRUCTURE_DATA": signals_with_micro,
        "STATE_COUNTS": sum(1 for s in states_list),
        "TRANSITION_COUNTS": len(transitions_list),
        "FORWARD_RETURN_SAMPLE_COUNTS": sum(r["N"] for r in state_fwd_results),
        "LOW_SAMPLE_COUNTS": sum(1 for r in state_fwd_results if r["sample_quality"] == "LOW_SAMPLE"),
        "NULL_COUNTS": sum(1 for s in states_list if s["state"] == "INSUFFICIENT_DATA"),
        "T0_BASED_COUNTS": sum(1 for s in states_list if s["regime_information_basis"] == "T0_BASED"),
        "PREVIOUS_HORIZON_FALLBACK_COUNTS": sum(1 for s in states_list if s["regime_information_basis"] == "PREVIOUS_HORIZON_FALLBACK"),
        "HYBRID_COUNTS": sum(1 for s in states_list if s["regime_information_basis"] == "HYBRID"),
        "INSUFFICIENT_DATA_COUNTS": sum(1 for s in states_list if s["regime_information_basis"] == "INSUFFICIENT_DATA")
    }

    return {
        "states": states_list,
        "transitions": transitions_list,
        "state_fwd_returns": state_fwd_results,
        "transition_fwd_returns": transition_fwd_results,
        "feature_buckets": feature_bucket_results,
        "combined_states": combined_state_results,
        "symbol_summary": symbol_summary_results,
        "dataset_summary": dataset_summary
    }

