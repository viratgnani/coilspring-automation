"""Exporter for Research Analysis results."""

import csv
from typing import Any, Sequence

def export_forward_returns(results: Sequence[dict[str, Any]], filename: str = "research_analysis_forward_returns.csv") -> None:
    headers = [
        "information_time_s", "future_horizon_s", "forward_duration_s", 
        "bucket_name", "N", "unique_symbols", "mean_forward_return", 
        "median_forward_return", "positive_return_pct", "stddev", 
        "sample_quality", "remaining_mfe", "reason"
    ]
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            b = r["bucket"]
            writer.writerow([
                r["information_time_s"], r["future_horizon_s"], r["forward_duration_s"],
                b.bucket_name, b.n, b.unique_symbols,
                b.mean_forward_return, b.median_forward_return, b.positive_return_pct, b.stddev,
                b.sample_quality, b.remaining_mfe, b.reason
            ])


def export_microstructure_buckets(feature_data: dict[str, Any], filename: str) -> None:
    headers = [
        "feature", "information_time_s", "future_horizon_s", "forward_duration_s", 
        "bucket_name", "N", "unique_symbols", "mean_forward_return", 
        "median_forward_return", "positive_return_pct", "stddev", 
        "sample_quality", "remaining_mfe", "reason"
    ]
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        feature = feature_data["feature"]
        for r in feature_data["results"]:
            b = r["bucket"]
            writer.writerow([
                feature, r["information_time_s"], r["future_horizon_s"], r["forward_duration_s"],
                b.bucket_name, b.n, b.unique_symbols,
                b.mean_forward_return, b.median_forward_return, b.positive_return_pct, b.stddev,
                b.sample_quality, b.remaining_mfe, b.reason
            ])


def export_combined_buckets(results: Sequence[dict[str, Any]], filename: str = "research_analysis_combined_buckets.csv") -> None:
    headers = [
        "information_time_s", "future_horizon_s", "forward_duration_s", 
        "bucket_name", "N", "unique_symbols", "mean_forward_return", 
        "median_forward_return", "positive_return_pct", "stddev", 
        "sample_quality", "remaining_mfe", "reason"
    ]
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            b = r["bucket"]
            writer.writerow([
                r["information_time_s"], r["future_horizon_s"], r["forward_duration_s"],
                b.bucket_name, b.n, b.unique_symbols,
                b.mean_forward_return, b.median_forward_return, b.positive_return_pct, b.stddev,
                b.sample_quality, b.remaining_mfe, b.reason
            ])

def export_regime_results(results: dict[str, Any]) -> None:
    # 1. States
    states = results.get("states")
    if states is not None:
        with open("research_regime_states.csv", 'w', newline='', encoding='utf-8') as f:
            if states:
                writer = csv.DictWriter(f, fieldnames=states[0].keys())
                writer.writeheader()
                writer.writerows(states)
            
    # 2. Transitions
    transitions = results.get("transitions")
    if transitions is not None:
        with open("research_regime_transitions.csv", 'w', newline='', encoding='utf-8') as f:
            if transitions:
                writer = csv.DictWriter(f, fieldnames=transitions[0].keys())
                writer.writeheader()
                writer.writerows(transitions)
            
    # 3. Forward Returns
    state_fwd_returns = results.get("state_fwd_returns")
    if state_fwd_returns is not None:
        with open("research_regime_forward_returns.csv", 'w', newline='', encoding='utf-8') as f:
            if state_fwd_returns:
                writer = csv.DictWriter(f, fieldnames=state_fwd_returns[0].keys())
                writer.writeheader()
                writer.writerows(state_fwd_returns)
            
    # 4. Transition Forward Returns
    transition_fwd_returns = results.get("transition_fwd_returns")
    if transition_fwd_returns is not None:
        with open("research_regime_transition_forward_returns.csv", 'w', newline='', encoding='utf-8') as f:
            if transition_fwd_returns:
                writer = csv.DictWriter(f, fieldnames=transition_fwd_returns[0].keys())
                writer.writeheader()
                writer.writerows(transition_fwd_returns)
            
    # 5. Feature Buckets
    feature_buckets = results.get("feature_buckets")
    if feature_buckets is not None:
        with open("research_regime_feature_buckets.csv", 'w', newline='', encoding='utf-8') as f:
            if feature_buckets:
                writer = csv.DictWriter(f, fieldnames=feature_buckets[0].keys())
                writer.writeheader()
                writer.writerows(feature_buckets)
            
    # 6. Combined States
    combined_states = results.get("combined_states")
    if combined_states is not None:
        with open("research_regime_combined_states.csv", 'w', newline='', encoding='utf-8') as f:
            if combined_states:
                writer = csv.DictWriter(f, fieldnames=combined_states[0].keys())
                writer.writeheader()
                writer.writerows(combined_states)
            
    # 7. Symbol Summary
    symbol_summary = results.get("symbol_summary")
    if symbol_summary is not None:
        with open("research_regime_symbol_summary.csv", 'w', newline='', encoding='utf-8') as f:
            if symbol_summary:
                writer = csv.DictWriter(f, fieldnames=symbol_summary[0].keys())
                writer.writeheader()
                writer.writerows(symbol_summary)
            
    # 8. Dataset Summary
    if results.get("dataset_summary"):
        with open("research_regime_dataset_summary.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for k, v in results["dataset_summary"].items():
                writer.writerow([k, v])
