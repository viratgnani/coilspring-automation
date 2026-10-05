"""Tests for Research Analysis V2 (Regime Engine)."""

import pytest
import aiosqlite
from coilbrain.research.analysis import regime_v1, regime_core

import asyncio

def test_regime_classification():
    # 3. EXPANSION classification
    assert regime_v1.classify_state(1.0, 1.0, 1.0)["state"] == "EXPANSION"
    # 4. OI_PRICE_DIVERGENCE classification
    assert regime_v1.classify_state(-1.0, 1.0, 1.0)["state"] == "OI_PRICE_DIVERGENCE"
    assert regime_v1.classify_state(0.0, 1.0, 1.0)["state"] == "OI_PRICE_DIVERGENCE"
    # 5. FLOW_PRICE_DIVERGENCE classification
    assert regime_v1.classify_state(1.0, 1.0, -1.0)["state"] == "FLOW_PRICE_DIVERGENCE"
    assert regime_v1.classify_state(1.0, 1.0, 0.0)["state"] == "FLOW_PRICE_DIVERGENCE"
    # 6. OI_UNWIND classification
    assert regime_v1.classify_state(-1.0, -1.0, -1.0)["state"] == "OI_UNWIND"
    # 7. PRICE_ONLY_EXPANSION classification
    assert regime_v1.classify_state(1.0, -1.0, 1.0)["state"] == "PRICE_ONLY_EXPANSION"
    assert regime_v1.classify_state(1.0, 0.0, 1.0)["state"] == "PRICE_ONLY_EXPANSION"
    # 8. MIXED classification
    assert regime_v1.classify_state(0.0, 0.0, 1.0)["state"] == "MIXED"
    assert regime_v1.classify_state(-1.0, 0.0, 1.0)["state"] == "MIXED"
    # 9. INSUFFICIENT_DATA classification
    assert regime_v1.classify_state(None, 1.0, 1.0)["state"] == "INSUFFICIENT_DATA"
    assert regime_v1.classify_state(1.0, None, 1.0)["state"] == "INSUFFICIENT_DATA"
    assert regime_v1.classify_state(1.0, 1.0, None)["state"] == "INSUFFICIENT_DATA"

    # 21. Regime definition versioning
    assert hasattr(regime_v1, "REGIME_DEFINITION_VERSION")


def test_regime_engine_db_processing():
    async def _run():
        async with aiosqlite.connect(":memory:") as db:
            await db.execute("""
                CREATE TABLE signals (
                    signal_id TEXT PRIMARY KEY
                )
            """)
            await db.execute("""
                CREATE TABLE research_signal_features (
                    signal_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    t0_timestamp REAL,
                    t0_price REAL,
                    score REAL,
                    grade TEXT,
                    mfe_pct REAL,
                    mae_pct REAL,
                    time_to_mfe_s REAL,
                    secs_to_1pct REAL,
                    secs_to_5pct REAL,
                    secs_to_10pct REAL,
                    t0_snapshot_ts REAL,
                    t0_oi_usdt REAL,
                    t0_oi_delta_pct REAL,
                    t0_funding_rate REAL,
                    t0_spread_abs REAL,
                    t0_spread_pct REAL,
                    t0_ob_imbalance REAL,
                    t0_trade_count INTEGER,
                    t0_traded_vol_usdt REAL,
                    t0_buy_vol_usdt REAL,
                    t0_sell_vol_usdt REAL,
                    t0_trade_imbalance REAL
                )
            """)
            await db.execute("""
                CREATE TABLE research_horizon_outcomes (
                    signal_id TEXT,
                    horizon_s INTEGER,
                    target_elapsed_s REAL,
                    actual_elapsed_s REAL,
                    alignment_error_s REAL,
                    observation_ts REAL,
                    price REAL,
                    return_pct REAL,
                    snapshot_ts REAL,
                    oi_usdt REAL,
                    oi_delta_pct REAL,
                    funding_rate REAL,
                    spread_abs REAL,
                    spread_pct REAL,
                    ob_imbalance REAL,
                    trade_count INTEGER,
                    traded_vol_usdt REAL,
                    buy_vol_usdt REAL,
                    sell_vol_usdt REAL,
                    trade_imbalance REAL,
                    oi_delta_from_t0_pct REAL,
                    spread_delta_from_t0_pct REAL,
                    ob_imbalance_delta_from_t0 REAL,
                    buy_volume_delta REAL,
                    sell_volume_delta REAL,
                    trade_imbalance_delta REAL,
                    PRIMARY KEY (signal_id, horizon_s)
                )
            """)
            await db.execute("""
                CREATE TABLE microstructure_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    binance_symbol TEXT,
                    snapshot_ts REAL,
                    oi_usdt REAL,
                    oi_delta_pct REAL,
                    funding_rate REAL,
                    mark_price REAL,
                    next_funding_ts REAL,
                    best_bid REAL,
                    best_ask REAL,
                    spread_abs REAL,
                    spread_pct REAL,
                    bid_depth_usdt REAL,
                    ask_depth_usdt REAL,
                    ob_imbalance REAL,
                    trade_count INTEGER,
                    traded_vol_usdt REAL,
                    buy_vol_usdt REAL,
                    sell_vol_usdt REAL,
                    trade_imbalance REAL
                )
            """)
            
            # Setup test data
            await db.execute(
                "INSERT INTO research_signal_features (signal_id, symbol, t0_timestamp, t0_price, score, grade, t0_snapshot_ts, t0_oi_usdt, t0_ob_imbalance, t0_trade_imbalance) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("SIG1", "GUNUSDT", 1000, 100.0, 0.9, "A", 1000, 500.0, 0.1, 0.1)
            )
            # 10. NULL handling / 11. no NULL -> zero conversion
            await db.execute(
                "INSERT INTO research_signal_features (signal_id, symbol, t0_timestamp, t0_price, score, grade, t0_snapshot_ts, t0_oi_usdt, t0_ob_imbalance, t0_trade_imbalance) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("SIG2_NULL", "BADUSDT", 1000, None, 0.9, "A", 1000, None, None, None)
            )
            
            # 30s horizon for SIG1
            await db.execute(
                "INSERT INTO research_horizon_outcomes (signal_id, horizon_s, target_elapsed_s, price, snapshot_ts, oi_usdt, oi_delta_from_t0_pct, ob_imbalance, trade_imbalance, buy_vol_usdt, sell_vol_usdt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("SIG1", 30, 30, 105.0, 1030, 550.0, 10.0, 0.2, 0.5, 100.0, 50.0) # Price +5%, OI +10%, Trade Imb > 0 => EXPANSION
            )
            # 60s horizon for SIG1
            await db.execute(
                "INSERT INTO research_horizon_outcomes (signal_id, horizon_s, target_elapsed_s, price, snapshot_ts, oi_usdt, oi_delta_from_t0_pct, ob_imbalance, trade_imbalance) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("SIG1", 60, 60, 110.0, 1060, 600.0, 20.0, 0.3, 0.6) 
            )
            
            # SIG3_GUN: Missing T0 OI and trade imbalance, but valid at 30s and 60s
            await db.execute(
                "INSERT INTO research_signal_features (signal_id, symbol, t0_timestamp, t0_price, score, grade, t0_snapshot_ts, t0_oi_usdt, t0_ob_imbalance, t0_trade_imbalance) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("SIG3_GUN", "GUNUSDT", 2000, 100.0, 0.9, "A", 2000, None, None, None)
            )
            await db.execute(
                "INSERT INTO research_horizon_outcomes (signal_id, horizon_s, target_elapsed_s, price, snapshot_ts, oi_usdt, oi_delta_from_t0_pct, ob_imbalance, trade_imbalance, trade_imbalance_delta, buy_vol_usdt, sell_vol_usdt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("SIG3_GUN", 30, 30, 105.0, 2030, 500.0, None, 0.1, None, 0.2, 100.0, 50.0) 
            )
            await db.execute(
                "INSERT INTO research_horizon_outcomes (signal_id, horizon_s, target_elapsed_s, price, snapshot_ts, oi_usdt, oi_delta_from_t0_pct, ob_imbalance, trade_imbalance, trade_imbalance_delta) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("SIG3_GUN", 60, 60, 110.0, 2060, 550.0, None, 0.2, None, 0.3) 
            )

            await db.commit()

            res = await regime_core.build_regime_dataset(db, min_sample=2)
            
            assert res["dataset_summary"]["TOTAL_SIGNALS"] == 3
            assert res["dataset_summary"]["NULL_COUNTS"] >= 1
            
            states = res["states"]
            
            # SIG1 30s: complete T0 data -> T0_BASED
            sig1_30 = next(s for s in states if s["signal_id"] == "SIG1" and s["information_horizon"] == 30)
            assert sig1_30["state"] == "EXPANSION"
            assert sig1_30["regime_information_basis"] == "T0_BASED"
            assert sig1_30["price_basis"] == "T0"
            assert sig1_30["oi_basis"] == "T0"
            assert sig1_30["trade_basis"] == "CURRENT"
            
            # SIG3_GUN 30s: no previous observation -> INSUFFICIENT_DATA
            sig3_30 = next(s for s in states if s["signal_id"] == "SIG3_GUN" and s["information_horizon"] == 30)
            assert sig3_30["state"] == "INSUFFICIENT_DATA"
            assert sig3_30["regime_information_basis"] == "INSUFFICIENT_DATA"
            
            # SIG3_GUN 60s: missing T0 OI/trade -> previous-horizon fallback (HYBRID because price has T0, others don't)
            sig3_60 = next(s for s in states if s["signal_id"] == "SIG3_GUN" and s["information_horizon"] == 60)
            assert sig3_60["state"] == "EXPANSION" # Price up (+4.7%), OI up (+10%), Trade delta > 0 (+0.3)
            assert sig3_60["regime_information_basis"] == "HYBRID"
            assert sig3_60["price_basis"] == "T0"
            assert sig3_60["oi_basis"] == "PREVIOUS_HORIZON"
            assert sig3_60["trade_basis"] == "DELTA_PREVIOUS_HORIZON"
            
            # Check effective values for SIG3_GUN 60s
            assert abs(sig3_60["effective_price_change"] - 10.0) < 0.001 # 110 / 100
            assert abs(sig3_60["effective_oi_change"] - 10.0) < 0.001 # (550 - 500) / 500
            assert abs(sig3_60["effective_trade_measure"] - 0.3) < 0.001
            
            # 13. forward return uses information-horizon price
            state_fwd = res["state_fwd_returns"]
            fwd_30 = next(r for r in state_fwd if r["information_horizon"] == 30 and r["future_horizon"] == 60 and r["state"] == "EXPANSION")
            expected_ret = (110.0 / 105.0 - 1.0) * 100.0
            assert abs(fwd_30["mean_return"] - expected_ret) < 0.001
            
            # 12. strict future_horizon > information_horizon
            for r in state_fwd:
                assert r["future_horizon"] > r["information_horizon"]

                
            # 16. transition generation
            transitions = res["transitions"]
            trans = next(t for t in transitions if t["signal_id"] == "SIG1" and t["from_horizon"] == 30 and t["to_horizon"] == 60)
            assert trans["from_state"] == "EXPANSION"
            # 19. LOW_SAMPLE classification / 20. configurable min_sample
            assert fwd_30["sample_quality"] == "LOW_SAMPLE"
            assert fwd_30["N"] == 1
            
            # 18. symbol aggregation
            sym = res["symbol_summary"]
            gun = next(s for s in sym if s["symbol"] == "GUNUSDT")
            assert gun["signal_count"] == 2
            assert gun["usable_microstructure_signal_count"] == 1
            assert gun["N_forward_returns"] > 0

    asyncio.run(_run())

def test_exporter_unicode(tmp_path):
    from coilbrain.research.analysis import exporter
    import os
    
    # Switch directory to tmp_path to not clutter workspace
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        results = {
            "dataset_summary": {"TEST_METRIC": "Unicode \u2014 \u2713 \u2605 test"}
        }
        exporter.export_regime_results(results)
        
        # Read it back
        with open("research_regime_dataset_summary.csv", "r", encoding="utf-8") as f:
            content = f.read()
            assert "Unicode \u2014 \u2713 \u2605 test" in content
    finally:
        os.chdir(old_cwd)

import pytest
import aiosqlite
import asyncio
from coilbrain.research.analysis import regime_core

def test_regime_robustness_clustering_and_splitting():
    async def _run():
        async with aiosqlite.connect(":memory:") as db:
            await db.execute("""
                CREATE TABLE research_signal_features (
                    signal_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    t0_timestamp REAL,
                    t0_price REAL,
                    score REAL,
                    grade TEXT,
                    t0_snapshot_ts REAL,
                    t0_oi_usdt REAL,
                    t0_oi_delta_pct REAL,
                    t0_funding_rate REAL,
                    t0_spread_abs REAL,
                    t0_spread_pct REAL,
                    t0_ob_imbalance REAL,
                    t0_trade_count INTEGER,
                    t0_traded_vol_usdt REAL,
                    t0_buy_vol_usdt REAL,
                    t0_sell_vol_usdt REAL,
                    t0_trade_imbalance REAL
                )
            """)
            await db.execute("""
                CREATE TABLE research_horizon_outcomes (
                    signal_id TEXT,
                    horizon_s INTEGER,
                    target_elapsed_s REAL,
                    actual_elapsed_s REAL,
                    alignment_error_s REAL,
                    observation_ts REAL,
                    price REAL,
                    return_pct REAL,
                    snapshot_ts REAL,
                    oi_usdt REAL,
                    oi_delta_pct REAL,
                    funding_rate REAL,
                    spread_abs REAL,
                    spread_pct REAL,
                    ob_imbalance REAL,
                    trade_count INTEGER,
                    traded_vol_usdt REAL,
                    buy_vol_usdt REAL,
                    sell_vol_usdt REAL,
                    trade_imbalance REAL,
                    oi_delta_from_t0_pct REAL,
                    spread_delta_from_t0_pct REAL,
                    ob_imbalance_delta_from_t0 REAL,
                    buy_volume_delta REAL,
                    sell_volume_delta REAL,
                    trade_imbalance_delta REAL,
                    PRIMARY KEY (signal_id, horizon_s)
                )
            """)
            await db.execute("""
                CREATE TABLE microstructure_snapshots (
                    binance_symbol TEXT,
                    snapshot_ts REAL,
                    bid_depth_usdt REAL,
                    ask_depth_usdt REAL,
                    PRIMARY KEY (binance_symbol, snapshot_ts)
                )
            """)
            
            # Create 5 distinct events. 
            # Event 1: SYM1, ts=1000, 1100 (same event)
            # Event 2: SYM2, ts=2000
            # Event 3: SYM1, ts=5000 (new event, > 3600s gap)
            # Event 4: SYM3, ts=6000
            # Event 5: SYM4, ts=7000
            
            signals = [
                ("SIG_1_A", "SYM1", 1000),
                ("SIG_1_B", "SYM1", 1100),
                ("SIG_2", "SYM2", 2000),
                ("SIG_3", "SYM1", 5000),
                ("SIG_4", "SYM3", 6000),
                ("SIG_5", "SYM4", 7000),
            ]
            
            for sig_id, sym, ts in signals:
                await db.execute(
                    "INSERT INTO research_signal_features (signal_id, symbol, t0_timestamp, t0_price, t0_snapshot_ts, t0_oi_usdt, t0_ob_imbalance, t0_trade_imbalance) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (sig_id, sym, ts, 100.0, ts, 500.0, 0.1, 0.1)
                )
                await db.execute(
                    "INSERT INTO research_horizon_outcomes (signal_id, horizon_s, target_elapsed_s, price, snapshot_ts, oi_usdt, oi_delta_from_t0_pct, ob_imbalance, trade_imbalance, trade_imbalance_delta) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (sig_id, 30, 30, 105.0, ts+30, 550.0, 10.0, 0.2, 0.5, 0.2) 
                )
                await db.execute(
                    "INSERT INTO research_horizon_outcomes (signal_id, horizon_s, target_elapsed_s, price, snapshot_ts, oi_usdt, oi_delta_from_t0_pct, ob_imbalance, trade_imbalance, trade_imbalance_delta) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (sig_id, 60, 60, 110.0, ts+60, 600.0, 20.0, 0.3, 0.6, 0.3) 
                )
            
            await db.commit()
            
            res = await regime_core.build_regime_dataset(db, min_sample=1)
            
            # Events chronologically: 
            # 1: SYM1(1000) -> SIG_1_A, SIG_1_B
            # 2: SYM2(2000) -> SIG_2
            # 3: SYM1(5000) -> SIG_3
            # 4: SYM3(6000) -> SIG_4
            # 5: SYM4(7000) -> SIG_5
            
            # n_events = 5
            # train_idx = int(5 * 0.6) = 3 -> indices 0, 1, 2 (Events 1, 2, 3)
            # val_idx = int(5 * 0.8) = 4 -> index 3 (Event 4)
            # test_idx -> index 4 (Event 5)
            
            # Let's check state_fwd_returns
            state_fwd = res["state_fwd_returns"]
            
            # Look at ALL split for state EXPANSION (since price+, oi+, trade+)
            all_exp = next(r for r in state_fwd if r["state"] == "EXPANSION" and r["split"] == "ALL" and r["information_horizon"] == 30 and r["future_horizon"] == 60)
            
            assert all_exp["N"] == 6 # 6 signals total
            assert all_exp["unique_symbols"] == 4
            assert all_exp["unique_events"] == 5
            
            # Look at TRAIN split
            train_exp = next(r for r in state_fwd if r["state"] == "EXPANSION" and r["split"] == "TRAIN" and r["information_horizon"] == 30 and r["future_horizon"] == 60)
            assert train_exp["N"] == 4 # SIG_1_A, SIG_1_B, SIG_2, SIG_3
            assert train_exp["unique_symbols"] == 2 # SYM1, SYM2
            assert train_exp["unique_events"] == 3
            
            # Look at VAL split
            val_exp = next(r for r in state_fwd if r["state"] == "EXPANSION" and r["split"] == "VAL" and r["information_horizon"] == 30 and r["future_horizon"] == 60)
            assert val_exp["N"] == 1 # SIG_4
            assert val_exp["unique_symbols"] == 1 # SYM3
            assert val_exp["unique_events"] == 1
            
            # Look at TEST split
            test_exp = next(r for r in state_fwd if r["state"] == "EXPANSION" and r["split"] == "TEST" and r["information_horizon"] == 30 and r["future_horizon"] == 60)
            assert test_exp["N"] == 1 # SIG_5
            assert test_exp["unique_symbols"] == 1 # SYM4
            assert test_exp["unique_events"] == 1

            # Feature buckets robustness
            feat_fwd = res["feature_buckets"]
            oi_bucket = next(r for r in feat_fwd if r["feature"] == "oi_delta" and r["bucket"] == "> +3%" and r["split"] == "ALL")
            assert oi_bucket["N"] == 6
            assert oi_bucket["unique_events"] == 5
            
    asyncio.run(_run())
