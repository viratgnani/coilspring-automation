"""Live session monitor for CoilBrain controlled demo session."""

import asyncio
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Set paths
REPO_ROOT = Path(__file__).resolve().parents[2]
COILBRAIN_ROOT = REPO_ROOT / "coilbrain"
DB_PATH = str(COILBRAIN_ROOT / "coilbrain.db")
sys.path.insert(0, str(COILBRAIN_ROOT / "src"))

load_dotenv(COILBRAIN_ROOT / ".env")
from coilbrain.bybit_demo import BybitDemoClient

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

async def verify_trade_execution(t_row):
    """Deep verification of the executed trade on Bybit Demo and local DB."""
    client = BybitDemoClient.from_env()
    symbol = t_row["bybit_symbol"] or t_row["symbol"]
    order_id = t_row["bybit_order_id"]
    
    print("\n" + "=" * 70)
    print(">>> VERIFYING TRADE EXECUTION ON BYBIT DEMO & DB <<<")
    print("=" * 70)
    
    # 1. Order status
    order_info = {}
    try:
        order_info = await client.get_order_status(symbol, order_id)
        print(f"1. Order ID:        {order_id}")
        print(f"   Order Status:    {order_info.get('orderStatus', 'UNKNOWN')}")
        print(f"   Avg Price:       {order_info.get('avgPrice', 'N/A')}")
        print(f"   Cum Exec Qty:    {order_info.get('cumExecQty', 'N/A')}")
    except Exception as e:
        print(f"1. Order status fetch error: {e}")
        
    # 2. Position
    pos_info = None
    try:
        pos_info = await client.get_position(symbol)
        if pos_info:
            print(f"2. Bybit Position:  Side={pos_info.side}, Size={pos_info.size}, AvgPrice={pos_info.avg_price}, LiqPrice={pos_info.raw.get('liqPrice')}")
        else:
            print("2. Bybit Position:  Flat or not found yet")
    except Exception as e:
        print(f"2. Position fetch error: {e}")
        
    # 3. Fill details from journal
    print("3. Fill Verification:")
    print(f"   Actual Entry:    {t_row['actual_entry_price']}")
    print(f"   Quantity:        {t_row['quantity']}")
    print(f"   Slippage Pct:    {t_row['slippage_pct']:.4f}%" if t_row['slippage_pct'] is not None else "   Slippage: N/A")
    print(f"   Fill Timestamp:  {t_row['fill_timestamp']}")

    # 4. SL / TP confirmations
    print("4. Protection Orders:")
    print(f"   SL Price:        {t_row['stop_loss']} | Confirmed TS: {t_row['sl_confirmed_timestamp']} | Order ID: {t_row['bybit_sl_order_id']}")
    print(f"   TP Price:        {t_row['take_profit']} | Confirmed TS: {t_row['tp_confirmed_timestamp']} | Order ID: {t_row['bybit_tp_order_id']}")

    # 5. Latencies
    print("5. Latency Breakdown:")
    print(f"   Signal -> Decision:   {t_row['signal_to_decision_ms']:.2f} ms" if t_row['signal_to_decision_ms'] else "   Signal -> Decision: N/A")
    print(f"   Signal -> Order:      {t_row['signal_to_order_ms']:.2f} ms" if t_row['signal_to_order_ms'] else "   Signal -> Order: N/A")
    print(f"   Signal -> Fill:       {t_row['signal_to_fill_ms']:.2f} ms" if t_row['signal_to_fill_ms'] else "   Signal -> Fill: N/A")
    print(f"   Signal -> Protection: {t_row['signal_to_protection_ms']:.2f} ms" if t_row['signal_to_protection_ms'] else "   Signal -> Protection: N/A")

    # 6. Demo guard
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT trade_count FROM demo_session_guard WHERE id = 1")
    g_row = cur.fetchone()
    guard_count = g_row[0] if g_row else -1
    print(f"6. Demo Guard Count: {guard_count} / 1")
    conn.close()
    print("=" * 70 + "\n")

def main():
    print("=" * 70)
    print("COILBRAIN LIVE DEMO SESSION MONITOR ACTIVE")
    print("=" * 70)
    sys.stdout.flush()

    seen_decisions = set()
    seen_trades = set()
    start_ts = time.time() - 2.0  # capture anything from start

    session_stats = {
        "signals_count": 0,
        "ignores_count": 0,
        "trades_count": 0,
        "signals": []
    }

    try:
        while True:
            conn = get_db()
            cur = conn.cursor()

            # Check new decisions
            cur.execute("SELECT * FROM decision_journal WHERE created_at >= ? ORDER BY id ASC", (start_ts,))
            decisions = cur.fetchall()

            for d in decisions:
                did = d["id"]
                if did in seen_decisions:
                    continue
                seen_decisions.add(did)

                # Fetch corresponding signal if present
                cur.execute("SELECT * FROM signals WHERE signal_id = ?", (d["signal_id"],))
                sig = cur.fetchone()

                session_stats["signals_count"] += 1
                if d["decision"] == "IGNORE":
                    session_stats["ignores_count"] += 1
                elif d["decision"] == "TRADE":
                    session_stats["trades_count"] += 1

                norm_sym = d["bybit_symbol"] or (sig["binance_symbol"] if sig and "binance_symbol" in sig.keys() else "N/A")
                
                # Buy/sell flow
                flow_str = f"Buy Vol: ${d['buy_vol_usdt']:,.2f} | Sell Vol: ${d['sell_vol_usdt']:,.2f} (Imbalance: {d['trade_imbalance']:.4f})" if d['buy_vol_usdt'] is not None and d['sell_vol_usdt'] is not None and d['trade_imbalance'] is not None else "N/A"

                sig_record = {
                    "id": d["id"],
                    "signal_id": d["signal_id"],
                    "symbol": d["symbol"],
                    "cs_score": d["cs_score"],
                    "volume_24h_usd": d["volume_24h_usd"],
                    "body_spike": d["body_spike"],
                    "volume_spike": d["volume_spike"],
                    "normalized_binance_symbol": norm_sym,
                    "data_fetch_latency_ms": d["data_fetch_latency_ms"],
                    "snapshot_age_ms": d["snapshot_age_ms"],
                    "oi_usdt": d["oi_usdt"],
                    "oi_delta_pct": d["oi_delta_pct"],
                    "funding_rate": d["funding_rate"],
                    "spread_pct": d["spread_pct"],
                    "ob_imbalance": d["ob_imbalance"],
                    "flow": flow_str,
                    "decision": d["decision"],
                    "decision_score": d["decision_score"],
                    "reasons": d["reasons_json"],
                }
                session_stats["signals"].append(sig_record)

                print("\n" + "-" * 60)
                print(f"[COILSPRING SIGNAL EVALUATED] Record #{session_stats['signals_count']} | Decision ID: {d['id']} | Signal ID: {d['signal_id']}")
                print(f"  Symbol:                    {d['symbol']}")
                print(f"  Signal ID:                 {d['signal_id']}")
                print(f"  CoilSpring Score:          {d['cs_score']}")
                print(f"  24h Volume USD:            ${d['volume_24h_usd']:,.2f}" if d['volume_24h_usd'] else "  24h Volume USD: None")
                print(f"  Body Spike:                {d['body_spike']:.2f}x" if d['body_spike'] is not None else "  Body Spike: None")
                print(f"  Volume Spike:              {d['volume_spike']:.2f}x" if d['volume_spike'] is not None else "  Volume Spike: None")
                print(f"  Normalized Binance Symbol: {norm_sym}")
                print(f"  Microstructure Latency:    {d['data_fetch_latency_ms']:.1f} ms" if d['data_fetch_latency_ms'] is not None else "  Latency: N/A")
                print(f"  Snapshot Age:              {d['snapshot_age_ms']:.1f} ms" if d['snapshot_age_ms'] is not None else "  Age: N/A")
                print(f"  OI USDT:                   ${d['oi_usdt']:,.2f}" if d['oi_usdt'] is not None else "  OI: None")
                print(f"  OI Delta:                  {d['oi_delta_pct']:.2f}%" if d['oi_delta_pct'] is not None else "  OI Delta: None")
                print(f"  Funding Rate:              {d['funding_rate']}")
                print(f"  Spread:                    {d['spread_pct']:.4f}%" if d['spread_pct'] is not None else "  Spread: None")
                print(f"  Order-book Imbalance:      {d['ob_imbalance']:.4f}" if d['ob_imbalance'] is not None else "  OB Imbalance: None")
                print(f"  Buy/Sell Flow:             {flow_str}")
                print(f"  DECISION:                  {d['decision']}")
                print(f"  Decision Score:            {d['decision_score']}")
                print(f"  Reasons:                   {d['reasons_json']}")
                print("-" * 60)
                sys.stdout.flush()

            # Check new trades
            cur.execute("SELECT * FROM trade_journal WHERE created_at >= ? ORDER BY id ASC", (start_ts,))
            trades = cur.fetchall()

            for t in trades:
                tid = t["id"]
                if tid in seen_trades:
                    continue
                seen_trades.add(tid)

                print("\n" + "=" * 70)
                print(f"[TRADE DETECTED] Trade Journal ID: {t['id']}")
                print(f"  Symbol:          {t['symbol']}")
                print(f"  Bybit Order ID:  {t['bybit_order_id']}")
                print(f"  Planned Entry:   {t['planned_entry']}")
                print(f"  Actual Entry:    {t['actual_entry_price']}")
                print(f"  Stop Loss:       {t['stop_loss']}")
                print(f"  Take Profit:     {t['take_profit']}")
                print(f"  Quantity:        {t['quantity']}")
                print(f"  Position Size:   ${t['position_size_usd']:,.2f}")
                print(f"  Fill Timestamp:  {t['fill_timestamp']}")
                print(f"  SL Confirmed:    {t['sl_confirmed_timestamp']}")
                print(f"  TP Confirmed:    {t['tp_confirmed_timestamp']}")
                print("=" * 70)
                sys.stdout.flush()

                # Verify trade execution via Bybit API and local DB
                asyncio.run(verify_trade_execution(dict(t)))

                # STOP PM2 IMMEDIATELY — EXACTLY ONE TRADE ALLOWED
                print("\n[STOPPING SESSION IMMEDIATELY] Exactly 1 trade executed. Halting PM2 processes now...")
                subprocess.run("pm2 stop all", shell=True, check=False)
                print("[SESSION STOPPED] PM2 processes stopped cleanly.")
                sys.stdout.flush()
                return

            conn.close()
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nMonitor stopped by user.")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
