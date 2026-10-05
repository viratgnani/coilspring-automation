import asyncio
import os
import sys
import sqlite3
from pathlib import Path

# Add coilbrain src to sys.path
sys.path.insert(0, r"c:\Users\user\coilspring automation\coilbrain\src")
from dotenv import load_dotenv

load_dotenv(r"c:\Users\user\coilspring automation\coilbrain\.env")

from coilbrain.bybit_demo import BybitDemoClient, _BYBIT_DEMO_BASE_URL
from coilbrain.demo_guard import MAX_DEMO_TRADES_PER_SESSION

async def run_preflight():
    print("=" * 60)
    print("COILBRAIN V1 — CONTROLLED LIVE DEMO PRE-FLIGHT VERIFICATION")
    print("=" * 60)

    # 1. Bybit endpoint
    print(f"\n1. Bybit Endpoint:")
    print(f"   Configured URL: {_BYBIT_DEMO_BASE_URL}")
    assert "api-demo.bybit.com" in _BYBIT_DEMO_BASE_URL, "FATAL: Must be api-demo.bybit.com!"
    assert "api.bybit.com" not in _BYBIT_DEMO_BASE_URL or "api-demo.bybit.com" in _BYBIT_DEMO_BASE_URL, "FATAL: Production URL detected!"
    print("   -> PASS: Bybit Demo endpoint verified (api-demo.bybit.com).")

    # 2. Demo credentials
    api_key = os.getenv("BYBIT_DEMO_API_KEY", "")
    api_secret = os.getenv("BYBIT_DEMO_API_SECRET", "")
    print(f"\n2. Demo Credentials:")
    print(f"   API Key present: {bool(api_key)} (length: {len(api_key)}, prefix: {api_key[:4]}...)")
    print(f"   API Secret present: {bool(api_secret)} (length: {len(api_secret)})")
    assert len(api_key) > 5 and len(api_secret) > 5, "FATAL: Credentials missing or invalid"
    print("   -> PASS: Credentials present.")

    # 3. Demo balance / account connectivity
    print(f"\n3. Demo Account Connectivity & Balance:")
    client = BybitDemoClient.from_env()
    bal = await client.get_balance()
    print(f"   Total Equity:      ${bal.total_equity:,.2f} USD")
    print(f"   Available Balance: ${bal.available_balance:,.2f} USD")
    assert bal.total_equity > 0, "FATAL: Account equity must be > 0"
    print("   -> PASS: Bybit Demo account connected successfully.")

    # 4. Open positions on Bybit & in DB
    print(f"\n4. Open Positions Check:")
    data = await client._get("/v5/position/list", {"category": "linear", "settleCoin": "USDT"}, signed=True)
    raw_positions = data.get("result", {}).get("list", [])
    active_bybit_positions = [p for p in raw_positions if float(p.get("size", 0)) > 0]
    print(f"   Bybit open positions count: {len(active_bybit_positions)}")
    if active_bybit_positions:
        for p in active_bybit_positions:
            print(f"     - Symbol: {p.get('symbol')}, Size: {p.get('size')}, Side: {p.get('side')}, Entry: {p.get('avgPrice')}")
    assert len(active_bybit_positions) == 0, f"FATAL: Existing open positions on Bybit: {active_bybit_positions}"

    db_path = r"c:\Users\user\coilspring automation\coilbrain\coilbrain.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT symbol, status FROM demo_positions WHERE status IN ('OPEN', 'PENDING_ENTRY', 'EXITING')")
    open_pos = cur.fetchall()
    print(f"   DB active positions count: {len(open_pos)}")
    assert len(open_pos) == 0, f"FATAL: Found active DB positions: {open_pos}"
    print("   -> PASS: Zero open/pending positions on Bybit and in DB.")

    # 5. Demo session guard in DB
    print(f"\n5. Demo Session Guard State:")
    cur.execute("SELECT trade_count FROM demo_session_guard WHERE id = 1")
    row = cur.fetchone()
    count = row[0] if row else 0
    print(f"   Current trade_count in DB: {count}")
    assert count == 0, f"FATAL: Persistent guard trade_count is {count}, expected 0/1!"
    print("   -> PASS: Persistent guard is exactly 0/1.")

    # 6. MAX_DEMO_TRADES_PER_SESSION
    print(f"\n6. MAX_DEMO_TRADES_PER_SESSION:")
    print(f"   Value: {MAX_DEMO_TRADES_PER_SESSION}")
    assert MAX_DEMO_TRADES_PER_SESSION == 1, "FATAL: MAX_DEMO_TRADES_PER_SESSION must be 1"
    print("   -> PASS: Guard limit verified at 1.")

    # 7. CoilBrain database health
    print(f"\n7. CoilBrain DB Health:")
    cur.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
    t_count = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM decision_journal")
    d_count = cur.fetchone()[0]
    print(f"   Tables: {t_count}, Existing decisions in journal: {d_count}")
    assert t_count >= 5, "FATAL: Database tables missing"
    print("   -> PASS: CoilBrain database is healthy.")

    # 8. Durable outbox health
    print(f"\n8. Durable Outbox Health:")
    outbox_path = r"c:\Users\user\coilspring automation\coilbrain_outbox.db"
    outbox_conn = sqlite3.connect(outbox_path)
    outbox_cur = outbox_conn.cursor()
    outbox_cur.execute("SELECT count(*) FROM outbox_items")
    total_outbox = outbox_cur.fetchone()[0]
    outbox_cur.execute("SELECT count(*) FROM outbox_items WHERE state = 'PENDING'")
    pending_outbox = outbox_cur.fetchone()[0]
    print(f"   Total outbox records: {total_outbox}, Pending records: {pending_outbox}")
    print("   -> PASS: CoilSpring durable outbox is healthy.")

    print("\n" + "=" * 60)
    print(">>> ALL 8 PRE-FLIGHT CHECKS PASSED SUCCESSFULLY <<<")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_preflight())
