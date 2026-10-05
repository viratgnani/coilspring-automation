"""Live session monitor for natural CoilSpring signals and CoilBrain decisions.

Outputs for every evaluated signal:
- Symbol
- Score
- Flow (state + value)
- OB (state + value)
- OI (state + delta)
- Supportive count
- Adverse count
- Normal qualification
- Squeeze qualification
- Final decision & path
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

DB_PATH = Path("coilbrain/coilbrain.db")
if not DB_PATH.exists():
    DB_PATH = Path("coilbrain.db")


def print_evaluation(row: dict):
    symbol = row["symbol"]
    score = row["cs_score"] if row["cs_score"] is not None else "N/A"
    flow_state = row["flow_state"] or "N/A"
    flow_val = f"{row['trade_imbalance']:+.3f}" if row["trade_imbalance"] is not None else "N/A"
    ob_state = row["ob_state"] or "N/A"
    ob_val = f"{row['ob_imbalance']:+.3f}" if row["ob_imbalance"] is not None else "N/A"
    oi_state = row["oi_state"] or "N/A"
    oi_val = f"{row['oi_delta_pct']:+.2f}%" if row["oi_delta_pct"] is not None else "N/A"
    intensity = row["intensity_regime"] or "NORMAL"
    decision = row["decision"]
    reasons = json.loads(row["reasons_json"] or "[]")
    
    # Calculate counts
    supportive = sum([
        oi_state == "SUPPORTIVE",
        flow_state == "SUPPORTIVE",
        ob_state == "SUPPORTIVE",
        intensity in ("ELEVATED", "EXTREME"),
    ])
    adverse = sum([
        oi_state == "ADVERSE",
        flow_state == "ADVERSE",
        ob_state == "ADVERSE",
    ])
    
    # Check qualifications
    score_num = float(score) if score != "N/A" else 0.0
    flow_num = row["trade_imbalance"]
    ob_num = row["ob_imbalance"]
    
    normal_qualifies = (score_num >= 55.0 and supportive >= 2 and adverse <= 1)
    is_squeeze = (
        score_num >= 85.0
        and flow_num is not None and flow_num >= 0.20
        and ob_num is not None and ob_num <= -0.10
        and adverse <= 1
    )
    squeeze_qualifies = is_squeeze and score_num >= 55.0
    
    # Determine path
    if decision == "TRADE":
        if "ASYMMETRIC_SQUEEZE" in reasons:
            path = "ASYMMETRIC_SQUEEZE"
        else:
            path = "NORMAL_V1"
    else:
        path = "NONE (IGNORED)"

    print("=" * 72)
    print(f"SIGNAL: {symbol:<16} | Decision: {decision:<6} | Path: {path}")
    print(f"  CoilSpring Score: {score}")
    print(f"  Trade Flow:       {flow_state:<10} (imbalance: {flow_val})")
    print(f"  Order Book:       {ob_state:<10} (imbalance: {ob_val})")
    print(f"  Open Interest:    {oi_state:<10} (delta: {oi_val})")
    print(f"  Intensity:        {intensity}")
    print(f"  Components:       Supportive = {supportive}/4 | Adverse = {adverse}/3")
    print(f"  Qualification:    Normal V1: {normal_qualifies} | Asymmetric Squeeze: {squeeze_qualifies}")
    print(f"  Reasons:          {reasons}")
    print("=" * 72)


def main():
    print(f"Starting CoilBrain live signal monitor (reading {DB_PATH})...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Find current max rowid
    r = c.execute("SELECT MAX(rowid) FROM decision_journal").fetchone()
    last_rowid = r[0] if r and r[0] is not None else 0
    print(f"Initialized monitor at rowid {last_rowid}. Waiting for natural signals...\n")
    conn.close()

    while True:
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            q = """
            SELECT 
                d.rowid,
                d.symbol,
                COALESCE(s.score, d.cs_score) as cs_score,
                d.trade_imbalance,
                d.ob_imbalance,
                d.oi_delta_pct,
                d.flow_state,
                d.ob_state,
                d.oi_state,
                d.intensity_regime,
                d.decision,
                d.reasons_json
            FROM decision_journal d
            LEFT JOIN signals s ON d.signal_id = s.signal_id
            WHERE d.rowid > ?
            ORDER BY d.rowid ASC
            """
            rows = c.execute(q, (last_rowid,)).fetchall()
            for row in rows:
                last_rowid = row["rowid"]
                print_evaluation(dict(row))
            conn.close()
        except Exception as exc:
            print(f"Monitor error: {exc}", file=sys.stderr)
        time.sleep(2.0)


if __name__ == "__main__":
    main()
