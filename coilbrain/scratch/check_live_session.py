import sqlite3

conn = sqlite3.connect("coilbrain.db")
c = conn.cursor()

print("=== MOVR SNAPSHOTS & DECISIONS ===")
q = """
SELECT 
    d.signal_id,
    d.symbol,
    s.score,
    d.trade_imbalance,
    d.ob_imbalance,
    d.oi_delta_pct,
    d.flow_state,
    d.ob_state,
    d.oi_state,
    d.decision,
    d.reasons_json
FROM decision_journal d
LEFT JOIN signals s ON d.signal_id = s.signal_id
WHERE d.symbol LIKE '%MOVR%'
ORDER BY d.rowid DESC
LIMIT 5
"""
for r in c.execute(q).fetchall():
    print(r)

print("\n=== LATEST 5 SNAPSHOTS FOR MOVRUSDT ===")
for r in c.execute("SELECT snapshot_ts, trade_imbalance, ob_imbalance, oi_delta_pct FROM microstructure_snapshots WHERE binance_symbol='MOVRUSDT' ORDER BY rowid DESC LIMIT 5").fetchall():
    print(r)

conn.close()
