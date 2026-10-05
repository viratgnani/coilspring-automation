import sqlite3

conn = sqlite3.connect("coilbrain.db")
c = conn.cursor()

print("=== DEMO POSITIONS ===")
for r in c.execute("SELECT position_id, symbol, status, created_at FROM demo_positions ORDER BY rowid DESC LIMIT 5").fetchall():
    print(r)

print("\n=== SESSION GUARD ===")
print(c.execute("SELECT * FROM demo_session_guard").fetchall())

print("\n=== TOTAL DECISIONS THIS SESSION ===")
# Epoch 1790820000 = approx 13:00 IST Oct 1 2026
print("Total:", c.execute("SELECT count(*) FROM decision_journal WHERE decision_ts > 1790820000").fetchone())

print("\n=== TRADE DECISIONS (ALL TIME) ===")
for r in c.execute("SELECT symbol, decision_score, reasons_json, decision_ts FROM decision_journal WHERE decision = 'TRADE' ORDER BY rowid DESC LIMIT 5").fetchall():
    import datetime
    ts = datetime.datetime.fromtimestamp(r[3]) if r[3] else "N/A"
    print(f"Symbol: {r[0]} | Score: {r[1]} | Time: {ts}")
    print(f"  Reasons: {r[2]}")

print("\n=== RECENT 10 DECISIONS (NEWEST FIRST) ===")
for r in c.execute("SELECT symbol, decision, flow_state, ob_state, oi_state, reasons_json FROM decision_journal ORDER BY rowid DESC LIMIT 10").fetchall():
    print(f"{r[0]}: {r[1]} | Flow={r[2]} OB={r[3]} OI={r[4]} | {r[5]}")

conn.close()
