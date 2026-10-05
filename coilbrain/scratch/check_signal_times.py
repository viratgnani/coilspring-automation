import sqlite3
import datetime

conn = sqlite3.connect("coilbrain.db")
c = conn.cursor()

q = "SELECT signal_id, symbol, score, grade, received_timestamp FROM signals ORDER BY rowid DESC LIMIT 10"
for r in c.execute(q).fetchall():
    ts = datetime.datetime.fromtimestamp(r[4]) if r[4] else "N/A"
    print(f"ID: {r[0][:8]} | Symbol: {r[1]} | Score: {r[2]} ({r[3]}) | Received: {ts}")

conn.close()
