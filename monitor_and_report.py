import sqlite3
import time
import json
import sys
import os

DB_PATH = 'coilbrain/coilbrain.db'

print("Monitoring CoilBrain DB for signals...")

def get_report():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    last_decision_id = 0
    trade_done = False
    
    while not trade_done:
        cur.execute('SELECT * FROM decision_journal WHERE id > ? ORDER BY id ASC', (last_decision_id,))
        decisions = cur.fetchall()
        for row in decisions:
            last_decision_id = row['id']
            decision = row['decision']
            print(f"\n[NEW DECISION] ID: {row['id']} | Symbol: {row['symbol']} | Decision: {decision}")
            print(f"Reason: {row['decision_reasons']}")
            print(f"Score: {row['decision_score']}")
            
            if decision == 'TRADE':
                print("\n>>> TRADE INITIATED! <<<")
                
                # Give it a couple of seconds to execute and record the trade journal
                time.sleep(3)
                
                cur.execute('SELECT * FROM trade_journal WHERE decision_id = ?', (row['id'],))
                trade = cur.fetchone()
                
                if trade:
                    print("\n--- TRADE EXECUTION REPORT ---")
                    for k in trade.keys():
                        print(f"{k}: {trade[k]}")
                else:
                    print("No trade journal entry found for this decision.")
                
                cur.execute('SELECT count FROM demo_session_guard')
                guard_row = cur.fetchone()
                guard_count = guard_row['count'] if guard_row else 0
                print(f"Demo guard count: {guard_count}")
                
                trade_done = True
                break
        
        time.sleep(5)

if __name__ == "__main__":
    get_report()
