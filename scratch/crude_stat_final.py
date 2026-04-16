import sqlite3

def check_win_rate():
    conn = sqlite3.connect('signals.db')
    cursor = conn.cursor()
    
    # Total resolved CrudeOIL BUY signals
    cursor.execute("""
        SELECT outcome, COUNT(*) 
        FROM signals 
        WHERE symbol = 'CrudeOIL' 
        AND outcome IN ('SUCCESS', 'FAIL') 
        GROUP BY outcome
    """)
    results = dict(cursor.fetchall())
    
    success = results.get('SUCCESS', 0)
    fail = results.get('FAIL', 0)
    total = success + fail
    win_rate = (success / total * 100) if total > 0 else 0
    
    print(f"CrudeOIL Resolved Stats:")
    print(f"  SUCCESS: {success}")
    print(f"  FAIL: {fail}")
    print(f"  TOTAL: {total}")
    print(f"  WIN RATE: {win_rate:.1f}%")
    
    conn.close()

if __name__ == "__main__":
    check_win_rate()
