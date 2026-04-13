
import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def check_active_types():
    conn = sqlite3.connect(DB_PATH)
    
    query = "SELECT signal, count(*) FROM signals WHERE outcome = 'ACTIVE' GROUP BY signal"
    
    df = pd.read_sql_query(query, conn)
    print(df)
    conn.close()

if __name__ == "__main__":
    check_active_types()
