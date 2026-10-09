import sys
sys.path.insert(0, '.')
import sqlite3
import pandas as pd
from datetime import datetime, timezone

# 1. Test database migration on signals.db
conn = sqlite3.connect('signals.db')
cursor = conn.cursor()

print("--- Migration: cleaning up corrupted records ---")
# Reset paper signals from August that had mt5_ticket = 0
cursor.execute("UPDATE signals SET mt5_ticket = NULL WHERE mt5_ticket IN (0, '0', '')")
print("Cleaned paper tickets with 0/blank:", cursor.rowcount)

# Fix ticket 152727004937 (GBPAUD): row 31138 was marked FAIL with $0.00 while row 31147 was SUCCESS with +$645.06
cursor.execute("""
    UPDATE signals 
    SET outcome = 'SUCCESS', exit_price = 1.89863, exit_reason = 'MT5 Native Close (Profit: $645.06)'
    WHERE id = 31138
""")
print("Fixed GBPAUD row 31138:", cursor.rowcount)

# Fix ticket 152727803409 (USDCHF): row 31150 was marked FAIL with $0.00 while row 31149 was SUCCESS with +$132.52
cursor.execute("""
    UPDATE signals 
    SET outcome = 'SUCCESS', exit_price = 0.83158, exit_reason = 'MT5 Native Close (Profit: $132.52)'
    WHERE id = 31150
""")
print("Fixed USDCHF row 31150:", cursor.rowcount)

# Fix ticket 152727549255 (AUDJPY): row 31146 had $0.00
cursor.execute("""
    UPDATE signals 
    SET exit_reason = 'MT5 Native Close (Profit: $-453.96)', exit_price = 110.251
    WHERE id = 31146
""")
print("Fixed AUDJPY row 31146:", cursor.rowcount)

# Fix ticket 152727004017 (AUDUSD): row 31134 had $0.00
cursor.execute("""
    UPDATE signals 
    SET exit_reason = 'MT5 Native Close (Profit: $-453.18)', exit_price = 0.69821
    WHERE id = 31134
""")
print("Fixed AUDUSD row 31134:", cursor.rowcount)

# Fix rows 31382, 31386 (USDCHF BE exits):
cursor.execute("""
    UPDATE signals
    SET outcome = 'SUCCESS', exit_reason = 'M15 BE hit ($0.00)'
    WHERE id IN (31382, 31386)
""")
print("Fixed BE rows 31382, 31386:", cursor.rowcount)

# Fix row 22442 (EURUSD BE exit):
cursor.execute("""
    UPDATE signals
    SET outcome = 'SUCCESS', exit_reason = 'MT5 Native Close (BE Profit: $0.00)'
    WHERE id = 22442
""")
print("Fixed BE row 22442:", cursor.rowcount)

conn.commit()
conn.close()
print("Migration completed.")
