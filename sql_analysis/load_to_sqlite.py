import pandas as pd
import sqlite3
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHES_PATH    = os.path.join(BASE_DIR, "data", "processed", "matches_clean.csv")
DELIVERIES_PATH = os.path.join(BASE_DIR, "data", "processed", "deliveries_clean.csv")
DB_PATH         = os.path.join(BASE_DIR, "sql_analysis", "ipl.db")

# Load CSVs
matches    = pd.read_csv(MATCHES_PATH)
deliveries = pd.read_csv(DELIVERIES_PATH)

# Write to SQLite
conn = sqlite3.connect(DB_PATH)
matches.to_sql("matches", conn, if_exists="replace", index=False)
deliveries.to_sql("deliveries", conn, if_exists="replace", index=False)
conn.close()

print("Database created successfully at:", DB_PATH)
print(f"matches table: {len(matches)} rows")
print(f"deliveries table: {len(deliveries)} rows")