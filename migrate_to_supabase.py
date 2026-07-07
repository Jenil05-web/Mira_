# migrate_to_supabase.py
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

sqlite_conn = sqlite3.connect("./mira_data/mimic.db")
pg_engine = create_engine("postgresql://postgres.tfrlbotgzxxdqwviemiz:EtnV1l2E7359NakT@aws-1-ap-southeast-2.pooler.supabase.com:6543/postgres")

for table in ["patients", "admissions", "diagnoses_icd", "labevents", "d_labitems"]:
    print(f"Migrating {table}...")
    df = pd.read_sql_query(f"SELECT * FROM {table}", sqlite_conn)
    df.to_sql(table, pg_engine, if_exists="replace", index=False)
    print(f"  ✅ {len(df)} rows")

print("Done — all tables in Supabase")