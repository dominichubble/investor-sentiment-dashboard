"""Read-only survey of the Neon DB. Never prints the connection string."""
import os, sys, psycopg

def load_url():
    for line in open("investor-sentiment-dashboard/.env"):
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip().strip("'\"")
    sys.exit("DATABASE_URL not found")

def connect():
    c = psycopg.connect(load_url(), connect_timeout=30, options="-c default_transaction_read_only=on")
    c.read_only = True
    return c

if __name__ == "__main__":
    with connect() as c, c.cursor() as cur:
        cur.execute("select table_name, column_name, data_type from information_schema.columns where table_schema='public' order by table_name, ordinal_position")
        tables = {}
        for t, col, dt in cur.fetchall():
            tables.setdefault(t, []).append(f"{col}:{dt}")
        for t, cols in tables.items():
            cur.execute(f'select count(*) from "{t}"')
            print(f"\n{t}  rows={cur.fetchone()[0]}\n  " + ", ".join(cols))
