#!/usr/bin/env python3
"""
fix_db.py
Safely back up instance/truthguard.db and add only missing columns to the 'analyses' table.
Run while your Flask app is stopped.

Usage:
    python fix_db.py
"""
import os
import sqlite3
import shutil
from datetime import datetime
import sys

DB_PATH = os.path.join("instance", "truthguard.db")

# name, sqlite_type, default_sql_or_None
EXPECTED_COLUMNS = [
    ("title", "TEXT", "''"),
    ("content", "TEXT", "''"),                # likely missing column
    ("article_data", "TEXT", None),
    ("source_url", "TEXT", None),
    ("classification", "TEXT", "'UNKNOWN'"),
    ("confidence_score", "REAL", None),
    ("sentiment_score", "REAL", None),
    ("sensationalism_score", "REAL", None),
    ("credibility_score", "REAL", None),
    ("key_findings", "TEXT", None),
    ("recommendations", "TEXT", None),
    ("fact_checks", "TEXT", None),
    ("analysis_metadata", "TEXT", None),
    ("user_id", "INTEGER", None),
    ("created_at", "DATETIME", None),
    ("updated_at", "DATETIME", None),
]

def fail(msg):
    print("ERROR:", msg)
    sys.exit(1)

def backup_db(path):
    if not os.path.exists(path):
        fail(f"DB not found at {path}")
    bak = f"{path}.bak.{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    shutil.copy2(path, bak)
    return bak

def get_existing_columns(conn, table="analyses"):
    cur = conn.execute(f"PRAGMA table_info('{table}')")
    rows = cur.fetchall()
    return [r[1] for r in rows]  # name is index 1

def add_column(conn, table, name, coltype, default):
    if default is not None:
        sql = f"ALTER TABLE {table} ADD COLUMN {name} {coltype} DEFAULT {default}"
    else:
        sql = f"ALTER TABLE {table} ADD COLUMN {name} {coltype}"
    conn.execute(sql)

def main():
    print("Starting DB fix script...")
    if not os.path.exists(DB_PATH):
        fail(f"Database file not found: {DB_PATH}")

    print("Creating backup...")
    bak = backup_db(DB_PATH)
    print("Backup created:", bak)

    conn = sqlite3.connect(DB_PATH)
    try:
        existing = get_existing_columns(conn, "analyses")
    except Exception as e:
        conn.close()
        fail(f"Could not read table info for 'analyses'. Error: {e}\nIf 'analyses' table doesn't exist, run your app with init_database(drop_in_dev=True) after backing up.")

    print("Existing columns in 'analyses':", existing)

    to_add = []
    for name, coltype, default in EXPECTED_COLUMNS:
        if name not in existing:
            to_add.append((name, coltype, default))

    if not to_add:
        print("No missing columns detected. Database schema looks up-to-date.")
        conn.close()
        return

    print("Columns to add:", [c[0] for c in to_add])
    try:
        for name, coltype, default in to_add:
            print(f"Adding column: {name} ...", end=" ")
            try:
                add_column(conn, "analyses", name, coltype, default)
                print("OK")
            except sqlite3.OperationalError as oe:
                # column may already exist (race) or SQL error
                print("SKIP (OperationalError):", oe)
            except Exception as e:
                print("FAILED:", e)
                raise
        conn.commit()
        print("All missing columns have been processed. Commit complete.")
    except Exception as e:
        print("Error during migration:", e)
        conn.rollback()
    finally:
        # show final columns
        try:
            final_cols = get_existing_columns(conn, "analyses")
            print("\nFinal 'analyses' columns:")
            for c in final_cols:
                print(" -", c)
        except Exception as e:
            print("Could not fetch final schema:", e)
        conn.close()

    print("Done. Please restart your Flask app and check /dashboard.")

if __name__ == "__main__":
    main()
