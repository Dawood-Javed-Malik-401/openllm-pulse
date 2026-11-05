import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join("data", "openllm_pulse.db")
CSV_PATH = os.path.join("data", "cleaned_models.csv")
TABLE_NAME = "llm_models"


def load_data(df: pd.DataFrame) -> None:
  """Loads the transformed Pandas DataFrame into a SQLite database and CSV file."""
  os.makedirs("data", exist_ok=True)

  # 1. Save to CSV file
  df.to_csv(CSV_PATH, index=False, encoding="utf-8")
  print(f"[LOAD] Saved cleaned dataset to CSV file: '{CSV_PATH}'")

  # 2. Save to SQLite Database
  conn = sqlite3.connect(DB_PATH)
  try:
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    row_count = cursor.fetchone()[0]
    print(
        f"[LOAD] Loaded {row_count} records into SQLite database '{DB_PATH}'"
        f" (table: '{TABLE_NAME}')."
    )
  finally:
    conn.close()


def read_from_database() -> pd.DataFrame:
  """Reads the structured dataset back from SQLite for Exploratory Data Analysis."""
  if not os.path.exists(DB_PATH):
    raise FileNotFoundError(
        f"Database not found at '{DB_PATH}'. Run the ETL pipeline first."
    )

  conn = sqlite3.connect(DB_PATH)
  try:
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    return df
  finally:
    conn.close()