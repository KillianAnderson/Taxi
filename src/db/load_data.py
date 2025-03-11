import sqlite3
import pandas as pd
from common import CONFIG

def load_data_from_db():
    conn = sqlite3.connect(CONFIG["paths"]["db_path"])
    df = pd.read_sql("SELECT * FROM taxi_data", conn)
    conn.close()
    print(f"{df.shape[0]} lignes, {df.shape[1]} colonnes.")
    return df
