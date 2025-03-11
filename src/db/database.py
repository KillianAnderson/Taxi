import sqlite3
from common import CONFIG

def save_to_db(df):
    conn = sqlite3.connect(CONFIG["paths"]["db_path"])
    df.to_sql("taxi_data", conn, if_exists="replace", index=False)
    conn.close()
    print("Données stockées dans SQLite.")
