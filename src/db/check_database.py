import sqlite3
from common import CONFIG

def check_database():
    conn = sqlite3.connect(CONFIG["paths"]["db_path"])
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM taxi_data LIMIT 5;")
    rows = cursor.fetchall()

    print("Données stockées ici :")
    for row in rows:
        print(row)

    conn.close()
