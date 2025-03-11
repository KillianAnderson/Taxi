import os
import pandas as pd
import requests
from common import CONFIG

def download_and_load_data():
    url = CONFIG["paths"]["dataset_url"]
    zip_path = "data.zip"

    if not os.path.exists(zip_path):
        print("Téléchargement")
        response = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(response.content)
        print("Téléchargement fin")

    df = pd.read_csv(zip_path, compression="zip")

    os.remove(zip_path)
    
    return df
