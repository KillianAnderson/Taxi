from fastapi import FastAPI, HTTPException
import uvicorn
import joblib
import os
import sys
from pydantic import BaseModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI()

import os

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "taxi_model.pkl")
model_path = os.path.abspath(model_path)  # Convertir en chemin absolu

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le fichier modèle {model_path} est introuvable.")

model = joblib.load(model_path)

class TripFeatures(BaseModel):
    vendor_id: int
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: int
    pickup_hour: int
    pickup_day: int
    pickup_month: int
    pickup_year: int
    weekday: int
    abnormal_period: int


@app.get("/")
def root():
    return {"message": "NYC Taxi Prediction API is running!"}

@app.post("/predict")
def predict_duration(features: TripFeatures):
    try:
        data = [[
            features.vendor_id,
            features.passenger_count,
            features.pickup_longitude,
            features.pickup_latitude,
            features.dropoff_longitude,
            features.dropoff_latitude,
            features.store_and_fwd_flag,
            features.pickup_hour, 
            features.pickup_day, 
            features.pickup_month, 
            features.pickup_year, 
            features.weekday,
            features.abnormal_period
        ]]
        
        prediction = model.predict(data)[0]
        
        return {"Prédiction ici : ": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
