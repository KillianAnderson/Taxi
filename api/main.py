from fastapi import FastAPI, HTTPException
import uvicorn
import pandas as pd
from pydantic import BaseModel
from datetime import datetime
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.TaxiModel import TaxiModel

app = FastAPI()

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "taxi_model.pkl"))
model = TaxiModel.load(model_path)

class TripFeatures(BaseModel):
    pickup_datetime: datetime

@app.get("/")
def root():
    return {"message": "Welcome"}

@app.post("/predict")
def predict_trip_duration(features: TripFeatures):
    try:
        input_data = pd.DataFrame([features.model_dump()])
        
        X_processed, _ = model.preprocess(input_data)

        predict = model.predict(X_processed)[0]

        return {"prediction": float(predict)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
