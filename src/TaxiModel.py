import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

class TaxiModel:
    def __init__(self, model=None):
        self.model = model or LinearRegression()
        self.features = ['hour', 'weekday', 'month', 'abnormal_period']

    def preprocess(self, df):
        df = df.copy()
    
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['hour'] = df['pickup_datetime'].dt.hour
        df['weekday'] = df['pickup_datetime'].dt.weekday
        df['month'] = df['pickup_datetime'].dt.month

        compteur = df['pickup_datetime'].dt.date.value_counts()
        max = compteur.quantile(0.02)
        abnormal_dates = compteur[compteur < max].index
        df['abnormal_period'] = df['pickup_datetime'].dt.date.isin(abnormal_dates).astype(int)
    
        y = df["trip_duration"] if "trip_duration" in df.columns else None
        return df[self.features], y

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"RMSE ici : {rmse:.4f}")

        self.save()

        return rmse

    def predict(self, df):
        if set(self.features).issubset(df.columns):  
            X_processed = df
        else:
            X_processed, _ = self.preprocess(df)
        
        return self.model.predict(X_processed)

    def save(self, path="models/taxi_model.pkl"):
        joblib.dump(self.model, path)

    @staticmethod
    def load(path="models/taxi_model.pkl"):
        model = joblib.load(path)
        return TaxiModel(model=model)
