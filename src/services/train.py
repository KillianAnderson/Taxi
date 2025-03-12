import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {rmse:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/taxi_model.pkl")
    print("Modèle sauvegardé")

    return model, rmse
