import joblib
from sklearn.metrics import mean_squared_error

def evaluate_model(X_test, y_test):
    
    model = joblib.load("models/taxi_model.pkl")

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"📊 RMSE sur les données de test : {rmse:.4f}")

    return rmse
