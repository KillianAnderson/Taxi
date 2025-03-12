from db.stock_data import download_and_load_data
from db.database import save_to_db
from db.check_database import check_database
from db.load_data import load_data_from_db
from services.train import train_model
from services.evaluate import evaluate_model

from services.data_preprocessing import clean_data, split_data

def main():
    print("Start ici")
    
    df = download_and_load_data()
    save_to_db(df)
    check_database()
    
    print("C'est en DB !")

    df = load_data_from_db()
    df_clean = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df_clean)

    print("C'est split ici")
    
    model, rmse = train_model(X_train, X_test, y_train, y_test)
    print("Modèle entraîné ici")
    print(f"RMSE: {rmse:.4f}")
    
    rmse_test = evaluate_model(X_test, y_test)
    print(f"Modèle évalué avec RMSE (test): {rmse_test:.4f}")

if __name__ == "__main__":
    main()
