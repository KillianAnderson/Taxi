from db.stock_data import download_and_load_data
from db.database import save_to_db
from db.check_database import check_database
from db.load_data import load_data_from_db
from TaxiModel import TaxiModel
from sklearn.model_selection import train_test_split

def main():
    df = download_and_load_data()
    
    save_to_db(df)
    check_database()
    
    df = load_data_from_db()
    
    model = TaxiModel()
    X, y = model.preprocess(df)  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model.train(X_train, y_train)  

    print("Sauvegarde du modèle")
    model.save()
    

if __name__ == "__main__":
    main()
