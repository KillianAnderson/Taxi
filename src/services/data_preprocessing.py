import pandas as pd
from sklearn.model_selection import train_test_split

def clean_data(df):
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'dropoff_datetime' in df.columns:
        df = df.drop(columns=['dropoff_datetime'])

    if 'pickup_datetime' in df.columns:
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    print("Clean terminé")
    return df

def split_data(df):
    X = df.drop(columns=['trip_duration'])
    y = df['trip_duration']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Données train ici après split ici : ")
    print(X_train.head())
    
    return X_train, X_test, y_train, y_test
