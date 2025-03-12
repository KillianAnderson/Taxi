import pandas as pd
from sklearn.model_selection import train_test_split

def clean_data(df):
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'dropoff_datetime' in df.columns:
        df = df.drop(columns=['dropoff_datetime'])

    if 'pickup_datetime' in df.columns:
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

        df["hour"] = df["pickup_datetime"].dt.hour
        df["day"] = df["pickup_datetime"].dt.day
        df["month"] = df["pickup_datetime"].dt.month
        df["year"] = df["pickup_datetime"].dt.year

        df = df.drop(columns=['pickup_datetime'])

    if 'store_and_fwd_flag' in df.columns:
        df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'N': 0, 'Y': 1})

    return df

def split_data(df):
    X = df.drop(columns=['trip_duration'])
    y = df['trip_duration']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(X_train.head())
    
    return X_train, X_test, y_train, y_test
