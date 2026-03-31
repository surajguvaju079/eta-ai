import datetime
import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


DATA_PATH = "data/real_training_data.csv"
MODEL_PATH = "models/real_eta_model.pkl"

REQUIRED_COLUMNS = [
    "distance_km",
    "speed_kmh",
    "hour",
    "weekday",
    "actual_eta_seconds"
]

OPTIONAL_COLUMNS = [
    "delay_seconds",
    "l_predicted_eta_seconds",
    "p_predicted_eta_seconds",
    "trip_id",
]

def add_time_features(df:pd.DataFrame) ->pd.DataFrame:
    df = df.copy()

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
    df["weekday_sin"] = np.sin(2*np.pi*df["weekday"]/7.0)
    df["weekday_cos"] = np.cos(2*np.pi*df["weekday"]/7.0)


    df["is_morning_peak"] = df["hour"].between(7,10).astype(int)
    df["is_evening_peak"] = df["hour"].between(16,19).astype(int)
    df["is_weekend"] = df["weekday"].between(5,6).astype(int)

    return df

def load_data(path:str)->pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in the dataset: {missing}")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[df["distance_km"].between(0.05, 100)]
    df = df[df["speed_kmh"].between(1, 120)]
    df = df[df["hour"].between(0,23)]
    df = df[df["weekday"].between(0,6)]
    df = df[df["actual_eta_seconds"].between(10, 7200)]

    for col in OPTIONAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return add_time_features(df)

def build_feature_columns(df:pd.DataFrame):
    features = [
        "distance_km",
        "speed_kmh",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "is_morning_peak",
        "is_evening_peak",
        "is_weekend"
    ]

    for col in ["delay_seconds", "l_predicted_eta_seconds", "p_predicted_eta_seconds"]:
        if col in df.columns:
            features.append(col)
    return features

def train():
    df = load_data(DATA_PATH)
    feature_columns = build_feature_columns(df)

    x = df[feature_columns]
    y = df["actual_eta_seconds"]

    X = df[feature_columns]
    y = df["actual_eta_seconds"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


    model = RandomForestRegressor(n_estimators=300,max_depth=18,min_samples_split=8,min_samples_leaf=3,random_state=42,n_jobs=-1)
    
    model.fit(X_train,y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test,preds)
    rmse = np.sqrt(mean_squared_error(y_test,preds))
    r2 = r2_score(y_test,preds)

    print("Model Evaluation")
    print(f"MAE:{mae:.2f} sec")
    print(f"RMSE:{rmse:.2f} sec")
    print(f"R2:{r2:.2f}")

    bundle = {
        "model":model,
        "feature_columns": feature_columns,
        "trained_at":datetime.utcnow().isoformat(),
        "target":"actual_eta_seconds",
        "metrics":{
            "mae_seconds": mae,
            "rmse_seconds": rmse,
            "r2_score": r2
        },
        "version": "2.0.0"
    }

    os.makedirs(os.path.dirname(MODEL_PATH),exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"Model saved to '{MODEL_PATH}'")


if __name__ == "__main__":
    train()