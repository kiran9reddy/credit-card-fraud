import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse
import os

MODEL_PATH = "models/fraud_model_lgbm.model"
SCALER_PATH = "models/scaler_lgbm.pkl"

def train_lgbm(X_path="data/processed/X.csv", y_path="data/processed/y.csv",
               model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    # Load features
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test)

    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'is_unbalance': True,
        'boosting_type': 'gbdt',
        'verbose': -1
    }

    # Train model
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    # Evaluate
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(classification_report(y_test, y_pred))

    # Save artifacts
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    model.save_model(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ LightGBM model saved at {model_path}")
    print(f"✅ Scaler saved at {scaler_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", default="data/processed/X.csv", help="Path to features CSV")
    parser.add_argument("--y", default="data/processed/y.csv", help="Path to target CSV")
    parser.add_argument("--model_path", default=MODEL_PATH, help="Path to save trained model")
    parser.add_argument("--scaler_path", default=SCALER_PATH, help="Path to save scaler")
    args = parser.parse_args()

    train_lgbm(args.X, args.y, args.model_path, args.scaler_path)
