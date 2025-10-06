import pandas as pd
import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os

MODEL_PATH = "models/fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"
os.makedirs("models", exist_ok=True)

def train_model(X_path="data/processed/X.csv", y_path="data/processed/y.csv"):
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")
    print(f"✅ Scaler saved at {SCALER_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", default="data/processed/X.csv")
    parser.add_argument("--y", default="data/processed/y.csv")
    args = parser.parse_args()
    train_model(args.X, args.y)
