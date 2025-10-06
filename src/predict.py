import joblib
import pandas as pd
import numpy as np
import argparse
import lightgbm as lgb
import os
from sklearn.metrics import classification_report, roc_auc_score

def load_artifacts(model_path, scaler_path=None):
    """Load model and scaler artifacts."""
    if model_path.endswith((".txt", ".model")):
        model = lgb.Booster(model_file=model_path)
    else:
        model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path else None
    return model, scaler

def predict_batch(model, scaler, data_path, output_path="data/processed/predictions.csv", y_path=None):
    """Predict fraud probabilities for batch CSV input and optionally evaluate."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ File not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"✅ Loaded feature data with shape: {df.shape}")

    # Match features to scaler training
    if scaler is not None:
        if hasattr(scaler, "feature_names_in_"):
            for col in scaler.feature_names_in_:
                if col not in df.columns:
                    df[col] = 0
            df = df[scaler.feature_names_in_]
        X_scaled = scaler.transform(df)
    else:
        X_scaled = df.values

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[:, 1]
    else:
        probs = model.predict(X_scaled)

    df_out = df.copy()
    df_out["fraud_probability"] = probs

    # Optional evaluation if y provided
    if y_path and os.path.exists(y_path):
        y_true = pd.read_csv(y_path).values.ravel()
        y_pred = (probs > 0.5).astype(int)
        print("\n📊 Model Evaluation Report:")
        print(classification_report(y_true, y_pred))
        auc = roc_auc_score(y_true, probs)
        print(f"ROC-AUC Score: {auc:.4f}")

    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Fraud Prediction & Evaluation")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--scaler", default=None, help="Path to trained scaler")
    parser.add_argument("--data", required=True, help="Path to CSV file for prediction")
    parser.add_argument("--output", default="data/processed/predictions.csv", help="Output CSV path")
    parser.add_argument("--y", default=None, help="Optional true labels CSV for evaluation")
    args = parser.parse_args()

    model, scaler = load_artifacts(args.model, args.scaler)
    predict_batch(model, scaler, args.data, args.output, args.y)
