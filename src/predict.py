import joblib
import pandas as pd
import numpy as np
import argparse

def load_artifacts(model_path, scaler_path=None):
    model = None
    if model_path.endswith(".txt") or model_path.endswith(".model"):
        import lightgbm as lgb
        model = lgb.Booster(model_file=model_path)
    else:
        model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path else None
    return model, scaler

def predict_single(model, scaler, record_dict):
    import pandas as pd
    X = pd.DataFrame([record_dict])
    if scaler is not None:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = scaler.transform(X[numeric_cols])
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:,1][0]
    else:
        prob = model.predict(X)  # lgb.Booster returns raw score by default
    return prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--scaler", default=None)
    parser.add_argument("--sample", default=None, help="JSON string or csv row")
    args = parser.parse_args()
    model, scaler = load_artifacts(args.model, args.scaler)
    # for demonstration, predict dummy
    sample = {"V1": 0.1, "V2": -0.2, "Amount": 100.0}
    prob = predict_single(model, scaler, sample)
    print("Fraud probability:", prob)
