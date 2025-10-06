import pandas as pd
import numpy as np

def add_time_features(df):
    # if 'Time' is seconds from first transaction:
    if "Time" in df.columns:
        df["hour"] = (df["Time"] // 3600) % 24
    return df

def generate_features(df):
    df = add_time_features(df)
    # Example: log transform Amount
    if "Amount" in df.columns:
        df["amount_log"] = np.log1p(df["Amount"])
    return df

def save_Xy(df, target_col="Class", out_dir="data/processed"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X.to_csv(f"{out_dir}/X.csv", index=False)
    y.to_csv(f"{out_dir}/y.csv", index=False)
    return X, y
