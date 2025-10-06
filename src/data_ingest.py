import pandas as pd
import os

def load_raw(path):
    return pd.read_csv(path)

def basic_clean(df):
    df = df.dropna(how="any")
    return df

def save_processed(df, out_dir="data/processed", filename="processed.csv"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    return path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/creditcard.csv")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()

    df = load_raw(args.input)
    df = basic_clean(df)
    save_path = save_processed(df, out_dir=args.output_dir)
    print("Saved processed data to", save_path)
