# prepare_features.py
from src.features import generate_features, save_Xy
import pandas as pd

# Load the processed CSV from ingestion
df = pd.read_csv("data/processed/processed.csv")

# Generate features
df = generate_features(df)

# Save X and y for training
X, y = save_Xy(df)
print("✅ Features generated and saved at data/processed/X.csv and y.csv")
