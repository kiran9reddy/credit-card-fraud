import os
from kaggle.api.kaggle_api_extended import KaggleApi

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Kaggle dataset path
dataset_name = "mldata/creditcardfraud"  # Update if needed

# Download and unzip
api.dataset_download_files(dataset_name, path=DATA_DIR, unzip=True)
print(f"✅ Dataset downloaded to {DATA_DIR}")
