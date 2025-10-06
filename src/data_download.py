import os
import zipfile
import kagglehub
import shutil

# Folder where raw data will be stored
DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

# Step 1: Download dataset (downloads to current directory)
zip_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("✅ Dataset downloaded at:", zip_path)

filename = "creditcard.csv"

# Step 2: Move zip to DATA_DIR
dst_path = os.path.join(DATA_DIR, filename, )
src_path = os.path.join(zip_path, filename)
shutil.copy(src_path, dst_path)

print(f"Data moved to {DATA_DIR}")

# # Step 3: Unzip
# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(DATA_DIR)
# print(f"✅ Dataset extracted to {DATA_DIR}")

# # Step 4: Remove zip to save space
# os.remove(zip_path)
