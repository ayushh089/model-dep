"""
Downloads the raw dataset from the Hugging Face dataset repo, cleans it,
fixes data-entry inconsistencies, drops non-predictive identifier columns,
performs a stratified train/test split, and pushes train.csv / test.csv
back to the same Hugging Face dataset repo.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download


HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-package-prediction"
TARGET_COL = "ProdTaken"

# 1. Download the raw dataset
raw_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="tourism.csv",
                            repo_type="dataset", token=HF_TOKEN)
df = pd.read_csv(raw_path)

# 2. Clean the data
drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
df = df.drop(columns=drop_cols)

df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

df = df.drop_duplicates()
df = df.dropna(subset=[TARGET_COL])

# 3. Stratified train/test split
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[TARGET_COL]
)

os.makedirs("tourism_project/data", exist_ok=True)
train_path = "tourism_project/data/train.csv"
test_path = "tourism_project/data/test.csv"
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# 4. Push processed files back to the Hugging Face dataset repo
api = HfApi(token=HF_TOKEN)
api.upload_file(path_or_fileobj=train_path, path_in_repo="train.csv",
                 repo_id=DATASET_REPO_ID, repo_type="dataset", token=HF_TOKEN)
api.upload_file(path_or_fileobj=test_path, path_in_repo="test.csv",
                 repo_id=DATASET_REPO_ID, repo_type="dataset", token=HF_TOKEN)
print(f"train.csv and test.csv uploaded to: https://huggingface.co/datasets/{DATASET_REPO_ID}")
