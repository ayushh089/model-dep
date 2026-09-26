"""Uploads the raw tourism.csv file to a Hugging Face Hub dataset repository."""
import os
from huggingface_hub import HfApi, create_repo



HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")

DATASET_REPO_ID = f"{HF_USERNAME}/tourism-package-prediction"
LOCAL_DATA_PATH = "tourism_project/data/tourism.csv"

api = HfApi(token=HF_TOKEN)

create_repo(
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    private=False,
    exist_ok=True,
    token=HF_TOKEN,
)

api.upload_file(
    path_or_fileobj=LOCAL_DATA_PATH,
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
)

print(f"Raw dataset uploaded successfully to: https://huggingface.co/datasets/{DATASET_REPO_ID}")
