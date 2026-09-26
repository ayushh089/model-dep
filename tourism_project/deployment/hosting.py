"""Creates/updates a Hugging Face Space and uploads the app files to it."""
import os
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
SPACE_REPO_ID = f"{HF_USERNAME}/tourism-package-prediction"

api = HfApi(token=HF_TOKEN)

create_repo(
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    space_sdk="docker",
    private=False,
    exist_ok=True,
    token=HF_TOKEN,
)

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    token=HF_TOKEN,
    allow_patterns=["app.py", "Dockerfile", "requirements.txt", "README.md"],
)

print(f"App deployed. Visit: https://huggingface.co/spaces/{SPACE_REPO_ID}")
