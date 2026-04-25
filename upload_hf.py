"""
GUARDIAN — HuggingFace Spaces Upload Utility
=============================================
Uploads the full Guardian project to a HuggingFace Space (Docker SDK).
Required env vars: HF_TOKEN (or HUGGING_FACE_HUB_TOKEN)
Required: pip install huggingface_hub python-dotenv

Usage:
    HF_TOKEN=your_token python upload_hf.py
    # or set HF_TOKEN in .env file
"""

import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

api = HfApi()

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not token:
    raise EnvironmentError(
        "HF_TOKEN not set. Add it to your .env file or export it:\n"
        "  export HF_TOKEN=hf_your_token_here"
    )

# Update this to your HuggingFace Space repo id (username/space-name)
repo_id = os.environ.get("HF_REPO_ID", "Yashashvi-05/Guardian")
repo_type = "space"

print(f"Uploading GUARDIAN to HF Space: {repo_id}...")
try:
    # Ensure the repo exists as a Space; create it if not
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type, token=token)
        print(f"Space {repo_id} found ✓")
    except Exception:
        print(f"Space {repo_id} not found. Creating it...")
        api.create_repo(repo_id=repo_id, repo_type=repo_type, token=token, space_sdk="docker")
        print(f"Space created ✓")

    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        ignore_patterns=[
            "upload_hf.py", "upload_out.txt",
            ".git", "__pycache__", "*.pyc",
            "outputs", "login_output.txt", ".env",
            "guardian/checkpoints",           # large model weights — upload separately
            "guardian/data/*.jsonl",          # session fingerprints (local only)
            "*.zip",
        ],
    )
    print("Upload successful! ✓")
    print(f"View your Space: https://huggingface.co/spaces/{repo_id}")
except Exception as e:
    print(f"Upload failed: {e}")
    raise
