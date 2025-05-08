from huggingface_hub import HfApi
import os
# Upload a folder to a Hugging Face Sp
# Import environment variables
from dotenv import load_dotenv
load_dotenv()

api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
api.upload_large_folder(
    folder_path="./outputs",
    repo_id="alejandroparedeslatorre/DeepSeek-R1-Distill-Qwen-1.5B-GRPO",
    repo_type="model",
)