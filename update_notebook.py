from huggingface_hub import login, create_repo, upload_folder

# 1) login with token string or call `login()` to be prompted
login(token="YOUR_HF_TOKEN")

# 2) create the repo (set repo_type to "model", "dataset", or "space")
create_repo(repo_id="codewithjarair/video_dubber", repo_type="model", exist_ok=True)

# 3) upload the folder; exclude big files via ignore_patterns
upload_folder(
    folder_path=".",                       # local project root
    repo_id="codewithjarair/video_dubber",
    repo_type="model",
    ignore_patterns=["**/*.pth","**/*.pt","**/checkpoints/**","**/temp/**"]
)