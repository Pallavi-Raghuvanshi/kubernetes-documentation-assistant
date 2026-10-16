
from pathlib import Path

from huggingface_hub import snapshot_download

# Project paths
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models" / "weights" / "qwen2.5-1.5b-instruct"

# Hugging Face repository
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def download_model() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model repository: {MODEL_ID}")
    print(f"Local directory: {MODEL_DIR}")
    print("Downloading model files...")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
    )

    print("Model files downloaded successfully.")


if __name__ == "__main__":
    download_model()
