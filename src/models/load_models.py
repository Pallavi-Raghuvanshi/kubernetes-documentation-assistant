#src\models\load_models.py
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

from src.utils.helpers import load_config

ROOT = Path(__file__).resolve().parents[2]
config = load_config()
model_name = config["generation"]["model_name"]
MODEL_DIR = ROOT / "models" / "weights" / model_name


def load_model():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}\n"
            "Run src/models/download_models.py first."
        )

    print(f"Loading tokenizer from models/weights/{model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    print("Loading model into CPU memory...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    model.eval()

    print("Model loaded successfully.")
    return tokenizer, model


if __name__ == "__main__":
    tokenizer, model = load_model()

    print(f"Device: {next(model.parameters()).device}")
    print(f"Parameters: {model.num_parameters():,}")
