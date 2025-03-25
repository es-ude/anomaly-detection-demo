import os
from pathlib import Path

import pandas as pd
import torch
from src.persistence import save_model as save_model_to_disk
from src.reproducibility import get_commit_hash

IMAGE_WIDTH = int(os.environ["IMAGE_WIDTH"])
IMAGE_HEIGHT = int(os.environ["IMAGE_HEIGHT"])
DEVICE = torch.device(os.environ["DEVICE"])
NUM_WORKERS = int(os.environ["NUM_WORKERS"])


def save_model(model: torch.nn.Module, destination: Path) -> None:
    destination.parent.mkdir(exist_ok=True)
    save_model_to_disk(model, destination)


def save_history(history: dict[str, list[float]], destination: Path) -> None:
    destination.parent.mkdir(exist_ok=True)
    df_history = pd.DataFrame.from_dict(history)
    df_history.to_csv(destination, index=False)


def save_version(destination: Path) -> None:
    commit_hash = get_commit_hash()
    if commit_hash is not None:
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(commit_hash)
