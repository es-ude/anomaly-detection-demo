import os
from pathlib import Path

import pandas as pd
import torch

from src.persistence import save_model as save_model_to_disk
from src.reproducibility import get_commit_hash

IMAGE_WIDTH = int(os.environ["IMAGE_WIDTH"])
IMAGE_HEIGHT = int(os.environ["IMAGE_HEIGHT"])


def save_model(model: torch.nn.Module) -> None:
    saved_model = Path(os.environ["SAVED_MODEL"])
    saved_model.parent.mkdir(exist_ok=True)
    save_model_to_disk(model, saved_model)


def save_history(history: dict[str, list[float]]) -> None:
    history_file = Path(os.environ["HISTORY_FILE"])
    df_history = pd.DataFrame.from_dict(history)
    df_history.to_csv(history_file, index=False)


def save_version() -> None:
    version_file = Path(os.environ["VERSION_FILE"])
    commit_hash = get_commit_hash()
    if commit_hash is not None:
        version_file.parent.mkdir(exist_ok=True)
        version_file.write_text(commit_hash)
