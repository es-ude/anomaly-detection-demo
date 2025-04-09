import os
from pathlib import Path

import pandas as pd
import torch

from src.anomaly_detection.persistence import save_model as save_model_to_disk
from src.anomaly_detection.reproducibility import get_commit_hash

DEVICE = torch.device(os.environ["DEVICE"])
NUM_WORKERS = int(os.environ["NUM_WORKERS"])

IMAGE_WIDTH = int(os.environ["IMAGE_WIDTH"])
IMAGE_HEIGHT = int(os.environ["IMAGE_HEIGHT"])

HISTORY_FILE_NAME = os.environ["HISTORY_FILE_NAME"]
VERSION_FILE_NAME = os.environ["VERSION_FILE_NAME"]
MODEL_FILE_NAME = os.environ["MODEL_FILE_NAME"]
QUANT_MODEL_FILE_NAME = os.environ["QUANT_MODEL_FILE_NAME"]


def history_file(output_dir: Path) -> Path:
    return output_dir / HISTORY_FILE_NAME


def version_file(output_dir: Path) -> Path:
    return output_dir / VERSION_FILE_NAME


def model_file(output_dir: Path) -> Path:
    return output_dir / MODEL_FILE_NAME


def quant_model_file(output_dir: Path) -> Path:
    return output_dir / QUANT_MODEL_FILE_NAME


def _save_model(model: torch.nn.Module, destination: Path) -> None:
    destination.parent.mkdir(exist_ok=True)
    save_model_to_disk(model, destination)


def save_model(model: torch.nn.Module, output_dir: Path) -> None:
    _save_model(model, model_file(output_dir))


def save_quant_model(model: torch.nn.Module, output_dir: Path) -> None:
    _save_model(model, quant_model_file(output_dir))


def save_history(history: dict[str, list[float]], output_dir: Path) -> None:
    destination = history_file(output_dir)
    destination.parent.mkdir(exist_ok=True)
    df_history = pd.DataFrame.from_dict(history)
    df_history.to_csv(destination, index=False)


def save_version(output_dir: Path) -> None:
    destination = version_file(output_dir)
    commit_hash = get_commit_hash()
    if commit_hash is not None:
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(commit_hash)
