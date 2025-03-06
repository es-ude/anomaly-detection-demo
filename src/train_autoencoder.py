import os
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from torchsummary import summary

from src.datasets.mvtec_ad import MVTecAD
from src.model import Autoencoder
from src.persistence import save_model
from src.preprocessing import TrainingPreprocessing
from src.reproducibility import get_commit_hash
from src.training import train_autoencoder

IMAGE_WIDTH = int(os.environ["IMAGE_WIDTH"])
IMAGE_HEIGHT = int(os.environ["IMAGE_HEIGHT"])


def main() -> None:
    model = Autoencoder()
    summary(model, input_size=(3, IMAGE_WIDTH, IMAGE_HEIGHT), device="cpu")

    ds_train, ds_test = _autoencoder_datasets()

    history = train_autoencoder(
        model=model,
        ds_train=ds_train,
        ds_test=ds_test,
        batch_size=32,
        epochs=500,
        learning_rate=1e-3,
        num_workers=int(os.environ["NUM_WORKERS"]),
        device=torch.device(os.environ["DEVICE"]),
    )

    _save_model(model)
    _save_history(history)
    _save_version()


def _autoencoder_datasets() -> tuple[Dataset, Dataset]:
    ds = MVTecAD(
        dataset_dir=Path(os.environ["AD_DATASET_DIR"]),
        object=os.environ["AD_OBJECT"],
        training_set=True,
        anomalies=["good"],
        sample_transform=TrainingPreprocessing(IMAGE_WIDTH, IMAGE_HEIGHT),
    )
    return cast(tuple[Dataset, Dataset], random_split(ds, lengths=[0.8, 0.2]))


def _save_model(model: torch.nn.Module) -> None:
    saved_model = Path(os.environ["SAVED_MODEL"])
    saved_model.parent.mkdir(exist_ok=True)
    save_model(model, saved_model)


def _save_history(history: dict[str, list[float]]) -> None:
    history_file = Path(os.environ["HISTORY_FILE"])
    df_history = pd.DataFrame.from_dict(history)
    df_history.to_csv(history_file, index=False)


def _save_version() -> None:
    version_file = Path(os.environ["VERSION_FILE"])
    commit_hash = get_commit_hash()
    if commit_hash is not None:
        version_file.parent.mkdir(exist_ok=True)
        version_file.write_text(commit_hash)


if __name__ == "__main__":
    main()
