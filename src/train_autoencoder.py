import os
from functools import partial
from pathlib import Path

import torch
from torchsummary import summary

from src.datasets.mvtec_ad import MVTecAD
from src.model import Autoencoder
from src.persistence import save_model
from src.preprocessing import ImagePreprocessing
from src.reproducibility import get_commit_hash
from src.training import train_autoencoder

SAVED_MODEL = Path(os.environ["SAVED_MODEL"])
VERSION_FILE = Path(os.environ["VERSION_FILE"])


def main() -> None:
    model = Autoencoder()
    summary(model, input_size=(3, 256, 256))

    ds_train, ds_test = _autoencoder_datasets()

    train_autoencoder(
        model=model,
        ds_train=ds_train,
        ds_test=ds_test,
        batch_size=32,
        epochs=10,
        learning_rate=1e-3,
        num_workers=0,
        device=torch.device(os.environ["DEVICE"]),
    )

    _save_model(model)
    _save_version()


def _autoencoder_datasets() -> tuple[MVTecAD, MVTecAD]:
    create_ds = partial(
        MVTecAD,
        dataset_dir=Path(os.environ["AD_DATASET_DIR"]),
        object=os.environ["AD_OBJECT"],
        anomalies=["good"],
        sample_transform=ImagePreprocessing(
            target_img_width=int(os.environ["IMAGE_WIDTH"]),
            target_img_height=int(os.environ["IMAGE_HEIGHT"]),
        ),
    )
    return create_ds(training_set=True), create_ds(training_set=False)


def _save_model(model: torch.nn.Module) -> None:
    SAVED_MODEL.parent.mkdir(exist_ok=True)
    save_model(model, SAVED_MODEL)


def _save_version() -> None:
    commit_hash = get_commit_hash()
    if commit_hash is not None:
        VERSION_FILE.parent.mkdir(exist_ok=True)
        VERSION_FILE.write_text(commit_hash)


if __name__ == "__main__":
    main()
