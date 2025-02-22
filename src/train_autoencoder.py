import os
from functools import partial
from pathlib import Path

import torch

from src.datasets.mvtec_ad import MVTecAD
from src.model import Autoencoder
from src.persistence import save_model
from src.preprocessing import ImagePreprocessing
from src.reproducibility import get_commit_hash
from src.training import train_autoencoder

OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    model = Autoencoder()
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
    save_model(model, destination=OUTPUT_DIR / "model.pt")


def _save_version() -> None:
    commit_hash = get_commit_hash()
    if commit_hash is not None:
        version_file = OUTPUT_DIR / "commit_hash.txt"
        version_file.write_text(commit_hash)


if __name__ == "__main__":
    main()
