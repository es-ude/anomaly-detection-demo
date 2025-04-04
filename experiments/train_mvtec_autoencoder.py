import os
from pathlib import Path

from src.datasets.mvtec_ad import MVTecAD
from src.model import Autoencoder
from src.preprocessing import TrainingPreprocessing
from src.training import train_autoencoder
from torch.utils.data import Dataset, random_split
from torchsummary import summary
from torchvision.transforms.v2 import RandomErasing

from experiments.definitions import (
    DEVICE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_WORKERS,
    save_history,
    save_model,
    save_version,
)

SAVED_MODEL_FILE = Path(os.environ["MVTEC_SAVED_MODEL"])
HISTORY_FILE = Path(os.environ["MVTEC_HISTORY_FILE"])


def main() -> None:
    model = Autoencoder()
    summary(model, input_size=(1, IMAGE_WIDTH, IMAGE_HEIGHT), device="cpu")

    ds_train, ds_test = _autoencoder_datasets()

    history = train_autoencoder(
        model=model,
        ds_train=ds_train,
        ds_test=ds_test,
        batch_size=32,
        epochs=7000,
        learning_rate=1e-3,
        weight_decay=0,
        augment_input_image=RandomErasing(p=1, scale=(0.25, 0.25), value="random"),
        num_workers=NUM_WORKERS,
        device=DEVICE,
    )

    save_model(model, SAVED_MODEL_FILE)
    save_history(history, HISTORY_FILE)


def _autoencoder_datasets() -> list[Dataset]:
    ds = MVTecAD(
        dataset_dir=Path(os.environ["MVTEC_DATASET_DIR"]),
        object=os.environ["MVTEC_OBJECT"],
        training_set=True,
        anomalies=["good"],
        sample_transform=TrainingPreprocessing(IMAGE_WIDTH, IMAGE_HEIGHT),
        in_memory=True,
    )
    return random_split(ds, lengths=[0.8, 0.2])


if __name__ == "__main__":
    main()
