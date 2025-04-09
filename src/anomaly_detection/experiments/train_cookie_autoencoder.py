import os
from pathlib import Path

from torch.utils.data import Dataset, random_split
from torchsummary import summary
from torchvision.transforms.v2 import RandomErasing

import src.anomaly_detection.experiments.training_definitions as defs
from src.anomaly_detection.datasets.cookie_ad import CookieAD
from src.anomaly_detection.model import Autoencoder
from src.anomaly_detection.preprocessing import TrainingPreprocessing
from src.anomaly_detection.training import train_autoencoder

OUTPUT_DIR = Path(os.environ["COOKIE_OUTPUT_DIR"])
DATASET_DIR = Path(os.environ["COOKIE_DATASET_DIR"])


def main() -> None:
    model = Autoencoder()
    summary(model, input_size=(1, defs.IMAGE_WIDTH, defs.IMAGE_HEIGHT), device="cpu")

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
        num_workers=defs.NUM_WORKERS,
        device=defs.DEVICE,
    )

    defs.save_model(model, OUTPUT_DIR)
    defs.save_history(history, OUTPUT_DIR)
    defs.save_version(OUTPUT_DIR)


def _autoencoder_datasets() -> list[Dataset]:
    ds = CookieAD(
        dataset_dir=DATASET_DIR,
        training_set=True,
        sample_transform=TrainingPreprocessing(defs.IMAGE_WIDTH, defs.IMAGE_HEIGHT),
        in_memory=True,
    )
    return random_split(ds, lengths=[0.8, 0.2])


if __name__ == "__main__":
    main()
