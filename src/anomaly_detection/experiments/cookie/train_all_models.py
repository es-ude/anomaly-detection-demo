import os
from pathlib import Path

from torch.utils.data import Dataset, random_split
from torchvision.transforms.v2 import RandomErasing

import src.anomaly_detection.experiments.training_definitions as defs
from src.anomaly_detection.datasets.cookie_ad import CookieAdDataset
from src.anomaly_detection.datasets.cookie_clf import CookieClfDataset
from src.anomaly_detection.model import Autoencoder, Classifier
from src.anomaly_detection.preprocessing import (
    InferencePreprocessing,
    TrainingPreprocessing,
)
from src.anomaly_detection.training import train_autoencoder, train_classifier

OUTPUT_DIR = Path(os.environ["COOKIE_OUTPUT_DIR"])
AE_DATASET_DIR = Path(os.environ["COOKIE_AE_DATASET_DIR"])
CLF_DATASET_DIR = Path(os.environ["COOKIE_CLF_DATASET_DIR"])


def main() -> None:
    defs.save_version(OUTPUT_DIR / "commit_hash.txt")

    autoencoder = Autoencoder()
    ds_train, ds_val = _autoencoder_datasets()
    history = train_autoencoder(
        model=autoencoder,
        ds_train=ds_train,
        ds_test=ds_val,
        batch_size=32,
        epochs=7000,
        learning_rate=1e-3,
        weight_decay=0,
        augment_input_image=RandomErasing(p=1, scale=(0.25, 0.25), value="random"),  # type: ignore
        num_workers=defs.NUM_WORKERS,
        device=defs.DEVICE,
    )
    defs.save_model(autoencoder, OUTPUT_DIR / "ae_model.pt")
    defs.save_history(history, OUTPUT_DIR / "ae_history.csv")

    classifier = Classifier()
    ds_train, ds_val = _classifier_datasets()
    history = train_classifier(
        classifier=classifier,
        encoder=autoencoder.encoder,
        ds_train=ds_train,
        ds_test=ds_val,
        batch_size=64,
        epochs=100,
        learning_rate=1e-3,
        num_workers=defs.NUM_WORKERS,
        device=defs.DEVICE,
    )
    defs.save_model(classifier, OUTPUT_DIR / "clf_model.pt")
    defs.save_history(history, OUTPUT_DIR / "clf_history.csv")


def _autoencoder_datasets() -> tuple[Dataset, Dataset]:
    ds = CookieAdDataset(
        dataset_dir=AE_DATASET_DIR,
        training_set=True,
        sample_transform=TrainingPreprocessing(defs.IMAGE_HEIGHT, defs.IMAGE_WIDTH),
        in_memory=True,
    )
    return tuple(random_split(ds, lengths=[0.8, 0.2]))  # type: ignore


def _classifier_datasets() -> tuple[Dataset, Dataset]:
    ds_train = CookieClfDataset(
        dataset_dir=CLF_DATASET_DIR,
        training_set=True,
        sample_transform=TrainingPreprocessing(defs.IMAGE_HEIGHT, defs.IMAGE_WIDTH),
    )
    ds_val = CookieClfDataset(
        dataset_dir=CLF_DATASET_DIR,
        training_set=False,
        sample_transform=InferencePreprocessing(defs.IMAGE_HEIGHT, defs.IMAGE_WIDTH),
    )
    return ds_train, ds_val


if __name__ == "__main__":
    main()
