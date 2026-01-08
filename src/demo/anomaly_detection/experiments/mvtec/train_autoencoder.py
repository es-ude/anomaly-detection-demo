import os
from pathlib import Path

from torch.utils.data import Dataset, random_split
from torchsummary import summary
from torchvision.transforms.v2 import RandomErasing

import demo.anomaly_detection.experiments.training_definitions as defs
from demo.anomaly_detection.datasets.mvtec_ad import MVTecAdDataset
from demo.anomaly_detection.model import Autoencoder
from demo.anomaly_detection.preprocessing import TrainingPreprocessing
from demo.anomaly_detection.training import train_autoencoder

OUTPUT_DIR = Path(os.environ["MVTEC_OUTPUT_DIR"])
DATASET_DIR = Path(os.environ["MVTEC_DATASET_DIR"])
OBJECT = os.environ["MVTEC_OBJECT"]


def main() -> None:
    defs.save_version(OUTPUT_DIR / "commit_hash.txt")

    model = Autoencoder()
    summary(model, input_size=(1, defs.IMAGE_HEIGHT, defs.IMAGE_WIDTH), device="cpu")

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

    defs.save_model(model, OUTPUT_DIR / "ae_model.pt")
    defs.save_history(history, OUTPUT_DIR / "ae_history.csv")


def _autoencoder_datasets() -> tuple[Dataset, Dataset]:
    ds = MVTecAdDataset(
        dataset_dir=DATASET_DIR,
        object=OBJECT,
        training_set=True,
        anomalies=["good"],
        sample_transform=TrainingPreprocessing(defs.IMAGE_HEIGHT, defs.IMAGE_WIDTH),
        in_memory=True,
    )
    return tuple(random_split(ds, lengths=[0.8, 0.2]))  # type: ignore


if __name__ == "__main__":
    main()
