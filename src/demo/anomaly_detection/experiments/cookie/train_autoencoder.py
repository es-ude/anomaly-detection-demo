import os
from functools import partial
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, random_split
from torchsummary import summary
from torchvision.transforms.v2 import RandomErasing

import demo.anomaly_detection.experiments.training_definitions as defs
from demo.anomaly_detection.datasets.cookie_ad import CookieAdDataset
from demo.anomaly_detection.model import Autoencoder
from demo.anomaly_detection.preprocessing import TrainingPreprocessing
from demo.anomaly_detection.training import train_autoencoder

OUTPUT_DIR = Path(os.environ["COOKIE_OUTPUT_DIR"])
AE_DATASET_DIR = Path(os.environ["COOKIE_AE_DATASET_DIR"])
CLF_DATASET_DIR = Path(os.environ["COOKIE_CLF_DATASET_DIR"])


def main() -> None:
    defs.save_version(OUTPUT_DIR / "commit_hash.txt")

    model = Autoencoder()
    summary(model, input_size=(1, defs.IMAGE_HEIGHT, defs.IMAGE_WIDTH), device="cpu")

    (ds_train, ds_val), ds_clf = _autoencoder_datasets()

    history = train_autoencoder(
        model=model,
        ds_train=ds_train,
        ds_test=ds_val,
        batch_size=32,
        epochs=3000,
        learning_rate=1e-3,
        weight_decay=0,
        augment_input_image=RandomErasing(p=1, scale=(0.25, 0.25), value="random"),  # type: ignore
        num_workers=defs.NUM_WORKERS,
        device=defs.DEVICE,
    )
    model.to("cpu").eval()

    model.determine_decision_boundary(calibration_data=ds_val[:][0], quantile=0.95)

    _write_classification_report(
        autoencoder=model,
        ds=ds_clf,
        report_file=OUTPUT_DIR / "classification_report.txt",
    )
    defs.save_model(model, OUTPUT_DIR / "ae_model.pt")
    defs.save_history(history, OUTPUT_DIR / "ae_history.csv")


def _autoencoder_datasets() -> tuple[tuple[Dataset, Dataset], Dataset]:
    load_dataset = partial(
        CookieAdDataset,
        dataset_dir=AE_DATASET_DIR,
        sample_transform=TrainingPreprocessing(defs.IMAGE_HEIGHT, defs.IMAGE_WIDTH),
        in_memory=True,
    )
    ds_ae = load_dataset(training_set=True)
    ds_clf = load_dataset(training_set=False)
    ds_ae_train, ds_ae_val = random_split(ds_ae, lengths=[0.8, 0.2])
    return (ds_ae_train, ds_ae_val), ds_clf


def _write_classification_report(
    autoencoder: Autoencoder, ds: Dataset, report_file: Path
) -> None:
    samples = torch.stack([x for x, _ in ds])
    labels = torch.stack([y for _, y in ds])
    predictions = autoencoder.classify(samples)
    report = classification_report(
        labels, predictions, target_names=["good", "damaged"]
    )
    report_file.write_text(str(report))


if __name__ == "__main__":
    main()
