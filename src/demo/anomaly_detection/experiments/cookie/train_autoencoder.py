import os
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
DATASET_DIR = Path(os.environ["COOKIE_AE_DATASET_DIR"])
FINETUNE_DATASET_DIR = Path(os.environ["COOKIE_AE_FINETUNE_DATASET_DIR"])


def main() -> None:
    defs.save_version(OUTPUT_DIR / "commit_hash.txt")

    model = Autoencoder()
    summary(model, input_size=(1, defs.IMAGE_HEIGHT, defs.IMAGE_WIDTH), device="cpu")

    ds_train, ds_val = _load_base_dataset()

    history = _run_training(model=model, ds_train=ds_train, ds_val=ds_val, epochs=3000)

    defs.save_model(model, OUTPUT_DIR / "ae_base_model.pt")
    defs.save_history(history, OUTPUT_DIR / "ae_base_history.csv")

    ds_train, ds_val = _load_finetune_dataset()

    history = _run_training(model=model, ds_train=ds_train, ds_val=ds_val, epochs=1000)
    model.to("cpu").eval()

    model.determine_decision_boundary(calibration_data=ds_val[:][0], quantile=0.95)

    _write_classification_report(
        autoencoder=model,
        ds=_load_clf_dataset(),
        report_file=OUTPUT_DIR / "classification_report.txt",
    )
    defs.save_model(model, OUTPUT_DIR / "ae_model.pt")
    defs.save_history(history, OUTPUT_DIR / "ae_history.csv")


def _cookie_dataset(dataset_dir: Path, training_set: bool) -> CookieAdDataset:
    return CookieAdDataset(
        dataset_dir=dataset_dir,
        training_set=training_set,
        sample_transform=TrainingPreprocessing(defs.IMAGE_HEIGHT, defs.IMAGE_WIDTH),
        in_memory=True,
    )


def _load_base_dataset() -> tuple[Dataset, Dataset]:
    ds = _cookie_dataset(dataset_dir=DATASET_DIR, training_set=True)
    ds_train, ds_val = random_split(ds, lengths=[0.8, 0.2])
    return ds_train, ds_val


def _load_finetune_dataset() -> tuple[Dataset, Dataset]:
    ds_train = _cookie_dataset(dataset_dir=FINETUNE_DATASET_DIR, training_set=True)
    ds_val = _cookie_dataset(dataset_dir=FINETUNE_DATASET_DIR, training_set=False)
    return ds_train, ds_val


def _load_clf_dataset() -> Dataset:
    return _cookie_dataset(dataset_dir=DATASET_DIR, training_set=False)


def _run_training(
    model: torch.nn.Module, ds_train: Dataset, ds_val: Dataset, epochs: int
) -> dict[str, list[float]]:
    return train_autoencoder(
        model=model,
        ds_train=ds_train,
        ds_test=ds_val,
        batch_size=32,
        epochs=epochs,
        learning_rate=1e-3,
        weight_decay=0,
        augment_input_image=RandomErasing(p=1, scale=(0.25, 0.25), value="random"),  # type: ignore
        num_workers=defs.NUM_WORKERS,
        device=defs.DEVICE,
    )


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
