from datetime import datetime
from pathlib import Path

import torch
from torch.nn import Module
from torch.utils.data import Dataset, random_split

import demo.anomaly_detection.experiments.training_definitions as defs
from demo.anomaly_detection.anomaly_detector import AnomalyDetector
from demo.anomaly_detection.datasets.cookie_ad import CookieAdDataset
from demo.anomaly_detection.preprocessing import TrainingPreprocessing
from demo.anomaly_detection.training import train_autoencoder


def train(
    anomaly_detector: AnomalyDetector, dataset_dir: Path, device: str, epochs: int = 10
) -> None:
    ds_train, ds_val = _load_dataset(dataset_dir=dataset_dir)
    snapshot_dir = Path(__file__).parent.parent.joinpath("snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    ae_anomaly_detector = anomaly_detector._autoencoder
    if ae_anomaly_detector is None:
        raise ValueError("Autoencoder model is not loaded in the anomaly detector.")

    history = _run_training(
        model=ae_anomaly_detector,
        ds_train=ds_train,
        ds_val=ds_val,
        epochs=epochs,
        device=device,
    )
    _save_model(
        model=ae_anomaly_detector,
        history=history,
        base_dir=snapshot_dir.joinpath(f"model_{int(datetime.now().timestamp())}"),
    )


def _load_dataset(dataset_dir) -> tuple[Dataset, Dataset]:
    ds = CookieAdDataset(
        dataset_dir=dataset_dir,
        training_set=True,
        sample_transform=TrainingPreprocessing(defs.IMAGE_HEIGHT, defs.IMAGE_WIDTH),
        in_memory=True,
    )
    ds_train, ds_val = random_split(ds, lengths=[0.6, 0.4])
    return ds_train, ds_val


def _run_training(
    model: Module, ds_train: Dataset, ds_val: Dataset, epochs: int, device: str
) -> dict[str, list[float]]:
    return train_autoencoder(
        model=model,
        ds_train=ds_train,
        ds_test=ds_val,
        batch_size=32,
        epochs=epochs,
        learning_rate=1e-3,
        weight_decay=1e-5,
        augment_input_image=lambda x: x + 0.05 * torch.randn_like(x),
        num_workers=4,
        device=torch.device(device),
    )


def _save_model(model: Module, history: dict[str, list[float]], base_dir: Path) -> None:
    defs.save_model(model, base_dir.joinpath("ae_model.pt"))
    defs.save_history(history, base_dir.joinpath("ae_history.csv"))
