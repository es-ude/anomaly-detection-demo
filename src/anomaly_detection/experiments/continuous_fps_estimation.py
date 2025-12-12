import os
from pathlib import Path
from time import time_ns

import numpy as np
import numpy.typing as npt
import torch

from src.anomaly_detection.anomaly_detector import AnomalyDetector

BATCH_SIZE = 100
IMAGE_WIDTH = int(os.environ["IMAGE_WIDTH"])
IMAGE_HEIGHT = int(os.environ["IMAGE_HEIGHT"])
DEVICE = torch.device(os.environ["DEVICE"])
CKPT_DIR = Path(os.environ["COOKIE_CKPT_DIR"])


def main() -> None:
    model = _get_model()

    while True:
        start_time = time_ns()

        for _ in range(BATCH_SIZE):
            _ = model.detect(_get_image())

        end_time = time_ns()

        print(f"{BATCH_SIZE * 1e9 / (end_time - start_time):.02f} FPS")


def _get_model() -> AnomalyDetector:
    return AnomalyDetector(
        autoencoder_file=CKPT_DIR / "ae_model.pt",
        classifier_file=CKPT_DIR / "clf_model.pt",
        inference_image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        device=DEVICE,
    )


def _get_image() -> npt.NDArray[np.uint8]:
    return np.random.randint(
        low=0, high=256, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    ).astype(np.uint8)


if __name__ == "__main__":
    main()
