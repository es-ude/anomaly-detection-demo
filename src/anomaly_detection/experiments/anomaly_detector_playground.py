import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from src.anomaly_detection.anomaly_detector import AnomalyDetector, DetectionResult

IMAGE_WIDTH = int(os.environ["IMAGE_WIDTH"])
IMAGE_HEIGHT = int(os.environ["IMAGE_HEIGHT"])
DEVICE = torch.device(os.environ["DEVICE"])

CKPT_DIR = Path(os.environ["COOKIE_CKPT_DIR"])
MODEL_FILE_NAME = os.environ["MODEL_FILE_NAME"]

DATASET_DIR = Path(os.environ["COOKIE_DATASET_DIR"])
IMAGE_PATH = DATASET_DIR / "test" / "bad"


def main() -> None:
    all_images = list(IMAGE_PATH.glob("*.jpg"))

    anomaly_detector = AnomalyDetector(
        model_file=CKPT_DIR / MODEL_FILE_NAME,
        input_img_size=(800, 800),
        inference_img_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        device=DEVICE,
    )
    anomaly_detector.load_model()

    image = _load_image(all_images[0])
    result = anomaly_detector.detect(image)

    _plot(result)


def _load_image(path: Path) -> npt.NDArray[np.uint8]:
    with Image.open(path, "r") as img:
        return np.array(img, dtype=np.uint8)


def _plot(result: DetectionResult) -> None:
    fig, axs = plt.subplots(nrows=1, ncols=5, tight_layout=True, figsize=(15, 3))
    axs[0].imshow(result.original)
    axs[0].set_title("Original")
    axs[1].imshow(result.preprocessed)
    axs[1].set_title("Preprocessed")
    axs[2].imshow(result.reconstructed)
    axs[2].set_title("Reconstructed")
    axs[3].imshow(result.residuals)
    axs[3].set_title("Residuals")
    axs[4].imshow(result.result)
    axs[4].set_title("Superimposed")
    plt.show()


if __name__ == "__main__":
    main()
