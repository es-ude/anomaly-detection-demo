import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torchvision.transforms.functional import crop

from src.anomaly_detector import AnomalyDetector, DetectionResult

IMAGE_PATH = Path(os.environ["COOKIE_DATASET_DIR"]) / "test" / "bad"
SAVED_MODEL = Path(os.environ["COOKIE_SAVED_MODEL"])
INFERENCE_IMG_SIZE = int(os.environ["IMAGE_WIDTH"]), int(os.environ["IMAGE_HEIGHT"])
DEVICE = torch.device(os.environ["DEVICE"])
_AREA_TO_CROP = (180, 490, 900, 900)


def main() -> None:
    all_images = list(IMAGE_PATH.glob("*.jpg"))

    anomaly_detector = AnomalyDetector(
        saved_model=SAVED_MODEL,
        input_img_size=(900, 900),
        inference_img_size=INFERENCE_IMG_SIZE,
        device=DEVICE,
    )
    anomaly_detector.load_model()

    image = _load_image(all_images[0])
    result = anomaly_detector.detect(image)

    _plot(result)


def _load_image(path: Path) -> npt.NDArray[np.uint8]:
    with Image.open(path, "r") as img:
        img = crop(img, *_AREA_TO_CROP)
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
    axs[4].imshow(result.superimposed)
    axs[4].set_title("Superimposed")
    plt.show()


if __name__ == "__main__":
    main()
