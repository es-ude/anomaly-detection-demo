import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image
from torchvision.transforms.functional import crop

from src.anomaly_detector import AnomalyDetector

IMAGE_PATH = Path(os.environ["COOKIE_DATASET_DIR"]) / "test" / "bad"
SAVED_MODEL = Path(os.environ["COOKIE_SAVED_MODEL"])
INFERENCE_IMG_SIZE = int(os.environ["IMAGE_WIDTH"]), int(os.environ["IMAGE_HEIGHT"])
_AREA_TO_CROP = (180, 490, 900, 900)


def main() -> None:
    all_images = list(IMAGE_PATH.glob("*.jpg"))

    anomaly_detector = AnomalyDetector(
        saved_model=SAVED_MODEL,
        input_img_size=(900, 900),
        inference_img_size=INFERENCE_IMG_SIZE,
    )
    anomaly_detector.load_model()

    image = _load_image(all_images[0])
    anomaly_image = anomaly_detector.detect(image)

    _plot(anomaly_image)


def _load_image(path: Path) -> npt.NDArray[np.uint8]:
    with Image.open(path, "r") as img:
        img = crop(img, *_AREA_TO_CROP)
        return np.array(img, dtype=np.uint8)


def _plot(image: npt.NDArray[np.uint8]) -> None:
    fig = plt.figure(frameon=False)
    plt.axis("off")
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    main()
