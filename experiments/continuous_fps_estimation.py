from time import time_ns

import numpy as np
import numpy.typing as npt
import torch
from src.model import Autoencoder
from src.preprocessing import InferencePreprocessing

import definitions as defs

BATCH_SIZE = 100


def main() -> None:
    model = _get_compiled_model()
    preprocess = InferencePreprocessing(defs.IMAGE_WIDTH, defs.IMAGE_HEIGHT)

    while True:
        start_time = time_ns()

        for _ in range(BATCH_SIZE):
            image = _get_image()
            image = preprocess(image)

            with torch.no_grad():
                image = image.to(defs.DEVICE)
                _ = model(image)

        end_time = time_ns()

        print(f"{BATCH_SIZE * 1e9 / (end_time - start_time):.02f} FPS")


def _get_model() -> Autoencoder:
    return Autoencoder().eval().to(defs.DEVICE)


def _get_compiled_model() -> torch.nn.Module:
    return torch.compile(_get_model(), backend="aot_eager")


def _get_image() -> npt.NDArray[np.uint8]:
    return np.random.randint(
        low=0,
        high=256,
        size=(defs.IMAGE_HEIGHT, defs.IMAGE_WIDTH, 3),
    ).astype(np.uint8)


if __name__ == "__main__":
    main()
