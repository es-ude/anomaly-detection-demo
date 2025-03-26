from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms.v2.functional as tv_func

from src.model import Autoencoder
from src.persistence import load_model
from src.preprocessing import InferencePreprocessing


@dataclass
class DetectionResult:
    original: npt.NDArray[np.uint8]
    preprocessed: npt.NDArray[np.uint8]
    reconstructed: npt.NDArray[np.uint8]
    residuals: npt.NDArray[np.uint8]
    result: npt.NDArray[np.uint8]


class AnomalyDetector:
    def __init__(
        self,
        saved_model: Path,
        input_img_size: tuple[int, int],
        inference_img_size: tuple[int, int],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._saved_model = saved_model
        self._input_img_size = list(input_img_size)
        self._device = device
        self._model = None
        self._preprocessing = InferencePreprocessing(*inference_img_size)

    def load_model(self) -> None:
        self._model = Autoencoder()
        load_model(self._model, self._saved_model)
        # self._model = torch.compile(self._model, backend="aot_eager")
        self._model.eval()
        self._model.to(self._device)

    def detect(self, image: npt.NDArray[np.uint8]) -> DetectionResult:
        if self._model is None:
            self.load_model()

        preprocessed_image = self._preprocessing(image)
        reconstructed_image = self._perform_inference(preprocessed_image)
        residuals = _compute_residuals(preprocessed_image, reconstructed_image)
        resized_residuals = self._resize_residuals(residuals)
        return DetectionResult(
            original=image,
            preprocessed=_to_rgb(_to_numpy(preprocessed_image)),
            reconstructed=_to_rgb(_to_numpy(reconstructed_image)),
            residuals=_apply_colormap(_to_numpy(residuals)),
            result=_superimpose(image, _to_numpy(resized_residuals)),
        )

    def _perform_inference(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self._device)
        with torch.no_grad():
            result = self._model(image)
        return result.cpu()

    def _resize_residuals(self, residuals: torch.Tensor) -> torch.Tensor:
        return tv_func.resize(
            residuals,
            size=self._input_img_size,
            interpolation=tv_func.InterpolationMode.BILINEAR,
        )


def _compute_residuals(
    original_image: torch.Tensor, reconstructed_image: torch.Tensor
) -> torch.Tensor:
    return (original_image - reconstructed_image).abs()


def _superimpose(
    image: npt.NDArray[np.uint8], residuals: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    residuals = _apply_colormap(residuals)
    combined = cv2.addWeighted(image, 0.4, residuals, 0.6, 0)
    return cv2.cvtColor(combined, cv2.COLOR_RGBA2RGB)


def _to_rgb(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def _apply_colormap(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    heatmap_image = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
    return cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2RGB)


def _to_numpy(image: torch.Tensor) -> npt.NDArray[np.uint8]:
    return np.array(image.movedim(0, -1) * 255, dtype=np.uint8).squeeze()
