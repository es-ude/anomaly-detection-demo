from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.v2.functional as tv_func

from src.anomaly_detection.model import Autoencoder
from src.anomaly_detection.persistence import load_model
from src.anomaly_detection.preprocessing import InferencePreprocessing

type Image = cv2.typing.MatLike


@dataclass
class DetectionResult:
    original: Image
    preprocessed: Image
    reconstructed: Image
    residuals: Image
    superimposed: Image
    damaged: Optional[bool]


class AnomalyDetector:
    def __init__(
        self,
        autoencoder_file: Path,
        use_classifier: bool,
        inference_image_size: tuple[int, int],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.autoencoder_file = autoencoder_file
        self.use_classifier = use_classifier
        self.inference_image_size = inference_image_size
        self.device = device
        self._autoencoder = None
        self._preprocessing = InferencePreprocessing(*inference_image_size)

    def load_model(self) -> None:
        self._autoencoder = Autoencoder()
        load_model(self._autoencoder, self.autoencoder_file)
        self._autoencoder.eval().to(self.device)

    def detect(self, image: Image) -> DetectionResult:
        if self._autoencoder is None:
            self.load_model()

        preprocessed = self._preprocessing(image)
        reconstructed, prediction = self._perform_inference(preprocessed)
        residuals = _compute_residuals(preprocessed, reconstructed)
        resized_residuals = _resize_residuals(residuals, size=image.shape)
        return DetectionResult(
            original=image,
            preprocessed=_to_rgb(_to_numpy(preprocessed)),
            reconstructed=_to_rgb(_to_numpy(reconstructed)),
            residuals=_apply_colormap(_to_numpy(residuals)),
            superimposed=_superimpose(image, _to_numpy(resized_residuals)),
            damaged=int(prediction.item()) == 1 if prediction is not None else None,
        )

    def _perform_inference(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._autoencoder is None:
            raise RuntimeError("Autoencoder not loaded. Call load_model() first.")

        image = image.to(self.device)

        with torch.no_grad():
            _, reconstructed = self._autoencoder(image)
            reconstructed = reconstructed.clamp(min=0, max=1).cpu()

            prediction = (
                self._autoencoder.classify(image).cpu() if self.use_classifier else None
            )

        return reconstructed, prediction


def _resize_residuals(residuals: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
    return tv_func.resize(
        residuals, size=list(size), interpolation=tv_func.InterpolationMode.BILINEAR
    )


def _compute_residuals(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> torch.Tensor:
    return (original - reconstructed).abs()


def _superimpose(image: Image, residuals: Image) -> Image:
    residuals = _apply_colormap(residuals)
    combined = cv2.addWeighted(image, 0.4, residuals, 0.6, 0)
    return cv2.cvtColor(combined, cv2.COLOR_RGBA2RGB)


def _to_rgb(image: Image) -> Image:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def _apply_colormap(image: Image) -> Image:
    heatmap_image = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
    return cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2RGB)


def _to_numpy(image: torch.Tensor) -> Image:
    return np.array(image.movedim(0, -1) * 255, dtype=np.uint8).squeeze()
