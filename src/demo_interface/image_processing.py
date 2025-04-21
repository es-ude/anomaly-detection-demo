from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt
import torch

from src.anomaly_detection.anomaly_detector import AnomalyDetector, DetectionResult

type Image = npt.NDArray[np.uint8]


class ImageProcessor(Protocol):
    def process(self, image: Image) -> DetectionResult | Image | None: ...


class _BaseImageProcessor(ABC):
    def __init__(
        self,
        target_image_size: tuple[int, int],
        flip_horizontal: bool = True,
        flip_vertical: bool = True,
    ) -> None:
        self.target_image_size = target_image_size
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical

    @abstractmethod
    def _process(self, image: Image) -> DetectionResult | Image:
        pass

    def process(self, image: Image) -> DetectionResult | Image | None:
        if image is None:
            return None
        cropped_image = _center_crop(image, self.target_image_size)
        flipped_image = _flip(cropped_image, self.flip_horizontal, self.flip_vertical)
        return self._process(flipped_image)


class BasicProcessor(_BaseImageProcessor):
    def _process(self, image: Image) -> Image:
        return image


class CalibrationProcessor(_BaseImageProcessor):
    def _process(self, image: Image) -> Image:
        height, width, _ = image.shape
        image_with_circle = cv2.circle(
            img=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            center=(height // 2, width // 2),
            radius=175,
            color=(0, 0, 255),
            thickness=5,
        )
        return cv2.cvtColor(image_with_circle, cv2.COLOR_BGR2RGB)


class AnomalyDetectorProcessor(_BaseImageProcessor):
    def __init__(
        self,
        model_file: Path,
        target_image_size: tuple[int, int],
        inference_image_size: tuple[int, int],
    ) -> None:
        super().__init__(target_image_size)
        self.anomaly_detector = AnomalyDetector(
            model_file=model_file,
            input_image_size=self.target_image_size,
            inference_image_size=inference_image_size,
            device=torch.device("cpu"),
        )
        self.anomaly_detector.load_model()

    def _process(self, image: Image) -> DetectionResult:
        return self.anomaly_detector.detect(image)


def _center_crop(image: Image, image_size: tuple[int, int]) -> Image:
    h_input, w_input, _ = image.shape
    h_output, w_output = image_size

    x_mid, y_mid = w_input // 2, h_input // 2
    x_delta, y_delta = w_output // 2, h_output // 2
    x_min = x_mid - x_delta
    x_max = x_mid + x_delta
    y_min = y_mid - y_delta
    y_max = y_mid + y_delta

    cropped_frame = image[y_min:y_max, x_min:x_max]

    return cropped_frame


def _flip(image: Image, horizontal: bool = False, vertical: bool = False) -> Image:
    image = cv2.flip(image, 1) if horizontal else image
    image = cv2.flip(image, 0) if vertical else image
    return image
