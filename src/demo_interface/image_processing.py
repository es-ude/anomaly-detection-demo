from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import torch

from src.anomaly_detection.anomaly_detector import AnomalyDetector, DetectionResult

type Image = npt.NDArray[np.uint8]


@dataclass
class AnomalyResult:
    result: bytes | None
    original: bytes
    preprocessed: bytes
    reconstructed: bytes
    residuals: bytes


class AbstractImageProcessor(ABC):
    def __init__(self, target_image_size: tuple[int, int]) -> None:
        self.target_image_size = target_image_size

    @abstractmethod
    def _process_image(self, image: Image) -> AnomalyResult | bytes:
        pass

    def process_image(self, image: Image) -> AnomalyResult | bytes | None:
        if image is None:
            return None
        cropped_image = _center_crop(image, self.target_image_size)
        return self._process_image(cropped_image)


class BasicProcessor(AbstractImageProcessor):
    def _process_image(self, image: Image) -> bytes:
        return _rgb_image_to_bytes(image)


class CalibrationProcessor(AbstractImageProcessor):
    def _process_image(self, image: Image) -> bytes:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width, _ = image.shape
        processed_frame = cv2.circle(
            image.copy(),
            center=(height // 2, width // 2),
            radius=175,
            color=(0, 0, 255),
            thickness=5,
        )
        return _bgr_image_to_bytes(processed_frame)


class AnomalyDetectorProcessor(AbstractImageProcessor):
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

    def _process_image(self, image: Image) -> AnomalyResult:
        image = _flip(image, horizontal=True, vertical=True)
        result = self._detect_anomaly(image)
        images = {
            field.name: _rgb_image_to_bytes(getattr(result, field.name))
            for field in fields(result)  # type: ignore
        }
        return AnomalyResult(**images)

    def _detect_anomaly(self, image: Image) -> DetectionResult:
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


def _bgr_image_to_bytes(frame: Image) -> bytes:
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to convert image to bytes.")
    return buffer.tobytes()


def _rgb_image_to_bytes(image: Image) -> bytes:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return _bgr_image_to_bytes(image)


def _flip(image: Image, horizontal: bool = False, vertical: bool = False) -> Image:
    image = cv2.flip(image, 1) if horizontal else image
    image = cv2.flip(image, 0) if vertical else image
    return image
