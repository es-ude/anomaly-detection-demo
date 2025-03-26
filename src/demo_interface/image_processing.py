import cv2
import numpy as np

from pathlib import Path
from dataclasses import asdict, dataclass
from abc import ABC, abstractmethod

from src.anomaly_detector import AnomalyDetector, DetectionResult
from src.demo_interface.utils import convert_image_to_bytes


@dataclass
class SingleImageResult:
    """
    Dataclass for only one image result (BaseProcessor and CalibrationProcessor)
    """
    result: bytes | None


@dataclass
class AnomalyResult:
    """Dataclass for anomaly detection result with multiple images."""
    result: bytes | None
    original: bytes
    preprocessed: bytes
    reconstructed: bytes
    residuals: bytes


@dataclass
class ProcessedImagesResult:
    result: bytes | None
    original: bytes | None = None
    preprocessed: bytes | None = None
    reconstructed: bytes | None = None
    residuals: bytes | None = None


class AbstractImageProcessor(ABC):
    """
    Abstract base class for all image processor classes.
    """

    def __init__(self, cropped_frame_length: int = 800):
        self.cropped_frame_length = cropped_frame_length

    def _crop_center_square(self, frame: np.ndarray) -> np.ndarray:
        """
        Crops a square frame from the input frame based on its center and cropped_frame_length.
        """
        height, width, _ = frame.shape
        center = (width // 2, height // 2)
        half_cropped_frame_length = self.cropped_frame_length // 2

        x_min = center[0] - half_cropped_frame_length
        x_max = center[0] + half_cropped_frame_length
        y_min = center[1] - half_cropped_frame_length
        y_max = center[1] + half_cropped_frame_length

        cropped_frame = frame[y_min:y_max, x_min:x_max]

        return cropped_frame

    @abstractmethod
    def _process_frame(self, frame: np.ndarray) -> SingleImageResult | AnomalyResult:
        """
        Abstract method for processing a single frame.
        """
        pass

    def process_frame(self, frame: np.ndarray) -> SingleImageResult | AnomalyResult:
        if frame is None:
            return SingleImageResult(
                result=None,
            )
        cropped_frame = self._crop_center_square(frame)
        return self._process_frame(cropped_frame)


class BasicProcessor(AbstractImageProcessor):
    """
    Simple Processing class for just return the received frame.
    """

    def _process_frame(self, frame: np.ndarray) -> SingleImageResult | AnomalyResult:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return SingleImageResult(result=convert_image_to_bytes(frame))


class CalibrationProcessor(AbstractImageProcessor):

    def _process_frame(self, frame: np.ndarray) -> SingleImageResult | AnomalyResult:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        processed_frame = cv2.circle(
            frame.copy(),
            center=center,
            radius=175,
            color=(0, 0, 255),
            thickness=5
        )
        return SingleImageResult(result=convert_image_to_bytes(processed_frame))


class AnomalyDetectorProcessor(AbstractImageProcessor):

    def __init__(self, inference_img_size: tuple[int, int] = (128, 128)):
        super().__init__()
        self.anomaly_detector = AnomalyDetector(
            saved_model=Path("/Users/florianhettstedt/projects/anomaly-detection-demo/model_checkpoints/cookie/model.pt"),
            input_img_size=(self.cropped_frame_length, self.cropped_frame_length),
            inference_img_size=inference_img_size,
        )
        self.anomaly_detector.load_model()

    def _detect_anomaly(self, frame: np.ndarray) -> DetectionResult:
        return self.anomaly_detector.detect(frame)

    def _process_frame(self, frame: np.ndarray) -> SingleImageResult | AnomalyResult:
        # images are flipped for a more intuitive look ine the ui
        frame = cv2.flip(frame, 1)
        frame = cv2.flip(frame, 0)
        frame = np.array(frame, dtype=np.uint8)
        result = self._detect_anomaly(frame)
        result = asdict(result)

        for key in result.keys():
            img = result[key]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_bytes = convert_image_to_bytes(img)
            result[key] = img_bytes

        processed_images_result = AnomalyResult(**result)
        # processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        return processed_images_result
