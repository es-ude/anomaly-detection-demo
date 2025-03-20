import cv2
import numpy as np

from pathlib import Path
from dataclasses import asdict, dataclass, fields

from src.anomaly_detector import AnomalyDetector, DetectionResult


@dataclass
class ProcessedImagesResult:
    result: bytes | None
    original: bytes | None = None
    preprocessed: bytes | None = None
    reconstructed: bytes | None = None
    residuals: bytes | None = None


class ImageBaseProcessor:
    def __init__(self, cropped_frame_length: int = 800):
        self.cropped_frame_length = cropped_frame_length

    def _crop_quadratic__centered_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width, _ = frame.shape
        center = (width // 2, height // 2)
        half_cropped_frame_length = self.cropped_frame_length // 2

        x_min = center[0] - half_cropped_frame_length
        x_max = center[0] + half_cropped_frame_length
        y_min = center[1] - half_cropped_frame_length
        y_max = center[1] + half_cropped_frame_length

        cropped_frame = frame[y_min:y_max, x_min:x_max]

        return cropped_frame

    @staticmethod
    def _convert_image_to_bytes(frame: np.ndarray) -> bytes | None:
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        return buffer.tobytes()

    def _process_frame(self, frame: np.ndarray) -> ProcessedImagesResult:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        processed_frame = cv2.circle(
            frame.copy(),
            center=center,
            radius=30,
            color=(255, 0, 0),
            thickness=5
        )
        return ProcessedImagesResult(
            result=self._convert_image_to_bytes(processed_frame),
        )

    def process_frame(self, frame: np.ndarray) -> ProcessedImagesResult:
        if frame is None:
            return None

        cropped_frame = self._crop_quadratic__centered_frame(frame)
        processed_images_result = self._process_frame(cropped_frame)

        return processed_images_result


class CookieCalibrator(ImageBaseProcessor):

    def __init__(self):
        super().__init__()

    def _process_frame(self, frame: np.ndarray) -> ProcessedImagesResult:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        processed_frame = cv2.circle(
            frame.copy(),
            center=center,
            radius=250,
            color=(0, 0, 255),
            thickness=5
        )
        return ProcessedImagesResult(
            result=self._convert_image_to_bytes(processed_frame),
        )


class ImageAnomalyDetector(ImageBaseProcessor):

    def __init__(self):
        super().__init__()
        self.anomaly_detector = AnomalyDetector(
            saved_model=Path("/Users/florianhettstedt/projects/anomaly-detection-demo/model_checkpoints/cookie/model.pt"),
            input_img_size=(800, 800),
            inference_img_size=(128, 128),
        )
        self.anomaly_detector.load_model()

    def _detect_anomaly(self, frame: np.ndarray) -> DetectionResult:
        return self.anomaly_detector.detect(frame)


    def _process_frame(self, frame: np.ndarray) -> ProcessedImagesResult:
        frame = np.array(frame, dtype=np.uint8)
        detection_result = self._detect_anomaly(frame)
        detection_result = asdict(detection_result)

        for key in detection_result.keys():
            img = detection_result[key]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_bytes = self._convert_image_to_bytes(img)
            detection_result[key] = img_bytes

        processed_images_result = ProcessedImagesResult(**detection_result)
        # processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        return processed_images_result
