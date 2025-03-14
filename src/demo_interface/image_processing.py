import cv2
import numpy as np


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

        cropped_frame = frame[y_min:y_max + 1, x_min:x_max + 1]

        return cropped_frame

    @staticmethod
    def _convert_image_to_bytes(frame: np.ndarray) -> bytes | None:
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        return buffer.tobytes()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        processed_frame = cv2.circle(
            frame.copy(),
            center=center,
            radius=30,
            color=(255, 0, 0),
            thickness=5
        )
        return processed_frame

    def process_frame(self, frame: np.ndarray) -> bytes | None:
        if frame is None:
            return None

        cropped_frame = self._crop_quadratic__centered_frame(frame)
        processed_frame = self._process_frame(cropped_frame)

        return self._convert_image_to_bytes(processed_frame)


class CookieCalibrator(ImageBaseProcessor):

    def __init__(self):
        super().__init__()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        processed_frame = cv2.circle(
            frame.copy(),
            center=center,
            radius=250,
            color=(0, 0, 255),
            thickness=5
        )
        return processed_frame


class AnomalyDetector(ImageBaseProcessor):

    def __init__(self):
        super().__init__()

    def _detect_cookie(self, frame: np.ndarray) -> bool:
        # placeholder for classificator model
        return True

    def _detect_anomaly(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame