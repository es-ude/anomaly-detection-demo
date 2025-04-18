import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch

from camera import Camera
from src.anomaly_detection.anomaly_detector import AnomalyDetector
from src.demo_interface.image_processing import AnomalyDetectorProcessor


def bytes_to_cv2_image(byte_data: bytes) -> np.ndarray:
    # Wandelt die Byte-Daten in ein 1D-Numpy-Array um.
    nparr = np.frombuffer(byte_data, np.uint8)
    # Decodiert das Array zu einem Bild.
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


cam = Camera()
anomaly_detector = AnomalyDetector(
    model_file=Path(__file__).parents[2]
    / "src/anomaly_detection/model_checkpoints/cookie/model.pt",
    input_img_size=(800, 800),
    inference_img_size=(128, 128),
    device=torch.device("cpu"),
)

anomaly_detector_processor = AnomalyDetectorProcessor()

while True:
    frame = cam.read_frame()
    if frame is not None:
        detection_result = anomaly_detector_processor.process(frame)
        detection_result = asdict(detection_result)
        detection_result = {
            key: bytes_to_cv2_image(value) for key, value in detection_result.items()
        }

        original = detection_result["original"]
        preprocessed = detection_result["preprocessed"]
        reconstructed = detection_result["reconstructed"]
        residuals = detection_result["residuals"]
        result = detection_result["result"]

        cv2.imshow("Original", original)
        cv2.imshow("Preprocessed", preprocessed)
        cv2.imshow("Reconstructed", reconstructed)
        cv2.imshow("Residuals", residuals)
        cv2.imshow("Result", result)
        cv2.waitKey(1)
        time.sleep(0.1)
