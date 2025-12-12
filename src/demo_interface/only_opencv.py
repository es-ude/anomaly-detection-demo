from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
from camera import Camera

from src.demo_interface.image_processing import AnomalyDetectorProcessor


def bytes_to_cv2_image(byte_data: bytes) -> np.ndarray:
    nparr = np.frombuffer(byte_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


cam = Camera(cam_port=0)
anomaly_detector_processor = AnomalyDetectorProcessor(
    model_file=Path(__file__).parents[2]
    / "src/anomaly_detection/model_checkpoints/cookie/model.pt",
    target_image_size=(800, 800),
    inference_image_size=(128, 128),
)

while True:
    frame = cam.read_frame()
    if frame is not None:
        detection_result = anomaly_detector_processor.process(frame)
        result = {
            field.name: bytes_to_cv2_image(getattr(detection_result, field.name))
            for field in fields(detection_result)  # type: ignore
        }

        original = result["original"]
        preprocessed = result["preprocessed"]
        reconstructed = result["reconstructed"]
        residuals = result["residuals"]
        superimposed = result["superimposed"]

        # cv2.imshow("Original", original)
        # cv2.imshow("Preprocessed", preprocessed)
        # cv2.imshow("Reconstructed", reconstructed)
        # cv2.imshow("Residuals", residuals)
        cv2.imshow("Superimposed", superimposed)
        cv2.waitKey(1)
