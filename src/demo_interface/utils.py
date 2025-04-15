import base64
from pathlib import Path

import cv2
import numpy as np


def convert_image_to_bytes(frame: np.ndarray) -> bytes | None:
	ret, buffer = cv2.imencode('.jpg', frame)
	if not ret:
		return None
	return buffer.tobytes()


def load_image_as_bytes(image_path: Path) -> bytes:
	with open(image_path, "rb") as file:
		return file.read()


def convert_bytes_to_base64(data: bytes) -> str:
	return base64.b64encode(data).decode('utf-8')