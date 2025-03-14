import cv2
import numpy as np

class Camera:

	def __init__(self, width: int = 1920, height: int = 1080, cam_port: int = 0):
		self.capture = cv2.VideoCapture(cam_port)
		self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

	def is_opened(self) -> bool:
		return self.capture.isOpened()

	def read_frame(self) -> np.ndarray | None:
		ret, frame = self.capture.read()
		return frame if ret else None

	def release(self) -> None:
		self.capture.release()

