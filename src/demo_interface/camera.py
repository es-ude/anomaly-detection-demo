import cv2
import numpy as np

class Camera:

	def __init__(self, width: int = 1920, height: int = 1080, cam_port: int = 0):
		"""
		Initialises the camera on the specified port with the given width and height.
		"""
		self.capture = cv2.VideoCapture(cam_port)
		self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

	def is_opened(self) -> bool:
		"""
		Checks if the camera is opened.
		"""
		return self.capture.isOpened()

	def read_frame(self) -> np.ndarray | None:
		"""
		Reads the current frame from the camera as a cv2 Image (= numpy array).
		:return:
		"""
		ret, frame = self.capture.read()
		if ret:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		return frame if ret else None

	def release(self) -> None:
		"""
		Releases the camera.
		"""
		self.capture.release()

