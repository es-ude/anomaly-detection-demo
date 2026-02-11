import cv2

from .camera import Camera
from .image import Image, convert_bgr_to_rgb


class CVCamera(Camera):
    def __init__(self, cam_port: int | str, width: int = 1920, height: int = 1080):
        self.capture = cv2.VideoCapture(cam_port)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def is_opened(self) -> bool:
        return self.capture.isOpened()

    def read_frame(self) -> Image | None:
        success, frame = self.capture.read()

        if not success:
            return None

        return convert_bgr_to_rgb(frame)

    def release(self) -> None:
        self.capture.release()
