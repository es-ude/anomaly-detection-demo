import cv2
import numpy as np
import numpy.typing as npt


class Camera:
    def __init__(self, cam_port: int | str, width: int = 1920, height: int = 1080):
        self.capture = cv2.VideoCapture(cam_port)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def is_opened(self) -> bool:
        return self.capture.isOpened()

    def read_frame(self) -> npt.NDArray[np.uint8] | None:
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame if ret else None

    def release(self) -> None:
        self.capture.release()
