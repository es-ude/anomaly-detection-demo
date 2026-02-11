from time import sleep

import numpy as np
from libcamera import controls
from picamera2 import Picamera2

from demo.camera.image import Image

from .camera import Camera


class PiCamera(Camera):
    def __init__(self, cam_port: int | str, width: int = 1920, height: int = 1080):
        self.camera = Picamera2(camera_num=cam_port)
        capture_config = self.camera.create_still_configuration(
            {"size": (width, height), "format": "RGB888"}
        )
        self.camera.configure(capture_config)
        self.camera.start()

        self.camera.set_controls({"AfMode": controls.AfModeEnum.Auto})
        sleep(1)
        if not self.camera.autofocus_cycle():
            print("Autofocus failed")

        self.open = True

    def is_opened(self) -> bool:
        return self.camera is not None and self.open

    def read_frame(self) -> Image | None:
        image: np.ndarray = self.camera.capture()

        return image

    def release(self) -> None:
        self.camera.stop()
        self.camera.close()
        self.open = False
