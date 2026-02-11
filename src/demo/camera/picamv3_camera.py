from libcamera import controls
from picamera2 import Picamera2

from demo.camera.image import Image


class Camera:
    def __init__(
        self,
        cam_port: int | str,
        width: int = 1920,
        height: int = 1080,
        lens_position: None | float = None,
    ):
        self.camera = Picamera2(camera_num=cam_port)
        self.config = self.camera.create_video_configuration(
            main={"size": (width, height), "format": "BGR888"}
        )
        self.camera.configure(self.config)
        self.camera.start()
        if lens_position is not None:
            self.camera.set_controls(
                {"AfMode": controls.AfModeEnum.Manual, "LensPosition": lens_position}
            )
        else:
            self.camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        self.is_open = True

    def is_opened(self) -> bool:
        return self.is_open

    def read_frame(self) -> Image | None:
        if not self.is_opened():
            return None

        return self.camera.capture_array()

    def release(self) -> None:
        self.camera.stop()
        self.camera.close()
        self.is_open = False
