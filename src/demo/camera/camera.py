from abc import ABC, abstractmethod

from .image import Image


class Camera(ABC):
    @abstractmethod
    def __init__(
        self, cam_port: int | str, image_height: int, image_width: int
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def is_opened(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def read_frame(self) -> Image | None:
        """Get the next frame from the camera.

        INFO: Image is in RGB888 format

        Returns:
            image (Image): The next frame from the camera.
        """
        raise NotImplementedError()

    @abstractmethod
    def release(self) -> None:
        """Release the camera."""
        raise NotImplementedError()
