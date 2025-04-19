import base64
from collections.abc import Callable
from dataclasses import fields

from nicegui import run

from src.demo_interface.camera import Camera
from src.demo_interface.image_processing import AnomalyResult, Image, ImageProcessor

type UpdateUICallback = Callable[[dict[str, str] | str], None]


class _NoneImageProcessor:
    def process(self, image: Image) -> None:
        return None


class DemoApplicationController:
    def __init__(self, cam_port: int | str, placeholder_image: bytes) -> None:
        self._camera = Camera(cam_port, width=1920, height=1080)
        self._placeholder_image = placeholder_image
        self._image_processor: ImageProcessor = _NoneImageProcessor()
        self._update_ui_callback: UpdateUICallback = lambda _: None

    async def run(self) -> None:
        while True:
            processed_frame = await self._take_and_process_frame()
            ui_data = self._frame_to_ui_data(processed_frame)
            self._update_ui_callback(ui_data)

    def set_update_ui_callback(self, callback: UpdateUICallback) -> None:
        self._update_ui_callback = callback

    def set_image_processor(self, image_processor: ImageProcessor) -> None:
        self._image_processor = image_processor

    def close(self) -> None:
        self._camera.release()

    async def _take_and_process_frame(self) -> AnomalyResult | bytes | None:
        frame = await run.io_bound(self._camera.read_frame)
        return await run.cpu_bound(self._image_processor.process, frame)

    def _frame_to_ui_data(
        self, frame: AnomalyResult | bytes | None
    ) -> dict[str, str] | str:
        if frame is None:
            return _bytes_to_base64(self._placeholder_image)
        elif isinstance(frame, bytes):
            return _bytes_to_base64(frame)
        else:
            return {
                field.name: _bytes_to_base64(getattr(frame, field.name))
                for field in fields(frame)  # type: ignore
            }


def _bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")
