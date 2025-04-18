import base64
from collections.abc import Callable
from dataclasses import fields

from nicegui import run

from src.demo_interface.camera import Camera
from src.demo_interface.image_processing import AbstractImageProcessor, AnomalyResult

type UpdateUICallback = Callable[[dict[str, str] | str], None]


class DemoApplicationController:
    def __init__(self, camera: Camera, placeholder_image: bytes) -> None:
        self._camera = camera
        self._placeholder_image = placeholder_image
        self._image_processor: AbstractImageProcessor | None = None
        self._update_ui_callback: UpdateUICallback = lambda _: None

    async def run(self) -> None:
        while True:
            processed_frame = await self._take_and_process_frame()
            ui_data = self._frame_to_ui_data(processed_frame)
            self._update_ui_callback(ui_data)

    def set_update_ui_callback(self, callback: UpdateUICallback) -> None:
        self._update_ui_callback = callback

    def set_image_processor(self, image_processor: AbstractImageProcessor) -> None:
        self._image_processor = image_processor

    async def _take_and_process_frame(self) -> AnomalyResult | bytes | None:
        if self._image_processor is None:
            return None
        frame = await _cpu_bound(self._camera.read_frame)
        frame = await _cpu_bound(self._image_processor.process_image, frame)
        return frame

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


async def _cpu_bound[**P, R](
    func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
) -> R:
    return await run.cpu_bound(func, *args, **kwargs)


def _bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")
